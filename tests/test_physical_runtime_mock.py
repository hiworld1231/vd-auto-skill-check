import threading
import time
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vd import input as vd_input
from vd.capture import Frame, default_roi
from vd.runtime import replay, run_session
from vd.dispatch import dispatch
from vd.engine import Engine
from vd.motion import Motion
from vd.vision import Arc, Measurement, Needle


class FakeProcess:
    @staticmethod
    def poll():
        return None


class FakeCapture:
    def __init__(self, *, synthetic, fps, priority=5, source='monitor',
                 normalize_window_scale=False):
        assert synthetic is False
        assert source in ('monitor', 'window', 'region')
        self.priority = priority
        self.roi = default_roi(source)
        self.cv = threading.Condition()
        self.proc = FakeProcess()
        self.stopping = False
        self.error = None
        self.latest = None
        self.sequence = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stopping = True

    def next(self, after=-1, timeout=1.0):
        time.sleep(.005)
        self.sequence = after + 1
        now = time.monotonic()
        frame = Frame(self.sequence, np.zeros((self.roi[3], self.roi[2], 3), dtype=np.uint8), now,
                      now, now, 'test_clock', False, None, 'test_caps',
                      source_size=(1920,1080))
        with self.cv:
            self.latest = frame
        return frame


class FakeMouse:
    def __init__(self):
        self.lock = threading.RLock()
        self.state = vd_input.MouseState(False, True, 'fake_mouse', None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def snapshot(self):
        return self.state


class FakeOutput:
    error = None

    def __init__(self):
        self.pulses = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def pulse(self):
        self.pulses += 1
        raise AssertionError('Blank test frames must never schedule physical input')


def test_run_mode_initializes_capture_mouse_and_output_without_physical_presses(monkeypatch, tmp_path):
    output = FakeOutput()
    monkeypatch.setattr('vd.runtime.PortalCapture', FakeCapture)
    monkeypatch.setattr(vd_input, 'MouseMonitor', FakeMouse)
    monkeypatch.setattr(vd_input, 'SpaceOutput', lambda: output)

    result = run_session(seconds=.06, synthetic=False, fps=60, directory=tmp_path / 'run',
                         lead_seconds=.06, lead_uncertainty=.015, physical=True)

    assert result['mode'] == 'run'
    assert result['frames'] > 0
    assert result['physical_presses'] == 0
    assert output.pulses == 0
    assert result['performance']['frame_processing_ms']['p50'] > 0
    assert result['performance']['frame_processing_ms']['p95'] >= result['performance']['frame_processing_ms']['p50']
    assert result['performance']['frame_age_ms']['p50'] >= 0
    assert result['performance']['capture_pipe_age_ms']['samples'] > 0
    assert result['performance']['frame_delivery_gap_ms']['samples'] == result['frames'] - 1
    assert result['performance']['frame_delivery_gap_ms']['p95'] > 0
    assert 0 <= result['performance']['main_cpu_percent']
    assert result['performance']['elapsed_seconds'] > 0


def test_runtime_passes_selected_capture_source_to_portal(monkeypatch, tmp_path):
    selected=[]

    class SourceCapture(FakeCapture):
        def __init__(self, *, synthetic, fps, priority=5, source='monitor',
                     normalize_window_scale=False):
            selected.append((source, normalize_window_scale))
            super().__init__(synthetic=synthetic,fps=fps,priority=priority,source=source)

    monkeypatch.setattr('vd.runtime.PortalCapture', SourceCapture)
    run_session(seconds=.025,synthetic=False,fps=60,directory=tmp_path/'window-source',
                lead_seconds=.035,lead_uncertainty=.020,recording=False,
                capture_source='window', normalize_window_scale=True)

    assert selected==[('window', True)]


def test_runtime_reports_capture_worker_cpu_share(monkeypatch, tmp_path):
    class CpuCapture:
        def __init__(self, **_):
            self.cv = threading.Condition()
            self.proc = FakeProcess()
            self.stopping = False
            self.error = None
            self.latest = None
            self.cpu_time_ns = 0
            self.last_received = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stopping = True

        def next(self, after=-1, timeout=1.0):
            time.sleep(.005)
            received = time.monotonic()
            if self.last_received is not None:
                self.cpu_time_ns += int((received - self.last_received) * 500_000_000)
            self.last_received = received
            self.latest = SimpleNamespace(sequence=after + 1,
                image=np.zeros((240, 320, 3), dtype=np.uint8),
                media_time=received, received_time=received, consumed_time=received,
                timestamp_kind='test_clock', synthetic=False, pts_ns=None,
                    negotiated_caps='test_caps',worker_cpu_time_ns=self.cpu_time_ns)
            return self.latest

    class BlankDetector:
        def __init__(self, *, roi_offset=(0, 0)):
            self.roi_offset=roi_offset

        def measure(self, image, timestamp, *, center_hint=None):
            return Measurement(timestamp,None,0,None,None,(),'NO_PROMPT')

    monkeypatch.setattr('vd.runtime.PortalCapture', CpuCapture)
    monkeypatch.setattr('vd.runtime.Detector', BlankDetector)
    result=run_session(seconds=.05,synthetic=False,fps=60,directory=tmp_path/'cpu',
                       lead_seconds=.035,lead_uncertainty=.020,recording=False,
                       video_recording=False)

    cpu=result['performance']['capture_worker_cpu_percent']
    assert cpu['samples']==result['frames']-1
    assert 48<=cpu['p50']<=52
    assert result['performance']['lifetime_max']['capture_worker_cpu_percent']>=cpu['max']


@pytest.mark.parametrize(('speed', 'fps'), ((700, 60), (1300, 60), (700, 30), (1300, 30)))
def test_runtime_timer_dispatch_hits_white_between_capture_frames(monkeypatch, tmp_path, speed, fps):
    class TimerCapture:
        def __init__(self, *, synthetic, fps, priority=5, source='monitor',
                     normalize_window_scale=False):
            assert synthetic is False
            assert source in ('monitor', 'window')
            self.fps = fps
            self.origin = time.monotonic() + .002
            self.next_at = self.origin
            self.sequence = 0
            self.cv = threading.Condition()
            self.proc = FakeProcess()
            self.stopping = False
            self.error = None
            self.latest = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stopping = True

        def next(self, after=-1, timeout=1.0):
            deadline = time.monotonic() + timeout
            target = min(self.next_at, deadline)
            if target > time.monotonic():
                time.sleep(target - time.monotonic())
            received = time.monotonic()
            if self.next_at > deadline:
                return None
            media = self.next_at
            self.next_at += 1 / self.fps
            frame = Frame(self.sequence, np.zeros((240, 320, 3), dtype=np.uint8),
                          media, received, received, 'test_clock', False,
                          None, 'test_caps')
            self.sequence += 1
            with self.cv:
                self.latest = frame
            return frame

    class HeldMouse(FakeMouse):
        def __init__(self):
            self.lock = threading.RLock()
            self.state = vd_input.MouseState(True, True, 'fake_mouse', None)

    class RecordingOutput:
        error = None

        def __init__(self):
            self.keydowns = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def pulse(self):
            at = time.monotonic()
            self.keydowns.append(at)
            return vd_input.Keydown(at, at)

    capture = None
    output = RecordingOutput()

    def create_capture(**kwargs):
        nonlocal capture
        capture = TimerCapture(**kwargs)
        return capture

    class SpinnerDetector:
        def __init__(self, *, roi_offset=(0, 0)):
            self.roi_offset=roi_offset

        def measure(self, image, timestamp, *, center_hint=None):
            angle = (270 + speed * (timestamp - capture.origin)) % 360
            return Measurement(timestamp, (160, 162.5), .99, Arc(40, 10),
                               Arc(51, 42),
                               (Needle(angle, 100, 1, 50, 2),), 'OK')

    monkeypatch.setattr('vd.runtime.PortalCapture', create_capture)
    monkeypatch.setattr('vd.runtime.Detector', SpinnerDetector)
    monkeypatch.setattr(vd_input, 'MouseMonitor', HeldMouse)
    monkeypatch.setattr(vd_input, 'SpaceOutput', lambda: output)
    result = run_session(seconds=.35, synthetic=False, fps=fps,
                         directory=tmp_path / f'timer-{speed}-{fps}',
                         lead_seconds=.035, lead_uncertainty=.020,
                         physical=True, learn_lead=False, recording=False,
                         video_recording=False)

    assert result['physical_presses'] == 1
    assert len(output.keydowns) == 1
    phase = (270 + speed * (output.keydowns[0] - capture.origin + .035)) % 360
    assert Arc(40, 10).contains(phase)
    frame_phase = (output.keydowns[0] - capture.origin) * fps
    assert abs(frame_phase - round(frame_phase)) > .05


def test_no_recording_skips_video_writer_and_still_reports_performance(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr('vd.runtime.PortalCapture', FakeCapture)

    result = run_session(seconds=.03, synthetic=False, fps=60, directory=tmp_path / 'unused',
                         lead_seconds=.06, lead_uncertainty=.015, recording=False)

    assert result['recording'] is None
    assert not (tmp_path / 'unused').exists()
    diagnostic = json.loads(capsys.readouterr().out.strip())
    assert diagnostic['kind'] == 'PERFORMANCE'
    assert diagnostic['frame_processing_ms']['samples'] > 0
    assert diagnostic['frame_delivery_gap_ms']['samples'] > 0


def test_run_session_applies_and_records_capture_profile(monkeypatch, tmp_path):
    seen = {}

    def fake_capture(**kwargs):
        seen.update(kwargs)
        return FakeCapture(**kwargs)

    monkeypatch.setattr('vd.runtime.PortalCapture', fake_capture)
    directory = tmp_path / 'quiet-profile'
    run_session(seconds=.025, synthetic=False, fps=30, directory=directory,
                lead_seconds=.06, lead_uncertainty=.015, recording=True,
                capture_priority=10, variant='deep-quiet',capture_source='window',
                normalize_window_scale=True)

    manifest = json.loads((directory / 'manifest.json').read_text())
    assert seen == {'synthetic': False, 'fps': 30, 'priority': 10,'source':'window',
                    'normalize_window_scale': True}
    assert manifest['requested_fps'] == 30
    assert manifest['capture_priority'] == 10
    assert manifest['variant'] == 'deep-quiet'
    assert manifest['capture_source'] == 'window'
    assert manifest['normalize_window_scale'] is True
    assert manifest['capture_source_size'] == [1920,1080]
    assert manifest['capture_frame_size'] == [320,240]
    assert manifest['capture_roi'] == [800,420,320,240]


def test_region_runtime_records_the_compact_roi_and_detector_offset(monkeypatch, tmp_path):
    monkeypatch.setattr('vd.runtime.PortalCapture', FakeCapture)
    directory = tmp_path / 'region'
    run_session(seconds=.025, synthetic=False, fps=60, directory=directory,
                lead_seconds=.035, lead_uncertainty=.020, recording=True,
                capture_source='region', video_recording=False)

    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['capture_roi'] == [875,418,180,250]
    assert manifest['capture_frame_size'] == [180,250]
    assert manifest['detector_roi_offset'] == [75,-2]


def test_region_runtime_scales_detector_for_1600px_source(monkeypatch, tmp_path):
    class LowResolutionCapture(FakeCapture):
        def next(self, after=-1, timeout=1.0):
            return replace(super().next(after, timeout), source_size=(1600,900))

    monkeypatch.setattr('vd.runtime.PortalCapture', LowResolutionCapture)
    directory = tmp_path / 'region-1600'
    run_session(seconds=.025, synthetic=False, fps=60, directory=directory,
                lead_seconds=.035, lead_uncertainty=.020, recording=True,
                capture_source='region', video_recording=False)

    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['capture_source_size'] == [1600,900]
    assert manifest['detector_ui_scale'] == 1.2


def test_replay_restores_the_recorded_region_detector_offset(monkeypatch, tmp_path):
    seen = {}

    class OffsetDetector:
        def __init__(self, *, roi_offset, ui_scale=1.0):
            seen['roi_offset'] = roi_offset
            seen['ui_scale'] = ui_scale

        def measure(self, image, timestamp, *, center_hint=None):
            return Measurement(timestamp, None, 0, None, None, (), 'NO_PROMPT')

    recording = tmp_path / 'region-replay'
    recording.mkdir()
    (recording / 'manifest.json').write_text(json.dumps({
        'lead_seconds': .035, 'lead_uncertainty': .020,
        'learn_lead': False, 'detector_roi_offset': [80, 0],
        'detector_ui_scale': 1.2,
    }))
    row = {'sequence': 0, 'decision_time': 1.0, 'held': True, 'media_time': 1.0}
    monkeypatch.setattr('vd.runtime.Detector', OffsetDetector)
    monkeypatch.setattr('vd.runtime.read_recording',
                        lambda _: iter(((row, np.zeros((240,180,3), np.uint8)),)))

    replay(recording)

    assert seen['roi_offset'] == (80, 0)
    assert seen['ui_scale'] == 1.2


def test_events_only_run_does_not_submit_video_frames(monkeypatch, tmp_path):
    monkeypatch.setattr('vd.runtime.PortalCapture', FakeCapture)
    directory = tmp_path / 'events-only'
    run_session(seconds=.025, synthetic=False, fps=60, directory=directory,
                lead_seconds=.06, lead_uncertainty=.015, recording=True,
                video_recording=False)

    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['video_recorded'] is False
    assert manifest['video_sampling'] == 'disabled'
    assert manifest['frames_written'] == 0
    assert manifest['events_written'] == 0
    assert manifest['measurement_reasons']['NO_PROMPT'] > 0


def test_run_session_without_seconds_keeps_running_until_interrupted(monkeypatch, tmp_path):
    class InterruptingCapture(FakeCapture):
        def next(self, after=-1, timeout=1.0):
            if after >= 3:
                raise KeyboardInterrupt
            return super().next(after, timeout)

    monkeypatch.setattr('vd.runtime.PortalCapture', InterruptingCapture)
    recording = tmp_path / 'run-until-stopped'
    with pytest.raises(KeyboardInterrupt):
        run_session(seconds=None, synthetic=False, fps=60, directory=recording,
                    lead_seconds=.06, lead_uncertainty=.015, physical=False)

    manifest = json.loads((recording / 'manifest.json').read_text())
    assert manifest['frames_written'] == 1
    assert manifest['video_sampling'] == 'full_rate_during_check_1fps_while_idle'
    assert manifest['runtime_error'] == 'KeyboardInterrupt'
    assert manifest['complete'] is False
    assert manifest['performance']['frame_processing_ms']['samples'] <= 600
    assert manifest['performance']['frame_delivery_gap_ms']['samples'] > 0
    assert manifest['performance']['profile_window_frames'] == 600


class DispatchCapture:
    def __init__(self, *, media_time=1.0, sequence=7, stopping=False, error=None,
                 process_alive=True):
        self.cv = threading.Condition()
        self.latest = Frame(sequence, np.zeros((240, 320, 3), dtype=np.uint8),
                            media_time, media_time, media_time, 'test_clock', False,
                            None, 'test_caps')
        self.sequence = sequence
        self.stopping = stopping
        self.error = error
        self.proc = SimpleNamespace(poll=lambda: None if process_alive else 1)


class DispatchMouse:
    def __init__(self, *, held=True, healthy=True):
        self.lock = threading.RLock()
        self.state = vd_input.MouseState(held, healthy, 'fake_mouse', None)

    def snapshot(self):
        return self.state


class DispatchOutput:
    error = None

    def __init__(self):
        self.pulses = 0

    def pulse(self):
        self.pulses += 1
        return SimpleNamespace(requested_at=1.0, syn_completed_at=1.0)


def planned_engine():
    engine = Engine(lead_seconds=.1, lead_uncertainty=0)
    engine.planner.begin(Arc(96, 8), 60)
    motion = Motion(1.0, 70, 300, 0, 0, 1.0, 6)
    assert engine.planner.update(motion, frame_at=1.0, now=1.0) is not None
    return engine


def test_dispatch_sends_one_pulse_only_for_fresh_capture_and_held_mouse():
    engine = planned_engine()
    output = DispatchOutput()
    result = dispatch(engine, DispatchCapture(), 7, mouse=DispatchMouse(held=True),
                      output=output, clock=lambda: 1.0)
    assert result is not None
    assert output.pulses == 1
    assert [event['kind'] for event in engine.take_events()] == ['PRESS_CLAIM', 'KEYDOWN']


def test_dispatch_never_pulses_when_mouse_is_released_or_frame_is_stale():
    engine = planned_engine()
    output = DispatchOutput()
    result = dispatch(engine, DispatchCapture(), 7, mouse=DispatchMouse(held=False),
                      output=output, clock=lambda: 1.0)
    assert result is None
    assert output.pulses == 0

    engine = planned_engine()
    result = dispatch(engine, DispatchCapture(media_time=.9), 7,
                      mouse=DispatchMouse(held=True), output=output, clock=lambda: 1.0)
    assert result is None
    assert output.pulses == 0


def test_dispatch_never_pulses_if_capture_or_mouse_source_is_unhealthy():
    output = DispatchOutput()
    for capture, mouse in (
        (DispatchCapture(stopping=True), DispatchMouse(held=True)),
        (DispatchCapture(error='capture failed'), DispatchMouse(held=True)),
        (DispatchCapture(process_alive=False), DispatchMouse(held=True)),
        (DispatchCapture(), DispatchMouse(held=True, healthy=False)),
    ):
        engine = planned_engine()
        result = dispatch(engine, capture, 7, mouse=mouse, output=output,
                          clock=lambda: 1.0)
        assert result is None
    assert output.pulses == 0


def test_dispatch_sends_blind_attempt_after_startup_grace_and_marks_timing_unknown():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)
    for at in (1.0,1.02):
        engine.observe(Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(),
                                   'NO_LINE_CANDIDATE'),now=at)
    early_capture=DispatchCapture(media_time=1.02,sequence=7)

    class Output:
        error=None
        def pulse(self):
            self.pulses+=1
            return SimpleNamespace(requested_at=1.321,syn_completed_at=1.322)

        def __init__(self):
            self.pulses=0

    output=Output()
    early=dispatch(engine,early_capture,7,mouse=DispatchMouse(held=True),output=output,
                   clock=lambda:1.02)
    assert early is None
    assert output.pulses==0

    for at in (1.08,1.14,1.20,1.26,1.30):
        engine.observe(Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(),
                                   'NO_LINE_CANDIDATE'),now=at)
    at=1.321
    engine.observe(Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(),
                               'NO_LINE_CANDIDATE'),now=at)
    capture=DispatchCapture(media_time=at,sequence=8)
    result=dispatch(engine,capture,8,mouse=DispatchMouse(held=True),output=output,
                    clock=lambda:at)
    events=engine.take_events()
    keydown=next(event for event in events if event['kind']=='KEYDOWN')

    assert result is not None
    assert result.timing_mode=='BLIND_NO_NEEDLE'
    assert output.pulses==1
    assert keydown['outside_target_window'] is None
    assert keydown['motion_diagnostics']['reason']=='NO_NEEDLE'
    assert keydown['motion_diagnostics']['samples']==0


def test_released_mouse_skips_expensive_detector(monkeypatch, tmp_path):
    monkeypatch.setattr('vd.runtime.PortalCapture', FakeCapture)
    monkeypatch.setattr(vd_input, 'MouseMonitor', FakeMouse)
    monkeypatch.setattr(vd_input, 'SpaceOutput', FakeOutput)
    def forbidden_measure(*args, **kwargs):
        raise AssertionError('CV must sleep while LMB is released')
    monkeypatch.setattr('vd.runtime.Detector.measure', forbidden_measure)
    result=run_session(seconds=.03,synthetic=False,fps=60,directory=tmp_path/'idle',
                       lead_seconds=.06,lead_uncertainty=.015,physical=True,
                       recording=False)
    assert result['frames']>0


def test_released_mouse_records_only_a_sparse_idle_sample(monkeypatch, tmp_path):
    monkeypatch.setattr('vd.runtime.PortalCapture', FakeCapture)
    monkeypatch.setattr(vd_input, 'MouseMonitor', FakeMouse)
    monkeypatch.setattr(vd_input, 'SpaceOutput', FakeOutput)
    directory=tmp_path/'idle-recording'
    result=run_session(seconds=.06,synthetic=False,fps=60,directory=directory,
                       lead_seconds=.06,lead_uncertainty=.015,physical=True)
    manifest=json.loads((directory/'manifest.json').read_text())
    assert result['frames']>4
    assert manifest['frames_written']==1
