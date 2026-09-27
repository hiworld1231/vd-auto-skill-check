import threading
import time
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vd import input as vd_input
from vd.capture import Frame
from vd.runtime import run_session
from vd.dispatch import dispatch
from vd.engine import Engine
from vd.motion import Motion
from vd.vision import Arc, Measurement


class FakeProcess:
    @staticmethod
    def poll():
        return None


class FakeCapture:
    def __init__(self, *, synthetic, fps, priority=5):
        assert synthetic is False
        self.priority = priority
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
        frame = Frame(self.sequence, np.zeros((240, 320, 3), dtype=np.uint8), now,
                      now, now, 'test_clock', False, None, 'test_caps')
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
                capture_priority=10, variant='deep-quiet')

    manifest = json.loads((directory / 'manifest.json').read_text())
    assert seen == {'synthetic': False, 'fps': 30, 'priority': 10}
    assert manifest['requested_fps'] == 30
    assert manifest['capture_priority'] == 10
    assert manifest['variant'] == 'deep-quiet'


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


def test_dispatch_sends_blind_attempt_and_marks_timing_as_unknown():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)
    for at in (1.0,1.02):
        engine.observe(Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(),
                                   'NO_LINE_CANDIDATE'),now=at)
    capture=DispatchCapture(media_time=1.02,sequence=7)

    class Output:
        error=None
        def pulse(self):
            self.pulses+=1
            return SimpleNamespace(requested_at=1.02,syn_completed_at=1.021)

        def __init__(self):
            self.pulses=0

    output=Output()
    result=dispatch(engine,capture,7,mouse=DispatchMouse(held=True),output=output,
                    clock=lambda:1.02)
    events=engine.take_events()
    keydown=next(event for event in events if event['kind']=='KEYDOWN')

    assert result is not None
    assert result.timing_mode=='BLIND_NO_NEEDLE'
    assert output.pulses==1
    assert keydown['outside_target_window'] is None


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
