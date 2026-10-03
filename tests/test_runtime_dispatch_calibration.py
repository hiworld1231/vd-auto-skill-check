import threading
import time
from types import SimpleNamespace

import numpy as np

from vd import input as vd_input
from vd.capture import Frame
from vd.runtime import run_session
from vd.vision import Measurement


class _Process:
    @staticmethod
    def poll():
        return None


class _Capture:
    def __init__(self, **_):
        self.cv = threading.Condition()
        self.proc = _Process()
        self.stopping = False
        self.error = None
        self.latest = None
        self.sequence = -1
        self.roi = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stopping = True

    def next(self, after=-1, timeout=1.0):
        time.sleep(min(.005, timeout))
        now = time.monotonic()
        self.sequence = after + 1
        self.latest = Frame(
            self.sequence,
            np.zeros((240, 320, 3), dtype=np.uint8),
            now, now, now, 'test_clock', False, None, 'test_caps',
        )
        return self.latest


class _HeldMouse:
    def __init__(self):
        self.lock = threading.RLock()
        self.state = vd_input.MouseState(True, True, 'fake_mouse', None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def snapshot(self):
        return self.state


class _Output:
    error = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _BlankDetector:
    def __init__(self, *, roi_offset=(0, 0), ui_scale=1.0):
        self.roi_offset = roi_offset
        self.ui_scale = ui_scale

    def measure(self, image, timestamp, *, center_hint=None):
        return Measurement(timestamp, None, 0, None, None, (), 'NO_PROMPT')


def test_runtime_learns_scheduler_late_dispatch_when_plan_deadline_was_clean(monkeypatch, tmp_path):
    observed = []

    class SpyEstimator:
        def __init__(self, lead, uncertainty):
            self.lead = lead
            self.uncertainty = uncertainty
            self.samples = []
            self.reason = 'INITIAL_MODEL'

        def observe_dispatch(self, **kwargs):
            observed.append(kwargs)
            self.reason = 'SEEN'
            return False

    emitted = False

    def fake_dispatch(engine, capture, sequence, **_):
        nonlocal emitted
        if emitted:
            return None
        emitted = True
        intended = time.monotonic()
        completed = intended + .008
        engine.emit(
            'KEYDOWN', completed,
            requested_at=intended,
            plan={
                'generation': 1,
                'version': 1,
                'press_at': intended,
                'intended_press_at': intended,
                'valid_until': intended + .080,
                'target_phase': 400.0,
                'uncertainty_degrees': 2.0,
                'latest_press_at': intended + .004,
                'target_grade': 'GREAT',
                'target_window_start': 395.0,
                'target_window_width': 10.0,
                'timing_mode': 'PREDICTED',
            },
            frame_at=intended,
            prefire_motion=None,
            motion_diagnostics={'reason': 'MEASURED'},
            great={'start': 35.0, 'width': 10.0},
            good={'start': 46.0, 'width': 42.0},
            dispatch_lag_ms=8.0,
            outside_target_window=True,
        )
        return None

    monkeypatch.setattr('vd.runtime.PortalCapture', _Capture)
    monkeypatch.setattr('vd.runtime.Detector', _BlankDetector)
    monkeypatch.setattr('vd.runtime.LeadEstimator', SpyEstimator)
    monkeypatch.setattr('vd.runtime.dispatch', fake_dispatch)
    monkeypatch.setattr(vd_input, 'MouseMonitor', _HeldMouse)
    monkeypatch.setattr(vd_input, 'SpaceOutput', _Output)

    run_session(
        seconds=.03,
        synthetic=False,
        fps=60,
        directory=tmp_path / 'scheduler-tail',
        lead_seconds=.002,
        lead_uncertainty=.003,
        physical=True,
        learn_lead=True,
        recording=False,
        video_recording=False,
    )

    assert observed
    assert observed[0]['physical'] is True
    assert observed[0]['dispatch_lag'] >= .008
    assert observed[0]['eligible'] is True
