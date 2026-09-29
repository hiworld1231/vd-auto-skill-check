import threading
import time

import numpy as np

from vd.capture import Frame
from vd.runtime import run_session
from vd.vision import Measurement


class _Proc:
    @staticmethod
    def poll():
        return None


class _ScaledCapture:
    def __init__(self, **_):
        self.ui_scale = 1.25
        self.roi = (800, 420, 320, 240)
        self.cv = threading.Condition()
        self.proc = _Proc()
        self.stopping = False
        self.error = None
        self.latest = None
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stopping = True

    def next(self, after=-1, timeout=1.0):
        if self.sent:
            time.sleep(min(timeout, 0.001))
            return None
        self.sent = True
        now = time.monotonic()
        frame = Frame(
            0, np.zeros((240, 320, 3), dtype=np.uint8), now, now, now,
            "test", False, None, "vulkan-test", source_size=(1600, 878),
        )
        with self.cv:
            self.latest = frame
        return frame


def test_runtime_passes_capture_ui_scale_to_detector(monkeypatch, tmp_path):
    seen = []

    class _Detector:
        def __init__(self, *, roi_offset=(0, 0), ui_scale):
            self.roi_offset = roi_offset
            seen.append(ui_scale)

        def measure(self, image, timestamp, *, center_hint=None):
            return Measurement(timestamp, None, 0.0, None, None, (), "NO_PROMPT")

    monkeypatch.setattr("vd.runtime.PortalCapture", _ScaledCapture)
    monkeypatch.setattr("vd.runtime.Detector", _Detector)

    run_session(
        seconds=0.01, synthetic=False, fps=60, directory=tmp_path / "runtime-scale",
        lead_seconds=0.035, lead_uncertainty=0.020, recording=False,
        capture_source="vulkan", video_recording=False,
    )

    assert seen == [1.25]
