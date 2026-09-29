import json
import threading
import time
from dataclasses import replace

import numpy as np

from vd.capture import Frame, default_roi
from vd.runtime import run_session


class FakeProcess:
    @staticmethod
    def poll():
        return None


class Region1600Capture:
    def __init__(self, *, synthetic, fps, priority=5, source='monitor',
                 normalize_window_scale=False):
        assert synthetic is False
        assert source == 'region'
        assert normalize_window_scale is False
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
        frame = Frame(
            self.sequence,
            np.zeros((self.roi[3], self.roi[2], 3), dtype=np.uint8),
            now, now, now, 'test_clock', False, None, 'test_caps',
            source_size=(1600, 900),
        )
        with self.cv:
            self.latest = frame
        return frame


def test_region_1600_capture_keeps_detector_at_normalized_canonical_scale(monkeypatch, tmp_path):
    monkeypatch.setattr('vd.runtime.PortalCapture', Region1600Capture)
    directory = tmp_path / 'region-1600'

    run_session(
        seconds=.025,
        synthetic=False,
        fps=60,
        directory=directory,
        lead_seconds=.035,
        lead_uncertainty=.020,
        recording=True,
        capture_source='region',
        video_recording=False,
    )

    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['capture_source_size'] == [1600, 900]
    assert manifest['capture_frame_size'] == [180, 250]
    assert manifest['detector_ui_scale'] == 1.0
