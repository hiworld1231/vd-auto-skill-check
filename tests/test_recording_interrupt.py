import json
import time

import numpy as np

from vd.recording import Recorder


def test_ctrl_c_marks_recording_partial_and_returns_quickly(tmp_path):
    recorder = Recorder(tmp_path / "interrupt", metadata={"requested_fps": 60}, capacity=4)
    frame = type("Frame", (), {"sequence": 0, "media_time": 0.0,
                               "received_time": 0.0, "consumed_time": 0.0,
                               "timestamp_kind": "test", "pts_ns": 0,
                               "negotiated_caps": "test", "synthetic": True,
                               "image": np.zeros((240, 320, 3), dtype=np.uint8)})()
    recorder.submit(frame, decision_time=0.0, held=False, state={}, events=[])

    started = time.monotonic()
    recorder.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)

    assert time.monotonic() - started < 1.0
    manifest = json.loads((tmp_path / "interrupt" / "manifest.json").read_text())
    assert manifest["interrupted"] is True
    assert manifest["complete"] is False


def test_event_evicts_video_when_recording_queue_is_full():
    import queue
    from vd.recording import Recorder
    recorder=Recorder.__new__(Recorder)
    recorder.closed=False
    recorder.error=None
    recorder.dropped=0
    recorder.events_dropped=0
    recorder.queue=queue.Queue(maxsize=1)
    recorder.queue.put_nowait((object(),{'sequence':1}))
    recorder.event({'kind':'KEYDOWN'})
    image,row=recorder.queue.get_nowait()
    assert image is None
    assert row['kind']=='KEYDOWN'
    assert recorder.dropped==1
    assert recorder.events_dropped==0
