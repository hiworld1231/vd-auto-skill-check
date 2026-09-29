import json
import time

import numpy as np
import pytest

from vd.recording import Recorder, read_recording


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


def test_events_only_recording_skips_frame_copy_and_video_encoder(tmp_path, monkeypatch):
    class Frame:
        sequence = 1
        media_time = 1.0
        received_time = 1.0
        consumed_time = 1.0
        timestamp_kind = "test"
        pts_ns = 1
        negotiated_caps = "test"
        synthetic = False

        @property
        def image(self):
            raise AssertionError("events-only recording must not copy frame pixels")

    monkeypatch.setattr("vd.recording._ffmpeg_encoder",
                        lambda: pytest.fail("events-only mode must not probe FFmpeg"))
    recorder = Recorder(tmp_path / "events-only",
                        metadata={"requested_fps": 60}, record_video=False)
    assert recorder.submit(Frame(), decision_time=1.1, held=True, state={}, events=[])
    recorder.event({"kind": "KEYDOWN", "generation": 1})
    recorder.close()

    manifest = json.loads((tmp_path / "events-only" / "manifest.json").read_text())
    events = [json.loads(line) for line in
              (tmp_path / "events-only" / "events.jsonl").read_text().splitlines()]
    assert manifest["video_recorded"] is False
    assert manifest["frames_written"] == 0
    assert manifest["events_written"] == 1
    assert events == [{"kind": "KEYDOWN", "generation": 1}]
    with pytest.raises(ValueError, match="events-only recording"):
        list(read_recording(tmp_path / "events-only"))
