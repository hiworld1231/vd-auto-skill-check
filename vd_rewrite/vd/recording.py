"""Lossless ROI + timestamps. Bounded writing never stalls the decision loop."""
import hashlib
import json
from pathlib import Path
import queue
import threading

import cv2


FORMAT='vd-roi-v1'


def source_digest():
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for path in sorted([root/'run.py',*root.glob('vd/*.py'),*root.glob('native/*.py')]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Recorder:
    def __init__(self, directory, *, metadata, capacity=120):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        (self.directory/'frames').mkdir()
        self.metadata=dict(format=FORMAT,source_sha256=source_digest(),**metadata)
        self.queue=queue.Queue(maxsize=capacity)
        self.dropped=0
        self.written=0
        self.events_dropped=0
        self.events_written=0
        self.error=None
        self.closed=False
        self.thread=threading.Thread(target=self._write,name='VD-recording',daemon=True)
        self.thread.start()

    def submit(self, frame, *, decision_time, held, state, events):
        if self.closed:
            raise RuntimeError('Recorder closed')
        if self.error:
            raise RuntimeError('Recorder failed: '+self.error)
        row=dict(sequence=frame.sequence,media_time=frame.media_time,
                 received_time=frame.received_time,consumed_time=frame.consumed_time,
                 timestamp_kind=frame.timestamp_kind,pts_ns=frame.pts_ns,
                 negotiated_caps=frame.negotiated_caps,synthetic=frame.synthetic,
                 decision_time=decision_time,held=held,state=state,events=events)
        try:
            self.queue.put_nowait((frame.image.copy(),row))
        except queue.Full:
            self.dropped+=1
            return False
        return True

    def event(self, event):
        if self.closed:
            raise RuntimeError('Recorder closed')
        if self.error:
            raise RuntimeError('Recorder failed: '+self.error)
        try:
            self.queue.put_nowait((None,dict(event)))
        except queue.Full:
            self.events_dropped+=1

    def _write(self):
        try:
            with (self.directory/'frames.jsonl').open('x') as log, \
                    (self.directory/'events.jsonl').open('x') as events:
                while True:
                    item=self.queue.get()
                    if item is None:
                        break
                    image,row=item
                    if image is None:
                        events.write(json.dumps(row,allow_nan=False)+'\n')
                        events.flush()
                        self.events_written+=1
                        continue
                    filename=f'frames/{self.written:08d}.png'
                    if not cv2.imwrite(str(self.directory/filename),image,
                                       [cv2.IMWRITE_PNG_COMPRESSION,1]):
                        raise OSError('PNG encoder failed')
                    row['image']=filename
                    log.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                    self.written+=1
        except Exception as exc:
            self.error=str(exc)

    def close(self):
        if self.closed:
            return
        self.closed=True
        while self.thread.is_alive():
            try:
                self.queue.put(None,timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        writer_complete=self.error is None and self.dropped==0 and self.events_dropped==0
        result=dict(self.metadata,frames_written=self.written,frames_dropped=self.dropped,
                    events_written=self.events_written,events_dropped=self.events_dropped,
                    writer_error=self.error,
                    writer_complete=writer_complete,
                    complete=(writer_complete and not self.metadata.get('runtime_error')
                              and self.metadata.get('capture_sequences_skipped',0)==0))
        (self.directory/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        if self.error:
            raise RuntimeError('Recorder failed: '+self.error)

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()


def read_recording(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('format')!=FORMAT:
        raise ValueError('Unsupported recording format')
    with (directory/'frames.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            path=(directory/row['image']).resolve()
            if not path.is_relative_to(directory/'frames'):
                raise ValueError('Image outside recording directory')
            image=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Missing or invalid recording image: '+str(path))
            yield row,image
