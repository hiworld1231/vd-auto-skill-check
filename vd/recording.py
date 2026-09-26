"""Compact video + exact decision timestamps. Writing never blocks the decision loop.

New recordings store one H.264/MP4 stream instead of thousands of PNG files.
The JSONL sidecar remains authoritative for capture/decision timing, so replay
uses video only as the image transport. Legacy vd-roi-v1 recordings remain
readable.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import queue
import shutil
import subprocess
import threading

import cv2


FORMAT='vd-video-v2'
LEGACY_FORMAT='vd-roi-v1'
VIDEO_NAME='frames.mp4'


def source_digest():
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for path in sorted([root/'run.py',*root.glob('vd/*.py'),*root.glob('native/*.py')]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ffmpeg_encoder():
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required for compact recording')
    try:
        probe=subprocess.run([ffmpeg,'-hide_banner','-encoders'],capture_output=True,
                             text=True,timeout=10,check=False)
        if 'libx264' in probe.stdout:
            return ffmpeg,'libx264'
    except (OSError,subprocess.TimeoutExpired):
        pass
    return ffmpeg,'mpeg4'


class Recorder:
    def __init__(self, directory, *, metadata, capacity=120):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.ffmpeg,self.video_codec=_ffmpeg_encoder()
        self.metadata=dict(format=FORMAT,source_sha256=source_digest(),
                           video=VIDEO_NAME,video_codec=self.video_codec,
                           video_lossless=False,**metadata)
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

    def _start_encoder(self,image):
        if image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('Recorder expects BGR images')
        height,width=image.shape[:2]
        fps=float(self.metadata.get('requested_fps') or 60)
        if not 1<=fps<=240:
            fps=60.0
        command=[self.ffmpeg,'-hide_banner','-loglevel','error','-y',
                 '-f','rawvideo','-pix_fmt','bgr24','-s:v',f'{width}x{height}',
                 '-r',f'{fps:g}','-i','pipe:0','-an']
        if self.video_codec=='libx264':
            command += ['-c:v','libx264','-preset','veryfast','-crf','18',
                        '-pix_fmt','yuv420p','-movflags','+faststart']
        else:
            command += ['-c:v','mpeg4','-q:v','3','-pix_fmt','yuv420p']
        command += [str(self.directory/VIDEO_NAME)]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        self.metadata['video_width']=width
        self.metadata['video_height']=height
        self.metadata['video_nominal_fps']=fps
        return proc,(height,width)

    def _write(self):
        proc=None
        shape=None
        try:
            with (self.directory/'frames.jsonl').open('x') as log, \
                    (self.directory/'events.jsonl').open('x') as events:
                while True:
                    item=self.queue.get()
                    if item is None:
                        break
                    image,row=item
                    if image is None:
                        events.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                        events.flush()
                        self.events_written+=1
                        continue
                    if proc is None:
                        proc,shape=self._start_encoder(image)
                    if image.shape[:2]!=shape:
                        raise ValueError(f'Frame size changed from {shape[::-1]} to {image.shape[1::-1]}')
                    try:
                        proc.stdin.write(image.tobytes(order='C'))
                    except (BrokenPipeError,OSError) as exc:
                        detail=proc.stderr.read().decode(errors='replace').strip() if proc.stderr else ''
                        raise OSError('Video encoder failed'+(': '+detail if detail else '')) from exc
                    row['video_frame']=self.written
                    log.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
                    self.written+=1
                if proc is not None:
                    proc.stdin.close()
                    stderr=proc.stderr.read().decode(errors='replace').strip() if proc.stderr else ''
                    code=proc.wait(timeout=30)
                    if code:
                        raise OSError(f'Video encoder exited {code}'+(': '+stderr if stderr else ''))
        except Exception as exc:
            self.error=str(exc)
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

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
        video_path=self.directory/VIDEO_NAME
        result=dict(self.metadata,frames_written=self.written,frames_dropped=self.dropped,
                    events_written=self.events_written,events_dropped=self.events_dropped,
                    video_bytes=video_path.stat().st_size if video_path.exists() else 0,
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


def _read_legacy(directory):
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


def _read_video(directory,manifest):
    video=(directory/manifest.get('video',VIDEO_NAME)).resolve()
    if not video.is_relative_to(directory):
        raise ValueError('Video outside recording directory')
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError('Missing or invalid recording video: '+str(video))
    try:
        with (directory/'frames.jsonl').open() as stream:
            expected=0
            for line in stream:
                row=json.loads(line)
                if row.get('video_frame')!=expected:
                    raise ValueError('Non-sequential video frame index')
                ok,image=capture.read()
                if not ok or image is None:
                    raise ValueError(f'Video ended before frame {expected}')
                yield row,image
                expected+=1
            ok,_=capture.read()
            if ok:
                raise ValueError('Video contains more frames than frames.jsonl')
    finally:
        capture.release()


def read_recording(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    fmt=manifest.get('format')
    if fmt==LEGACY_FORMAT:
        yield from _read_legacy(directory)
        return
    if fmt==FORMAT:
        yield from _read_video(directory,manifest)
        return
    raise ValueError('Unsupported recording format')


def summarize_recording(directory):
    """Return compact diagnostics without decoding images/video."""
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    reasons=Counter()
    motion_reasons=Counter()
    plan_grades=Counter()
    frames=0
    first=None
    last=None
    with (directory/'frames.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            frames+=1
            at=row.get('decision_time')
            if isinstance(at,(int,float)):
                first=at if first is None else first
                last=at
            state=row.get('state') or {}
            reasons[state.get('reason') or 'UNKNOWN']+=1
            motion_reasons[state.get('motion_reason') or 'UNKNOWN']+=1
            plan=state.get('plan')
            if plan:
                plan_grades[plan.get('target_grade') or 'UNKNOWN']+=1
    event_kinds=Counter()
    end_reasons=Counter()
    events_path=directory/'events.jsonl'
    if events_path.exists():
        with events_path.open() as stream:
            for line in stream:
                event=json.loads(line)
                event_kinds[event.get('kind') or 'UNKNOWN']+=1
                if event.get('kind')=='END':
                    end_reasons[event.get('reason') or 'UNKNOWN']+=1
    total_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    return dict(format=manifest.get('format'),frames=frames,
                duration_seconds=(last-first if first is not None and last is not None else None),
                size_bytes=total_bytes,reasons=dict(reasons),
                motion_reasons=dict(motion_reasons),plan_grades=dict(plan_grades),
                event_kinds=dict(event_kinds),end_reasons=dict(end_reasons),
                frames_dropped=manifest.get('frames_dropped'),
                capture_sequences_skipped=manifest.get('capture_sequences_skipped'),
                complete=manifest.get('complete'))
