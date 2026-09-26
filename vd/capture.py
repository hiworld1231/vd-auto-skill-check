from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import subprocess
import threading
import time
from collections import deque

import numpy as np


@dataclass(frozen=True)
class Frame:
    sequence: int
    image: np.ndarray
    media_time: float | None
    received_time: float
    consumed_time: float
    timestamp_kind: str
    synthetic: bool
    pts_ns: int | None
    negotiated_caps: str


class PortalCapture:
    """Own one worker and one replaceable frame; never build a frame queue."""
    def __init__(self, roi=(800, 420, 320, 240), *, synthetic=False, fps=60):
        if type(fps) is not int or not 1 <= fps <= 240:
            raise ValueError('FPS must be an integer in [1, 240]')
        self.roi = tuple(roi)
        if len(self.roi) != 4 or any(type(x) is not int for x in self.roi):
            raise ValueError('ROI must contain four integers')
        if min(self.roi[:2]) < 0 or min(self.roi[2:]) <= 0:
            raise ValueError('Invalid ROI')
        self.cv = threading.Condition()
        self.latest = None
        self.error = None
        self.stopping = False
        self.stderr = deque(maxlen=15)
        worker = Path(__file__).resolve().parents[1] / 'native' / 'pipewire_worker.py'
        command = ['/usr/bin/python', str(worker), '--roi', ','.join(map(str, self.roi)),
                   '--fps', str(fps)]
        if synthetic:
            command.append('--synthetic')
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.reader = threading.Thread(target=self._read, name='VD-frame-reader', daemon=True)
        self.errors = threading.Thread(target=self._stderr, name='VD-capture-log', daemon=True)
        try:
            self.reader.start()
            self.errors.start()
        except BaseException:
            self.close()
            raise

    def _stderr(self):
        for line in iter(self.proc.stderr.readline, b''):
            self.stderr.append(line.decode(errors='replace').rstrip())

    def _exact(self, length):
        result = bytearray()
        while len(result) < length:
            data = self.proc.stdout.read(length - len(result))
            if not data:
                raise EOFError('Capture worker closed its stream')
            result.extend(data)
        return result

    def _read(self):
        try:
            while not self.stopping:
                size, = struct.unpack('!I', self._exact(4))
                if size > 16384:
                    raise ValueError('Invalid frame header size')
                h = json.loads(self._exact(size))
                w, height = self.roi[2:]
                if (h['width'], h['height'], h['stride'], h['bytes']) != (w, height, w*3, w*height*3):
                    raise ValueError('Invalid capture frame dimensions')
                image = np.frombuffer(self._exact(h['bytes']), np.uint8).reshape(height, w, 3)
                image.flags.writeable = False
                media = h['media_monotonic_ns']
                frame = Frame(int(h['seq']), image,
                    None if media is None else media / 1e9,
                    h['received_ns'] / 1e9, time.monotonic(),
                    h['timestamp_kind'], bool(h['synthetic']),
                    h['pts_ns'], h['negotiated_caps'])
                with self.cv:
                    self.latest = frame
                    self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                if not self.stopping:
                    self.error = str(exc)
                self.cv.notify_all()

    def next(self, after=-1, timeout=1.0):
        until = time.monotonic() + timeout
        with self.cv:
            while not self.stopping:
                if self.error:
                    raise RuntimeError(self.error + ': ' + ' | '.join(self.stderr))
                if self.latest is not None and self.latest.sequence > after:
                    return self.latest
                remaining = until - time.monotonic()
                if remaining <= 0:
                    return None
                self.cv.wait(remaining)
        return None

    def close(self):
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        if self.proc.poll() is None:
            # SIGINT allows the worker to close its portal session explicitly.
            import signal
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        for thread in (self.reader,self.errors):
            if thread.ident is not None:
                thread.join(timeout=2)
        self.proc.stdout.close()
        self.proc.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
