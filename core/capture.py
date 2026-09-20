"""Low-latency screen capture for Linux/Wayland with explicit health state.

Preferred path:
    gpu-screen-recorder (KMS/VFR) -> ffmpeg decode -> freshest raw BGR frame

The worker never hides child-process death behind a permanently cached last frame.
Transport frame IDs are explicit so the main loop can skip timed-out duplicates.
"""

from __future__ import annotations

import array
import collections
import fcntl
import os
import shutil
import subprocess
import termios
import threading
import time
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

CAPTURE_REGION = {"left": 800, "top": 420, "width": 320, "height": 240}
DEFAULT_CAPTURE_FPS = 120
F_SETPIPE_SZ = 1031
F_GETPIPE_SZ = 1032


class CaptureError(RuntimeError):
    pass


class ScreenGrabber:
    def __init__(
        self,
        region: dict = None,
        fps: int = DEFAULT_CAPTURE_FPS,
        framerate_mode: str = "vfr",
        keyint: str = "2.0",
        allow_mss_fallback: bool = False,
    ):
        self.region = region or CAPTURE_REGION
        self.fps = int(fps)
        self.framerate_mode = str(framerate_mode).lower()
        self.keyint = str(keyint)
        self.allow_mss_fallback = bool(allow_mss_fallback)

        self.use_gsr = False
        self.gsr_process: Optional[subprocess.Popen] = None
        self.ff_process: Optional[subprocess.Popen] = None
        self.fd: Optional[int] = None
        self.sct = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_decode_ts = 0.0
        self._last_new_frame_mono = 0.0
        self._frame_id = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread: Optional[threading.Thread] = None
        self._stderr_threads = []
        self._stderr_tail: Deque[str] = collections.deque(maxlen=32)
        self._worker_failed = False
        self._last_error: Optional[str] = None

        self.input_pipe_size = 65536
        self.output_pipe_size = 65536
        self.discarded_frames_count = 0
        self.max_backlog_bytes = 0
        self.transport_duplicate_waits = 0
        self._last_mss_grab_duration_ms = 0.0

        have_gsr = bool(shutil.which("gpu-screen-recorder"))
        have_ffmpeg = bool(shutil.which("ffmpeg"))
        if have_gsr and have_ffmpeg:
            try:
                self._init_gsr()
                return
            except Exception as exc:
                self.close()
                if not self.allow_mss_fallback:
                    raise CaptureError(f"GPU capture failed and MSS fallback is disabled: {exc}") from exc
                print(f"[CAPTURE] gpu-screen-recorder error ({exc}), falling back to mss...")
                # Re-open logical state after close() for the fallback.
                self._running = True
                self._worker_failed = False
                self._last_error = None
        elif not self.allow_mss_fallback:
            missing = []
            if not have_gsr:
                missing.append("gpu-screen-recorder")
            if not have_ffmpeg:
                missing.append("ffmpeg")
            raise CaptureError("Required low-latency capture tools missing: " + ", ".join(missing))

        self._init_mss()

    def _stderr_reader(self, proc: subprocess.Popen, label: str) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            while self._running:
                line = stream.readline()
                if not line:
                    break
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                msg = f"{label}: {line.strip()}"
                if msg.strip():
                    self._stderr_tail.append(msg[-500:])
        except Exception:
            pass

    def _start_stderr_reader(self, proc: subprocess.Popen, label: str) -> None:
        th = threading.Thread(target=self._stderr_reader, args=(proc, label), daemon=True, name=f"CaptureStderr-{label}")
        th.start()
        self._stderr_threads.append(th)

    def _init_gsr(self) -> None:
        w, h = self.region["width"], self.region["height"]
        x, y = self.region["left"], self.region["top"]
        reg_str = f"{w}x{h}+{x}+{y}"

        gsr_cmd = [
            "gpu-screen-recorder",
            "-w", reg_str,
            "-f", str(self.fps),
            # -c selects the FFmpeg muxer.  h264 intentionally means raw
            # Annex-B H.264 for the decoder pipe; -k explicitly locks the
            # video codec instead of relying on GSR's auto default.
            "-c", "h264",
            "-k", "h264",
            "-fm", self.framerate_mode,
            "-tune", "performance",
            "-keyint", self.keyint,
            "-o", "/dev/stdout",
        ]
        self.gsr_process = subprocess.Popen(
            gsr_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._start_stderr_reader(self.gsr_process, "GSR")
        assert self.gsr_process.stdout is not None

        in_pipe_fd = self.gsr_process.stdout.fileno()
        try:
            fcntl.fcntl(in_pipe_fd, F_SETPIPE_SZ, 65536)
            self.input_pipe_size = fcntl.fcntl(in_pipe_fd, F_GETPIPE_SZ)
        except Exception as exc:
            try:
                self.input_pipe_size = fcntl.fcntl(in_pipe_fd, F_GETPIPE_SZ)
            except Exception:
                self.input_pipe_size = 65536
            print(f"[CAPTURE] GSR pipe size unchanged ({exc}); {self.input_pipe_size} bytes")

        ff_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-avioflags", "direct",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-threads", "1",
            "-f", "h264",
        ]
        # VFR is authoritative for GEN_RUSH.  Only legacy CFR explicitly asks
        # ffmpeg to synthesize fixed input timestamps.
        if str(self.framerate_mode).lower() == "cfr":
            ff_cmd += ["-r", str(self.fps)]
        ff_cmd += [
            "-i", "pipe:0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-fps_mode", "passthrough",
            "-",
        ]

        self.ff_process = subprocess.Popen(
            ff_cmd,
            stdin=self.gsr_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=262144,
            start_new_session=True,
        )
        self._start_stderr_reader(self.ff_process, "FFMPEG")
        self.gsr_process.stdout.close()
        assert self.ff_process.stdout is not None

        self.fd = self.ff_process.stdout.fileno()
        os.set_blocking(self.fd, False)
        try:
            fcntl.fcntl(self.fd, F_SETPIPE_SZ, 262144)
            self.output_pipe_size = fcntl.fcntl(self.fd, F_GETPIPE_SZ)
        except Exception as exc:
            try:
                self.output_pipe_size = fcntl.fcntl(self.fd, F_GETPIPE_SZ)
            except Exception:
                self.output_pipe_size = 65536
            print(f"[CAPTURE] FFmpeg pipe size unchanged ({exc}); {self.output_pipe_size} bytes")

        self._thread = threading.Thread(target=self._worker, daemon=True, name="ScreenGrabberWorker")
        self._thread.start()

        deadline = time.monotonic() + 4.0
        with self._cond:
            while self._latest_frame is None and not self._worker_failed and time.monotonic() < deadline:
                self._cond.wait(timeout=0.1)

        if self._latest_frame is None:
            err = self._last_error or "stream initialization timeout"
            raise CaptureError(err)

        self.use_gsr = True
        print(
            f"[CAPTURE] Direct GPU KMS Pipe active ({reg_str} @ {self.fps} FPS, {self.framerate_mode.upper()}, "
            f"keyint={self.keyint}s, pipe={self.input_pipe_size//1024}K/{self.output_pipe_size//1024}K)."
        )

    def _init_mss(self) -> None:
        try:
            import mss
            self.sct = mss.mss()
            self.use_gsr = False
            self._frame_id = 0
            print("[CAPTURE] WARNING: using mss fallback; KMS/GSR path unavailable.")
        except Exception as exc:
            self._worker_failed = True
            self._last_error = f"No capture backend: {exc}"
            raise CaptureError(self._last_error)

    def _mark_failed(self, msg: str) -> None:
        with self._cond:
            self._worker_failed = True
            self._last_error = str(msg)
            self._cond.notify_all()

    def _worker(self) -> None:
        import select

        w, h = self.region["width"], self.region["height"]
        frame_bytes = w * h * 3
        buf = bytearray()
        ionread = array.array("i", [0])

        while self._running:
            try:
                if self.gsr_process is not None:
                    rc = self.gsr_process.poll()
                    if rc is not None:
                        self._mark_failed(f"gpu-screen-recorder exited with code {rc}")
                        break
                if self.ff_process is None:
                    self._mark_failed("ffmpeg process missing")
                    break
                rc = self.ff_process.poll()
                if rc is not None:
                    self._mark_failed(f"ffmpeg exited with code {rc}")
                    break

                r_fds, _, _ = select.select([self.fd], [], [], 0.02)
                if not r_fds:
                    continue
                fcntl.ioctl(self.fd, termios.FIONREAD, ionread)
                ready = int(ionread[0])
                if ready <= 0:
                    continue
                self.max_backlog_bytes = max(self.max_backlog_bytes, ready)

                chunk = os.read(self.fd, ready)
                if not chunk:
                    self._mark_failed("ffmpeg rawvideo pipe closed")
                    break
                buf.extend(chunk)

                if len(buf) >= frame_bytes:
                    num_frames = len(buf) // frame_bytes
                    if num_frames > 1:
                        self.discarded_frames_count += num_frames - 1
                    start = (num_frames - 1) * frame_bytes
                    frame_data = bytes(buf[start:start + frame_bytes])
                    del buf[:num_frames * frame_bytes]
                    arr = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, 3))
                    ts = time.monotonic()
                    with self._cond:
                        self._latest_frame = arr
                        self._latest_decode_ts = ts
                        self._last_new_frame_mono = ts
                        # Increment once for the newest delivered image.  Discarded
                        # decoder backlog is tracked separately rather than hidden in ID.
                        self._frame_id += 1
                        self._cond.notify_all()
            except Exception as exc:
                if not self._running:
                    break
                self._mark_failed(f"capture worker exception: {exc!r}")
                break

    def is_healthy(self) -> bool:
        if self.sct is not None:
            return True
        if self._worker_failed:
            return False
        if self.gsr_process is not None and self.gsr_process.poll() is not None:
            return False
        if self.ff_process is not None and self.ff_process.poll() is not None:
            return False
        return True

    def grab(
        self,
        wait_new: bool = False,
        last_id: Optional[int] = None,
        timeout: float = 0.02,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
        if self.use_gsr:
            with self._cond:
                if wait_new and last_id is not None and self._frame_id == last_id and self._running and not self._worker_failed:
                    self._cond.wait(timeout=timeout)
                if self._worker_failed:
                    tail = " | ".join(list(self._stderr_tail)[-4:])
                    raise CaptureError(f"{self._last_error or 'capture worker failed'}{(' | ' + tail) if tail else ''}")
                if self._latest_frame is not None:
                    if wait_new and last_id is not None and self._frame_id == last_id:
                        self.transport_duplicate_waits += 1
                    return self._latest_frame, self._frame_id, self._latest_decode_ts
            return None, None, None

        if self.sct is not None:
            t0 = time.monotonic()
            raw = self.sct.grab(self.region)
            t1 = time.monotonic()
            self._last_mss_grab_duration_ms = (t1 - t0) * 1000.0
            self._frame_id += 1
            ts = (t0 + t1) / 2.0
            self._last_new_frame_mono = ts
            return np.array(raw)[:, :, :3], self._frame_id, ts

        raise CaptureError(self._last_error or "capture backend unavailable")

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "healthy": self.is_healthy(),
            "worker_failed": self._worker_failed,
            "last_error": self._last_error,
            "stderr_tail": list(self._stderr_tail),
            "use_gsr": self.use_gsr,
            "framerate_mode": self.framerate_mode,
            "keyint": self.keyint,
            "input_pipe_size_bytes": self.input_pipe_size,
            "output_pipe_size_bytes": self.output_pipe_size,
            "discarded_stale_frames": self.discarded_frames_count,
            "transport_duplicate_waits": self.transport_duplicate_waits,
            "max_backlog_bytes": self.max_backlog_bytes,
            "last_mss_grab_duration_ms": self._last_mss_grab_duration_ms,
            "last_new_frame_age_ms": (
                (time.monotonic() - self._last_new_frame_mono) * 1000.0
                if self._last_new_frame_mono > 0 else None
            ),
        }

    def close(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()

        for proc in (self.ff_process, self.gsr_process):
            if proc is None:
                continue
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except Exception:
                pass
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

        self.ff_process = None
        self.gsr_process = None

        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        for th in self._stderr_threads:
            th.join(timeout=0.1)
        self._stderr_threads.clear()

        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass
            self.sct = None
