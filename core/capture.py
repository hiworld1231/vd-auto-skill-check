"""
High-Performance Zero-Lag Screen Grabber for Linux Wayland / KMS.
Uses gpu-screen-recorder KMS scanout piped to low-latency FFmpeg with monitored pipe buffers
and non-blocking FIONREAD queue draining.
Provides honest decode-ready timestamps and pipe backlog telemetry.
"""

import array
import fcntl
import os
import shutil
import subprocess
import termios
import threading
import time
from typing import Dict, Optional, Tuple, Any
import numpy as np

CAPTURE_REGION = {"left": 800, "top": 420, "width": 320, "height": 240}
DEFAULT_CAPTURE_FPS = 120
F_SETPIPE_SZ = 1031  # Linux F_SETPIPE_SZ constant
F_GETPIPE_SZ = 1032  # Linux F_GETPIPE_SZ constant


class ScreenGrabber:
    """
    KMS Direct Framebuffer Screen Grabber.
    Captures screen at high FPS (120 FPS) with hardware AMD VAAPI acceleration.
    Pipes into low-delay ffmpeg with zero-latency flags and drops stale frames.
    """

    def __init__(
        self,
        region: dict = None,
        fps: int = DEFAULT_CAPTURE_FPS,
        framerate_mode: str = "cfr",
        keyint: str = "2.0",
    ):
        self.region = region or CAPTURE_REGION
        self.fps = fps
        self.framerate_mode = framerate_mode
        self.keyint = str(keyint)
        self.use_gsr = False
        self.gsr_process = None
        self.ff_process = None
        self.fd = None
        self.sct = None
        self._latest_frame = None
        self._latest_decode_ts = time.monotonic()
        self._frame_id = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = None

        # Telemetry & queue diagnostics
        self.input_pipe_size = 65536
        self.output_pipe_size = 65536
        self.discarded_frames_count = 0
        self.max_backlog_bytes = 0
        self._last_mss_grab_duration_ms = 0.0

        if shutil.which("gpu-screen-recorder"):
            try:
                self._init_gsr()
            except Exception as e:
                print(f"[CAPTURE] gpu-screen-recorder error ({e}), falling back to mss...")
                self._init_mss()
        else:
            self._init_mss()

    def _init_gsr(self):
        w, h = self.region["width"], self.region["height"]
        x, y = self.region["left"], self.region["top"]
        reg_str = f"{w}x{h}+{x}+{y}"

        gsr_cmd = [
            "gpu-screen-recorder",
            "-w", reg_str,
            "-f", str(self.fps),
            "-c", "h264",
            "-fm", self.framerate_mode,
            "-tune", "performance",
            "-keyint", self.keyint,
            "-o", "/dev/stdout",
        ]

        self.gsr_process = subprocess.Popen(
            gsr_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        in_pipe_fd = self.gsr_process.stdout.fileno()
        try:
            fcntl.fcntl(in_pipe_fd, F_SETPIPE_SZ, 65536)
            self.input_pipe_size = fcntl.fcntl(in_pipe_fd, F_GETPIPE_SZ)
        except Exception as exc:
            try:
                self.input_pipe_size = fcntl.fcntl(in_pipe_fd, F_GETPIPE_SZ)
            except Exception:
                self.input_pipe_size = 65536
            print(f"[CAPTURE] Note: F_SETPIPE_SZ on GSR pipe failed ({exc}), using default {self.input_pipe_size} bytes")

        # IMPORTANT: in VFR mode, do not force FFmpeg input timestamps with -r.
        # GSR already decides when a content frame exists.  Forcing input -r makes
        # an irregular content stream look CFR again and destroys useful timing
        # semantics.  CFR keeps the historical command for non-GEN_RUSH modes.
        ff_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-avioflags", "direct",
            "-threads", "1",
            "-f", "h264",
        ]
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
            stderr=subprocess.DEVNULL,
            bufsize=262144,  # 256KB buffer
            start_new_session=True,
        )
        self.gsr_process.stdout.close()

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
            print(f"[CAPTURE] Note: F_SETPIPE_SZ on FFmpeg pipe failed ({exc}), using default {self.output_pipe_size} bytes")

        self._thread = threading.Thread(target=self._worker, daemon=True, name="ScreenGrabberWorker")
        self._thread.start()

        # Warm up first frame with timeout
        with self._cond:
            t_start = time.time()
            while self._latest_frame is None and (time.time() - t_start < 4.0):
                self._cond.wait(timeout=0.1)

        if self._latest_frame is not None:
            self.use_gsr = True
            print(
                f"[CAPTURE] Direct GPU KMS Pipe active ({reg_str} @ {self.fps} FPS, {self.framerate_mode.upper()}, "
                f"keyint={self.keyint}s, pipe={self.input_pipe_size//1024}K/{self.output_pipe_size//1024}K)."
            )
        else:
            self.close()
            raise RuntimeError("gpu-screen-recorder stream initialization timeout")

    def _init_mss(self):
        try:
            import mss
            self.sct = mss.mss()
            self.use_gsr = False
            print("[CAPTURE] Using standard fallback screen grabber.")
        except Exception as e:
            print(f"[CAPTURE] Fallback grabber failed: {e}")

    def _worker(self):
        import select
        w, h = self.region["width"], self.region["height"]
        frame_bytes = w * h * 3
        buf = bytearray()
        ionread = array.array("i", [0])

        while self._running:
            try:
                if self.ff_process is None or self.ff_process.poll() is not None:
                    break
                r_fds, _, _ = select.select([self.fd], [], [], 0.02)
                if not r_fds:
                    continue

                fcntl.ioctl(self.fd, termios.FIONREAD, ionread)
                ready = ionread[0]
                if ready == 0:
                    continue

                if ready > self.max_backlog_bytes:
                    self.max_backlog_bytes = ready

                chunk = os.read(self.fd, ready)
                if not chunk:
                    break
                buf.extend(chunk)

                if len(buf) >= frame_bytes:
                    num_frames = len(buf) // frame_bytes
                    if num_frames > 1:
                        self.discarded_frames_count += (num_frames - 1)

                    # Extract strictly the latest decoded frame
                    start_offset = (num_frames - 1) * frame_bytes
                    frame_data = buf[start_offset:start_offset + frame_bytes]
                    buf = buf[num_frames * frame_bytes:]

                    arr = np.frombuffer(frame_data, dtype=np.uint8).reshape((h, w, 3))
                    # Honest timestamp: this is when python received and unbuffered the decoded frame
                    decode_ready_ts = time.monotonic()
                    with self._cond:
                        self._latest_frame = arr
                        self._latest_decode_ts = decode_ready_ts
                        self._frame_id += num_frames
                        self._cond.notify_all()
            except Exception:
                if not self._running:
                    break
                time.sleep(0.001)

    def grab(
        self,
        wait_new: bool = False,
        last_id: Optional[int] = None,
        timeout: float = 0.02,
    ) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
        """
        Retrieves the freshest captured screen frame.
        wait_new: If True, waits for a new frame if current ID matches last_id.
        Returns: (frame_bgr, frame_id, decode_ready_ts)
        """
        if self.use_gsr:
            with self._cond:
                if wait_new and last_id is not None and self._frame_id == last_id and self._running:
                    self._cond.wait(timeout=timeout)
                if self._latest_frame is not None:
                    return self._latest_frame, self._frame_id, self._latest_decode_ts
            return None, None, None

        if self.sct is not None:
            t0 = time.monotonic()
            raw = self.sct.grab(self.region)
            t1 = time.monotonic()
            self._last_mss_grab_duration_ms = (t1 - t0) * 1000.0
            decode_ready_ts = (t0 + t1) / 2.0
            return np.array(raw)[:, :, :3], 1, decode_ready_ts

        return None, None, None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns capture pipeline queue metrics."""
        return {
            "use_gsr": self.use_gsr,
            "framerate_mode": self.framerate_mode,
            "keyint": self.keyint,
            "input_pipe_size_bytes": self.input_pipe_size,
            "output_pipe_size_bytes": self.output_pipe_size,
            "discarded_stale_frames": self.discarded_frames_count,
            "max_backlog_bytes": self.max_backlog_bytes,
            "last_mss_grab_duration_ms": self._last_mss_grab_duration_ms,
        }

    def close(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()

        if self.ff_process is not None:
            try:
                if self.ff_process.stdout:
                    self.ff_process.stdout.close()
            except Exception:
                pass
            try:
                self.ff_process.kill()
                self.ff_process.wait(timeout=0.2)
            except Exception:
                pass
            self.ff_process = None

        if self.gsr_process is not None:
            try:
                self.gsr_process.kill()
                self.gsr_process.wait(timeout=0.2)
            except Exception:
                pass
            self.gsr_process = None

        if self._thread is not None:
            try:
                self._thread.join(timeout=0.3)
            except Exception:
                pass
            self._thread = None

        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass
            self.sct = None
