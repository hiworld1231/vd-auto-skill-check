from __future__ import annotations

import mmap
from pathlib import Path
import struct
import threading
import time

import cv2
import numpy as np

from vd.capture import DEFAULT_ROI, Frame


SHM_MAGIC = 0x56444C59
SHM_VERSION = 2
STATUS_CAPTURE_ENABLED = 1 << 2
STATUS_PIXELS_READY = 1 << 3
HEADER_FORMAT = "=IIIIIIQQQfIIIIIII64sIIIIIII4xQQ112s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
BGRA_FORMATS = {44, 50}  # VK_FORMAT_B8G8R8A8_UNORM/SRGB
RGBA_FORMATS = {37, 43}  # VK_FORMAT_R8G8B8A8_UNORM/SRGB
DEFAULT_SHM = Path.home() / ".var" / "app" / "org.vinegarhq.Sober" / "data" / "vulkan" / "vd_layer_shm.dat"


class _CaptureProcessState:
    """Small compatibility surface used by vd.dispatch for capture liveness."""

    def __init__(self, owner):
        self.owner = owner

    def poll(self):
        return 0 if self.owner.stopping else None


class VulkanCapture:
    """Read the latest swapchain ROI published by the VD Vulkan layer.

    The layer publishes a physical-resolution center ROI. This reader converts the
    swapchain's 4-byte pixel format to BGR, center-crops it to the solver's 4:3
    geometry, then normalizes it back to the canonical 320x240 frame. The existing
    detector/engine therefore stays Vulkan-agnostic.

    It also exposes the same publication/liveness surface as PortalCapture
    (cv/latest/error/proc/stopping), because vd.dispatch intentionally verifies the
    exact frame being acted on while holding the capture publication lock.
    """

    def __init__(self, roi=None, *, synthetic=False, fps=60, priority=5,
                 source="vulkan", normalize_window_scale=False, shm_path=None):
        if source != "vulkan":
            raise ValueError("VulkanCapture requires source='vulkan'")
        if synthetic:
            raise ValueError("Vulkan capture cannot be synthetic")
        if normalize_window_scale:
            raise ValueError("Vulkan capture is normalized automatically")
        if type(fps) is not int or not 1 <= fps <= 240:
            raise ValueError("FPS must be an integer in [1, 240]")
        if type(priority) is not int or not 0 <= priority <= 19:
            raise ValueError("Capture priority must be an integer in [0, 19]")
        if roi is not None and tuple(roi) != DEFAULT_ROI:
            raise ValueError("Vulkan capture currently uses the canonical solver ROI")

        self.roi = DEFAULT_ROI
        self.shm_path = Path(shm_path) if shm_path is not None else DEFAULT_SHM
        self.cv = threading.Condition()
        self.error = None
        self.latest = None
        self.stopping = False
        self.proc = _CaptureProcessState(self)
        self._file = None
        self._mapping = None

    def _close_mapping(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _ensure_mapping(self):
        if self._mapping is not None:
            return True
        try:
            f = self.shm_path.open("rb")
            if self.shm_path.stat().st_size < HEADER_SIZE:
                f.close()
                return False
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (FileNotFoundError, OSError, ValueError):
            return False
        self._file = f
        self._mapping = mm
        return True

    def _snapshot(self):
        if not self._ensure_mapping():
            return None
        mm = self._mapping
        if mm is None or len(mm) < HEADER_SIZE:
            self._close_mapping()
            return None

        for _ in range(4):
            try:
                header = bytes(mm[:HEADER_SIZE])
                values = struct.unpack(HEADER_FORMAT, header)
            except (ValueError, struct.error):
                self._close_mapping()
                return None

            magic, version, struct_size, flags, _pid, seq1 = values[:6]
            if magic != SHM_MAGIC:
                raise RuntimeError("VD Vulkan shared-memory magic not found")
            if version != SHM_VERSION or struct_size != HEADER_SIZE:
                raise RuntimeError(
                    f"VD Vulkan shared-memory contract mismatch: version={version} size={struct_size}"
                )
            if seq1 & 1:
                time.sleep(0.001)
                continue
            if not (flags & STATUS_CAPTURE_ENABLED) or not (flags & STATUS_PIXELS_READY):
                return None

            sw_width, sw_height, sw_format = values[11:14]
            roi_x, roi_y, roi_w, roi_h, roi_stride, roi_offset, roi_size = values[18:25]
            capture_count, capture_ts = values[25:27]
            if roi_w <= 0 or roi_h <= 0 or roi_stride != roi_w * 4:
                return None
            if roi_size != roi_stride * roi_h:
                raise RuntimeError("Unexpected VD Vulkan ROI layout")
            end = roi_offset + roi_size
            if roi_offset < HEADER_SIZE or end > len(mm):
                raise RuntimeError("VD Vulkan ROI points outside shared-memory file")
            raw = bytes(mm[roi_offset:end])
            seq2 = struct.unpack_from("=I", mm, 20)[0]
            if seq1 == seq2 and not (seq2 & 1):
                return dict(
                    count=int(capture_count), timestamp_ns=int(capture_ts),
                    source_size=(int(sw_width), int(sw_height)), format=int(sw_format),
                    roi=(int(roi_x), int(roi_y), int(roi_w), int(roi_h)), raw=raw,
                )
            time.sleep(0.001)
        return None

    @staticmethod
    def _center_crop_aspect(image, target_w, target_h):
        height, width = image.shape[:2]
        lhs = width * target_h
        rhs = height * target_w
        if lhs == rhs:
            return image
        if lhs > rhs:
            crop_w = max(1, height * target_w // target_h)
            x = max(0, (width - crop_w) // 2)
            return image[:, x:x + crop_w]
        crop_h = max(1, width * target_h // target_w)
        y = max(0, (height - crop_h) // 2)
        return image[y:y + crop_h, :]

    @staticmethod
    def _to_bgr(snapshot):
        _x, _y, width, height = snapshot["roi"]
        pixels = np.frombuffer(snapshot["raw"], np.uint8).reshape(height, width, 4)
        vk_format = snapshot["format"]
        if vk_format in BGRA_FORMATS:
            bgr = pixels[:, :, :3]
        elif vk_format in RGBA_FORMATS:
            bgr = pixels[:, :, [2, 1, 0]]
        else:
            raise RuntimeError(f"Unsupported Vulkan swapchain format: {vk_format}")

        target_w, target_h = DEFAULT_ROI[2:]
        bgr = VulkanCapture._center_crop_aspect(bgr, target_w, target_h)
        if (bgr.shape[1], bgr.shape[0]) != (target_w, target_h):
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            bgr = np.array(bgr, copy=True)
        bgr.flags.writeable = False
        return bgr

    def next(self, after=-1, timeout=1.0):
        deadline = time.monotonic() + timeout
        while not self.stopping:
            snapshot = self._snapshot()
            if snapshot is not None and snapshot["count"] > after:
                received = time.monotonic()
                image = self._to_bgr(snapshot)
                timestamp_ns = snapshot["timestamp_ns"]
                frame = Frame(
                    sequence=snapshot["count"],
                    image=image,
                    media_time=timestamp_ns / 1e9 if timestamp_ns else None,
                    received_time=received,
                    consumed_time=time.monotonic(),
                    timestamp_kind="vulkan_monotonic",
                    synthetic=False,
                    pts_ns=timestamp_ns or None,
                    negotiated_caps=f"vulkan-format={snapshot['format']};raw-roi={snapshot['roi'][2]}x{snapshot['roi'][3]}",
                    worker_cpu_time_ns=None,
                    source_size=snapshot["source_size"],
                )
                with self.cv:
                    self.latest = frame
                    self.cv.notify_all()
                return frame
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)
        return None

    def close(self, *, fast=False):
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        self._close_mapping()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        self.close(fast=exc_type is not None and issubclass(exc_type, KeyboardInterrupt))
