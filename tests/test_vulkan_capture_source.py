import mmap
from pathlib import Path
import struct
import time

import numpy as np

from vd.capture import DEFAULT_ROI
from vd.vulkan_capture import VulkanCapture


HEADER_FORMAT = "=IIIIIIQQQfIIIIIII64sIIIIIII4xQQ112s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = 0x56444C59
VERSION = 2


def write_snapshot(path: Path, *, vk_format=44, capture_count=7, capture_ts_ns=None,
                   swapchain=(1600, 878), roi=(667, 341, 4, 3),
                   pixel=(51, 153, 26, 255), raw=None):
    capture_ts_ns = capture_ts_ns or time.monotonic_ns()
    x, y, w, h = roi
    stride = w * 4
    if raw is None:
        raw = bytes(pixel) * (w * h)
    assert len(raw) == stride * h
    values = (
        MAGIC, VERSION, HEADER_SIZE, 0b1111, 3, 2,
        capture_ts_ns - 1_000_000, capture_ts_ns, capture_count, 60.0,
        0, swapchain[0], swapchain[1], vk_format, 1, 4, 0,
        b"Main\0" + b"\0" * 59,
        x, y, w, h, stride, HEADER_SIZE, len(raw),
        capture_count, capture_ts_ns, b"\0" * 112,
    )
    path.write_bytes(struct.pack(HEADER_FORMAT, *values) + raw)


def test_vulkan_capture_reads_bgra_and_normalizes_to_solver_frame(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm)

    with VulkanCapture(shm_path=shm) as capture:
        frame = capture.next(timeout=0.05)

    assert frame is not None
    assert frame.sequence == 7
    assert frame.image.shape == (240, 320, 3)
    assert frame.image.dtype == np.uint8
    assert frame.image.flags.writeable is False
    assert tuple(frame.image[120, 160]) == (51, 153, 26)  # BGR from BGRA
    assert frame.source_size == (1600, 878)
    assert capture.roi == DEFAULT_ROI
    assert frame.timestamp_kind == "vulkan_monotonic"


def test_vulkan_capture_center_crops_square_layer_roi_before_resize(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    w = h = 8
    top = bytes((0, 0, 255, 255)) * (w * 1)
    middle = bytes((0, 255, 0, 255)) * (w * 6)
    bottom = bytes((255, 0, 0, 255)) * (w * 1)
    write_snapshot(shm, roi=(672, 311, w, h), raw=top + middle + bottom)

    with VulkanCapture(shm_path=shm) as capture:
        frame = capture.next(timeout=0.05)

    assert frame is not None
    # 4:3 normalization should discard the square ROI's top/bottom bands,
    # not stretch all 8 rows into the solver frame.
    assert tuple(frame.image[0, 160]) == (0, 255, 0)
    assert tuple(frame.image[-1, 160]) == (0, 255, 0)


def test_vulkan_capture_converts_rgba_to_bgr(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm, vk_format=37, pixel=(26, 153, 51, 255))

    with VulkanCapture(shm_path=shm) as capture:
        frame = capture.next(timeout=0.05)

    assert frame is not None
    assert tuple(frame.image[0, 0]) == (51, 153, 26)


def test_vulkan_capture_waits_for_new_capture_count(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm, capture_count=4)

    with VulkanCapture(shm_path=shm) as capture:
        first = capture.next(timeout=0.05)
        assert first is not None
        assert capture.next(after=first.sequence, timeout=0.02) is None


def test_vulkan_capture_rejects_missing_pixel_ready_flag(tmp_path):
    shm = tmp_path / "vd_layer_shm.dat"
    write_snapshot(shm)
    raw = bytearray(shm.read_bytes())
    struct.pack_into("=I", raw, 12, 0b0111)
    shm.write_bytes(raw)

    with VulkanCapture(shm_path=shm) as capture:
        assert capture.next(timeout=0.02) is None
