#!/usr/bin/env python3
"""Read the latest captured Vulkan ROI from vd_layer_shm.dat and save it as a PNG."""

import argparse
import binascii
import mmap
import struct
import time
import zlib
from pathlib import Path

SHM_MAGIC = 0x56444C59
HEADER_FORMAT = "=IIIIIIQQQfIIIIIII64sIIIIIII4xQQ112s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
BGRA_FORMATS = {44, 50}  # VK_FORMAT_B8G8R8A8_UNORM/SRGB
RGBA_FORMATS = {37, 43}  # VK_FORMAT_R8G8B8A8_UNORM/SRGB
DEFAULT_SHM = Path.home() / ".var/app/org.vinegarhq.Sober/data/vulkan/vd_layer_shm.dat"


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def write_rgba_png(path: Path, width: int, height: int, rgba: bytes):
    expected = width * height * 4
    if len(rgba) != expected:
        raise ValueError(f"RGBA length {len(rgba)} != {expected}")
    rows = b"".join(b"\x00" + rgba[y * width * 4:(y + 1) * width * 4] for y in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(rows, 6))
    png += png_chunk(b"IEND", b"")
    path.write_bytes(png)


def raw_to_rgba(raw: bytes, vk_format: int) -> bytes:
    if vk_format in RGBA_FORMATS:
        return raw
    if vk_format in BGRA_FORMATS:
        out = bytearray(len(raw))
        for i in range(0, len(raw), 4):
            b, g, r, a = raw[i:i + 4]
            out[i:i + 4] = bytes((r, g, b, a))
        return bytes(out)
    raise ValueError(f"Unsupported 4-byte swapchain format: {vk_format}")


def snapshot(shm_path: Path, retries: int = 50):
    with shm_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for _ in range(retries):
            header1 = bytes(mm[:HEADER_SIZE])
            values1 = struct.unpack(HEADER_FORMAT, header1)
            magic, version, struct_size, _flags, _pid, seq1 = values1[:6]
            if magic != SHM_MAGIC:
                raise RuntimeError("VD shared-memory magic not found")
            if struct_size != HEADER_SIZE:
                raise RuntimeError(f"Header size mismatch: producer={struct_size} viewer={HEADER_SIZE}")
            if seq1 & 1:
                time.sleep(0.01)
                continue
            sw_format = values1[13]
            roi_x, roi_y, roi_w, roi_h, roi_stride, roi_offset, roi_size = values1[18:25]
            capture_count, capture_ts = values1[25:27]
            if roi_w == 0 or roi_h == 0 or roi_size == 0:
                raise RuntimeError("No captured ROI yet. Start Sober with VD_CAPTURE_ENABLE=1 and enter a Vulkan scene.")
            if roi_stride != roi_w * 4 or roi_size != roi_stride * roi_h:
                raise RuntimeError("Unexpected ROI layout")
            end = roi_offset + roi_size
            if end > len(mm):
                raise RuntimeError("ROI points outside shared-memory file")
            raw = bytes(mm[roi_offset:end])
            seq2 = struct.unpack_from("=I", mm, 20)[0]
            if seq1 == seq2 and not (seq2 & 1):
                return {
                    "version": version,
                    "format": sw_format,
                    "x": roi_x,
                    "y": roi_y,
                    "width": roi_w,
                    "height": roi_h,
                    "capture_count": capture_count,
                    "capture_timestamp_ns": capture_ts,
                    "raw": raw,
                }
            time.sleep(0.01)
    raise RuntimeError("Could not obtain a stable ROI snapshot")


def main():
    parser = argparse.ArgumentParser(description="Save latest VD Vulkan ROI to PNG")
    parser.add_argument("--shm", type=Path, default=DEFAULT_SHM)
    parser.add_argument("--output", type=Path, default=Path("/tmp/vd-roi.png"))
    args = parser.parse_args()
    snap = snapshot(args.shm)
    rgba = raw_to_rgba(snap["raw"], snap["format"])
    write_rgba_png(args.output, snap["width"], snap["height"], rgba)
    print(f"saved={args.output} roi={snap['width']}x{snap['height']}@({snap['x']},{snap['y']}) format={snap['format']} capture={snap['capture_count']}")


if __name__ == "__main__":
    main()
