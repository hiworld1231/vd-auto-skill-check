#!/usr/bin/env python3
"""Inspect the live VD Vulkan layer telemetry produced by Sober/Roblox."""

import argparse
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path

DEFAULT_SOBER_DATA = Path(os.environ.get("HOME", "/home/oae")) / ".var" / "app" / "org.vinegarhq.Sober" / "data"
DEFAULT_VULKAN_DIR = DEFAULT_SOBER_DATA / "vulkan"
SHM_MAGIC = 0x56444C59
HEADER_FORMAT = "=IIIIIIQQQfIIIIIII64sIIIIIII4xQQ112s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def _decode_header(data: bytes):
    if len(data) < HEADER_SIZE:
        return None
    values = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    (
        magic, version, struct_size, status_flags, pid, sequence,
        init_ts, last_ts, frame_count, current_fps,
        dropped_frames, sw_width, sw_height, sw_format, sw_mode, sw_img_count, sw_cur_idx,
        proc_name_raw,
        roi_x, roi_y, roi_width, roi_height, roi_stride, roi_offset, roi_size,
        roi_capture_count, roi_capture_ts,
        _padding,
    ) = values
    if magic != SHM_MAGIC:
        return None
    return {
        "magic": hex(magic),
        "version": version,
        "struct_size": struct_size,
        "status_flags": status_flags,
        "pid": pid,
        "sequence": sequence,
        "process_name": proc_name_raw.split(b"\x00")[0].decode("utf-8", errors="ignore"),
        "init_timestamp_ns": init_ts,
        "last_present_ns": last_ts,
        "frame_count": frame_count,
        "current_fps": round(current_fps, 2),
        "dropped_frames": dropped_frames,
        "swapchain": {
            "width": sw_width,
            "height": sw_height,
            "format": sw_format,
            "present_mode": sw_mode,
            "image_count": sw_img_count,
            "current_image_index": sw_cur_idx,
        },
        "roi": {
            "x": roi_x,
            "y": roi_y,
            "width": roi_width,
            "height": roi_height,
            "stride": roi_stride,
            "data_offset": roi_offset,
            "data_size": roi_size,
            "capture_count": roi_capture_count,
            "capture_timestamp_ns": roi_capture_ts,
        },
    }


def read_shm_header(shm_path: Path):
    if not shm_path.exists():
        return None
    try:
        with open(shm_path, "rb") as f:
            return _decode_header(f.read(HEADER_SIZE))
    except Exception as exc:
        return {"error": str(exc)}


def extension_installed() -> bool:
    root = Path.home() / ".local/share/flatpak/extension/org.freedesktop.Platform.VulkanLayer.VDCapture"
    return any(root.glob("*/**/share/vulkan/implicit_layer.d/VkLayer_VD_capture.json")) if root.exists() else False


def print_status(vulkan_dir: Path):
    status_path = vulkan_dir / "vd_layer_status.json"
    shm_path = vulkan_dir / "vd_layer_shm.dat"
    log_path = vulkan_dir / "vd_layer.log"

    print("==================================================")
    print("       VD Vulkan Layer Diagnostic & Status        ")
    print("==================================================")
    print(f"Vulkan Directory : {vulkan_dir}")
    print(f"Extension Installed: {'YES' if extension_installed() else 'NO'}")

    if status_path.exists():
        try:
            status_json = json.loads(status_path.read_text())
            print("\n[Status JSON]")
            print(f"  PID            : {status_json.get('pid')} (Flatpak PID namespace)")
            print(f"  Process        : {status_json.get('process_name')}")
            print(f"  Frames Count   : {status_json.get('frames_presented')}")
            print(f"  Captures       : {status_json.get('captures', 0)}")
            print(f"  Current FPS    : {status_json.get('fps', 0):.1f}")
            sc = status_json.get("swapchain", {})
            print(f"  Resolution     : {sc.get('width')}x{sc.get('height')}")
            print(f"  Format / Mode  : Format={sc.get('format')} PresentMode={sc.get('present_mode')} Images={sc.get('image_count')}")
        except Exception as exc:
            print(f"\n[Status JSON Error]: {exc}")
    else:
        print("\n[Status JSON] Not generated yet")

    shm_data = read_shm_header(shm_path)
    if shm_data and "error" not in shm_data:
        print("\n[Live MMap Shared Memory Header]")
        print(f"  Magic / Version: {shm_data['magic']} / v{shm_data['version']} (Seq: {shm_data['sequence']})")
        print(f"  PID / Process  : {shm_data['pid']} ({shm_data['process_name']}) [PID is namespace-local]")
        print(f"  Status Flags   : {bin(shm_data['status_flags'])}")
        print(f"  Frame Counter  : {shm_data['frame_count']}")
        print(f"  Live FPS       : {shm_data['current_fps']}")
        print(f"  Dropped Capture: {shm_data['dropped_frames']}")
        sc = shm_data["swapchain"]
        print(f"  Swapchain      : {sc['width']}x{sc['height']} (Format {sc['format']}, ImgIdx {sc['current_image_index']})")
        roi = shm_data["roi"]
        print(f"  ROI            : {roi['width']}x{roi['height']} @ ({roi['x']},{roi['y']}) bytes={roi['data_size']} captures={roi['capture_count']}")
    elif shm_data and "error" in shm_data:
        print(f"\n[Live MMap Error]: {shm_data['error']}")

    if log_path.exists():
        print("\n[Recent Log Entries]")
        try:
            for line in log_path.read_text(errors="replace").splitlines()[-10:]:
                print("  " + line)
        except Exception as exc:
            print(f"  Error reading log: {exc}")
    print("==================================================")


def watch_status(vulkan_dir: Path, interval: float):
    shm_path = vulkan_dir / "vd_layer_shm.dat"
    print(f"Watching Vulkan layer live status (Interval {interval}s, Ctrl+C to stop)...")
    last_frames = 0
    last_time = time.time()
    try:
        while True:
            shm = read_shm_header(shm_path)
            now = time.time()
            if shm and "frame_count" in shm:
                frames = shm["frame_count"]
                fps = (frames - last_frames) / max(now - last_time, 1e-6)
                sc = shm["swapchain"]
                roi = shm["roi"]
                sys.stdout.write(
                    f"\r[PID {shm['pid']} {shm['process_name']}] Frames: {frames:<8} | FPS: {fps:5.1f} | "
                    f"Res: {sc['width']}x{sc['height']} | ImgIdx: {sc['current_image_index']} | "
                    f"Captures: {roi['capture_count']:<6} | ROI: {roi['width']}x{roi['height']}   "
                )
                sys.stdout.flush()
                last_frames = frames
                last_time = now
            else:
                sys.stdout.write("\rWaiting for active Vulkan layer process...   ")
                sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Inspect VD Vulkan layer status")
    parser.add_argument("--dir", type=Path, default=DEFAULT_VULKAN_DIR)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.watch:
        watch_status(args.dir, args.interval)
    else:
        print_status(args.dir)


if __name__ == "__main__":
    main()
