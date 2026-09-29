#!/usr/bin/env python3
"""
Inspect and monitor the VD Vulkan Implicit Layer status in Sober/Roblox.
Reads the status JSON, logs, and live mmap shared memory buffer.
"""

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
SHM_MAGIC = 0x56444C59  # 'VDLY'


def read_shm_header(shm_path: Path):
    if not shm_path.exists():
        return None
    try:
        with open(shm_path, "rb") as f:
            data = f.read(256)
            if len(data) < 120:
                return None
            (
                magic,
                version,
                struct_size,
                status_flags,
                pid,
                sequence,
                init_ts,
                last_ts,
                frame_count,
                current_fps,
                dropped_frames,
                sw_width,
                sw_height,
                sw_format,
                sw_mode,
                sw_img_count,
                sw_cur_idx,
                proc_name_raw,
            ) = struct.unpack("=IIII II QQ QfI IIIIII 64s", data[:144])

            if magic != SHM_MAGIC:
                return None

            proc_name = proc_name_raw.split(b"\x00")[0].decode("utf-8", errors="ignore")
            is_alive = False
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    is_alive = True
                except (ProcessLookupError, PermissionError):
                    is_alive = False

            return {
                "magic": hex(magic),
                "version": version,
                "status_flags": status_flags,
                "pid": pid,
                "sequence": sequence,
                "process_name": proc_name,
                "is_alive": is_alive,
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
            }
    except Exception as e:
        return {"error": str(e)}


def print_status(vulkan_dir: Path):
    manifest_path = vulkan_dir / "implicit_layer.d" / "VkLayer_VD_capture.json"
    status_path = vulkan_dir / "vd_layer_status.json"
    shm_path = vulkan_dir / "vd_layer_shm.dat"
    log_path = vulkan_dir / "vd_layer.log"

    print("==================================================")
    print("       VD Vulkan Layer Diagnostic & Status        ")
    print("==================================================")
    print(f"Vulkan Directory : {vulkan_dir}")
    print(f"Manifest Active  : {'YES' if manifest_path.exists() else 'NO (Run native/vulkan_layer/install_sober_layer.sh)'}")

    if status_path.exists():
        try:
            with open(status_path, "r") as f:
                status_json = json.load(f)
            print("\n[Status JSON]")
            print(f"  PID            : {status_json.get('pid')}")
            print(f"  Process        : {status_json.get('process_name')}")
            print(f"  Frames Count   : {status_json.get('frames_presented')}")
            print(f"  Current FPS    : {status_json.get('fps'):.1f}")
            sc = status_json.get("swapchain", {})
            print(f"  Resolution     : {sc.get('width')}x{sc.get('height')}")
            print(f"  Format / Mode  : Format={sc.get('format')} PresentMode={sc.get('present_mode')} Images={sc.get('image_count')}")
        except Exception as e:
            print(f"\n[Status JSON Error]: {e}")
    else:
        print("\n[Status JSON] Not generated yet (Waiting for Vulkan application to initialize)")

    shm_data = read_shm_header(shm_path)
    if shm_data and "error" not in shm_data:
        print("\n[Live MMap Shared Memory Header]")
        print(f"  Magic / Version: {shm_data['magic']} / v{shm_data['version']} (Seq: {shm_data['sequence']})")
        print(f"  PID / Process  : {shm_data['pid']} ({shm_data['process_name']}) [Alive: {'YES' if shm_data['is_alive'] else 'NO'}]")
        print(f"  Status Flags   : {bin(shm_data['status_flags'])}")
        print(f"  Frame Counter  : {shm_data['frame_count']}")
        print(f"  Live FPS       : {shm_data['current_fps']}")
        sc = shm_data["swapchain"]
        print(f"  Swapchain      : {sc['width']}x{sc['height']} (Format {sc['format']}, ImgIdx {sc['current_image_index']})")
    elif shm_data and "error" in shm_data:
        print(f"\n[Live MMap Error]: {shm_data['error']}")

    if log_path.exists():
        print("\n[Recent Log Entries]")
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print("  " + line.strip())
        except Exception as e:
            print(f"  Error reading log: {e}")
    print("==================================================")


def watch_status(vulkan_dir: Path, interval: float):
    shm_path = vulkan_dir / "vd_layer_shm.dat"
    print(f"Watching Vulkan layer live status (Interval {interval}s, Ctrl+C to stop)...")
    try:
        last_frames = 0
        last_time = time.time()
        while True:
            shm = read_shm_header(shm_path)
            now = time.time()
            if shm and "frame_count" in shm:
                frames = shm["frame_count"]
                delta_f = frames - last_frames
                delta_t = now - last_time
                fps = (delta_f / delta_t) if delta_t > 0 else 0.0
                sc = shm["swapchain"]
                sys.stdout.write(
                    f"\r[PID {shm['pid']} {shm['process_name']}] "
                    f"Frames: {frames:<8} | FPS: {fps:5.1f} | "
                    f"Res: {sc['width']}x{sc['height']} (Fmt {sc['format']}) | "
                    f"ImgIdx: {sc['current_image_index']}   "
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
    parser = argparse.ArgumentParser(description="Inspect VD Vulkan layer status in Sober Flatpak")
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_VULKAN_DIR,
        help="Base directory where layer status is stored",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor live frame rate and frame counter",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Poll interval for watch mode in seconds",
    )
    args = parser.parse_args()

    if args.watch:
        watch_status(args.dir, args.interval)
    else:
        print_status(args.dir)


if __name__ == "__main__":
    main()
