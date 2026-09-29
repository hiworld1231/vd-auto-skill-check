import json
from pathlib import Path
import re
import struct
import subprocess

import pytest

from vd import capture
from vd import kwin_region


PROJECT = Path(__file__).resolve().parents[1]


def test_region_capture_uses_the_native_kde_worker(monkeypatch, tmp_path):
    worker = tmp_path / "kwin-region-worker"
    monkeypatch.setattr(capture, "ensure_kwin_region_worker", lambda: worker)

    command = capture.capture_worker_command(
        roi=(880, 420, 180, 240), fps=45, priority=10, source="region",
        normalize_window_scale=False, synthetic=False,
    )

    assert command == [str(worker), "--roi", "880,420,180,240", "--fps", "45",
                       "--priority", "10"]


def test_region_capture_defaults_to_the_compact_detection_area():
    assert capture.default_roi("region") == (875, 418, 180, 250)
    assert capture.default_roi("monitor") == (800, 420, 320, 240)


def test_region_capture_rejects_non_kde_wayland(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")

    with pytest.raises(RuntimeError, match="KDE Wayland"):
        kwin_region.ensure_kwin_region_worker()


def test_capture_converts_only_the_selected_roi_before_emitting_bgr_frames():
    result = subprocess.run(
        ["/usr/bin/python", str(PROJECT / "native" / "pipewire_worker.py"),
         "--synthetic", "--frames", "2", "--roi", "800,420,320,240"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    match = re.search(rb"Capture negotiated: ([^\r\n]+)", result.stderr)
    assert match, result.stderr.decode(errors="replace")
    assert "width=(int)320" in match.group(1).decode()
    assert "height=(int)240" in match.group(1).decode()

    header_size, = struct.unpack("!I", result.stdout[:4])
    header = json.loads(result.stdout[4:4 + header_size])
    assert (header["width"], header["height"], header["bytes"]) == (320, 240, 320 * 240 * 3)
    assert (header["source_width"], header["source_height"]) == (1920, 1080)
    assert header["worker_cpu_time_ns"] > 0


def test_monitor_capture_scales_1080p_roi_to_1600x900_and_normalizes_output():
    result = subprocess.run(
        ["/usr/bin/python", str(PROJECT / "native" / "pipewire_worker.py"),
         "--synthetic", "--synthetic-size", "1600,900", "--frames", "2",
         "--roi", "800,420,320,240"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"Capture region: x=667 y=350 width=267 height=200 normalized=320x240" in result.stderr
    match = re.search(rb"Capture negotiated: ([^\r\n]+)", result.stderr)
    assert match, result.stderr.decode(errors="replace")
    assert "width=(int)320" in match.group(1).decode()
    assert "height=(int)240" in match.group(1).decode()
    header_size, = struct.unpack("!I", result.stdout[:4])
    header = json.loads(result.stdout[4:4 + header_size])
    assert (header["source_width"], header["source_height"]) == (1600, 900)
    assert (header["width"], header["height"]) == (320, 240)


def test_window_capture_centers_the_roi_in_the_selected_window():
    result = subprocess.run(
        ["/usr/bin/python", str(PROJECT / "native" / "pipewire_worker.py"),
         "--synthetic", "--frames", "2", "--source", "window",
         "--roi", "0,0,320,240"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"Capture region: x=800 y=420 width=320 height=240" in result.stderr


def test_scaled_window_capture_resizes_the_center_roi_to_the_solver_size():
    result = subprocess.run(
        ["/usr/bin/python", str(PROJECT / "native" / "pipewire_worker.py"),
         "--synthetic", "--synthetic-size", "1280,720", "--frames", "2",
         "--source", "window", "--normalize-window-scale",
         "--roi", "0,0,320,240"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"Capture region: x=533 y=280 width=213 height=160" in result.stderr
    match = re.search(rb"Capture negotiated: ([^\r\n]+)", result.stderr)
    assert match, result.stderr.decode(errors="replace")
    assert "width=(int)320" in match.group(1).decode()
    assert "height=(int)240" in match.group(1).decode()
    header_size, = struct.unpack("!I", result.stdout[:4])
    header = json.loads(result.stdout[4:4 + header_size])
    assert (header["source_width"], header["source_height"]) == (1280, 720)
    assert (header["width"], header["height"]) == (320, 240)
