from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from typing import List


@dataclass
class PreflightReport:
    ok: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = []
        if self.errors:
            bits.append("errors=" + "; ".join(self.errors))
        if self.warnings:
            bits.append("warnings=" + "; ".join(self.warnings))
        return " | ".join(bits) if bits else "OK"


def run_preflight(*, dry_run: bool, require_lmb: bool, require_capture_tools: bool = False) -> PreflightReport:
    errors: List[str] = []
    warnings: List[str] = []

    for mod in ("numpy", "cv2", "rich"):
        if importlib.util.find_spec(mod) is None:
            errors.append(f"missing Python module: {mod}")

    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "space_template.png")
    if not os.path.isfile(template_path):
        errors.append(f"required detector template missing: {template_path}")

    if shutil.which("gpu-screen-recorder") is None:
        if require_capture_tools:
            errors.append("gpu-screen-recorder not found")
        else:
            warnings.append("gpu-screen-recorder not found; capture will fall back to mss")
    if shutil.which("ffmpeg") is None:
        if require_capture_tools:
            errors.append("ffmpeg not found")
        else:
            warnings.append("ffmpeg not found; capture will fall back to mss")

    if importlib.util.find_spec("mss") is None:
        # mss is only a fallback if GSR works, so this is a warning rather than a
        # hard error when the preferred KMS path is present.
        if shutil.which("gpu-screen-recorder") is None or shutil.which("ffmpeg") is None:
            errors.append("mss missing and KMS capture tools are unavailable")
        else:
            warnings.append("mss missing; no capture fallback if GSR fails")

    if not dry_run and importlib.util.find_spec("evdev") is None and importlib.util.find_spec("pynput") is None:
        errors.append("no input backend (install evdev; pynput is fallback only)")

    if require_lmb and importlib.util.find_spec("evdev") is None and importlib.util.find_spec("pynput") is None:
        errors.append("require_lmb=true but no mouse input backend is installed")

    # /dev/uinput is the expected Linux/Wayland production path.  The final
    # authority remains HardwareTrigger, because ACLs/group permissions vary.
    if not dry_run and os.path.exists("/dev/uinput") and not os.access("/dev/uinput", os.W_OK):
        warnings.append("/dev/uinput exists but is not writable by this user")

    return PreflightReport(ok=not errors, warnings=warnings, errors=errors)
