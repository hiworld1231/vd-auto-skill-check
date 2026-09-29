#!/usr/bin/env python3
"""Run the existing VD solver using pixels from the Vulkan capture layer."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import vd.runtime as runtime
from vd.vulkan_capture import VulkanCapture


VARIANTS = {
    "baseline": (60, 5),
    "responsive": (60, 0),
    "quiet": (60, 10),
}


def main():
    parser = argparse.ArgumentParser(description="VD solver using Sober Vulkan shared-memory capture")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--variant", choices=VARIANTS, default="quiet")
    parser.add_argument("--lead-ms", type=float, default=35)
    parser.add_argument("--lead-uncertainty-ms", type=float, default=20)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--no-learn-lead", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    if args.seconds is not None and not 0 < args.seconds <= 86400:
        parser.error("seconds must be in (0, 86400]")
    if not 0 <= args.lead_ms <= 300 or not 0 <= args.lead_uncertainty_ms <= 100:
        parser.error("lead must be in [0, 300] ms; uncertainty in [0, 100] ms")

    fps, capture_priority = VARIANTS[args.variant]
    directory = args.recording or (
        Path(__file__).resolve().parent / "recordings" /
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )

    # run_session already contains the complete Detector -> Engine -> input pipeline.
    # Swap only its capture backend so the solver logic stays identical.
    runtime.PortalCapture = VulkanCapture

    print(
        "RUN Vulkan: зайди в игру и удерживай LMB; Ctrl+C для остановки; "
        f"profile={args.variant}; capture=swapchain; lead={args.lead_ms:g} ms.",
        flush=True,
    )

    result = runtime.run_session(
        seconds=args.seconds,
        synthetic=False,
        fps=fps,
        directory=directory,
        lead_seconds=args.lead_ms / 1000,
        lead_uncertainty=args.lead_uncertainty_ms / 1000,
        physical=True,
        learn_lead=not args.no_learn_lead,
        recording=True,
        capture_priority=capture_priority,
        variant=args.variant,
        capture_source="vulkan",
        normalize_window_scale=False,
        video_recording=args.video,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "events")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit("VD Vulkan: " + str(exc))
