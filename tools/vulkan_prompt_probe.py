#!/usr/bin/env python3
"""Live diagnostic for locating the VD prompt inside Vulkan swapchain capture."""

import argparse
from pathlib import Path
import sys
import time

# When this file is executed directly, Python puts tools/ on sys.path rather
# than the repository root. Add the root explicitly so `import vd` works the
# same way it does under pytest and the project entry points.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from vd.vulkan_capture import BGRA_FORMATS, RGBA_FORMATS, VulkanCapture
from vd.vision import Detector


def best_match(gray, template, scales):
    best = (-1.0, None, None, None)
    for scale in scales:
        width = max(1, round(template.shape[1] * scale))
        height = max(1, round(template.shape[0] * scale))
        if width > gray.shape[1] or height > gray.shape[0]:
            continue
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(template, (width, height), interpolation=interpolation)
        result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score > best[0]:
            best = (float(score), float(scale), tuple(map(int, location)), (width, height))
    return best


def raw_bgr(snapshot):
    _x, _y, width, height = snapshot["roi"]
    pixels = np.frombuffer(snapshot["raw"], np.uint8).reshape(height, width, 4)
    if snapshot["format"] in BGRA_FORMATS:
        return np.array(pixels[:, :, :3], copy=True)
    if snapshot["format"] in RGBA_FORMATS:
        return np.array(pixels[:, :, [2, 1, 0]], copy=True)
    raise RuntimeError(f"Unsupported Vulkan swapchain format: {snapshot['format']}")


def annotate(image, match, output):
    score, scale, location, size = match
    rendered = image.copy()
    if location is not None and size is not None:
        x, y = location
        w, h = size
        cv2.rectangle(rendered, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(rendered, f"score={score:.3f} scale={scale:.3f}",
                    (4, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .38,
                    (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), rendered)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Find the prompt template in live Vulkan capture")
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="optional timeout; by default run until Ctrl+C",
    )
    parser.add_argument("--raw-output", type=Path, default=Path("/tmp/vd-prompt-raw.png"))
    parser.add_argument("--normalized-output", type=Path, default=Path("/tmp/vd-prompt-normalized.png"))
    args = parser.parse_args(argv)
    if args.seconds is not None and not 0 < args.seconds <= 86400:
        parser.error("seconds must be in (0, 86400]")
    return args


def deadline_for_seconds(seconds, *, now=None):
    if seconds is None:
        return None
    if now is None:
        now = time.monotonic()
    return now + seconds


def main(argv=None):
    args = parse_args(argv)

    detector = Detector()
    raw_scales = np.linspace(.55, 1.15, 25)
    normalized_scales = np.linspace(.70, 1.30, 25)
    best_raw = (-1.0, None, None, None)
    best_norm = (-1.0, None, None, None)
    best_raw_image = None
    best_norm_image = None
    best_raw_meta = None
    best_norm_meta = None
    frames = 0
    after = -1
    deadline = deadline_for_seconds(args.seconds)
    interrupted = False

    if deadline is None:
        print("probe: без таймера; вызови skill check и нажми Ctrl+C после него", flush=True)
    else:
        print(f"probe: timeout={args.seconds:g}s", flush=True)

    try:
        with VulkanCapture() as capture:
            while deadline is None or time.monotonic() < deadline:
                snap = capture._snapshot()
                if snap is None or snap["count"] <= after:
                    time.sleep(.002)
                    continue
                after = snap["count"]
                frames += 1

                raw = raw_bgr(snap)
                normalized = capture._to_bgr(snap)
                raw_match = best_match(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), detector.template, raw_scales)
                norm_match = best_match(cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY), detector.template,
                                        normalized_scales)

                if raw_match[0] > best_raw[0]:
                    best_raw = raw_match
                    best_raw_image = raw
                    best_raw_meta = (snap["count"], snap["roi"], snap["source_size"])
                    print(f"raw new best score={raw_match[0]:.3f} scale={raw_match[1]:.3f} "
                          f"loc={raw_match[2]} frame={snap['count']}", flush=True)
                if norm_match[0] > best_norm[0]:
                    best_norm = norm_match
                    best_norm_image = normalized
                    best_norm_meta = (snap["count"], snap["roi"], snap["source_size"])
                    print(f"norm new best score={norm_match[0]:.3f} scale={norm_match[1]:.3f} "
                          f"loc={norm_match[2]} frame={snap['count']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True

    if best_raw_image is not None:
        annotate(best_raw_image, best_raw, args.raw_output)
    if best_norm_image is not None:
        annotate(best_norm_image, best_norm, args.normalized_output)

    print("=== Vulkan prompt probe ===")
    if interrupted:
        print("stopped=Ctrl+C")
    print(f"frames={frames}")
    print(f"raw_best score={best_raw[0]:.4f} scale={best_raw[1]} loc={best_raw[2]} size={best_raw[3]} meta={best_raw_meta}")
    print(f"normalized_best score={best_norm[0]:.4f} scale={best_norm[1]} loc={best_norm[2]} size={best_norm[3]} meta={best_norm_meta}")
    print(f"raw_png={args.raw_output}")
    print(f"normalized_png={args.normalized_output}")


if __name__ == "__main__":
    main()
