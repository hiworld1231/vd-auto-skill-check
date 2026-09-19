#!/usr/bin/env python3
"""Violence District Skill Check — CLEAN V5 entry point.

The clean runtime is speed-agnostic: normal and perk-accelerated checks use the
same continuous measured-speed predictor. ``--gen-rush`` is retained only for
command compatibility with previous releases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from core.genrush_runtime import run_genrush_clean

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def _load_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.json must contain a JSON object")
    return data


def _region(value: str) -> Dict[str, int]:
    try:
        left, top, width, height = (int(x.strip()) for x in value.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("region must be left,top,width,height") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("region width/height must be positive")
    return {"left": left, "top": top, "width": width, "height": height}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Violence District external CV skill-check bot — CLEAN V5"
    )
    ap.add_argument(
        "--gen-rush", "--speed-perk", dest="gen_rush", action="store_true",
        help="compatibility flag; CLEAN V5 automatically handles normal and accelerated speeds",
    )
    ap.add_argument("--dry-run", action="store_true", help="do not send Space")
    ap.add_argument("--fps", type=int, default=None, help="capture FPS ceiling (VFR)")
    ap.add_argument("--region", type=_region, default=None, help="left,top,width,height")
    ap.add_argument("--detector", choices=("hybrid", "baseline"), default=None)
    ap.add_argument("--record-all", action="store_true", help="persist media for every check")

    hud = ap.add_mutually_exclusive_group()
    hud.add_argument("--hud", dest="show_hud", action="store_true")
    hud.add_argument("--no-hud", dest="show_hud", action="store_false")
    ap.set_defaults(show_hud=None)

    lmb = ap.add_mutually_exclusive_group()
    lmb.add_argument("--require-lmb", dest="require_lmb", action="store_true")
    lmb.add_argument("--no-require-lmb", dest="require_lmb", action="store_false")
    ap.set_defaults(require_lmb=None)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config()
    try:
        run_genrush_clean(
            root=ROOT,
            config=config,
            dry_run=args.dry_run,
            fps=args.fps,
            region=args.region,
            show_hud=args.show_hud,
            require_lmb=args.require_lmb,
            record_all=args.record_all,
            detector_name=args.detector,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
