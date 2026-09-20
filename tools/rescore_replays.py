#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _signed_delta(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def _median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else None


def _mad(values: Iterable[float], center: Optional[float] = None) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    c = float(statistics.median(vals) if center is None else center)
    return float(statistics.median(abs(v - c) for v in vals))


def _post_fire_samples(doc: Dict[str, Any]) -> List[Tuple[float, float]]:
    trigger = (doc.get("trigger_event") or {}).get("trigger_time")
    if trigger is None:
        return []
    out: List[Tuple[float, float]] = []
    for frame in doc.get("frames") or []:
        t, a = frame.get("t"), frame.get("needle_angle")
        if t is None or a is None or float(t) < float(trigger) - 0.005:
            continue
        out.append((float(t), float(a) % 360.0))
    return out


def _has_plateau(samples: List[Tuple[float, float]], width_deg: float = 1.4) -> bool:
    if len(samples) < 4:
        return False
    for i in range(len(samples) - 3):
        chunk = samples[i:i + 4]
        ref = chunk[0][1]
        vals = [ref + _signed_delta(a, ref) for _, a in chunk]
        if max(vals) - min(vals) <= width_deg:
            return True
    return False


def _impossible_jump(doc: Dict[str, Any]) -> bool:
    samples = _post_fire_samples(doc)
    speed = (doc.get("outcome_info") or {}).get("speed_at_fire")
    if speed is None:
        speed = (doc.get("trigger_event") or {}).get("speed_deg_s")
    speed = max(20.0, abs(float(speed or 278.0)))
    for (t0, a0), (t1, a1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        step = abs(_signed_delta(a1, a0))
        # Generous guard: real motion + detector slack.  The known 8° -> 292°
        # post-hit decoy is still far outside this envelope.
        allowed = max(12.0, speed * dt * 1.8 + 6.0)
        if step > allowed:
            return True
    return False


def _speed_bin(speed: Optional[float]) -> str:
    if speed is None:
        return "unknown"
    s = float(speed)
    if s <= 320:
        return "<=320"
    if s <= 450:
        return "320-450"
    if s <= 650:
        return "450-650"
    if s <= 800:
        return "650-800"
    return "800+"


def _trusted_ideal(doc: Dict[str, Any]) -> Optional[float]:
    info = doc.get("outcome_info") or {}
    fit = info.get("fit_telemetry") or {}
    if int(doc.get("chain_count") or 1) != 1:
        return None
    if info.get("plateau_found") is not True:
        return None
    if info.get("outcome") not in {"GREAT", "GOOD", "MISS"}:
        return None
    if info.get("detector_fallback"):
        return None
    if info.get("white_source") and not str(info.get("white_source")).startswith("MEASURED"):
        return None
    jitter = info.get("scheduler_jitter_ms")
    if jitter is None or abs(float(jitter)) > 4.5:
        return None
    if int(fit.get("fit_sample_count") or 0) < 5:
        return None
    resid = fit.get("fit_residual_mad_deg")
    if resid is not None and float(resid) > 3.0:
        return None
    spread = fit.get("live_fit_spread")
    if spread is not None and float(spread) > 60.0:
        return None
    delta = fit.get("short_vs_long_delta")
    if delta is not None and abs(float(delta)) > 75.0:
        return None
    used = info.get("effective_dispatch_lead_ms", info.get("actual_used_delay_ms"))
    error = info.get("center_error_ms")
    if used is None or error is None:
        return None
    if abs(float(error)) > 80.0:
        return None
    ideal = float(used) + float(error)
    return ideal if 35.0 <= ideal <= 160.0 else None


def analyze(paths: Iterable[Path]) -> Dict[str, Any]:
    docs: List[Dict[str, Any]] = []
    bad_json: List[str] = []
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                doc["_path"] = str(path)
                docs.append(doc)
        except Exception:
            bad_json.append(str(path))

    outcomes = Counter(str((d.get("outcome_info") or {}).get("outcome", "UNKNOWN")) for d in docs)
    impossible = [d["check_id"] for d in docs if _impossible_jump(d)]
    false_frenzy = [
        d["check_id"] for d in docs
        if str((d.get("outcome_info") or {}).get("outcome")) == "FRENZY_TRANSITION"
        and _has_plateau(_post_fire_samples(d))
    ]

    speed_stats: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in docs:
        info = d.get("outcome_info") or {}
        speed = info.get("speed_at_fire")
        if speed is None:
            speed = (d.get("trigger_event") or {}).get("speed_deg_s")
        grouped[_speed_bin(speed)].append(d)

    for name, group in grouped.items():
        errors = [
            (d.get("outcome_info") or {}).get("center_error_ms")
            for d in group
            if (d.get("outcome_info") or {}).get("center_error_ms") is not None
        ]
        speed_stats[name] = {
            "count": len(group),
            "median_center_error_ms": _median(errors),
            "misses": sum(1 for d in group if (d.get("outcome_info") or {}).get("outcome") == "MISS"),
            "greats": sum(1 for d in group if (d.get("outcome_info") or {}).get("outcome") == "GREAT"),
            "goods": sum(1 for d in group if (d.get("outcome_info") or {}).get("outcome") == "GOOD"),
        }

    ideals = [x for x in (_trusted_ideal(d) for d in docs) if x is not None]
    ideal_med = _median(ideals)
    return {
        "files": len(docs),
        "bad_json": bad_json,
        "outcomes": dict(outcomes),
        "dataset_has_great": outcomes.get("GREAT", 0) > 0,
        "impossible_post_fire_jump_count": len(impossible),
        "impossible_post_fire_jump_ids": impossible,
        "false_frenzy_plateau_count": len(false_frenzy),
        "false_frenzy_plateau_ids": false_frenzy,
        "trusted_ideal_lead_samples": len(ideals),
        "trusted_ideal_lead_median_ms": ideal_med,
        "trusted_ideal_lead_mad_ms": _mad(ideals, ideal_med),
        "speed_bins": speed_stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit VD skill-check replay JSON telemetry")
    ap.add_argument("path", nargs="?", default="replays", help="replay directory or JSON file")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()

    root = Path(args.path)
    paths = [root] if root.is_file() else sorted(root.rglob("check_*.json"))
    result = analyze(paths)

    if args.as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(f"Replay files: {result['files']}")
    print("Outcomes:", ", ".join(f"{k}={v}" for k, v in sorted(result["outcomes"].items())))
    if not result["dataset_has_great"]:
        print("WARNING: no GREAT rows are present; GREAT-rate statistics are survivorship-biased/incomplete.")
    print(f"Impossible post-fire jumps: {result['impossible_post_fire_jump_count']}")
    print(f"Frenzy transitions with a freeze plateau: {result['false_frenzy_plateau_count']}")
    print(
        "Trusted ideal lead: "
        f"n={result['trusted_ideal_lead_samples']} "
        f"median={result['trusted_ideal_lead_median_ms']}ms "
        f"MAD={result['trusted_ideal_lead_mad_ms']}ms"
    )
    for name in ("<=320", "320-450", "450-650", "650-800", "800+", "unknown"):
        if name not in result["speed_bins"]:
            continue
        row = result["speed_bins"][name]
        print(
            f"speed {name:>7}: n={row['count']:3d} "
            f"GREAT={row['greats']:3d} GOOD={row['goods']:3d} MISS={row['misses']:3d} "
            f"median_error={row['median_center_error_ms']}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
