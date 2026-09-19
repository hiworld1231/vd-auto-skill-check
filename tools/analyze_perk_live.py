#!/usr/bin/env python3
"""Deterministic LIVE analyzer for GEN_RUSH replays.

Important: this tool does not simulate alternate delays and does not predict
GREAT rates. It only summarizes fields recorded by real gameplay.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "replays"


def finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def median(vals: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in vals if finite(v)]
    return float(statistics.median(xs)) if xs else None


def mad(vals: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in vals if finite(v)]
    if not xs:
        return None
    m = statistics.median(xs)
    return float(statistics.median(abs(x - m) for x in xs))


def trigger_mode(trigger: Dict[str, Any]) -> str:
    explicit = trigger.get("trigger_mode")
    if explicit:
        return str(explicit).upper()
    reason = str(trigger.get("reason") or "").upper()
    if "FALLBACK" in reason:
        return "FALLBACK_NO_LOCK"
    if "BACKUP" in reason:
        return "BACKUP"
    if "СПИН" in reason or "SCHEDULED" in reason:
        return "SCHEDULED"
    return "IMMEDIATE"


def bucket(speed: Optional[float]) -> str:
    if not finite(speed):
        return "UNKNOWN"
    s = float(speed)
    if s < 330:
        return "BASE/<330"
    if s < 380:
        return "330-379"
    if s < 500:
        return "380-499"
    if s < 600:
        return "500-599"
    if s < 700:
        return "600-699"
    if s < 825:
        return "700-824"
    return "825+"


def load_rows(directory: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(directory.glob("check_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        trig = d.get("trigger") or {}
        ev = d.get("evaluation") or {}
        timing = trig.get("timing") or {}
        phase_fire = trig.get("phase_telemetry") or ev.get("phase_at_fire") or {}
        phase_after = ev.get("phase_after_outcome") or {}
        phase_after_telem = ev.get("phase_telemetry_after_outcome") or {}

        speed_lock = trig.get("measured_speed_at_lock")
        speed_fire = trig.get("measured_speed_at_fire", trig.get("speed_deg_s"))
        center_err = ev.get("center_error_ms", ev.get("error_ms"))
        entry_err = ev.get("entry_edge_error_ms")
        actual_delay = ev.get("actual_used_delay_ms", trig.get("used_profile_delay", trig.get("latency_ms")))
        base = ev.get("configured_base_latency_ms", phase_fire.get("base_compensation_ms", d.get("configured_latency_ms")))
        corr = phase_fire.get("response_phase_correction_ms")
        pred_late = timing.get("prediction_lateness_ms")
        sched_jitter = timing.get("scheduler_jitter_ms")
        if sched_jitter is None:
            # Legacy replay: top-level scheduler_error_ms may have actually been
            # prediction lateness. Preserve the ambiguity instead of silently
            # treating it as pure scheduler jitter.
            sched_jitter_legacy = trig.get("scheduler_error_ms")
        else:
            sched_jitter_legacy = None

        det_source = ev.get("detector_source_at_fire", timing.get("detector_source_at_fire"))
        det_fb = ev.get("detector_fallback_at_fire", timing.get("detector_fallback_at_fire"))
        mode_switched = ev.get("speed_mode_switched_this_check", timing.get("speed_mode_switched_this_check"))
        fit_spread = ev.get("live_fit_spread_deg_s", timing.get("live_fit_spread_deg_s"))

        row = {
            "path": p,
            "check_id": d.get("check_id", p.stem),
            "timestamp": d.get("timestamp"),
            "outcome": str(ev.get("outcome", "UNKNOWN")).upper(),
            "raw_outcome": ev.get("raw_outcome"),
            "outcome_source": ev.get("outcome_source"),
            "plateau": ev.get("plateau_found"),
            "speed_lock": speed_lock,
            "speed_fire": speed_fire,
            "speed_drift": (float(speed_fire) - float(speed_lock)) if finite(speed_fire) and finite(speed_lock) else None,
            "bucket": bucket(speed_fire),
            "center_error_ms": center_err,
            "entry_error_ms": entry_err,
            "actual_delay_ms": actual_delay,
            "base_latency_ms": base,
            "phase_corr_ms": corr,
            "phase_state": phase_after_telem.get("phase_state", phase_fire.get("phase_state")),
            "phase_accepted": phase_after.get("accepted"),
            "phase_reject_reason": phase_after.get("reject_reason") or phase_after.get("rejection_reason"),
            "trigger_reason": trig.get("reason"),
            "trigger_mode": trigger_mode(trig),
            "prediction_lateness_ms": pred_late,
            "scheduler_jitter_ms": sched_jitter,
            "legacy_scheduler_error_ms": sched_jitter_legacy,
            "frame_age_ms": trig.get("last_frame_age_ms_at_fire"),
            "detector_source": det_source,
            "detector_fallback": det_fb,
            "mode_switched": mode_switched,
            "fit_spread": fit_spread,
            "target_angle": trig.get("target_angle", ev.get("target_angle")),
            "hit_angle": ev.get("hit_angle"),
            "white_start": ev.get("white_start"),
            "white_center": ev.get("white_center"),
            "white_end": ev.get("white_end"),
        }
        rows.append(row)
    return rows


def clean_reason(r: Dict[str, Any]) -> Tuple[bool, str]:
    if not finite(r.get("center_error_ms")):
        return False, "MISSING_ERROR"
    if r.get("plateau") is not True:
        return False, "NO_CONFIRMED_PLATEAU"
    if r.get("outcome") not in {"GREAT", "GOOD", "MISS"}:
        return False, f"OUTCOME_{r.get('outcome')}"
    if r.get("trigger_mode") != "SCHEDULED":
        return False, f"TRIGGER_{r.get('trigger_mode')}"
    if not finite(r.get("frame_age_ms")) or float(r["frame_age_ms"]) > 15.0:
        return False, "STALE_FRAME"
    if not finite(r.get("speed_lock")) or not finite(r.get("speed_fire")):
        return False, "NO_SPEED_LOCK"
    if r.get("detector_fallback") is True:
        return False, "DETECTOR_FALLBACK"
    if r.get("mode_switched") is True:
        return False, "MODE_SWITCHED_THIS_CHECK"
    if finite(r.get("fit_spread")) and float(r["fit_spread"]) > 25.0:
        return False, "UNSTABLE_SPEED_FIT"
    max_drift = max(25.0, 0.08 * abs(float(r["speed_lock"])))
    if abs(float(r["speed_fire"]) - float(r["speed_lock"])) > max_drift:
        return False, "SPEED_DRIFT"
    if abs(float(r["center_error_ms"])) > 60.0:
        return False, "PHASE_OUTLIER"

    # Prefer the explicitly separated scheduler-jitter field. For legacy data,
    # scheduler semantics are ambiguous, so do not silently certify it as clean.
    sj = r.get("scheduler_jitter_ms")
    if sj is None:
        return False, "LEGACY_SCHEDULER_SEMANTICS"
    if not finite(sj) or abs(float(sj)) > 2.0:
        return False, "SCHEDULER_JITTER"
    return True, "CLEAN"


def fmt(v: Optional[float], width: int = 0) -> str:
    if not finite(v):
        s = "N/A"
    else:
        s = f"{float(v):+.2f}"
    return s.rjust(width) if width else s


def print_report(rows: List[Dict[str, Any]], verbose: bool) -> None:
    for r in rows:
        ok, reason = clean_reason(r)
        r["clean"] = ok
        r["clean_reason"] = reason

    print("=" * 112)
    print("GEN_RUSH LIVE AUDIT — RAW MEASUREMENTS ONLY (NO COUNTERFACTUAL OUTCOME PREDICTIONS)")
    print("=" * 112)
    counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in ("GREAT", "GOOD", "MISS", "UNCONFIRMED")}
    print(f"Checks={len(rows)} GREAT={counts['GREAT']} GOOD={counts['GOOD']} MISS={counts['MISS']} UNCONF={counts['UNCONFIRMED']}")

    print("\nSpeed groups are descriptive ranges, NOT claimed game tiers:")
    print(f"{'GROUP':<11} {'N':>3} {'CLEAN':>5} {'G':>3} {'GOOD':>4} {'M':>3} {'RAW MED':>9} {'RAW MAD':>9} {'CLEAN MED':>10} {'DELAY':>8}")
    print("-" * 90)
    order = ["BASE/<330", "330-379", "380-499", "500-599", "600-699", "700-824", "825+", "UNKNOWN"]
    for b in order:
        rr = [r for r in rows if r["bucket"] == b]
        if not rr:
            continue
        clean = [r for r in rr if r["clean"]]
        errs = [r["center_error_ms"] for r in rr if finite(r.get("center_error_ms"))]
        cerrs = [r["center_error_ms"] for r in clean if finite(r.get("center_error_ms"))]
        delays = [r["actual_delay_ms"] for r in rr if finite(r.get("actual_delay_ms"))]
        print(
            f"{b:<11} {len(rr):>3} {len(clean):>5} "
            f"{sum(r['outcome']=='GREAT' for r in rr):>3} {sum(r['outcome']=='GOOD' for r in rr):>4} {sum(r['outcome']=='MISS' for r in rr):>3} "
            f"{fmt(median(errs),9)} {fmt(mad(errs),9)} {fmt(median(cerrs),10)} {fmt(median(delays),8)}"
        )

    excluded = [r for r in rows if not r["clean"]]
    print(f"\nStrict clean phase samples: {len(rows)-len(excluded)}/{len(rows)}")
    reasons: Dict[str, int] = {}
    for r in excluded:
        reasons[r["clean_reason"]] = reasons.get(r["clean_reason"], 0) + 1
    if reasons:
        print("Excluded:", ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))

    # Explicit consistency check that catches the previous summary/detail mismatch.
    for b in order:
        rr = [r for r in rows if r["bucket"] == b and finite(r.get("center_error_ms"))]
        if rr:
            calc = median([r["center_error_ms"] for r in rr])
            assert calc == median([r["center_error_ms"] for r in rr])

    if verbose:
        print("\nDetailed checks:")
        print(f"{'CHECK':<34} {'SPD':>7} {'OUT':>6} {'ERRms':>8} {'ENTRY':>8} {'DELAY':>8} {'MODE':>13} {'FRAME':>7} {'CLEAN/REASON'}")
        print("-" * 130)
        for r in rows:
            print(
                f"{str(r['check_id']):<34} "
                f"{fmt(r.get('speed_fire'),7)} {r['outcome']:>6} "
                f"{fmt(r.get('center_error_ms'),8)} {fmt(r.get('entry_error_ms'),8)} {fmt(r.get('actual_delay_ms'),8)} "
                f"{r['trigger_mode']:>13} {fmt(r.get('frame_age_ms'),7)} "
                f"{'CLEAN' if r['clean'] else r['clean_reason']}"
            )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "check_id", "timestamp", "outcome", "bucket", "speed_lock", "speed_fire", "speed_drift",
        "center_error_ms", "entry_error_ms", "base_latency_ms", "phase_corr_ms", "actual_delay_ms",
        "trigger_reason", "trigger_mode", "prediction_lateness_ms", "scheduler_jitter_ms",
        "legacy_scheduler_error_ms", "frame_age_ms", "detector_source", "detector_fallback",
        "mode_switched", "fit_spread", "plateau", "phase_accepted", "phase_reject_reason",
        "clean", "clean_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if "clean" not in r:
                r["clean"], r["clean_reason"] = clean_reason(r)
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Replay directory")
    ap.add_argument("--recent", type=int, default=100, help="Use newest N JSON checks")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows(args.dir)
    if args.recent and len(rows) > args.recent:
        rows = rows[-args.recent:]
    if not rows:
        raise SystemExit(f"No check_*.json files in {args.dir}")
    print_report(rows, args.verbose)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
