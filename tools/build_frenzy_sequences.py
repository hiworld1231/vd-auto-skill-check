#!/usr/bin/env python3
"""Build measured Frenzy traces without inventing links between recordings."""
import argparse
from collections import Counter
import json
import math
import statistics
import re
from pathlib import Path


def _session_key(metrics, path=""):
    if metrics.get("session_id"):
        return str(metrics["session_id"])
    match = re.search(r"check_(\d{8})_", str(metrics.get("check_id", "")))
    return "legacy:" + (match.group(1) if match else str(path))


def _check(metrics, source_path=""):
    great, good = metrics.get("great"), metrics.get("good")
    if not great or not good or great.get("source") != "MEASURED":
        return None
    try:
        start = float(great["start"]) % 360
        great_width = float(great["width"])
        good_width = float(good["width"])
        gap = float(metrics["gap"])
        chain = int(metrics["chain_count"])
        at = float(metrics["start_monotonic"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (9 <= great_width <= 12 and 35 <= good_width <= 45 and 0 <= gap <= 8
            and chain >= 1 and math.isfinite(at)):
        return None
    speed = metrics.get("observed_speed_median")
    return {"great_start_deg": start, "great_width_deg": great_width,
            "good_width_deg": good_width, "gap_deg": gap, "chain_count": chain,
            "speed_deg_s": float(speed) if isinstance(speed, (int, float)) and speed > 0 else None,
            "outcome": str(metrics.get("outcome", "UNKNOWN")).upper(), "at": at,
            "source_path": str(source_path)}


def cross_validate_relative_positions(routes):
    """Predict each check's offset from its own route start, holding out a session."""
    errors = {}
    observed_sessions = {}
    for held_out in routes:
        checks = held_out.get("checks", [])
        if len(checks) < 2:
            continue
        session = held_out.get("session_id")
        origin = float(checks[0]["great_start_deg"])
        for check in checks[1:]:
            position = int(check["chain_count"])
            offsets = []
            for route in routes:
                if route.get("session_id") == session or not route.get("checks"):
                    continue
                other_origin = float(route["checks"][0]["great_start_deg"])
                for other in route["checks"]:
                    if int(other["chain_count"]) == position:
                        offsets.append((float(other["great_start_deg"]) - other_origin) % 360)
                        break
            if not offsets:
                continue
            sin_sum = sum(math.sin(math.radians(value)) for value in offsets)
            cos_sum = sum(math.cos(math.radians(value)) for value in offsets)
            if abs(sin_sum) + abs(cos_sum) < 1e-9:
                continue
            predicted = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
            actual = (float(check["great_start_deg"]) - origin) % 360
            error = abs((actual - predicted + 180) % 360 - 180)
            errors.setdefault(position, []).append(error)
            observed_sessions.setdefault(position, set()).add(session)
    return {
        str(position): {
            "sessions": len(observed_sessions[position]),
            "samples": len(values),
            "median_absolute_error_deg": round(statistics.median(values), 3),
            "mean_absolute_error_deg": round(statistics.mean(values), 3),
        }
        for position, values in sorted(errors.items())
    }


def build_routes(records, *, max_gap=1.5):
    """Return contiguous measured routes, retaining the first observed terminal check."""
    sessions = {}
    for row in records:
        metrics = row.get("metrics", row)
        if metrics.get("start_monotonic") is None:
            continue
        check = _check(metrics, row.get("path", ""))
        if check is None:
            continue
        session = _session_key(metrics, row.get("path", ""))
        sessions.setdefault(session, []).append((check, metrics))

    routes = []
    for session, items in sessions.items():
        items.sort(key=lambda item: item[0]["at"])
        route = None
        for check, metrics in items:
            outcome = check["outcome"]
            if route is None:
                if outcome == "FRENZY_TRANSITION":
                    route = {"session_id": session, "checks": [check],
                             "last_observed_outcome": None, "truncated": True}
                continue

            previous = route["checks"][-1]
            continues = (check["chain_count"] == previous["chain_count"] + 1
                         and 0 < check["at"] - previous["at"] < max_gap)
            if continues:
                route["checks"].append(check)
                if outcome != "FRENZY_TRANSITION":
                    route["last_observed_outcome"] = outcome
                    route["truncated"] = False
                    if len(route["checks"]) >= 2:
                        routes.append(route)
                    route = None
            else:
                if len(route["checks"]) >= 2:
                    routes.append(route)
                route = ({"session_id": session, "checks": [check],
                          "last_observed_outcome": None, "truncated": True}
                         if outcome == "FRENZY_TRANSITION" else None)
        if route is not None:
            if len(route["checks"]) >= 2:
                routes.append(route)

    routes.sort(key=lambda route: (-len(route["checks"]), route["session_id"],
                                  route["checks"][0]["at"]))
    return routes


def make_audit(routes):
    checks = [check for route in routes for check in route["checks"]]
    separations = [abs((right["great_start_deg"] - left["great_start_deg"] + 180) % 360 - 180)
                   for route in routes for left, right in zip(route["checks"], route["checks"][1:])]
    speeds = {}
    for check in checks:
        if check["speed_deg_s"] is not None:
            speeds.setdefault(check["chain_count"], []).append(check["speed_deg_s"])
    if not speeds:
        speeds = {1: [280.0]}
    fallback_values = [value for count, values in speeds.items() if count >= 4 for value in values]
    fallback = statistics.median(fallback_values or [value for values in speeds.values() for value in values])
    speed_by_count = [statistics.median(speeds.get(count, [fallback]))
                      for count in range(1, max(speeds) + 1)] + [fallback]

    def distribution(values):
        if not values:
            return {"count": 0, "min": None, "median": None, "max": None}
        return {"count": len(values), "min": round(min(values), 3),
                "median": round(statistics.median(values), 3), "max": round(max(values), 3)}

    return {
        "routes": len(routes), "checks": len(checks),
        "observed_transitions": len(separations),
        "longest_route": max((len(route["checks"]) for route in routes), default=0),
        "routes_ending_in_observed_outcome": sum(not route["truncated"] for route in routes),
        "truncated_routes": sum(route["truncated"] for route in routes),
        "observed_great_start_separation_deg": distribution(separations),
        "transition_steps_for_infinite_mode": sum(max(0, len(route["checks"]) - 1)
                                                   for route in routes),
        "recorded_angles_synthetic": False,
        "speed_medians_deg_s": [round(speed, 3) for speed in speed_by_count],
        "cross_session_relative_position_cv": cross_validate_relative_positions(routes),
        "uniform_random_angle_baseline": {
            "median_absolute_error_deg": 90.0,
            "mean_absolute_error_deg": 90.0,
        },
    }


def build_transition_pool(routes):
    """Export only observed consecutive steps as relative angles plus target geometry."""
    transitions = []
    for route in routes:
        for previous, following in zip(route["checks"], route["checks"][1:]):
            delta = (following["great_start_deg"] - previous["great_start_deg"]) % 360
            transitions.append([
                round(following["great_width_deg"], 3),
                round(following["good_width_deg"], 3),
                round(following["gap_deg"], 3),
                round(delta, 3),
                previous["chain_count"],
                round(previous["great_start_deg"], 3),
            ])
    return transitions


def build_transition_timing(routes):
    """Summarize observed keydown-to-next-ring delays for contiguous checks."""
    delays = []
    cache = {}

    def replay(path):
        if path not in cache:
            cache[path] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cache[path]

    for route in routes:
        for previous, following in zip(route["checks"], route["checks"][1:]):
            if not previous.get("source_path") or not following.get("source_path"):
                continue
            try:
                pressed_at = (replay(previous["source_path"])
                              .get("trigger_event", {}).get("keydown_syn_time"))
                next_at = replay(following["source_path"]).get("start_monotonic")
                delay_ms = (float(next_at) - float(pressed_at)) * 1000
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if math.isfinite(delay_ms) and 0 < delay_ms < 2000:
                delays.append(delay_ms)
    if not delays:
        return {"count": 0, "p10_ms": None, "median_ms": None, "p90_ms": None,
                "min_ms": None, "max_ms": None}
    ordered = sorted(delays)
    def quantile(p):
        position = (len(ordered) - 1) * p
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    return {"count": len(ordered), "p10_ms": round(quantile(.1), 1),
            "median_ms": round(statistics.median(ordered), 1),
            "p90_ms": round(quantile(.9), 1),
            "min_ms": round(ordered[0], 1), "max_ms": round(ordered[-1], 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="audit_replay_geometry.py JSON report")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    routes = build_routes(report["records"])
    transitions = build_transition_pool(routes)
    transition_timing = build_transition_timing(routes)
    transition_levels = Counter(int(row[4]) for row in transitions)
    audit = make_audit(routes)
    speed_by_count = audit.pop("speed_medians_deg_s")
    output_routes = [{"checks": [[round(check["great_width_deg"], 3),
                                  round(check["good_width_deg"], 3),
                                  round(check["gap_deg"], 3),
                                  round(check["great_start_deg"], 3),
                                  check["chain_count"]]
                                for check in route["checks"]],
                      "last_observed_outcome": route["last_observed_outcome"],
                      "truncated": route["truncated"]}
                     for route in routes]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("// Exact measured, contiguous replay traces.\n"
                        "window.VDFrenzySequences = " + json.dumps(output_routes, separators=(",", ":")) + ";\n"
                        "// Median observed speeds by recorded chain_count; final entry is pooled fallback.\n"
                        "window.VDFrenzySpeeds = " + json.dumps(speed_by_count, separators=(",", ":")) + ";\n"
                        "// Infinite continuation reuses these measured relative steps in fixed order, relative to the current angle.\n"
                        "// Each row: destination geometry, angle delta, source chain_count and source angle.\n"
                        "window.VDFrenzyTransitions = " + json.dumps(transitions, separators=(",", ":")) + ";\n"
                        "// Replay-observed keydown-to-next-check timing (milliseconds).\n"
                        "window.VDFrenzyTiming = " + json.dumps(transition_timing, separators=(",", ":")) + ";\n",
                        encoding="utf-8")
    audit["speed_medians_deg_s"] = speed_by_count
    audit["transition_steps_for_infinite_mode"] = len(transitions)
    audit["infinite_continuation"] = (
        "Cyclic, fixed-order replay of measured relative angle transitions applied to the current GREAT start; "
        "the continuation is an empirical simulator model, not an observed game run.")
    audit["infinite_continuation_generates_absolute_angles"] = True
    audit["transition_timing_ms"] = transition_timing
    audit["transition_source_chain_count_distribution"] = {
        str(level): count for level, count in sorted(transition_levels.items())
    }
    audit["highest_observed_transition_source_chain_count"] = max(transition_levels, default=None)
    if args.audit_report:
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
