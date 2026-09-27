#!/usr/bin/env python3
"""Bundle ordered, contiguous ordinary skill-check routes from replay manifests."""
import argparse
import json
import math
from pathlib import Path
import re

if __package__:
    from .audit_replay_manifest_order import build_normal_routes, load_manifest_sessions
else:
    from audit_replay_manifest_order import build_normal_routes, load_manifest_sessions


def build_measured_normal_routes(records):
    """Export only explicit-session, consecutive ordinary checks with measured zones."""
    sessions = {}
    for row in records:
        metrics = row.get("metrics", row)
        great, good = metrics.get("great"), metrics.get("good")
        if not isinstance(great, dict) or not isinstance(good, dict):
            continue
        if (not metrics.get("session_id") or metrics.get("synthetic_test_record")
                or metrics.get("chain_count") != 1
                or str(metrics.get("outcome", "")).upper() not in {"GREAT", "GOOD", "MISS"}
                or great.get("source") != "MEASURED" or good.get("source") != "MEASURED"):
            continue
        try:
            check_id = str(metrics["check_id"])
            match = re.search(r"_(\d+)$", check_id)
            at = float(metrics["start_monotonic"])
            width = float(great["width"])
            good_width = float(good["width"])
            gap = float(metrics["gap"])
            start = float(great["start"]) % 360
            if match is None:
                continue
            check = {"at": at, "index": int(match.group(1)), "width": width,
                     "good_width": good_width, "gap": gap, "start": start}
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(at) and 9 <= width <= 12 and 40 <= good_width <= 44
                and 0 <= gap <= 2):
            continue
        sessions.setdefault(str(metrics["session_id"]), []).append(check)

    routes = []
    for checks in sessions.values():
        checks.sort(key=lambda check: check["at"])
        route = []
        for check in checks:
            adjacent = bool(route and check["index"] == route[-1]["index"] + 1
                            and 1.5 <= check["at"] - route[-1]["at"] <= 60)
            if adjacent:
                route.append(check)
            else:
                if len(route) >= 2:
                    routes.append(route)
                route = [check]
        if len(route) >= 2:
            routes.append(route)

    routes.sort(key=lambda route: (-len(route), route[0]["at"], route[0]["index"]))
    return [[[
        round(check["width"], 3), round(check["good_width"], 3),
        round(check["gap"], 3), round(check["start"], 3),
    ] for check in route] for route in routes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("--geometry-report", type=Path,
                        help="optional all-replay geometry report with explicit session IDs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sessions, inventory = load_manifest_sessions(args.manifest_root)
    manifest_routes = build_normal_routes(sessions)
    measured_routes = []
    if args.geometry_report is not None:
        report = json.loads(args.geometry_report.read_text(encoding="utf-8"))
        measured_routes = build_measured_normal_routes(report["records"])
    routes = measured_routes + manifest_routes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "// Contiguous ordinary routes: explicit-session measured traces first, then manifest traces.\n"
        "window.VDDefaultZoneSequences = " + json.dumps(routes, separators=(",", ":")) + ";\n"
        "window.VDDefaultZoneSequenceMeta = " + json.dumps({
            "explicit_session_measured_routes": len(measured_routes),
            "explicit_session_measured_checks": sum(map(len, measured_routes)),
            "manifest_routes": len(manifest_routes),
            "manifest_checks": sum(map(len, manifest_routes)),
        }, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    lengths = [len(route) for route in routes]
    print(json.dumps({
        "ordered_routes": len(routes),
        "checks": sum(lengths),
        "explicit_session_measured_routes": len(measured_routes),
        "explicit_session_measured_checks": sum(map(len, measured_routes)),
        "manifest_routes": len(manifest_routes),
        "manifest_checks": sum(map(len, manifest_routes)),
        "median_route_checks": sorted(lengths)[len(lengths) // 2] if lengths else None,
        "longest_route_checks": max(lengths, default=0),
        "capture_inventory": inventory,
        "route_boundary_transitions_captured": False,
        "playback_cycles_routes_back_to_back": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
