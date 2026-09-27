from tools.build_default_sequences import build_measured_normal_routes


def record(session, index, at, *, outcome="GREAT", source="MEASURED", chain=1,
           great_width=10.5, good_width=42, gap=1.0, start=20):
    return {
        "path": f"check_20260920_120000_{index:04d}.json",
        "metrics": {
            "check_id": f"check_20260920_120000_{index:04d}",
            "session_id": session,
            "start_monotonic": at,
            "chain_count": chain,
            "outcome": outcome,
            "synthetic_test_record": False,
            "great": {"start": start, "width": great_width, "source": source},
            "good": {"width": good_width, "source": source},
            "gap": gap,
        },
    }


def test_measured_normal_routes_require_explicit_session_and_strict_adjacency():
    rows = [
        record("session-a", 1, 1.0, start=20.25),
        record("session-a", 2, 3.0, outcome="GOOD", start=190.75),
        record("session-a", 4, 4.0, start=80),  # skipped ID breaks the route
        record("session-a", 5, 70.0, start=250),  # long gap breaks the route
        record("session-b", 3, 3.5, start=45),  # another session never joins
        record("session-a", 6, 71.0, source="UNSPECIFIED", start=100),
        record("session-a", 7, 72.0, chain=2, start=110),
        record("session-a", 8, 73.0, outcome="UNKNOWN", start=120),
        record(None, 1, 1.1, start=30),  # no implicit session inference
    ]

    assert build_measured_normal_routes(rows) == [[
        [10.5, 42.0, 1.0, 20.25],
        [10.5, 42.0, 1.0, 190.75],
    ]]
