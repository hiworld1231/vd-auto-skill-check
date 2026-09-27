import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_frenzy_sequences.py"
SPEC = importlib.util.spec_from_file_location("build_frenzy_sequences", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def record(session, at, chain, outcome, start, *, source="MEASURED"):
    return {
        "metrics": {
            "session_id": session,
            "start_monotonic": at,
            "check_id": f"check_20260920_120000_{chain:04d}",
            "chain_count": chain,
            "outcome": outcome,
            "great": {"start": start, "width": 10.0, "source": source},
            "good": {"start": (start + 11.0) % 360, "width": 42.0, "source": source},
            "gap": 1.0,
            "observed_speed_median": 300.0 + chain,
        }
    }


def test_frenzy_route_includes_the_observed_terminal_check_without_joining_sessions():
    records = [
        record("session-a", 10.0, 1, "FRENZY_TRANSITION", 20.0),
        record("session-a", 10.4, 2, "FRENZY_TRANSITION", 190.0),
        record("session-a", 10.9, 3, "MISS", 35.0),
        record("session-b", 11.0, 1, "FRENZY_TRANSITION", 280.0),
        record("session-b", 11.4, 2, "GOOD", 100.0),
    ]

    routes = builder.build_routes(records)

    assert [route["session_id"] for route in routes] == ["session-a", "session-b"]
    assert [check["great_start_deg"] for check in routes[0]["checks"]] == [20.0, 190.0, 35.0]
    assert routes[0]["last_observed_outcome"] == "MISS"
    assert routes[0]["truncated"] is False
    assert [route["session_id"] for route in routes] == ["session-a", "session-b"]
    assert [check["great_start_deg"] for check in routes[1]["checks"]] == [280.0, 100.0]
    assert routes[1]["last_observed_outcome"] == "GOOD"


def test_frenzy_route_breaks_on_a_chain_gap_and_skips_unmeasured_geometry():
    records = [
        record("session-a", 10.0, 1, "FRENZY_TRANSITION", 20.0),
        record("session-a", 12.0, 2, "FRENZY_TRANSITION", 190.0),
        record("session-a", 12.4, 3, "FRENZY_TRANSITION", 35.0, source="UNSPECIFIED"),
    ]

    routes = builder.build_routes(records)

    assert routes == []


def test_frenzy_route_keeps_measured_zone_transitions_with_wider_good_arc_geometry():
    records = [
        record("session-a", 10.0, 1, "FRENZY_TRANSITION", 20.0),
        record("session-a", 10.4, 2, "FRENZY_TRANSITION", 190.0),
        record("session-a", 10.8, 3, "GOOD", 35.0),
    ]
    records[1]["metrics"]["good"]["width"] = 38.0
    records[1]["metrics"]["gap"] = 5.5

    routes = builder.build_routes(records)
    transitions = builder.build_transition_pool(routes)

    assert len(routes) == 1
    assert len(routes[0]["checks"]) == 3
    assert transitions == [
        [10.0, 38.0, 5.5, 170.0, 1, 20.0],
        [10.0, 42.0, 1.0, 205.0, 2, 190.0],
    ]


def test_infinite_transition_pool_keeps_only_measured_adjacent_route_steps():
    records = [
        record("session-a", 10.0, 1, "FRENZY_TRANSITION", 20.0),
        record("session-a", 10.4, 2, "FRENZY_TRANSITION", 190.0),
        record("session-a", 10.8, 3, "FRENZY_TRANSITION", 35.0, source="UNSPECIFIED"),
    ]

    transitions = builder.build_transition_pool(builder.build_routes(records))

    assert transitions == [[10.0, 42.0, 1.0, 170.0, 1, 20.0]]


def test_transition_timing_uses_recorded_keydown_and_next_check_start(tmp_path):
    records = []
    for index, (at, pressed_at, outcome) in enumerate([
        (10.0, 10.25, "FRENZY_TRANSITION"),
        (10.4, 10.62, "FRENZY_TRANSITION"),
        (10.8, None, "MISS"),
    ]):
        path = tmp_path / f"check-{index}.json"
        raw = {"start_monotonic": at}
        if pressed_at is not None:
            raw["trigger_event"] = {"keydown_syn_time": pressed_at}
        path.write_text(json.dumps(raw), encoding="utf-8")
        row = record("session-a", at, index + 1, outcome, 20.0 + index * 160)
        row["path"] = str(path)
        records.append(row)

    timing = builder.build_transition_timing(builder.build_routes(records))

    assert timing == {
        "count": 2, "p10_ms": 153.0, "median_ms": 165.0,
        "p90_ms": 177.0, "min_ms": 150.0, "max_ms": 180.0,
    }


def test_relative_position_cross_validation_holds_out_the_whole_session():
    routes = []
    for session, start in (("a", 10.0), ("b", 120.0), ("c", 250.0)):
        routes.append({"session_id": session, "checks": [
            {"chain_count": 1, "great_start_deg": start},
            {"chain_count": 2, "great_start_deg": (start + 170.0) % 360},
            {"chain_count": 3, "great_start_deg": (start + 340.0) % 360},
        ]})

    result = builder.cross_validate_relative_positions(routes)

    assert result["2"]["sessions"] == 3
    assert result["2"]["median_absolute_error_deg"] == 0.0
    assert result["3"]["median_absolute_error_deg"] == 0.0
