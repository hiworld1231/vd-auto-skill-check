from tools.audit_replay_sequences import (
    _legacy_frenzy_geometry_hint, analyze_records, infer_legacy_sessions,
    strict_normal_placements,
)


def record(session, at, chain, outcome, start):
    return {"metrics": {
        "session_id": session,
        "start_monotonic": at,
        "chain_count": chain,
        "outcome": outcome,
        "synthetic_test_record": False,
        "great": {"start": start, "source": "MEASURED"} if start is not None else None,
    }}


def test_sequence_audit_keeps_missing_geometry_as_a_chain_break():
    rows = [
        record("run", 1.0, 1, "FRENZY_TRANSITION", 120),
        record("run", 1.2, 2, "NO_FIRE", None),
        record("run", 1.5, 3, "GREAT", 280),
        record("run", 1.7, 4, "GREAT", 150),
    ]

    result = analyze_records(rows, permutations=0)

    assert result["frenzy"]["consecutive_pairs"] == 3
    assert result["frenzy"]["pairs_with_two_measured_great_zones"] == 1
    assert result["frenzy"]["missing_geometry_pairs"] == 2
    assert result["frenzy"]["within_run_shuffle_under_50_deg"]["permutations"] == 0


def test_sequence_audit_counts_only_completed_normal_chain_transitions():
    rows = [
        record("run", 1.0, 1, "GREAT", 10),
        record("run", 8.0, 1, "GOOD", 170),
        record("run", 14.0, 1, "MISS", 80),
        record("run", 15.0, 2, "GREAT", 260),
    ]

    result = analyze_records(rows, permutations=0)

    assert result["normal"]["eligible_checks"] == 3
    assert result["normal"]["adjacent_pairs"] == 2
    assert result["normal"]["great_start_separation_deg"]["under_50"] == 0


def test_legacy_session_inference_uses_consecutive_check_ids_and_splits_gaps():
    rows = [
        {"path": "check_20260920_120000_0001.json", "metrics": {
            "check_id": "check_20260920_120000_0001", "session_id": None,
            "chain_count": 1, "outcome": "GREAT", "great": {"start": 20, "source": "MEASURED"}}},
        {"path": "check_20260920_120001_0002.json", "metrics": {
            "check_id": "check_20260920_120001_0002", "session_id": None,
            "chain_count": 1, "outcome": "GOOD", "great": {"start": 190, "source": "MEASURED"}}},
        {"path": "check_20260920_120002_0004.json", "metrics": {
            "check_id": "check_20260920_120002_0004", "session_id": None,
            "chain_count": 1, "outcome": "MISS", "great": {"start": 35, "source": "MEASURED"}}},
    ]

    inferred, audit = infer_legacy_sessions(rows)

    assert audit["inferred_sessions_kept"] == 1
    assert audit["records_in_kept_sessions"] == 2
    assert [row["metrics"]["check_id"] for row in inferred] == [
        "check_20260920_120000_0001", "check_20260920_120001_0002"]


def test_legacy_frenzy_hint_reports_unspecified_geometry_and_repeated_zones():
    rows = [
        {"path": "check_20260920_120000_0001.json", "metrics": {
            "check_id": "check_20260920_120000_0001", "session_id": None,
            "start_monotonic": 1.0, "chain_count": 1, "outcome": "FRENZY_TRANSITION",
            "great": {"start": 20, "width": 10, "source": "UNSPECIFIED"},
            "good": {"width": 42}}},
        {"path": "check_20260920_120000_0002.json", "metrics": {
            "check_id": "check_20260920_120000_0002", "session_id": None,
            "start_monotonic": 1.4, "chain_count": 2, "outcome": "UNKNOWN",
            "great": {"start": 20, "width": 10, "source": "UNSPECIFIED"},
            "good": {"width": 42}}},
    ]
    inferred, _ = infer_legacy_sessions(rows)

    result = _legacy_frenzy_geometry_hint(inferred)

    assert result["geometry_status"] == "plausible zone widths; legacy source labels are unspecified"
    assert result["paired_zone_transitions"] == 1
    assert result["all_pairs"]["under_50"] == 1
    assert result["repeated_same_zone_pairs"] == 1


def test_strict_normal_cohort_requires_measured_zones_sequential_ids_and_plausible_gap():
    def measured(index, at, angle, *, source="MEASURED", session="run"):
        return {"metrics": {
            "session_id": session,
            "check_id": f"check_20260920_120000_{index:04d}",
            "start_monotonic": at,
            "chain_count": 1,
            "outcome": "GREAT",
            "great": {"start": angle, "source": source},
            "good": {"start": (angle + 12) % 360, "source": source},
        }}

    result = strict_normal_placements([
        measured(1, 1.0, 30.0), measured(2, 3.0, 30.2),
        measured(4, 5.0, 31.0), measured(5, 70.0, 32.0),
        measured(6, 72.0, 40.0, source="UNSPECIFIED"),
    ])

    assert result["adjacent_pairs"] == 1
    assert result["pairs_under_1deg"] == 1
    assert result["closest_pairs"][0]["left_check_id"].endswith("_0001")
    assert result["closest_pairs"][0]["right_check_id"].endswith("_0002")
