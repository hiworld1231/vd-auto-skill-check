from tools.audit_session_captures import merge_target_spans
from tools.audit_solver_session_captures import candidate_targets


def span(start, end, great=120.0):
    return {
        "start_s": start,
        "end_s": end,
        "duration_s": end - start,
        "sampled_frames": 20,
        "needle_sweep_deg": 160.0,
        "needle_speed_deg_s": 320.0,
        "great_start_median_deg": great,
        "good_start_median_deg": (great + 12.0) % 360,
        "great_width_median_deg": 10.5,
        "good_width_median_deg": 42.0,
        "center_median_px": [160.0, 120.0],
    }


def test_brief_detector_dropout_with_same_ring_merges_into_one_target():
    parts = [span(1.00, 1.30), span(1.35, 1.65), span(1.70, 2.00)]

    merged = merge_target_spans(parts)

    assert len(merged) == 1
    assert merged[0]["start_s"] == 1.00
    assert merged[0]["end_s"] == 2.00
    assert merged[0]["sampled_frames"] == 60
    assert merged[0]["merged_segments"] == 3


def test_changed_angle_is_a_new_target_even_after_a_brief_gap():
    merged = merge_target_spans([span(1.00, 1.30), span(1.35, 1.65, great=270.0)])

    assert len(merged) == 2


def test_same_target_after_a_longer_gap_is_not_merged():
    merged = merge_target_spans([span(1.00, 1.30), span(1.50, 1.80)])

    assert len(merged) == 2


def test_angle_comparison_wraps_at_zero_degrees():
    merged = merge_target_spans([span(1.00, 1.30, 359.5), span(1.35, 1.65, 0.5)])

    assert len(merged) == 1


def test_solver_video_audit_prefers_merged_targets_over_raw_detector_spans():
    raw = [span(1.00, 1.30), span(1.35, 1.65)]
    session = {"candidate_skillcheck_spans": raw,
               "candidate_target_spans": merge_target_spans(raw)}

    assert len(candidate_targets(session)) == 1
