import json

from tools.audit_solver_coverage import audit


def test_coverage_report_distinguishes_predicted_and_blind_attempts(tmp_path):
    events = [
        {'kind': 'BEGIN', 'generation': 1},
        {'kind': 'KEYDOWN', 'generation': 1,
         'plan': {'target_grade': 'GREAT', 'timing_mode': 'BLIND_NO_MOTION'}},
        {'kind': 'BEGIN', 'generation': 2},
        {'kind': 'KEYDOWN', 'generation': 2,
         'plan': {'target_grade': 'GREAT'}},
    ]
    (tmp_path / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)) + '\n')

    result = audit(tmp_path)

    assert result['checks_with_keydown'] == 2
    assert result['timing_modes'] == {'BLIND_NO_MOTION': 1, 'PREDICTED': 1}
