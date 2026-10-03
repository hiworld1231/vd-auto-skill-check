import pytest

from vd.motion import Motion
from vd.planning import Planner
from vd.vision import Arc


def test_subframe_great_keeps_precommitted_timer_through_one_noisy_frame():
    p = Planner(lead_seconds=0, lead_uncertainty=.003)
    p.begin(Arc(20, 10), 300, Arc(31, 42))

    committed = p.update(
        Motion(1.000, 350.0, 900, .10, 4, 1.000, 10),
        frame_at=1.000,
        now=1.000,
    )
    noisy = p.update(
        Motion(1.017, 365.3, 900, .90, 35, 1.017, 11),
        frame_at=1.017,
        now=1.017,
    )

    assert committed is not None
    assert committed.target_grade == 'GREAT'
    assert committed.press_at == pytest.approx(1.038888888888889)
    assert noisy is committed
    assert noisy.target_grade == 'GREAT'


def test_subframe_great_does_not_keep_timer_when_fresh_mean_misses_window():
    p = Planner(lead_seconds=0, lead_uncertainty=.003)
    p.begin(Arc(20, 10), 300, Arc(31, 42))

    committed = p.update(
        Motion(1.000, 350.0, 900, .10, 4, 1.000, 10),
        frame_at=1.000,
        now=1.000,
    )
    diverted = p.update(
        Motion(1.017, 367.5, 1050, .90, 35, 1.017, 11),
        frame_at=1.017,
        now=1.017,
    )

    assert committed is not None and committed.target_grade == 'GREAT'
    assert diverted is not committed
    assert diverted is None or diverted.target_grade != 'GREAT'
