import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_practice_zones.py"
SPEC = importlib.util.spec_from_file_location("build_practice_zones", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def record(outcome, chain, start, *, source="MEASURED"):
    return {"metrics": {
        "outcome": outcome,
        "chain_count": chain,
        "great": {"start": start, "width": 10.5, "source": source},
        "good": {"start": start + 11.5, "width": 42.0, "source": source},
        "gap": 1.0,
    }}


def test_default_pool_uses_completed_normal_checks_and_keeps_their_saved_angles():
    records = [record("GREAT", 1, 20), record("GOOD", 1, 180), record("MISS", 1, 300),
               record("NO_FIRE", 1, 45), record("FRENZY_TRANSITION", 1, 90),
               record("GREAT", 2, 120), record("UNCONFIRMED", 1, 240)]

    samples = builder.build_samples(records)

    assert [sample[3] for sample in samples] == [20.0, 180.0, 300.0]
