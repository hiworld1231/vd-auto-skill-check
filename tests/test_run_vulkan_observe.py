import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RUN_VULKAN = PROJECT / "run-vulkan.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_vulkan", RUN_VULKAN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_observe_mode_disables_physical_input_and_learning(monkeypatch, tmp_path):
    module = load_module()
    seen = {}

    def fake_run_session(**kwargs):
        seen.update(kwargs)
        return {
            "mode": "dry-run",
            "frames": 1,
            "claims": 0,
            "physical_presses": 0,
            "capture_sequences_skipped": 0,
            "reasons": {},
            "measurement_reasons": {},
            "performance": {},
            "recording": None,
            "recording_frames_dropped": 0,
        }

    monkeypatch.setattr(module.runtime, "run_session", fake_run_session)
    module.main(["--observe", "--seconds", "0.1", "--recording", str(tmp_path / "rec")])

    assert seen["physical"] is False
    assert seen["learn_lead"] is False
    assert seen["capture_source"] == "vulkan"
    assert seen["synthetic"] is False
