import importlib.util
import json
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


def test_ctrl_c_prints_saved_manifest_summary(monkeypatch, tmp_path, capsys):
    module = load_module()
    recording = tmp_path / "rec"

    def fake_run_session(**kwargs):
        recording.mkdir(parents=True, exist_ok=True)
        (recording / "manifest.json").write_text(json.dumps({
            "complete": False,
            "frames_written": 42,
            "frames_dropped": 0,
            "capture_sequences_skipped": 3,
            "capture_source": "vulkan",
            "capture_source_size": [1600, 878],
            "capture_frame_size": [320, 240],
            "capture_roi": [800, 420, 320, 240],
            "measurement_reasons": {"NO_PROMPT": 100, "OK": 12},
            "detector_ui_scale": 1.0,
            "performance": {"elapsed_seconds": 2.5},
        }))
        raise KeyboardInterrupt

    monkeypatch.setattr(module.runtime, "run_session", fake_run_session)
    module.main(["--observe", "--recording", str(recording)])

    output = capsys.readouterr().out
    assert "Stopped." in output
    assert '"recording"' in output
    assert '"measurement_reasons"' in output
    assert '"OK": 12' in output
    assert '"detector_ui_scale": 1.0' in output
