from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_solver_launcher_never_restarts_sober():
    text = (ROOT / "start-vulkan-solver.sh").read_text()
    assert "flatpak kill" not in text
    assert "flatpak run" not in text
    assert "start-vulkan-sober.sh" in text


def test_sober_launcher_starts_capture_once_without_forced_kill():
    text = (ROOT / "start-vulkan-sober.sh").read_text()
    assert "VD_CAPTURE_ENABLE=1" in text
    assert "VD_CAPTURE_INTERVAL_MS=0" in text
    assert "flatpak run" in text
    assert "flatpak kill" not in text
