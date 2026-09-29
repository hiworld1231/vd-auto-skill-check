import importlib.util
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vulkan_prompt_probe", ROOT / "tools" / "vulkan_prompt_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(probe)


def test_best_match_recovers_known_template_location_and_scale():
    rng = np.random.default_rng(1234)
    template = rng.integers(0, 256, size=(9, 11), dtype=np.uint8)
    image = rng.integers(0, 30, size=(48, 64), dtype=np.uint8)
    x, y = 23, 17
    image[y:y + template.shape[0], x:x + template.shape[1]] = template

    score, scale, location, size = probe.best_match(
        image, template, [0.8, 1.0, 1.2]
    )

    assert score > 0.99
    assert scale == 1.0
    assert location == (x, y)
    assert size == (template.shape[1], template.shape[0])
