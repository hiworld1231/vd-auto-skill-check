"""Production detector registry for Violence District CLEAN V5.

Only the two detector paths used by the production runtime are shipped:
- ``hybrid``: BASELINE zone acquisition + HYBRID needle tracking (default)
- ``baseline``: full baseline detector for diagnostics/fallback testing
"""
from typing import Dict, Type

from core.detectors.base import (
    BaseDetector,
    GeometryConfig,
    DEFAULT_GEOMETRY,
    SPACE_TEMPLATE,
    TEMPLATE_H,
    TEMPLATE_W,
    is_angle_in_arc,
)
from core.detectors.baseline import BaselineDetector
from core.detectors.hybrid import HybridDetector

DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
    "baseline": BaselineDetector,
    "contour": BaselineDetector,
    "contour_baseline": BaselineDetector,
    "hybrid": HybridDetector,
}


def get_detector(name: str = "hybrid") -> BaseDetector:
    key = (name or "hybrid").lower().strip().replace("-", "_")
    cls = DETECTOR_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown detector backend '{name}'. Available: {sorted(DETECTOR_REGISTRY)}"
        )
    return cls()
