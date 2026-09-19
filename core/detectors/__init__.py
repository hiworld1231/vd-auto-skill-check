"""
Skill Check Detector Backends Registry.
Provides access to multiple computer vision detection architectures:
1. CONTOUR_BASELINE: Original OpenCV matchTemplate + warpPolar pipeline.
2. FIXED_POLAR: Precomputed polar coordinate lookup table.
3. PRECOMPUTED_RAYS: Discrete radial ray scoring.
4. LOCAL_TRACKER: Dynamic local angular window tracking.
5. HYBRID: Primary candidate combining fixed geometry, local search, and lifecycle zone caching.
6. MINIMAL_COLOR: Direct integer BGR indexing without cvtColor.
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
from core.detectors.polar import FixedPolarDetector
from core.detectors.rays import PrecomputedRaysDetector
from core.detectors.local import LocalTrackerDetector
from core.detectors.hybrid import HybridDetector
from core.detectors.color_index import MinimalColorDetector

DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
    "baseline": BaselineDetector,
    "contour": BaselineDetector,
    "contour_baseline": BaselineDetector,
    "polar": FixedPolarDetector,
    "fixed_polar": FixedPolarDetector,
    "rays": PrecomputedRaysDetector,
    "precomputed_rays": PrecomputedRaysDetector,
    "local": LocalTrackerDetector,
    "local_tracker": LocalTrackerDetector,
    "hybrid": HybridDetector,
    "color_index": MinimalColorDetector,
    "minimal_color": MinimalColorDetector,
}


def get_detector(name: str = "hybrid") -> BaseDetector:
    """Instantiates a detector backend by name (case-insensitive)."""
    key = (name or "hybrid").lower().strip().replace("-", "_")
    cls = DETECTOR_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown detector backend '{name}'. Available: {list(DETECTOR_REGISTRY.keys())}")
    return cls()
