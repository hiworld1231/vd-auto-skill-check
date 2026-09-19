"""
Violent District Skill Check Autonomous System Core Package.
"""

from .capture import ScreenGrabber, CAPTURE_REGION
from .vision import VisionEngine, is_angle_in_arc
from .predictor import SkillCheckPredictor, DEFAULT_SPEED_DEG_S, DEFAULT_LATENCY_MS
from .trigger import HardwareTrigger, PreciseTriggerScheduler
from .mouse_tracker import MouseTracker

__all__ = [
    "ScreenGrabber",
    "CAPTURE_REGION",
    "VisionEngine",
    "is_angle_in_arc",
    "SkillCheckPredictor",
    "DEFAULT_SPEED_DEG_S",
    "DEFAULT_LATENCY_MS",
    "HardwareTrigger",
    "PreciseTriggerScheduler",
    "MouseTracker",
]
