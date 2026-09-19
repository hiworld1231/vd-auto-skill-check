#!/usr/bin/env python3
"""
Rebuild clean speed_profiles.json from real historical gameplay replays.
Applies Schema Version 2 and Calibration Epoch 2.
Historical data is marked as epoch 1 shadow priors (trusted=False) until
confirmed by live gameplay in epoch 2.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learner import SpeedProfileManager, PROFILES_PATH, REPLAYS_DIR, SCHEMA_VERSION, CURRENT_CALIBRATION_EPOCH


def rebuild_profiles(target_path: Path = PROFILES_PATH):
    print(f"[REBUILD] Building clean speed profiles at: {target_path}")
    mgr = SpeedProfileManager(profiles_path=target_path, auto_bootstrap=False, save_to_disk=True)
    mgr.schema_version = SCHEMA_VERSION
    mgr.calibration_epoch = CURRENT_CALIBRATION_EPOCH
    mgr.bootstrap_from_replays(REPLAYS_DIR)
    mgr.save()
    print("[REBUILD] Successfully rebuilt speed_profiles.json:")
    print(mgr.get_summary())


if __name__ == "__main__":
    rebuild_profiles()
