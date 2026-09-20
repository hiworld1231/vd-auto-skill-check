from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_MODEL = "v6-keydown-syn-great-center-predictive-unc-v2"


@dataclass(frozen=True)
class CalibrationLoadResult:
    accepted: bool
    reason: str
    lead_ms: Optional[float] = None
    uncertainty_ms: Optional[float] = None
    age_s: Optional[float] = None
    trusted_sample_count: int = 0

    def telemetry(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "lead_ms": self.lead_ms,
            "uncertainty_ms": self.uncertainty_ms,
            "age_s": self.age_s,
            "trusted_sample_count": self.trusted_sample_count,
        }


class LeadCalibrationStore:
    """Small fail-closed persistent store for trusted session lead calibration."""

    def __init__(
        self,
        path: Path,
        fingerprint: Dict[str, Any],
        *,
        max_age_s: float = 7.0 * 24.0 * 3600.0,
        min_lead_ms: float = 35.0,
        max_lead_ms: float = 160.0,
    ) -> None:
        self.path = Path(path)
        self.fingerprint = self._canonical_fingerprint(fingerprint)
        self.max_age_s = max(0.0, float(max_age_s))
        self.min_lead_ms = float(min_lead_ms)
        self.max_lead_ms = float(max_lead_ms)

    @staticmethod
    def _canonical_fingerprint(value: Dict[str, Any]) -> Dict[str, Any]:
        # JSON round-trip gives stable primitive-only data and catches objects
        # that should not accidentally participate in calibration identity.
        return json.loads(json.dumps(dict(value), sort_keys=True))

    @staticmethod
    def _finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def load(self, *, now_epoch: Optional[float] = None) -> CalibrationLoadResult:
        now = float(time.time() if now_epoch is None else now_epoch)
        if not self.path.is_file():
            return CalibrationLoadResult(False, "NOT_FOUND")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return CalibrationLoadResult(False, "UNREADABLE")
        if not isinstance(payload, dict):
            return CalibrationLoadResult(False, "INVALID_FORMAT")
        if int(payload.get("schema_version", -1)) != CALIBRATION_SCHEMA_VERSION:
            return CalibrationLoadResult(False, "SCHEMA_MISMATCH")
        if payload.get("calibration_model") != CALIBRATION_MODEL:
            return CalibrationLoadResult(False, "MODEL_MISMATCH")
        if payload.get("fingerprint") != self.fingerprint:
            return CalibrationLoadResult(False, "FINGERPRINT_MISMATCH")

        saved = payload.get("saved_at_epoch")
        lead = payload.get("lead_ms")
        unc = payload.get("uncertainty_ms")
        if not self._finite(saved):
            return CalibrationLoadResult(False, "INVALID_TIMESTAMP")
        age_s = max(0.0, now - float(saved))
        if age_s > self.max_age_s:
            return CalibrationLoadResult(False, "EXPIRED", age_s=age_s)
        if not self._finite(lead) or not (
            self.min_lead_ms <= float(lead) <= self.max_lead_ms
        ):
            return CalibrationLoadResult(False, "INVALID_LEAD", age_s=age_s)
        if not self._finite(unc) or not (0.5 <= float(unc) <= 25.0):
            return CalibrationLoadResult(False, "INVALID_UNCERTAINTY", age_s=age_s)

        return CalibrationLoadResult(
            True,
            "RESTORED",
            lead_ms=float(lead),
            uncertainty_ms=float(unc),
            age_s=age_s,
            trusted_sample_count=max(0, int(payload.get("trusted_sample_count", 0))),
        )

    def save(
        self,
        *,
        lead_ms: float,
        uncertainty_ms: float,
        trusted_sample_count: int,
        now_epoch: Optional[float] = None,
    ) -> None:
        if not self._finite(lead_ms) or not (
            self.min_lead_ms <= float(lead_ms) <= self.max_lead_ms
        ):
            raise ValueError("lead_ms out of calibration range")
        if not self._finite(uncertainty_ms) or not (
            0.5 <= float(uncertainty_ms) <= 25.0
        ):
            raise ValueError("uncertainty_ms out of calibration range")

        payload = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibration_model": CALIBRATION_MODEL,
            "saved_at_epoch": float(
                time.time() if now_epoch is None else now_epoch
            ),
            "fingerprint": self.fingerprint,
            "lead_ms": float(lead_ms),
            "uncertainty_ms": float(uncertainty_ms),
            "trusted_sample_count": max(0, int(trusted_sample_count)),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)


def make_calibration_fingerprint(
    *,
    fps: int,
    region: Dict[str, int],
    detector: str,
    capture_backend: str,
    input_backend: str,
) -> Dict[str, Any]:
    return {
        "calibration_model": CALIBRATION_MODEL,
        "fps": int(fps),
        "region": {
            "left": int(region["left"]),
            "top": int(region["top"]),
            "width": int(region["width"]),
            "height": int(region["height"]),
        },
        "detector": str(detector),
        "capture_backend": str(capture_backend),
        "input_backend": str(input_backend),
    }
