from __future__ import annotations

import time
from collections import Counter
from typing import Optional

try:
    from rich.console import Console
except Exception:
    Console = None


class SkillCheckTUI:
    """Small None-safe console/TUI facade used by CLEAN V5."""
    def __init__(self, *_, **__):
        self.console = Console() if Console else None
        self.counts = Counter()
        self.started = time.monotonic()
        self.last_status = "IDLE"

    def log(self, msg: str, style: Optional[str] = None) -> None:
        if self.console:
            self.console.print(msg, style=style)
        else:
            print(msg)

    def set_status(self, text: str, **_: object) -> None:
        self.last_status = str(text)

    def record_fire_telemetry(self, *_: object, **__: object) -> None:
        pass

    def record_hit(self, outcome: str, fact_angle=None, target_angle=None, error_deg=None,
                   error_ms=None, chain: int = 1, latency_ms: float = 0.0, **_: object) -> None:
        self.counts[str(outcome)] += 1
        def f(v, spec=".1f"):
            try:
                return format(float(v), spec)
            except Exception:
                return "N/A"
        self.log(
            f"{outcome}: fact={f(fact_angle)}° target={f(target_angle)}° "
            f"err={f(error_deg, '+.1f')}°/{f(error_ms, '+.1f')}ms chain={chain} lead={f(latency_ms)}ms"
        )

    def finish(self) -> None:
        total = sum(self.counts.values())
        self.log(
            f"SESSION: total={total} GREAT={self.counts['GREAT']} GOOD={self.counts['GOOD']} "
            f"MISS={self.counts['MISS']} UNCONF={self.counts['UNCONFIRMED']} NO_FIRE={self.counts['NO_FIRE']}"
        )

    close = finish
