from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class InputDispatchResult:
    success: bool
    backend: str
    started_at: float
    finished_at: float
    error: Optional[str] = None


class HardwareTrigger:
    """Fail-closed Space dispatcher. Linux production path is evdev/UInput."""
    def __init__(self, dry_run: bool = False, hold_seconds: float = 0.035, allow_pynput_fallback: bool = False):
        self.dry_run = bool(dry_run)
        self.hold_seconds = max(0.005, float(hold_seconds))
        self.allow_pynput_fallback = bool(allow_pynput_fallback)
        self._ui = None
        self._key = None
        self._keyboard = None
        self.backend = "DRY_RUN" if self.dry_run else "UNAVAILABLE"
        if self.dry_run:
            return
        try:
            import evdev
            from evdev import ecodes
            self._evdev = evdev
            self._ecodes = ecodes
            self._ui = evdev.UInput({ecodes.EV_KEY: [ecodes.KEY_SPACE]}, name="vd-skillcheck")
            self.backend = "EVDEV_UINPUT"
            return
        except Exception:
            self._ui = None
        if self.allow_pynput_fallback:
            try:
                from pynput.keyboard import Controller, Key
                self._keyboard = Controller()
                self._key = Key.space
                self.backend = "PYNPUT"
            except Exception:
                self._keyboard = None

    def trigger(self) -> InputDispatchResult:
        start = time.monotonic()
        if self.dry_run:
            return InputDispatchResult(True, self.backend, start, time.monotonic())
        try:
            if self.backend == "EVDEV_UINPUT" and self._ui is not None:
                self._ui.write(self._ecodes.EV_KEY, self._ecodes.KEY_SPACE, 1)
                self._ui.syn()
                time.sleep(self.hold_seconds)
                self._ui.write(self._ecodes.EV_KEY, self._ecodes.KEY_SPACE, 0)
                self._ui.syn()
            elif self.backend == "PYNPUT" and self._keyboard is not None:
                self._keyboard.press(self._key)
                time.sleep(self.hold_seconds)
                self._keyboard.release(self._key)
            else:
                raise RuntimeError("no usable keyboard backend")
            return InputDispatchResult(True, self.backend, start, time.monotonic())
        except Exception as exc:
            return InputDispatchResult(False, self.backend, start, time.monotonic(), str(exc))

    def close(self) -> None:
        if self._ui is not None:
            try:
                self._ui.close()
            except Exception:
                pass
            self._ui = None


class PreciseTriggerScheduler:
    """Single-deadline scheduler. Callback is invoked exactly once per dispatch."""
    def __init__(self, callback: Callable[..., None], spin_window_s: float = 0.002):
        self.callback = callback
        self.spin_window_s = float(spin_window_s)
        self._cv = threading.Condition()
        self._deadline: Optional[float] = None
        self._reason = "SCHEDULED"
        self._desired: Optional[float] = None
        self._generation = 0
        self._armed = True
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="PreciseTriggerScheduler", daemon=True)
        self._thread.start()

    def rearm(self) -> None:
        with self._cv:
            self._armed = True

    def cancel_pending(self) -> None:
        with self._cv:
            self._generation += 1
            self._deadline = None
            self._cv.notify_all()

    def schedule(self, when: float, reason: str = "SCHEDULED", desired_press_time: Optional[float] = None) -> None:
        with self._cv:
            if not self._armed:
                return
            self._generation += 1
            self._deadline = float(when)
            self._reason = reason
            self._desired = float(desired_press_time if desired_press_time is not None else when)
            self._cv.notify_all()

    def trigger_now(self, reason: str = "IMMEDIATE", desired_press_time: Optional[float] = None) -> None:
        with self._cv:
            if not self._armed:
                return
            self._armed = False
            self._generation += 1
            self._deadline = None
        now = time.monotonic()
        self.callback(
            reason,
            desired_press_time=desired_press_time,
            scheduler_dispatch_target=now,
            callback_entry_time=now,
        )

    def _worker(self) -> None:
        while True:
            with self._cv:
                while self._running and (self._deadline is None or not self._armed):
                    self._cv.wait(timeout=0.2)
                if not self._running:
                    return
                deadline = self._deadline
                reason = self._reason
                desired = self._desired
                gen = self._generation
            while True:
                with self._cv:
                    if not self._running:
                        return
                    if gen != self._generation or self._deadline is None or not self._armed:
                        break
                remain = deadline - time.monotonic()
                if remain <= 0:
                    with self._cv:
                        if gen != self._generation or not self._armed:
                            break
                        self._armed = False
                        self._deadline = None
                    entry = time.monotonic()
                    self.callback(
                        reason,
                        desired_press_time=desired,
                        scheduler_dispatch_target=deadline,
                        callback_entry_time=entry,
                    )
                    break
                if remain > self.spin_window_s:
                    time.sleep(max(0.0, remain - self.spin_window_s))
                else:
                    pass

    def close(self) -> None:
        with self._cv:
            self._running = False
            self._deadline = None
            self._cv.notify_all()
        self._thread.join(timeout=1.0)
