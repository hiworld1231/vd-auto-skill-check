"""
Hardware Input Injection and Microsecond-Precision Spin-Wait Trigger Scheduler.
Uses Linux Kernel evdev UInput for zero-latency Wayland/Sober key injection,
and a generation-tracked hybrid sleep/spin-wait timer with sub-millisecond precision.
"""

import threading
import time
from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np


class HardwareTrigger:
    """
    Direct Linux Kernel evdev UInput Hardware Keyboard Controller.
    Completely bypasses Wayland/XWayland compositor input limitations.
    Records precise monotonic dispatch timestamps around the kernel keydown event.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.backend = "DRY_RUN"
        self._press_fn = None
        self._cleanup_fn = lambda: None
        self._init_backend()

    def _init_backend(self):
        if self.dry_run:
            print("[TRIGGER] Hardware input in DRY-RUN mode (emulation only).")
            return

        # 1. evdev UInput (Direct Linux Kernel Hardware Device)
        try:
            import evdev
            from evdev import UInput, ecodes as e

            keys = list(range(1, 248))
            ui = UInput({e.EV_KEY: keys}, name="VD SkillCheck Keyboard")

            def press_space() -> Tuple[float, float]:
                t_dispatch_start = time.monotonic()
                try:
                    ui.write(e.EV_KEY, e.KEY_SPACE, 1)
                    ui.syn()
                except Exception:
                    pass
                t_dispatch_done = time.monotonic()

                def _release():
                    time.sleep(0.025)
                    try:
                        ui.write(e.EV_KEY, e.KEY_SPACE, 0)
                        ui.syn()
                    except Exception:
                        pass

                threading.Thread(target=_release, daemon=True).start()
                return t_dispatch_start, t_dispatch_done

            def cleanup():
                time.sleep(0.06)
                try:
                    ui.close()
                except Exception:
                    pass

            self.backend = "evdev UInput (Kernel)"
            self._press_fn = press_space
            self._cleanup_fn = cleanup
            print("[TRIGGER] Keyboard: evdev UInput (Linux Kernel Level - Zero-Latency Wayland).")
            return
        except Exception as exc:
            print(f"[TRIGGER] evdev UInput unavailable ({exc}), trying pynput fallback...")

        # 2. pynput (X11 / XWayland)
        try:
            from pynput.keyboard import Controller, Key
            kbd = Controller()

            def press_space() -> Tuple[float, float]:
                t_dispatch_start = time.monotonic()
                try:
                    kbd.press(Key.space)
                except Exception:
                    pass
                t_dispatch_done = time.monotonic()

                def _release():
                    time.sleep(0.025)
                    try:
                        kbd.release(Key.space)
                    except Exception:
                        pass

                threading.Thread(target=_release, daemon=True).start()
                return t_dispatch_start, t_dispatch_done

            self.backend = "pynput (X11)"
            self._press_fn = press_space
            print("[TRIGGER] Keyboard: pynput fallback active.")
            return
        except Exception as exc:
            print(f"[TRIGGER] pynput fallback unavailable: {exc}")

        self.backend = "DRY_RUN"
        self.dry_run = True

    def trigger(self) -> Tuple[float, float]:
        """
        Dispatches an instantaneous Space key press.
        Returns (uinput_dispatch_time, uinput_dispatch_done_time) monotonic timestamps.
        """
        if self._press_fn is not None and not self.dry_run:
            return self._press_fn()
        now = time.monotonic()
        return now, now

    def benchmark_dispatch(self, iterations: int = 50) -> Dict[str, float]:
        """
        Benchmarks the actual input dispatch latency across multiple calls.
        Returns latency percentiles in microseconds.
        """
        durations_us = []
        for _ in range(iterations):
            t0, t1 = self.trigger()
            durations_us.append((t1 - t0) * 1_000_000.0)

        arr = np.array(durations_us)
        return {
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "max_us": float(np.max(arr)),
            "min_us": float(np.min(arr)),
        }

    def close(self):
        self._cleanup_fn()


def init_keyboard(dry_run: bool = False):
    hw = HardwareTrigger(dry_run=dry_run)
    return hw.backend, hw.trigger, hw.close


class PreciseTriggerScheduler:
    """
    Sub-millisecond hybrid spin-wait scheduler for the Space trigger.
    - Uses generation tokens to guarantee clean cancel, reschedule, and thread-safe exactly-once execution.
    - Slices coarse sleep (up to 4ms chunks) so dynamic schedule adjustments from fresh frames apply in real time.
    - Switches to busy-wait spin in the final 2.5ms before target.
    - Tracks desired_press_time, scheduler_dispatch_target, and callback_entry_time for honest telemetry.
    """

    def __init__(self, action_callback: Callable[..., None]):
        self.action = action_callback
        self.lock = threading.Lock()
        self.target_t: Optional[float] = None
        self.desired_press_time: Optional[float] = None
        self.reason = "SCHEDULED"
        self.active = False
        self.fired = False
        self.generation = 0
        self.thread: Optional[threading.Thread] = None

    def cancel(self):
        """
        Cancels any pending schedule and advances the generation counter.
        Kills any running background spin threads from earlier checks and resets
        the fired state so the subsequent check can trigger immediately if needed.
        """
        with self.lock:
            self.generation += 1
            self.target_t = None
            self.desired_press_time = None
            self.active = False
            self.fired = False

    def schedule(
        self,
        target_monotonic_time: float,
        reason: str = "PRECISE_SPIN_TIMER",
        desired_press_time: Optional[float] = None,
    ):
        """Schedules or dynamically updates the target fire timestamp for the active check."""
        with self.lock:
            if self.fired:
                return
            if target_monotonic_time < time.monotonic() - 0.050:
                return
            self.target_t = target_monotonic_time
            self.desired_press_time = desired_press_time if desired_press_time is not None else target_monotonic_time
            self.reason = reason
            current_gen = self.generation
            if not self.active:
                self.active = True
                self.thread = threading.Thread(target=self._run, args=(current_gen,), daemon=True)
                self.thread.start()

    def trigger_now(
        self,
        reason: str = "DIRECT_TRIGGER",
        planned_t: Optional[float] = None,
        desired_press_time: Optional[float] = None,
    ) -> bool:
        """
        Immediately fires the action if not already fired in this check generation.
        Thread-safe and guaranteed exactly-once.
        Preserves original desired_press_time from predictor.
        """
        target_desired = desired_press_time if desired_press_time is not None else planned_t
        with self.lock:
            if self.fired:
                return False
            self.fired = True
            self.active = False
            sched_target = self.target_t
            self.target_t = None

        callback_entry_t = time.monotonic()
        pt = target_desired if target_desired is not None else callback_entry_t
        self._invoke_action(
            reason=reason,
            planned_t=pt,
            desired_press_time=pt,
            scheduler_dispatch_target=sched_target,
            callback_entry_time=callback_entry_t,
            diag_timestamps={"callback_entry_mono": callback_entry_t},
        )
        return True

    def _invoke_action(
        self,
        reason: str,
        planned_t: float,
        desired_press_time: float,
        scheduler_dispatch_target: Optional[float],
        callback_entry_time: float,
        diag_timestamps: Optional[Dict[str, float]] = None,
    ):
        try:
            self.action(
                reason,
                planned_t=planned_t,
                desired_press_time=desired_press_time,
                scheduler_dispatch_target=scheduler_dispatch_target,
                callback_entry_time=callback_entry_time,
                diag_timestamps=diag_timestamps,
            )
        except TypeError:
            try:
                self.action(
                    reason,
                    planned_t=planned_t,
                    desired_press_time=desired_press_time,
                    scheduler_dispatch_target=scheduler_dispatch_target,
                    callback_entry_time=callback_entry_time,
                )
            except TypeError:
                try:
                    self.action(reason, planned_t=planned_t)
                except TypeError:
                    self.action(reason)

    def _run(self, my_gen: int):
        diag_timestamps: Dict[str, float] = {}
        while True:
            with self.lock:
                if self.generation != my_gen or self.fired or self.target_t is None:
                    self.active = False
                    return
                tgt = self.target_t
                fire_reason = self.reason
                desired_pt = self.desired_press_time

            now = time.monotonic()
            rem = tgt - now

            # Switch to tight microsecond spin-wait in the final 2.5 ms
            if rem <= 0.0025:
                spin_entry_t = time.monotonic()
                diag_timestamps["spin_entry_mono"] = spin_entry_t
                while time.monotonic() < tgt:
                    with self.lock:
                        if self.generation != my_gen or self.fired:
                            self.active = False
                            return
                        if self.target_t != tgt:
                            break

                with self.lock:
                    if self.target_t != tgt:
                        continue
                    if self.generation != my_gen or self.fired:
                        self.active = False
                        return
                    self.fired = True
                    self.active = False
                    self.target_t = None

                callback_entry_t = time.monotonic()
                diag_timestamps["callback_entry_mono"] = callback_entry_t
                pt = desired_pt if desired_pt is not None else tgt
                self._invoke_action(
                    reason=fire_reason,
                    planned_t=pt,
                    desired_press_time=pt,
                    scheduler_dispatch_target=tgt,
                    callback_entry_time=callback_entry_t,
                    diag_timestamps=diag_timestamps,
                )
                return

            # Sliced sleep leaving buffer for the 2.5ms spin-wait transition
            sleep_chunk = min(0.005, max(0.0005, rem - 0.0020))
            sleep_start_t = time.monotonic()
            diag_timestamps["last_sleep_start_mono"] = sleep_start_t
            time.sleep(sleep_chunk)
            sleep_wake_t = time.monotonic()
            diag_timestamps["last_sleep_wake_mono"] = sleep_wake_t
            diag_timestamps["last_sleep_overshoot_ms"] = ((sleep_wake_t - sleep_start_t) - sleep_chunk) * 1000.0
