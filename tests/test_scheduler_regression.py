#!/usr/bin/env python3
"""
Regression test suite for PreciseTriggerScheduler:
- Generation state machine & clean cancellation
- Thread-safe exactly-once execution
- Stale thread rejection
- Desired press time preservation and latency calculation
"""

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trigger import PreciseTriggerScheduler


class TestSchedulerRegression(unittest.TestCase):
    def test_cancel_does_not_block_subsequent_trigger_now(self):
        """Cancelling a check must reset fired=False and allow immediate trigger in next check."""
        fired_reasons = []

        def callback(reason, **kwargs):
            fired_reasons.append(reason)

        scheduler = PreciseTriggerScheduler(callback)
        scheduler.schedule(time.monotonic() + 0.100, reason="OLD_CHECK")
        scheduler.cancel()

        # In new check, trigger_now must succeed immediately
        res = scheduler.trigger_now(reason="NEW_CHECK_IMMEDIATE")
        self.assertTrue(res)
        self.assertEqual(fired_reasons, ["NEW_CHECK_IMMEDIATE"])

    def test_cancel_kills_stale_spin_thread(self):
        """A thread from an older generation must exit without firing callback."""
        fired_reasons = []

        def callback(reason, **kwargs):
            fired_reasons.append(reason)

        scheduler = PreciseTriggerScheduler(callback)
        # Schedule fire in 25ms
        target = time.monotonic() + 0.025
        scheduler.schedule(target, reason="OLD_CHECK_SPIN")

        # Cancel after 5ms
        time.sleep(0.005)
        scheduler.cancel()

        # Wait past the original target time
        time.sleep(0.035)
        self.assertEqual(fired_reasons, [], "Old cancelled spin thread must NOT fire callback")

    def test_strictly_exactly_once_under_race(self):
        """Concurrent trigger_now calls must result in exactly 1 fire."""
        fire_count = 0
        lock = threading.Lock()

        def callback(reason, **kwargs):
            nonlocal fire_count
            with lock:
                fire_count += 1

        scheduler = PreciseTriggerScheduler(callback)
        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda idx=i: scheduler.trigger_now(f"RACE_{idx}"))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(fire_count, 1, "Scheduler must fire callback strictly once per generation")

    def test_spin_wait_precision_and_target_delivery(self):
        """Spin timer fires within sub-millisecond precision and passes honest targets."""
        delivered_info = {}

        def callback(reason, planned_t=None, desired_press_time=None, scheduler_dispatch_target=None, callback_entry_time=None):
            delivered_info["reason"] = reason
            delivered_info["planned_t"] = planned_t
            delivered_info["desired_press_time"] = desired_press_time
            delivered_info["scheduler_dispatch_target"] = scheduler_dispatch_target
            delivered_info["callback_entry_time"] = callback_entry_time

        scheduler = PreciseTriggerScheduler(callback)
        target = time.monotonic() + 0.015  # 15ms in future
        desired = target - 0.002  # desired 2ms earlier

        scheduler.schedule(target, reason="PRECISE_SPIN", desired_press_time=desired)
        time.sleep(0.025)

        self.assertEqual(delivered_info.get("reason"), "PRECISE_SPIN")
        self.assertEqual(delivered_info.get("desired_press_time"), desired)
        self.assertEqual(delivered_info.get("scheduler_dispatch_target"), target)
        self.assertIsNotNone(delivered_info.get("callback_entry_time"))

        # Jitter between callback entry and scheduler dispatch target should be tiny (< 1.5ms)
        jitter_ms = abs(delivered_info["callback_entry_time"] - target) * 1000.0
        self.assertLess(jitter_ms, 1.5, f"Spin timer jitter too high: {jitter_ms:.3f}ms")

    def test_backup_timer_preserves_original_desired_time(self):
        """Backup trigger preserves original desired_press_time to compute true lateness."""
        delivered_info = {}

        def callback(reason, planned_t=None, desired_press_time=None, **kwargs):
            delivered_info["reason"] = reason
            delivered_info["desired_press_time"] = desired_press_time

        scheduler = PreciseTriggerScheduler(callback)
        desired_t = time.monotonic() - 0.010  # 10ms in the past (overdue backup)
        scheduler.trigger_now("ТАЙМЕР-BACKUP", desired_press_time=desired_t)

        self.assertEqual(delivered_info["reason"], "ТАЙМЕР-BACKUP")
        self.assertEqual(delivered_info["desired_press_time"], desired_t)


if __name__ == "__main__":
    unittest.main()
