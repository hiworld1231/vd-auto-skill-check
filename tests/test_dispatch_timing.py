import unittest

from core.dispatch_timing import DispatchTimingModel


class DispatchTimingModelTests(unittest.TestCase):
    def test_no_compensation_until_enough_scheduled_samples(self):
        m = DispatchTimingModel(min_samples=3)
        self.assertFalse(m.initialized)
        self.assertEqual(m.compensation_ms(), 0.0)
        self.assertEqual(m.deadline_for_physical_press(10.0), 10.0)
        m.record(1.0000, 1.0010)
        m.record(2.0000, 2.0012)
        self.assertFalse(m.initialized)
        self.assertEqual(m.compensation_ms(), 0.0)

    def test_robust_median_moves_deadline_earlier(self):
        m = DispatchTimingModel(min_samples=3)
        m.record(1.0000, 1.0010)
        m.record(2.0000, 2.0012)
        m.record(3.0000, 3.0008)
        self.assertTrue(m.initialized)
        self.assertAlmostEqual(m.compensation_ms(), 1.0, delta=0.01)
        desired = 20.0
        self.assertAlmostEqual(
            m.deadline_for_physical_press(desired),
            desired - 0.001,
            delta=0.00001,
        )

    def test_single_large_but_valid_lag_does_not_drag_median(self):
        m = DispatchTimingModel(min_samples=3)
        for i, lag_ms in enumerate([0.9, 1.0, 1.1, 25.0, 1.2], start=1):
            m.record(float(i), float(i) + lag_ms / 1000.0)
        self.assertAlmostEqual(m.compensation_ms(), 1.1, delta=0.01)
        self.assertLess(m.uncertainty_ms(), 1.0)

    def test_impossible_lag_is_rejected(self):
        m = DispatchTimingModel(max_sample_ms=50.0)
        self.assertIsNone(m.record(1.0, 1.080))
        self.assertEqual(m.rejected_total, 1)
        self.assertEqual(len(m.samples_ms), 0)

    def test_telemetry_exposes_compensation_and_residual_scale(self):
        m = DispatchTimingModel(min_samples=3)
        for i, lag_ms in enumerate([0.8, 1.0, 1.2], start=1):
            m.record(float(i), float(i) + lag_ms / 1000.0)
        t = m.telemetry()
        self.assertTrue(t["initialized"])
        self.assertEqual(t["sample_count"], 3)
        self.assertAlmostEqual(t["compensation_ms"], 1.0, delta=0.01)
        self.assertGreaterEqual(t["uncertainty_ms"], 0.2)


if __name__ == "__main__":
    unittest.main()
