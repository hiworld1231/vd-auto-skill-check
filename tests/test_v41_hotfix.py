import pathlib
import unittest

from core.tui import SkillCheckTUI


class TestV41Hotfix(unittest.TestCase):
    def test_tui_unconfirmed_none_geometry_does_not_crash(self):
        tui = SkillCheckTUI()
        tui.record_hit(
            outcome="UNCONFIRMED",
            fact_angle=None,
            target_angle=337.5,
            error_deg=None,
            error_ms=None,
            chain=2,
            latency_ms=64.4,
        )
        self.assertEqual(tui.last_outcome, "UNCONFIRMED")
        self.assertIsNone(tui.last_hit_angle)
        self.assertTrue(any("N/A" in str(x) for x in tui.events))

    def test_new_ring_no_longer_uses_1p5s_chain_heuristic(self):
        text = pathlib.Path("skillcheck_bot.py").read_text(encoding="utf-8")
        self.assertNotIn("if (now - last_check_end_t) <= 1.5", text)
        self.assertIn("[FRENZY_CONFIRM]", text)


if __name__ == "__main__":
    unittest.main()
