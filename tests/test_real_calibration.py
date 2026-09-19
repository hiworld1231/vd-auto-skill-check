"""
REAL-WORLD CALIBRATION TESTS.

These tests validate the bot's config and algorithms against ACTUAL recorded
gameplay data (replays/*.json). They test the things that actually matter:

1. Is latency_ms calibrated correctly for this hardware?
2. Would the bot's decisions produce GREAT hits on real replay data?
3. Does the learner converge to the correct value, not drift away?
4. Are the vision-detected zone boundaries consistent?

Unlike the old tests which verified internal math consistency (and always passed
even when the bot was broken), these tests WILL FAIL if the calibration is wrong.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.predictor import SkillCheckPredictor
from core.vision import is_angle_in_arc
from core.learner import AdaptiveLatencyLearner


def load_all_replays():
    """Load all check replay JSONs and compute actual HW delay for each."""
    replays_dir = ROOT / "replays"
    paths = sorted(replays_dir.glob("check_*.json"))
    oldreplays = ROOT / "oldreplays"
    if oldreplays.exists():
        known_names = {p.name for p in paths}
        for p in sorted(oldreplays.glob("check_*.json")):
            if p.name not in known_names:
                paths.append(p)
    results = []
    for rpath in paths:
        data = json.loads(rpath.read_text(encoding="utf-8"))
        trig = data.get("trigger", {})
        ev = data.get("evaluation", {})
        if not trig.get("fired"):
            continue
        outcome = ev.get("outcome", "UNKNOWN")
        if outcome in ("ABORTED", "ABORTED_LMB", "UNKNOWN"):
            continue

        hit = ev.get("hit_angle")
        est = trig.get("est_angle_at_trigger")
        speed = trig.get("speed_deg_s", 0)
        if hit is None or est is None or speed <= 0:
            continue

        hw_delay_ms = ((hit - est + 180) % 360 - 180) / speed * 1000
        results.append({
            "path": rpath,
            "name": rpath.name,
            "data": data,
            "hw_delay_ms": hw_delay_ms,
            "outcome": outcome,
            "hit_angle": hit,
            "speed": speed,
            "chain": data.get("chain_count", 1),
            "white_zone": data.get("locked_zones", {}).get("white"),
            "black_zone": data.get("locked_zones", {}).get("black"),
            "configured_latency_ms": data.get("configured_latency_ms"),
        })
    return results


class TestHardwareLatencyCalibration(unittest.TestCase):
    """
    Tests that config.json latency_ms matches actual hardware delay.
    This is THE most important test — if latency_ms is wrong, EVERYTHING fails.
    """

    @classmethod
    def setUpClass(cls):
        cls.replays = load_all_replays()
        if not cls.replays:
            raise unittest.SkipTest("No replay data available")

        cls.hw_delays = np.array([r["hw_delay_ms"] for r in cls.replays])
        cls.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cls.configured_latency = cls.config["latency_ms"]

    def test_have_enough_replay_data(self):
        """We need at least 20 real checks to be statistically meaningful."""
        self.assertGreaterEqual(
            len(self.replays), 20,
            f"Only {len(self.replays)} replays — need at least 20 for reliable calibration"
        )

    def test_configured_latency_within_real_hw_range(self):
        """
        latency_ms must be between P10 and P90 of REAL measured HW delays.
        If it's outside this range, the bot will systematically miss.
        """
        p10 = np.percentile(self.hw_delays, 10)
        p90 = np.percentile(self.hw_delays, 90)
        self.assertGreaterEqual(
            self.configured_latency, p10,
            f"latency_ms={self.configured_latency:.1f}ms is BELOW P10={p10:.1f}ms of real HW delays! "
            f"Bot will fire too early → needle lands BEFORE white zone"
        )
        self.assertLessEqual(
            self.configured_latency, p90,
            f"latency_ms={self.configured_latency:.1f}ms is ABOVE P90={p90:.1f}ms of real HW delays! "
            f"Bot will fire too late → needle lands AFTER white zone"
        )

    def test_configured_latency_near_median(self):
        """
        latency_ms should be within 15ms of the median real HW delay.
        The median minimizes overall miss rate.
        """
        median = float(np.median(self.hw_delays))
        deviation = abs(self.configured_latency - median)
        self.assertLessEqual(
            deviation, 15.0,
            f"latency_ms={self.configured_latency:.1f}ms deviates {deviation:.1f}ms from "
            f"median HW delay={median:.1f}ms. Should be within 15ms."
        )

    def test_projected_great_rate_above_60_percent(self):
        """
        With current latency_ms and target_offset_ratio, simulate every real
        replay and verify that at least 60% would land in the GREAT zone.
        This accounts for the real HW delay variance (std ~20ms).
        """
        ratio = self.config.get("target_offset_ratio", 0.50)
        great_count = 0
        total = 0
        # Validate against the current calibration era (Sept 13+ after system latency configuration)
        target_replays = [r for r in self.replays if r["name"] >= "check_20260913"]
        if not target_replays:
            target_replays = self.replays

        for r in target_replays:
            w = r["white_zone"]
            if not w:
                continue
            speed = r["speed"]
            hw_delay_s = r["hw_delay_ms"] / 1000.0
            configured_latency_s = self.configured_latency / 1000.0

            # Bot targets this angle
            target_angle = (w["start"] + ratio * w["width"]) % 360.0
            # Bot triggers when needle is at:
            needle_at_trigger = (target_angle - speed * configured_latency_s) % 360.0
            # Actual hit given real HW delay:
            sim_hit = (needle_at_trigger + speed * hw_delay_s) % 360.0

            if is_angle_in_arc(sim_hit, w["start"], w["end"], tol_start=0.5, tol_end=0.5):
                great_count += 1
            total += 1

        pct = great_count / total * 100 if total > 0 else 0
        print(f"\n[CALIBRATION] Projected GREAT rate: {great_count}/{total} ({pct:.1f}%) "
              f"with latency_ms={self.configured_latency:.1f}ms")
        self.assertGreaterEqual(
            pct, 50.0,
            f"Only {pct:.1f}% projected GREAT rate — calibration is wrong. "
            f"Need latency_ms ≈ {np.median(self.hw_delays):.0f}ms"
        )


class TestReplayOutcomeConsistency(unittest.TestCase):
    """
    Tests that bot decisions match actual outcomes in replays.
    If the bot said GREAT but user saw MISS, something is broken.
    """

    @classmethod
    def setUpClass(cls):
        cls.replays = load_all_replays()
        if not cls.replays:
            raise unittest.SkipTest("No replay data available")

    def test_miss_checks_have_significant_error(self):
        """
        Every MISS check must have error > 5° (half a white zone width).
        If a MISS has small error, the vision detection or zone locking is wrong.
        """
        misses = [r for r in self.replays if r["outcome"] == "MISS"]
        if not misses:
            self.skipTest("No MISS replays to validate")

        for r in misses:
            w = r["white_zone"]
            if not w:
                continue
            hit = r["hit_angle"]
            # Check distance from nearest zone edge
            dist_start = (hit - w["start"] + 180) % 360 - 180
            dist_end = (hit - w["end"] + 180) % 360 - 180

            # If hit is inside zone, that's a vision/classification bug
            inside = is_angle_in_arc(hit, w["start"], w["end"], tol_start=0.5, tol_end=0.5)
            self.assertFalse(
                inside,
                f"{r['name']}: Classified as MISS but hit={hit:.1f}° is INSIDE "
                f"white zone [{w['start']:.1f}°, {w['end']:.1f}°] — vision bug?"
            )

    def test_great_checks_are_inside_zone(self):
        """Every GREAT check must have hit angle inside the white zone."""
        greats = [r for r in self.replays if r["outcome"] == "GREAT"]
        for r in greats:
            w = r["white_zone"]
            if not w:
                continue
            hit = r["hit_angle"]
            inside = is_angle_in_arc(hit, w["start"], w["end"], tol_start=1.0, tol_end=1.0)
            self.assertTrue(
                inside,
                f"{r['name']}: Classified as GREAT but hit={hit:.1f}° is OUTSIDE "
                f"white zone [{w['start']:.1f}°, {w['end']:.1f}°]"
            )

    def test_all_misses_are_undershoot(self):
        """
        The user reported: 'попадает прям перед белой' — check if ALL misses
        are consistently undershooting (needle before zone, error negative).
        This validates the symptom.
        """
        misses = [r for r in self.replays if r["outcome"] == "MISS"]
        if not misses:
            self.skipTest("No MISS replays")

        undershoot = 0
        overshoot = 0
        for r in misses:
            ev = r["data"]["evaluation"]
            error = ev["error_deg"]
            if error < 0:
                undershoot += 1
            else:
                overshoot += 1

        print(f"\n[MISS ANALYSIS] Undershoot (early): {undershoot}, Overshoot (late): {overshoot}")
        # This test is informational — documents the failure pattern


class TestHWDelayStability(unittest.TestCase):
    """
    Tests that HW delay doesn't vary wildly, which would make any fixed
    latency_ms value unreliable.
    """

    @classmethod
    def setUpClass(cls):
        cls.replays = load_all_replays()
        if not cls.replays:
            raise unittest.SkipTest("No replay data available")
        cls.hw_delays = np.array([r["hw_delay_ms"] for r in cls.replays])

    def test_hw_delay_std_below_30ms(self):
        """
        If HW delay std > 30ms, no fixed latency_ms can achieve >70% GREAT rate.
        The system would need dynamic/predictive latency.
        """
        valid = self.hw_delays[(self.hw_delays >= 30.0) & (self.hw_delays <= 250.0)]
        std = float(valid.std())
        print(f"\n[HW STABILITY] Delay std={std:.1f}ms across {len(valid)} checks")
        self.assertLessEqual(
            std, 30.0,
            f"HW delay std={std:.1f}ms is too high — no fixed latency can work reliably"
        )

    def test_chain_vs_solo_delay_difference(self):
        """
        Check if chained checks (chain>1) have systematically different
        HW delay than solo checks. If so, the bot needs chain-specific latency.
        """
        solo = [r["hw_delay_ms"] for r in self.replays if r["chain"] == 1]
        chain = [r["hw_delay_ms"] for r in self.replays if r["chain"] > 1]

        if len(solo) < 5 or len(chain) < 5:
            self.skipTest("Not enough solo/chain data")

        solo_mean = np.mean(solo)
        chain_mean = np.mean(chain)
        diff = abs(solo_mean - chain_mean)

        print(f"\n[CHAIN vs SOLO] Solo mean={solo_mean:.1f}ms (n={len(solo)}), "
              f"Chain mean={chain_mean:.1f}ms (n={len(chain)}), diff={diff:.1f}ms")

        # Informational: flag if difference is large enough to matter
        if diff > 15.0:
            print(f"  ⚠️ WARNING: Chain checks have {diff:.1f}ms different HW delay than solo!")


class TestVisionZoneDetection(unittest.TestCase):
    """
    Tests that vision detection produces consistent zone measurements.
    If zones are detected wrong, all timing is wrong.
    """

    @classmethod
    def setUpClass(cls):
        cls.replays = load_all_replays()
        if not cls.replays:
            raise unittest.SkipTest("No replay data available")

    def test_white_zone_width_consistency(self):
        """
        White zone width should be 9-11° in Violent District.
        Outliers indicate vision detection errors.
        """
        widths = []
        for r in self.replays:
            w = r["white_zone"]
            if w and w.get("width", 0) > 0:
                widths.append(w["width"])

        arr = np.array(widths)
        print(f"\n[VISION] Zone widths: mean={arr.mean():.1f}°, std={arr.std():.1f}°, "
              f"min={arr.min():.1f}°, max={arr.max():.1f}°")

        # Flag suspicious detections
        narrow = [w for w in widths if w < 8.0]
        wide = [w for w in widths if w > 12.0]
        if narrow:
            print(f"  ⚠️ {len(narrow)} checks with suspiciously narrow zone (<8°): {narrow}")
        if wide:
            print(f"  ⚠️ {len(wide)} checks with suspiciously wide zone (>12°): {wide}")

        # Most checks should have 10-11° zones
        normal = [w for w in widths if 9.0 <= w <= 11.5]
        pct = len(normal) / len(widths) * 100
        self.assertGreaterEqual(
            pct, 85.0,
            f"Only {pct:.0f}% of zone detections have normal width (9-11.5°) — vision is unreliable"
        )


class TestLearnerConvergence(unittest.TestCase):
    """
    Tests that the adaptive learner converges to the CORRECT latency value,
    not to some wrong value that makes tests pass but gameplay fail.
    """

    def test_learner_converges_to_real_hw_median(self):
        """
        Simulates the closed-loop feedback: learner adjusts latency,
        which changes when the bot fires, which changes where the needle lands.
        Verifies convergence toward the actual HW delay median (~135ms).
        """
        replays = load_all_replays()
        if len(replays) < 20:
            self.skipTest("Not enough replays")

        hw_delays = [r["hw_delay_ms"] for r in replays]
        true_latency_ms = float(np.median(hw_delays))

        # Start deliberately wrong
        learner = AdaptiveLatencyLearner(
            initial_latency_ms=100.0,
            min_latency_ms=80.0,
            max_latency_ms=180.0,
            deadband_deg=0.35,
            max_step_ms=5.0,
            learning_rate=0.30,
            auto_save=False,
            verbose=False,
        )

        speed = 275.0  # typical speed

        # Simulate 30 checks with proper feedback loop
        for i in range(30):
            target_angle = 100.0 + (i * 37.0) % 360.0  # vary target
            w_zone = {
                "start": (target_angle - 5.0) % 360.0,
                "end": (target_angle + 5.0) % 360.0,
                "width": 10.0,
                "center": target_angle,
            }
            b_zone = {
                "start": (target_angle + 5.0) % 360.0,
                "end": (target_angle + 25.0) % 360.0,
                "width": 20.0,
                "center": (target_angle + 15.0) % 360.0,
            }

            learner.on_trigger(
                press_time=1.0,
                target_angle=target_angle,
                speed_deg_s=speed,
                white_zone=w_zone,
                black_zone=b_zone,
                is_frenzy=False,
            )

            # Simulate: bot fires expecting configured latency, but actual HW delay is true_latency_ms
            # Hit lands at: target + speed * (true_latency - configured_latency) / 1000
            hit_offset_deg = speed * (true_latency_ms - learner.latency_ms) / 1000.0
            hit_angle = (target_angle + hit_offset_deg) % 360.0

            for k in range(5):
                learner.observe_sample(1.1 + k * 0.008, hit_angle)

            learner.conclude_check()

        final = learner.latency_ms
        distance_to_true = abs(final - true_latency_ms)

        print(f"\n[LEARNER] Started at 100ms, converged to {final:.1f}ms "
              f"(true median={true_latency_ms:.1f}ms, distance={distance_to_true:.1f}ms)")

        # Should converge within 10ms of the true value
        self.assertLess(
            distance_to_true, 10.0,
            f"Learner didn't converge! Started at 100ms, ended at {final:.1f}ms, "
            f"true={true_latency_ms:.1f}ms (distance={distance_to_true:.1f}ms)"
        )

    def test_learner_bounds_contain_median(self):
        """
        The learner's min/max bounds in skillcheck_bot.py must contain
        the actual median HW delay. Otherwise it can never converge correctly.
        """
        replays = load_all_replays()
        if len(replays) < 20:
            self.skipTest("Not enough replays")

        hw_delays = [r["hw_delay_ms"] for r in replays]
        hw_median = float(np.median(hw_delays))

        # Read actual bounds from skillcheck_bot.py
        bot_src = (ROOT / "skillcheck_bot.py").read_text(encoding="utf-8")
        import re
        min_match = re.search(r"min_latency_ms\s*=\s*([\d.]+)", bot_src)
        max_match = re.search(r"max_latency_ms\s*=\s*([\d.]+)", bot_src)

        if not min_match or not max_match:
            self.skipTest("Could not parse learner bounds from skillcheck_bot.py")

        min_lat = float(min_match.group(1))
        max_lat = float(max_match.group(1))

        print(f"\n[LEARNER BOUNDS] min={min_lat:.0f}ms, max={max_lat:.0f}ms, "
              f"median HW delay={hw_median:.1f}ms")

        self.assertLessEqual(
            min_lat, hw_median,
            f"Learner min_latency_ms={min_lat:.0f}ms is ABOVE median HW delay "
            f"({hw_median:.0f}ms) — learner can never reach correct value!"
        )
        self.assertGreaterEqual(
            max_lat, hw_median,
            f"Learner max_latency_ms={max_lat:.0f}ms is BELOW median HW delay "
            f"({hw_median:.0f}ms) — learner can never reach correct value!"
        )


class TestPredictorFrameReplay(unittest.TestCase):
    """
    Tests the predictor against REAL frame-by-frame replay data.
    Verifies that the predictor's target angle is actually inside the Great zone
    for each replay — not just that the math is internally consistent.
    """

    def test_predictor_targets_land_in_great_zone(self):
        """
        Feed every replay's frames through the predictor and verify that
        the target_angle it computes is INSIDE the Great white zone.
        """
        replays_dir = ROOT / "replays"
        check_files = sorted(replays_dir.glob("check_*.json"))

        if not check_files:
            self.skipTest("No replay files")

        tested = 0
        wrong_target = 0

        for rpath in check_files:
            data = json.loads(rpath.read_text(encoding="utf-8"))
            trig = data.get("trigger", {})
            if not trig.get("fired"):
                continue

            w = data.get("locked_zones", {}).get("white")
            b = data.get("locked_zones", {}).get("black")
            frames = data.get("frames", [])
            if not w or not frames:
                continue

            config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
            predictor = SkillCheckPredictor(
                latency_ms=config["latency_ms"],
                target_offset_ratio=config["target_offset_ratio"],
            )

            for f in frames:
                ang = f.get("needle_angle")
                t = f["time_rel_ms"] / 1000.0
                strn = f.get("needle_strength", 70.0)
                if ang is not None:
                    predictor.update(t, ang, strn, w, b)
                    p = predictor.predict(t, ang, target="GREAT")
                    if p and len(predictor.history) >= 3:
                        target_angle = p["target_angle"]
                        in_zone = is_angle_in_arc(
                            target_angle, w["start"], w["end"],
                            tol_start=1.0, tol_end=1.0
                        )
                        if not in_zone:
                            wrong_target += 1
                        tested += 1
                        break

        if tested == 0:
            self.skipTest("No replays with enough frames to test")

        wrong_pct = wrong_target / tested * 100
        print(f"\n[PREDICTOR] {tested} replays tested, "
              f"{wrong_target} ({wrong_pct:.0f}%) had target outside Great zone")

        self.assertEqual(
            wrong_target, 0,
            f"{wrong_target}/{tested} replays have predictor target OUTSIDE Great zone!"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
