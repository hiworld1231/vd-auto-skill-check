import unittest

import numpy as np

from core.detectors.base import refine_zone_from_score


class SubdegreeZoneGeometryTests(unittest.TestCase):
    def test_interpolates_both_boundaries_between_degree_bins(self):
        score = np.full(360, -1.0, dtype=np.float64)
        score[10:18] = 1.0
        score[9] = -0.6
        score[10] = 0.4
        score[17] = 0.25
        score[18] = -0.75
        zone = {
            "start": 10.0,
            "end": 17.0,
            "width": 8.0,
            "center": 14.0,
            "source": "MEASURED",
        }

        refined = refine_zone_from_score(
            zone, score, min_width=5.0, max_width=16.0
        )
        self.assertTrue(refined["geometry_refined"])
        self.assertAlmostEqual(refined["start"], 9.6, places=6)
        self.assertAlmostEqual(refined["end"], 17.25, places=6)
        self.assertAlmostEqual(refined["width"], 7.65, places=6)
        self.assertAlmostEqual(refined["center"], 13.425, places=6)

    def test_wraparound_refinement_keeps_circular_center(self):
        score = np.full(360, -1.0, dtype=np.float64)
        for i in [357, 358, 359, 0, 1, 2, 3, 4]:
            score[i] = 1.0
        score[356] = -0.25
        score[357] = 0.75
        score[4] = 0.4
        score[5] = -0.6
        zone = {
            "start": 357.0,
            "end": 4.0,
            "width": 8.0,
            "center": 1.0,
            "source": "MEASURED",
        }

        refined = refine_zone_from_score(
            zone, score, min_width=5.0, max_width=16.0
        )
        self.assertAlmostEqual(refined["start"], 356.25, places=6)
        self.assertAlmostEqual(refined["end"], 4.4, places=6)
        self.assertAlmostEqual(refined["width"], 8.15, places=6)
        self.assertAlmostEqual(refined["center"], 0.325, places=6)

    def test_malformed_crossing_falls_back_to_integer_geometry(self):
        score = np.ones(360, dtype=np.float64)
        zone = {
            "start": 10.0,
            "end": 17.0,
            "width": 8.0,
            "center": 14.0,
            "source": "MEASURED",
        }
        refined = refine_zone_from_score(
            zone, score, min_width=5.0, max_width=16.0
        )
        self.assertEqual(refined, zone)


if __name__ == "__main__":
    unittest.main()
