"""
Unit tests verifying memory ownership and buffer independence in FlightRecorder.
Confirms that caller buffer mutations and capture buffer reuses do not corrupt
stored diagnostic and replay frames.
"""

import unittest
from pathlib import Path
import numpy as np

from core.flight_recorder import FlightRecorder


class TestRecorderMemoryOwnership(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("/tmp/test_flight_recorder_memory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = FlightRecorder(output_dir=self.output_dir)

    def test_buffer_mutation_isolation(self):
        # 1. Create a frame filled with pixel value 128
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
        now = 100.0

        # Feed to pre-roll
        self.recorder.on_frame(now, frame, None)
        self.assertEqual(len(self.recorder._pre_roll), 1)

        # Start a check
        self.recorder.start_check(now + 0.01)
        self.recorder.on_frame(now + 0.015, frame, {"needle_angle": 45.0})

        # 2. Mutate original frame in place to 0
        frame[:] = 0

        # 3. Verify that stored pre-roll and active episode frames retain pixel value 128
        pre_roll_frame = self.recorder._pre_roll[0][1]
        episode_frame = self.recorder._episode["frames"][0][1]

        self.assertEqual(float(np.mean(pre_roll_frame)), 128.0, "Pre-roll frame was corrupted by caller buffer mutation!")
        self.assertEqual(float(np.mean(episode_frame)), 128.0, "Episode frame was corrupted by caller buffer mutation!")
        self.assertFalse(np.shares_memory(frame, pre_roll_frame), "Pre-roll frame shares memory with caller buffer!")
        self.assertFalse(np.shares_memory(frame, episode_frame), "Episode frame shares memory with caller buffer!")


if __name__ == "__main__":
    unittest.main()
