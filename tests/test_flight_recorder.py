import shutil
import tempfile
import time
import unittest
from pathlib import Path

from core.flight_recorder import FlightRecorder


class FlightRecorderResilienceTests(unittest.TestCase):
    def test_recreates_replay_directory_if_removed_mid_session(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "replays"
            rec = FlightRecorder(
                out,
                save_video=False,
                save_diagnostic_strip=False,
                record_all=True,
            )
            shutil.rmtree(out)
            now = time.monotonic()
            rec.start_check(now, chain_count=1, latency_ms=100.0)
            rec.end_check(now + 0.01, {"outcome": "GOOD"})
            rec.close()

            self.assertTrue(out.is_dir())
            self.assertEqual(rec.write_failures, 0)
            self.assertEqual(len(list(out.glob("check_*.json"))), 1)

    def test_one_write_exception_does_not_kill_worker(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "replays"
            rec = FlightRecorder(
                out,
                save_video=False,
                save_diagnostic_strip=False,
                record_all=True,
            )
            original_write = rec._write
            calls = {"n": 0}

            def flaky_write(ep):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("synthetic recorder failure")
                return original_write(ep)

            rec._write = flaky_write
            now = time.monotonic()
            rec.start_check(now, chain_count=1, latency_ms=100.0)
            rec.end_check(now + 0.01, {"outcome": "GOOD"})
            rec.start_check(now + 0.02, chain_count=1, latency_ms=100.0)
            rec.end_check(now + 0.03, {"outcome": "GOOD"})
            rec.close()

            self.assertEqual(rec.write_failures, 1)
            self.assertIsNone(rec.last_write_error)
            self.assertEqual(len(list(out.glob("check_*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
