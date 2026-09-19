import unittest
from pathlib import Path

class TestCaptureSource(unittest.TestCase):
    def test_vfr_does_not_force_ffmpeg_input_r(self):
        src=(Path(__file__).parents[1]/'core'/'capture.py').read_text()
        self.assertIn('if str(self.framerate_mode).lower() == "cfr"',src)
        self.assertIn('ff_cmd += ["-r", str(self.fps)]',src)
        self.assertIn('"-fm", self.framerate_mode',src)

if __name__=='__main__': unittest.main()
