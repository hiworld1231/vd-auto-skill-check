"""Rendered browser pixels must be readable by the actual solver detector."""
from pathlib import Path
import shutil
import subprocess

import cv2
import pytest

from vd.vision import Detector


BENCH=Path(__file__).resolve().parents[1]/'simulator/skillcheck.html'


def test_browser_bench_renders_a_detectable_check(tmp_path):
    browser=shutil.which('chromium')
    if browser is None:
        pytest.skip('Chromium is not installed')
    screenshot=tmp_path/'bench.png'
    subprocess.run([browser,'--headless','--no-sandbox','--disable-gpu',
        '--hide-scrollbars','--window-size=1920,1080','--virtual-time-budget=700',
        f'--screenshot={screenshot}',BENCH.as_uri()],check=True,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
    frame=cv2.imread(str(screenshot))[420:660,800:1120]
    m=Detector().measure(frame,1.)
    assert m.reason=='OK'
    assert m.prompt_score>.9
    assert m.center==(160,162.5)
    assert m.great is not None and 5<=m.great.width<=16
    assert abs((m.great.start-40+180)%360-180)<2
    assert len(m.candidates)==1
