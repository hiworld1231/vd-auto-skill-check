import cv2

import vd.vision as vision


def test_realtime_vision_limits_opencv_worker_threads():
    # The detector runs in the latency-sensitive solver process. OpenCV is free
    # to create a worker pool by default, which can briefly consume many cores
    # and starve the game/dispatch thread. Keep its internal parallelism bounded.
    vision.configure_realtime_opencv()
    assert cv2.getNumThreads() == 1
