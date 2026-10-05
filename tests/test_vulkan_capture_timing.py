import pytest

from vd.vulkan_capture import VulkanCapture


class FakeClock:
    def __init__(self, start=10.0):
        self.now = start
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_vulkan_capture_timeout_does_not_oversleep_short_dispatch_deadline(monkeypatch):
    capture = VulkanCapture.__new__(VulkanCapture)
    capture.stopping = False
    capture._snapshot = lambda: None

    clock = FakeClock()
    monkeypatch.setattr('vd.vulkan_capture.time.monotonic', clock.monotonic)
    monkeypatch.setattr('vd.vulkan_capture.time.sleep', clock.sleep)

    started = clock.monotonic()
    assert capture.next(after=1, timeout=.0007) is None
    elapsed = clock.monotonic() - started

    assert elapsed == pytest.approx(.0007)
    assert clock.sleeps
    assert max(clock.sleeps) <= .0007 + 1e-12
