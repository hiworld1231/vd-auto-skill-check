import threading
from queue import Queue
from types import SimpleNamespace
import time

import pytest

from vd import input as vd_input


class FakeUInput:
    def __init__(self):
        self.events = []
        self.released = threading.Event()

    def write(self, event_type, code, value):
        self.events.append((event_type, code, value))

    def syn(self):
        if self.events and self.events[-1][2] == 0:
            self.released.set()

    def close(self):
        pass


def test_space_output_sends_short_press_then_release(monkeypatch):
    fake = FakeUInput()
    monkeypatch.setattr(vd_input.evdev, 'UInput', lambda *a, **k: fake)
    output = vd_input.SpaceOutput(pulse_seconds=.01)
    result = output.pulse()
    assert result.syn_completed_at >= result.requested_at
    assert fake.events == [(vd_input.e.EV_KEY, vd_input.e.KEY_SPACE, 1)]
    assert fake.released.wait(.5)
    assert fake.events[-1] == (vd_input.e.EV_KEY, vd_input.e.KEY_SPACE, 0)
    output.close()


def test_close_releases_space_if_timer_has_not_fired(monkeypatch):
    fake = FakeUInput()
    monkeypatch.setattr(vd_input.evdev, 'UInput', lambda *a, **k: fake)
    output = vd_input.SpaceOutput(pulse_seconds=.1)
    output.pulse()
    with pytest.raises(RuntimeError, match='not been released'):
        output.pulse()
    output.close()
    assert fake.events[-1] == (vd_input.e.EV_KEY, vd_input.e.KEY_SPACE, 0)


def test_mouse_monitor_combines_buttons_without_grabbing(monkeypatch):
    opened = []

    class FakeDevice:
        def __init__(self, path):
            self.path = path
            self.events = Queue()
            opened.append(self)

        def capabilities(self):
            return {vd_input.e.EV_KEY: [vd_input.e.BTN_LEFT]}

        def active_keys(self):
            return []

        def read_loop(self):
            while True:
                event = self.events.get()
                if event is None:
                    return
                yield event

        def close(self):
            self.events.put(None)

    monkeypatch.setattr(vd_input.evdev, 'list_devices', lambda: ['/dev/fake0', '/dev/fake1'])
    monkeypatch.setattr(vd_input.evdev, 'InputDevice', FakeDevice)
    monitor = vd_input.MouseMonitor()
    first, second = monitor.devices

    def send(device, value):
        device.events.put(SimpleNamespace(type=vd_input.e.EV_KEY,
                                         code=vd_input.e.BTN_LEFT, value=value))

    def wait_for(held):
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline:
            if monitor.snapshot().held is held:
                return
            time.sleep(.005)
        pytest.fail(f'mouse held state did not become {held}')

    send(first, 1)
    wait_for(True)
    send(first, 0)
    wait_for(False)
    send(first, 1)
    send(second, 1)
    wait_for(True)
    send(first, 0)
    wait_for(True)
    monitor.close()
    assert len(opened) == 4  # two capability probes and two live readers
