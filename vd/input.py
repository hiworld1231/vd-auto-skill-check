"""Linux mouse-state monitor and short Space pulses through evdev/uinput.

This module is only used by explicit ``run`` mode. It does not inspect or
communicate with the game process; mouse state comes from normal input events.
"""
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import evdev
from evdev import ecodes as e


@dataclass(frozen=True)
class MouseState:
    held: bool
    healthy: bool
    device: str | None
    error: str | None


class MouseMonitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.devices = []
        self.held_by_device = {}
        self.healthy = True
        self.error = None
        self.stopping = threading.Event()
        self.threads = []
        paths = []
        for path in evdev.list_devices():
            dev = None
            try:
                dev = evdev.InputDevice(path)
                if e.BTN_LEFT in dev.capabilities().get(e.EV_KEY, []):
                    paths.append(path)
            except OSError:
                pass
            finally:
                if dev is not None:
                    dev.close()
        if not paths:
            raise RuntimeError('No readable mouse with a left button was found')
        try:
            for path in paths:
                device = evdev.InputDevice(path)
                self.devices.append(device)
                self.held_by_device[path] = e.BTN_LEFT in device.active_keys()
            self.threads = [threading.Thread(target=self._read, args=(device,),
                name='VD-mouse-monitor-'+Path(device.path).name, daemon=True)
                for device in self.devices]
            for thread in self.threads:
                thread.start()
        except BaseException:
            self.close()
            raise

    def _read(self, device):
        try:
            for event in device.read_loop():
                if self.stopping.is_set():
                    break
                if event.type == e.EV_KEY and event.code == e.BTN_LEFT and event.value in (0, 1):
                    with self.lock:
                        self.held_by_device[device.path] = bool(event.value)
        except OSError as exc:
            with self.lock:
                if not self.stopping.is_set():
                    self.error = str(exc)
                    self.healthy = False
                    self.held_by_device[device.path] = False

    def snapshot(self):
        with self.lock:
            return MouseState(any(self.held_by_device.values()), self.healthy,
                              ','.join(d.path for d in self.devices) or None, self.error)

    def close(self):
        self.stopping.set()
        for device in self.devices:
            device.close()
        self.devices.clear()
        for thread in getattr(self, 'threads', []):
            if thread.is_alive():
                thread.join(timeout=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class Keydown:
    requested_at: float
    syn_completed_at: float


class SpaceOutput:
    def __init__(self, pulse_seconds=.035):
        if not 0.01 <= pulse_seconds <= 0.1:
            raise ValueError('Space pulse must be in [10, 100] ms')
        self.pulse_seconds = pulse_seconds
        self.ui = evdev.UInput({e.EV_KEY: [e.KEY_SPACE]}, name='VD Skill Check Space')
        self.lock = threading.Lock()
        self.error = None
        self.closed = False
        self.pressed = False
        self.release_timer = None

    def _release(self):
        with self.lock:
            if self.closed or not self.pressed:
                return
            try:
                self.ui.write(e.EV_KEY, e.KEY_SPACE, 0)
                self.ui.syn()
                self.pressed = False
            except OSError as exc:
                self.error = str(exc)

    def pulse(self):
        with self.lock:
            if self.closed:
                raise RuntimeError('Space output is closed')
            if self.pressed:
                raise RuntimeError('Previous Space pulse has not been released')
            requested = time.monotonic()
            self.ui.write(e.EV_KEY, e.KEY_SPACE, 1)
            self.ui.syn()
            completed = time.monotonic()
            self.pressed = True
            self.release_timer = threading.Timer(self.pulse_seconds, self._release)
            self.release_timer.daemon = True
            self.release_timer.start()
            return Keydown(requested, completed)

    def close(self):
        with self.lock:
            if not self.closed:
                if self.pressed:
                    try:
                        self.ui.write(e.EV_KEY, e.KEY_SPACE, 0)
                        self.ui.syn()
                    except OSError as exc:
                        self.error = str(exc)
                    self.pressed = False
                if self.release_timer is not None:
                    self.release_timer.cancel()
                self.closed = True
                self.ui.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
