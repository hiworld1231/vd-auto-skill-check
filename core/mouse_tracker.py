"""
Kernel-Level Mouse Button Tracking Module.
Uses Linux evdev /dev/input directly for zero-latency detection of Left Mouse Button
(BTN_LEFT) across all physical and virtual mouse devices, with automatic hotplug rescan
and graceful fallback to XQueryPointer / pynput.
"""

import os
import select
import threading
import time
from typing import Any, List, Optional


class MouseTracker:
    """
    High-performance, zero-latency mouse button tracker.
    Monitors all system pointer devices simultaneously via Linux kernel evdev.
    """

    def __init__(self, enabled: bool = True, rescan_interval: float = 4.0):
        self.enabled = enabled
        self.rescan_interval = rescan_interval
        self._running = False
        self._lock = threading.Lock()

        # State
        self._held = False
        self._last_press_t = 0.0
        self._last_release_t = 0.0
        self._devices: List[Any] = []
        self._device_paths: set = set()
        self._last_rescan_t = 0.0
        self._backend = "DISABLED"

        # Fallback objects
        self._fallback_listener = None
        self._x11_display = None
        self._x11_root = None
        self._x11_lib = None

        if self.enabled:
            self.start()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="MouseTracker")
        self._thread.start()

    def stop(self):
        self._running = False
        self._close_devices()
        self._close_fallbacks()

    close = stop

    # =========================================================================
    # PUBLIC QUERY APIS
    # =========================================================================

    def is_held(self) -> bool:
        """Returns True if Left Mouse Button is currently held down."""
        if not self.enabled:
            return True  # If tracking is disabled, never block execution
        with self._lock:
            return self._held

    def was_released_since(self, timestamp: float) -> bool:
        """Returns True if LMB was released at or after the given monotonic timestamp."""
        if not self.enabled:
            return False
        with self._lock:
            return self._last_release_t >= timestamp

    def last_press_time(self) -> float:
        with self._lock:
            return self._last_press_t

    def last_release_time(self) -> float:
        with self._lock:
            return self._last_release_t

    def time_since_release(self) -> float:
        with self._lock:
            if self._last_release_t == 0.0:
                return float("inf")
            return time.monotonic() - self._last_release_t

    @property
    def backend_name(self) -> str:
        return self._backend

    # =========================================================================
    # EVDEV KERNEL WORKER
    # =========================================================================

    def _worker_loop(self):
        # 1. Try initializing evdev devices
        has_evdev = self._rescan_evdev_devices()
        if has_evdev:
            self._backend = "evdev (Kernel)"
        else:
            # Try X11 / pynput fallback
            self._init_fallbacks()

        while self._running:
            now = time.monotonic()

            # Periodic hotplug check (handles wireless mouse sleep/wake or USB reconnection)
            if now - self._last_rescan_t > self.rescan_interval:
                self._last_rescan_t = now
                self._rescan_evdev_devices()

            if self._devices:
                # Kernel event monitoring via epoll/select
                try:
                    r, _, _ = select.select(self._devices, [], [], 0.05)
                except (OSError, ValueError):
                    self._rescan_evdev_devices()
                    continue

                for dev in r:
                    try:
                        for ev in dev.read():
                            if ev.type == 1 and ev.code == 272:  # EV_KEY, BTN_LEFT
                                t_event = time.monotonic()
                                with self._lock:
                                    if ev.value == 1:
                                        self._held = True
                                        self._last_press_t = t_event
                                    elif ev.value == 0:
                                        self._held = False
                                        self._last_release_t = t_event
                    except (OSError, IOError):
                        # Device disconnected
                        self._rescan_evdev_devices()
                        break
            else:
                # Fallback polling if no evdev device is open
                self._poll_fallback()
                time.sleep(0.01)

    def _rescan_evdev_devices(self) -> bool:
        try:
            import evdev

            available_paths = set(evdev.list_devices())
            # Close removed devices
            active = []
            for d in self._devices:
                if d.path in available_paths:
                    active.append(d)
                else:
                    try:
                        d.close()
                    except Exception:
                        pass
            self._devices = active
            self._device_paths = {d.path for d in self._devices}

            # Open newly discovered devices
            for path in available_paths:
                if path not in self._device_paths:
                    try:
                        dev = evdev.InputDevice(path)
                        caps = dev.capabilities()
                        if evdev.ecodes.EV_KEY in caps and evdev.ecodes.BTN_LEFT in caps[evdev.ecodes.EV_KEY]:
                            self._devices.append(dev)
                            self._device_paths.add(path)
                            # Initial state sync
                            if evdev.ecodes.BTN_LEFT in dev.active_keys():
                                with self._lock:
                                    self._held = True
                    except Exception:
                        pass

            if self._devices:
                self._backend = f"evdev ({len(self._devices)} dev)"
                return True
        except Exception:
            pass

        return len(self._devices) > 0

    def _close_devices(self):
        for d in self._devices:
            try:
                d.close()
            except Exception:
                pass
        self._devices.clear()
        self._device_paths.clear()

    # =========================================================================
    # FALLBACK ENGINE (X11 / PYNPUT)
    # =========================================================================

    def _init_fallbacks(self):
        # 1. Try pynput
        try:
            from pynput import mouse

            def on_click(x, y, button, pressed):
                if button == mouse.Button.left:
                    now = time.monotonic()
                    with self._lock:
                        self._held = pressed
                        if pressed:
                            self._last_press_t = now
                        else:
                            self._last_release_t = now

            self._fallback_listener = mouse.Listener(on_click=on_click)
            self._fallback_listener.daemon = True
            self._fallback_listener.start()
            self._backend = "pynput (Fallback)"
            return
        except Exception:
            pass

        # 2. Try XQueryPointer
        try:
            import ctypes
            import ctypes.util

            x11_path = ctypes.util.find_library("X11")
            if x11_path:
                self._x11_lib = ctypes.cdll.LoadLibrary(x11_path)
                self._x11_lib.XOpenDisplay.restype = ctypes.c_void_p
                self._x11_lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
                self._x11_lib.XDefaultRootWindow.restype = ctypes.c_ulong
                self._x11_lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
                self._x11_lib.XQueryPointer.restype = ctypes.c_int
                self._x11_lib.XQueryPointer.argtypes = [
                    ctypes.c_void_p, ctypes.c_ulong,
                    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_uint)
                ]
                self._x11_lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
                self._x11_display = self._x11_lib.XOpenDisplay(None)
                if self._x11_display:
                    self._x11_root = self._x11_lib.XDefaultRootWindow(self._x11_display)
                    self._backend = "XQueryPointer (Fallback)"
        except Exception:
            pass

    def _poll_fallback(self):
        if self._x11_display and self._x11_lib:
            import ctypes
            r_r = ctypes.c_ulong()
            c_r = ctypes.c_ulong()
            rx = ctypes.c_int()
            ry = ctypes.c_int()
            wx = ctypes.c_int()
            wy = ctypes.c_int()
            mask = ctypes.c_uint()
            try:
                self._x11_lib.XQueryPointer(
                    self._x11_display, self._x11_root,
                    ctypes.byref(r_r), ctypes.byref(c_r),
                    ctypes.byref(rx), ctypes.byref(ry),
                    ctypes.byref(wx), ctypes.byref(wy),
                    ctypes.byref(mask)
                )
                held = bool(mask.value & (1 << 8))
                now = time.monotonic()
                with self._lock:
                    if held != self._held:
                        self._held = held
                        if held:
                            self._last_press_t = now
                        else:
                            self._last_release_t = now
            except Exception:
                pass

    def _close_fallbacks(self):
        if self._fallback_listener is not None:
            try:
                self._fallback_listener.stop()
            except Exception:
                pass
            self._fallback_listener = None
        if self._x11_display and self._x11_lib:
            try:
                self._x11_lib.XCloseDisplay(self._x11_display)
            except Exception:
                pass
            self._x11_display = None
