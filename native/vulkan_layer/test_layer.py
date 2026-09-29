#!/usr/bin/env python3
"""Tests for the VD Vulkan layer and the opt-in ROI pixel-capture path."""

import ctypes
import importlib.util
import json
import os
import struct
import subprocess
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT_DIR / "build" / "vulkan-layer"
LAYER_LIB = BUILD_DIR / "libVkLayer_VD_capture.so"
LAYER_MANIFEST = BUILD_DIR / "VkLayer_VD_capture.json"
SOBER_DATA_DIR = Path.home() / ".var" / "app" / "org.vinegarhq.Sober" / "data"
SOBER_VULKAN_DIR = SOBER_DATA_DIR / "vulkan"
VIEWER_PATH = ROOT_DIR / "tools" / "vulkan_roi_viewer.py"
HEADER_FORMAT = "=IIIIIIQQQfIIIIIII64sIIIIIII4xQQ112s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class VkLayerProperties(ctypes.Structure):
    _fields_ = [
        ("layerName", ctypes.c_char * 256),
        ("specVersion", ctypes.c_uint32),
        ("implementationVersion", ctypes.c_uint32),
        ("description", ctypes.c_char * 256),
    ]


class VkNegotiateLayerInterface(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("loaderLayerInterfaceVersion", ctypes.c_uint32),
        ("pfnGetInstanceProcAddr", ctypes.c_void_p),
        ("pfnGetDeviceProcAddr", ctypes.c_void_p),
        ("pfnGetPhysicalDeviceProcAddr", ctypes.c_void_p),
    ]


def have_flatpak_sober():
    try:
        result = subprocess.run(
            ["flatpak", "info", "org.vinegarhq.Sober"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def load_viewer_module():
    spec = importlib.util.spec_from_file_location("vd_vulkan_roi_viewer", VIEWER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestVulkanLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_script = ROOT_DIR / "native" / "vulkan_layer" / "build_layer.sh"
        result = subprocess.run([str(build_script)], capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"Build failed:\n{result.stdout}\n{result.stderr}")
        if not LAYER_LIB.exists() or not LAYER_MANIFEST.exists():
            raise AssertionError("Vulkan layer build outputs are missing")

    def test_01_symbols_and_loader_negotiation(self):
        lib = ctypes.CDLL(str(LAYER_LIB))
        self.assertTrue(hasattr(lib, "vkNegotiateLoaderLayerInterfaceVersion"))
        self.assertTrue(hasattr(lib, "vkGetInstanceProcAddr"))
        self.assertTrue(hasattr(lib, "vkGetDeviceProcAddr"))
        self.assertTrue(hasattr(lib, "vkEnumerateInstanceLayerProperties"))
        self.assertFalse(hasattr(lib, "vk_layerGetPhysicalDeviceProcAddr"))

        count = ctypes.c_uint32(0)
        enum_props = lib.vkEnumerateInstanceLayerProperties
        enum_props.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(VkLayerProperties)]
        enum_props.restype = ctypes.c_int
        self.assertEqual(enum_props(ctypes.byref(count), None), 0)
        self.assertEqual(count.value, 1)
        props = (VkLayerProperties * count.value)()
        self.assertEqual(enum_props(ctypes.byref(count), props), 0)
        self.assertEqual(props[0].layerName.decode(), "VK_LAYER_VD_capture")
        self.assertEqual(props[0].implementationVersion, 2)

        negotiate = lib.vkNegotiateLoaderLayerInterfaceVersion
        negotiate.argtypes = [ctypes.POINTER(VkNegotiateLayerInterface)]
        negotiate.restype = ctypes.c_int
        info = VkNegotiateLayerInterface()
        info.sType = 1  # LAYER_NEGOTIATE_INTERFACE_STRUCT
        info.loaderLayerInterfaceVersion = 2
        self.assertEqual(negotiate(ctypes.byref(info)), 0)
        self.assertEqual(info.loaderLayerInterfaceVersion, 2)
        self.assertTrue(info.pfnGetInstanceProcAddr)
        self.assertTrue(info.pfnGetDeviceProcAddr)
        self.assertFalse(info.pfnGetPhysicalDeviceProcAddr)

    def test_02_shared_memory_contract_is_stable(self):
        self.assertEqual(HEADER_SIZE, 304)
        header_text = (ROOT_DIR / "native" / "vulkan_layer" / "vd_capture_layer.h").read_text()
        self.assertIn("VD_LAYER_SHM_VERSION 2u", header_text)
        self.assertIn("VD_CAPTURE_MAX_WIDTH 256u", header_text)
        self.assertIn("VD_CAPTURE_MAX_HEIGHT 256u", header_text)
        self.assertIn("roi_capture_count", header_text)
        self.assertIn("roi_capture_timestamp_ns", header_text)

    def test_03_manifest_and_capture_are_opt_in(self):
        manifest = json.loads(LAYER_MANIFEST.read_text())
        layer = manifest["layer"]
        self.assertEqual(layer["name"], "VK_LAYER_VD_capture")
        self.assertEqual(layer["api_version"], "1.4.0")
        self.assertEqual(layer["implementation_version"], "2")
        self.assertEqual(layer["disable_environment"], {"DISABLE_VD_LAYER": "1"})

        source = (ROOT_DIR / "native" / "vulkan_layer" / "vd_capture_layer.cpp").read_text()
        self.assertIn('env_true("VD_CAPTURE_ENABLE")', source)
        self.assertIn("VK_IMAGE_USAGE_TRANSFER_SRC_BIT", source)
        self.assertIn("vkCmdCopyImageToBuffer", source)
        self.assertNotIn("vkQueueWaitIdle(", source)
        self.assertIn("g_log_mutex", source)

    def test_04_roi_viewer_converts_bgra_and_writes_png(self):
        viewer = load_viewer_module()
        raw = bytes((51, 153, 26, 255, 10, 20, 30, 40))
        rgba = viewer.raw_to_rgba(raw, 44)
        self.assertEqual(rgba, bytes((26, 153, 51, 255, 30, 20, 10, 40)))

        out = Path("/tmp/vd-roi-viewer-test.png")
        try:
            viewer.write_rgba_png(out, 2, 1, rgba)
            self.assertTrue(out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        finally:
            out.unlink(missing_ok=True)

    @unittest.skipUnless(have_flatpak_sober(), "Sober Flatpak not installed")
    def test_05_flatpak_extension_install_and_discovery(self):
        install = ROOT_DIR / "native" / "vulkan_layer" / "install_sober_layer.sh"
        result = subprocess.run([str(install)], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"Install failed:\n{result.stdout}\n{result.stderr}")
        self.assertIn("extension library + merged manifest visible inside Sober", result.stdout)

        check = subprocess.run(
            [
                "flatpak", "run", "--command=sh", "org.vinegarhq.Sober", "-c",
                "test -r /usr/lib/extensions/vulkan/VDCapture/lib/libVkLayer_VD_capture.so && "
                "test -r /usr/share/vulkan/implicit_layer.d/VkLayer_VD_capture.json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(check.returncode, 0, f"Flatpak extension discovery failed: {check.stderr}")

    @unittest.skipUnless(have_flatpak_sober(), "Sober Flatpak not installed")
    def test_06_deterministic_roi_pixel_readback(self):
        install = ROOT_DIR / "native" / "vulkan_layer" / "install_sober_layer.sh"
        result = subprocess.run([str(install)], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, f"Install failed:\n{result.stdout}\n{result.stderr}")

        app_src = ROOT_DIR / "native" / "vulkan_layer" / "test_capture_app.c"
        app_bin = SOBER_DATA_DIR / "test_capture_app"
        app_bin.parent.mkdir(parents=True, exist_ok=True)
        compile_result = subprocess.run(
            ["gcc", "-O2", str(app_src), "-lvulkan", "-o", str(app_bin)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(compile_result.returncode, 0, f"Test app compile failed: {compile_result.stderr}")

        for path in (
            SOBER_VULKAN_DIR / "vd_layer_shm.dat",
            SOBER_VULKAN_DIR / "vd_layer_status.json",
            SOBER_VULKAN_DIR / "vd_layer.log",
        ):
            path.unlink(missing_ok=True)

        run_result = subprocess.run(
            [
                "flatpak", "run",
                "--env=VD_CAPTURE_ENABLE=1",
                "--env=VD_CAPTURE_INTERVAL_MS=0",
                f"--command={app_bin}",
                "org.vinegarhq.Sober",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            run_result.returncode, 0,
            f"Capture app failed:\nSTDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}",
        )
        self.assertIn("CAPTURE_TEST_OK", run_result.stdout)

        viewer = load_viewer_module()
        snap = viewer.snapshot(SOBER_VULKAN_DIR / "vd_layer_shm.dat")
        self.assertGreater(snap["capture_count"], 0)
        self.assertGreater(snap["width"], 0)
        self.assertGreater(snap["height"], 0)
        self.assertEqual(snap["format"], 44)
        self.assertGreaterEqual(len(snap["raw"]), 4)

        # The test app clears B8G8R8A8_UNORM to logical RGBA ~= (0.10, 0.60, 0.20, 1.0),
        # therefore raw memory should be BGRA ~= (51, 153, 26, 255).
        b, g, r, a = snap["raw"][:4]
        self.assertLessEqual(abs(b - 51), 3)
        self.assertLessEqual(abs(g - 153), 3)
        self.assertLessEqual(abs(r - 26), 3)
        self.assertGreaterEqual(a, 250)


if __name__ == "__main__":
    unittest.main(verbosity=2)
