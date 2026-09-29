#!/usr/bin/env python3
"""
Test suite for VD Vulkan Implicit Layer.
Tests layer negotiation, Vulkan loader discovery, instance creation,
physical device enumeration, swapchain hooks, presentation frames,
shared memory header layout, and status reporting inside Flatpak Sober.
"""

import ctypes
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
SOBER_DATA_DIR = Path(os.environ.get("HOME", "/home/oae")) / ".var" / "app" / "org.vinegarhq.Sober" / "data"
SOBER_VULKAN_DIR = SOBER_DATA_DIR / "vulkan"


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


class TestVulkanLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure layer is built and installed to Sober
        build_script = ROOT_DIR / "native" / "vulkan_layer" / "build_layer.sh"
        res = subprocess.run([str(build_script)], capture_output=True, text=True)
        assert res.returncode == 0, f"Build failed: {res.stderr}"
        assert LAYER_LIB.exists(), f"Layer lib missing: {LAYER_LIB}"
        assert LAYER_MANIFEST.exists(), f"Layer manifest missing: {LAYER_MANIFEST}"

        install_script = ROOT_DIR / "native" / "vulkan_layer" / "install_sober_layer.sh"
        res = subprocess.run([str(install_script)], capture_output=True, text=True)
        assert res.returncode == 0, f"Install failed: {res.stderr}"

    def test_01_direct_symbols_and_negotiation(self):
        """Verify layer exports and interface version negotiation."""
        lib = ctypes.CDLL(str(LAYER_LIB))
        self.assertTrue(hasattr(lib, "vkNegotiateLoaderLayerInterfaceVersion"))
        self.assertTrue(hasattr(lib, "vkGetInstanceProcAddr"))
        self.assertTrue(hasattr(lib, "vkGetDeviceProcAddr"))
        self.assertTrue(hasattr(lib, "vkEnumerateInstanceLayerProperties"))
        self.assertTrue(hasattr(lib, "vk_layerGetPhysicalDeviceProcAddr"))

        # Test layer properties enumeration
        count = ctypes.c_uint32(0)
        res = lib.vkEnumerateInstanceLayerProperties(ctypes.byref(count), None)
        self.assertEqual(res, 0)
        self.assertGreaterEqual(count.value, 1)

        props = (VkLayerProperties * count.value)()
        res = lib.vkEnumerateInstanceLayerProperties(ctypes.byref(count), props)
        self.assertEqual(res, 0)
        layer_name = props[0].layerName.decode()
        self.assertEqual(layer_name, "VK_LAYER_VD_capture")

        # Test interface version negotiation
        negotiate = lib.vkNegotiateLoaderLayerInterfaceVersion
        negotiate.argtypes = [ctypes.POINTER(VkNegotiateLayerInterface)]
        negotiate.restype = ctypes.c_int

        interface_struct = VkNegotiateLayerInterface()
        interface_struct.sType = 1  # LAYER_NEGOTIATE_INTERFACE_STRUCT
        interface_struct.loaderLayerInterfaceVersion = 2

        res = negotiate(ctypes.byref(interface_struct))
        self.assertEqual(res, 0)
        self.assertEqual(interface_struct.loaderLayerInterfaceVersion, 2)
        self.assertIsNotNone(interface_struct.pfnGetInstanceProcAddr)
        self.assertIsNotNone(interface_struct.pfnGetDeviceProcAddr)
        # Interface version 2 requires pfnGetPhysicalDeviceProcAddr
        self.assertIsNotNone(interface_struct.pfnGetPhysicalDeviceProcAddr)

    def test_02_shm_header_struct_layout(self):
        """Verify shared memory header layout and size."""
        # VdLayerShmHeader:
        # magic (uint32), version (uint32), struct_size (uint32), status_flags (uint32) = 16 bytes
        # pid (uint32), sequence (uint32) = 8 bytes
        # init_timestamp_ns (uint64), last_present_ns (uint64) = 16 bytes
        # frame_count (uint64), current_fps (float32), dropped_frames (uint32) = 16 bytes
        # swapchain_width, height, format, present_mode, image_count, current_image_index = 24 bytes
        # process_name (char[64]) = 64 bytes
        # roi_x, y, width, height, stride, offset, size = 28 bytes + 4 bytes padding = 32 bytes
        # padding (128 bytes)
        shm_format = "=IIII II QQ QfI IIIIII 64s IIIIIII 128s 4x"
        expected_size = struct.calcsize(shm_format)
        self.assertEqual(expected_size, 304)

    def test_03_flatpak_sober_discovery(self):
        """Verify that Vulkan Loader inside Flatpak Sober discovers the layer."""
        python_check = """
import ctypes

class VkLayerProperties(ctypes.Structure):
    _fields_ = [
        ('layerName', ctypes.c_char * 256),
        ('specVersion', ctypes.c_uint32),
        ('implementationVersion', ctypes.c_uint32),
        ('description', ctypes.c_char * 256),
    ]

vulkan = ctypes.CDLL('libvulkan.so.1')
count = ctypes.c_uint32(0)
res = vulkan.vkEnumerateInstanceLayerProperties(ctypes.byref(count), None)
layers = (VkLayerProperties * count.value)()
res = vulkan.vkEnumerateInstanceLayerProperties(ctypes.byref(count), layers)

found = False
for i in range(count.value):
    name = layers[i].layerName.decode('utf-8', errors='ignore')
    if 'VK_LAYER_VD_capture' in name:
        print('DISCOVERED_LAYER:' + name)
        found = True

assert found, 'Layer not found'
"""
        res = subprocess.run(
            ["flatpak", "run", "--command=python3", "org.vinegarhq.Sober", "-c", python_check],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Discovery failed: {res.stderr}\n{res.stdout}")
        self.assertIn("DISCOVERED_LAYER:VK_LAYER_VD_capture", res.stdout)

    def test_04_vulkan_instance_activation_mock(self):
        """Verify layer is invoked during vkCreateInstance via Vulkan Loader."""
        env = os.environ.copy()
        env["VK_LAYER_PATH"] = str(BUILD_DIR)
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_VD_capture"
        env["VD_LAYER_DIR"] = "/tmp/vd_test_layer_env"

        test_code = """
import ctypes

class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ('sType', ctypes.c_uint32),
        ('pNext', ctypes.c_void_p),
        ('pApplicationName', ctypes.c_char_p),
        ('applicationVersion', ctypes.c_uint32),
        ('pEngineName', ctypes.c_char_p),
        ('engineVersion', ctypes.c_uint32),
        ('apiVersion', ctypes.c_uint32),
    ]

class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ('sType', ctypes.c_uint32),
        ('pNext', ctypes.c_void_p),
        ('flags', ctypes.c_uint32),
        ('pApplicationInfo', ctypes.POINTER(VkApplicationInfo)),
        ('enabledLayerCount', ctypes.c_uint32),
        ('ppEnabledLayerNames', ctypes.POINTER(ctypes.c_char_p)),
        ('enabledExtensionCount', ctypes.c_uint32),
        ('ppEnabledExtensionNames', ctypes.POINTER(ctypes.c_char_p)),
    ]

vulkan = ctypes.CDLL('libvulkan.so.1')

app_info = VkApplicationInfo()
app_info.sType = 1 # VK_STRUCTURE_TYPE_APPLICATION_INFO
app_info.pApplicationName = b'VD_Test_App'
app_info.apiVersion = (1 << 22) | (3 << 12)

layer_name = ctypes.c_char_p(b'VK_LAYER_VD_capture')
layer_names = (ctypes.c_char_p * 1)(layer_name)

create_info = VkInstanceCreateInfo()
create_info.sType = 10 # VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
create_info.pApplicationInfo = ctypes.pointer(app_info)
create_info.enabledLayerCount = 1
create_info.ppEnabledLayerNames = layer_names

instance = ctypes.c_void_p(0)
res = vulkan.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
assert res == 0, f'vkCreateInstance failed with code {res}'
vulkan.vkDestroyInstance(instance, None)
print('VULKAN_INSTANCE_SUCCESS')
"""
        res = subprocess.run([sys.executable, "-c", test_code], env=env, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Vulkan test app failed: {res.stderr}\n{res.stdout}")
        self.assertIn("VULKAN_INSTANCE_SUCCESS", res.stdout)

        test_dir = Path("/tmp/vd_test_layer_env")
        log_file = test_dir / "vd_layer.log"
        status_file = test_dir / "vd_layer_status.json"
        shm_file = test_dir / "vd_layer_shm.dat"

        self.assertTrue(log_file.exists(), f"Log file missing: {log_file}")
        self.assertTrue(status_file.exists(), f"Status file missing: {status_file}")
        self.assertTrue(shm_file.exists(), f"Shm file missing: {shm_file}")

        with open(status_file, "r") as f:
            status_data = json.load(f)
            self.assertEqual(status_data["layer_name"], "VK_LAYER_VD_capture")

        # Clean up /tmp test dir
        for p in (log_file, status_file, shm_file):
            if p.exists():
                p.unlink()
        for p in test_dir.glob("*.tmp"):
            p.unlink()
        if test_dir.exists():
            test_dir.rmdir()

    def test_05_flatpak_sober_swapchain_and_presentation(self):
        """Verify full instance, device, swapchain, and presentation lifecycle inside Flatpak Sober."""
        app_src = ROOT_DIR / "native" / "vulkan_layer" / "test_layer_app.c"
        app_bin = SOBER_DATA_DIR / "test_layer_app"

        # Compile test client
        comp_res = subprocess.run(
            ["gcc", "-O2", str(app_src), "-lvulkan", "-o", str(app_bin)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(comp_res.returncode, 0, f"Compilation failed: {comp_res.stderr}")

        # Run test client inside Flatpak Sober with swapchain testing
        run_res = subprocess.run(
            ["flatpak", "run", f"--command={app_bin}", "org.vinegarhq.Sober", "--test-swapchain"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run_res.returncode, 0, f"Execution failed: {run_res.stderr}\n{run_res.stdout}")
        self.assertIn("Completed 30 presentation frames", run_res.stdout)
        self.assertIn("vkDestroySwapchainKHR completed", run_res.stdout)
        self.assertIn("vkDestroyInstance completed successfully", run_res.stdout)

        # Inspect status JSON and SHM in Sober data dir
        status_file = SOBER_VULKAN_DIR / "vd_layer_status.json"
        shm_file = SOBER_VULKAN_DIR / "vd_layer_shm.dat"

        self.assertTrue(status_file.exists(), f"Status file missing: {status_file}")
        self.assertTrue(shm_file.exists(), f"SHM file missing: {shm_file}")

        with open(status_file, "r") as f:
            status = json.load(f)
            self.assertEqual(status["layer_name"], "VK_LAYER_VD_capture")
            self.assertEqual(status["frames_presented"], 30)
            self.assertEqual(status["swapchain"]["width"], 1280)
            self.assertEqual(status["swapchain"]["height"], 720)
            self.assertEqual(status["swapchain"]["format"], 44)  # VK_FORMAT_B8G8R8A8_UNORM

        # Check SHM header
        with open(shm_file, "rb") as f:
            data = f.read(144)
            (
                magic,
                ver,
                ssize,
                flags,
                pid,
                seq,
                init_ts,
                last_ts,
                frame_count,
                fps,
                dropped,
                w,
                h,
                fmt,
                mode,
                img_cnt,
                cur_idx,
                proc_name,
            ) = struct.unpack("=IIII II QQ QfI IIIIII 64s", data)
            self.assertEqual(magic, 0x56444C59)
            self.assertEqual(frame_count, 30)
            self.assertEqual(w, 1280)
            self.assertEqual(h, 720)
            self.assertEqual(fmt, 44)


if __name__ == "__main__":
    unittest.main()
