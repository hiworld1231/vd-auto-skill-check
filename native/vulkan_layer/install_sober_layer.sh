#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"

# Target directory inside Sober's sandboxed data home
sober_data="${SOBER_DATA_DIR:-$HOME/.var/app/org.vinegarhq.Sober/data}"
sober_vulkan_dir="$sober_data/vulkan"
manifest_dir="$sober_vulkan_dir/implicit_layer.d"
installed_lib="$sober_vulkan_dir/libVkLayer_VD_capture.so"
installed_manifest="$manifest_dir/VkLayer_VD_capture.json"

echo "=== Installing VD Vulkan Implicit Layer to Sober Flatpak ==="
echo "Target Vulkan directory: $sober_vulkan_dir"

# Build layer first
"$script_dir/build_layer.sh"

built_lib="$root/build/vulkan-layer/libVkLayer_VD_capture.so"
if [[ ! -f "$built_lib" ]]; then
    echo "ERROR: Built library not found at $built_lib" >&2
    exit 1
fi

mkdir -p "$manifest_dir"

# Copy library
cp -f "$built_lib" "$installed_lib"
chmod 755 "$installed_lib"
echo "[INSTALL] Copied shared library to $installed_lib"

# Generate and install manifest pointing to Sober internal path
sed "s|@LAYER_LIBRARY_PATH@|$installed_lib|g" "$script_dir/VkLayer_VD_capture.json.in" > "$installed_manifest"
chmod 644 "$installed_manifest"
echo "[INSTALL] Installed manifest at $installed_manifest"

echo "[VERIFY] Verifying layer discovery inside Flatpak Sober..."
python_check="
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
    desc = layers[i].description.decode('utf-8', errors='ignore')
    if 'VK_LAYER_VD_capture' in name:
        print(f'SUCCESS: Discovered inside Flatpak: {name} (v{layers[i].implementationVersion}) - {desc}')
        found = True

if not found:
    print('FAILURE: Layer not discovered by Vulkan Loader inside Flatpak')
    exit(1)
"

flatpak run --command=python3 org.vinegarhq.Sober -c "$python_check"

echo "=== Installation & Discovery Verification Complete! ==="
