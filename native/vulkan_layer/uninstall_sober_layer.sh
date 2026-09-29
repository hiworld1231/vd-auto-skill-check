#!/usr/bin/env bash
set -euo pipefail

sober_data="${SOBER_DATA_DIR:-$HOME/.var/app/org.vinegarhq.Sober/data}"
sober_vulkan_dir="$sober_data/vulkan"
manifest_dir="$sober_vulkan_dir/implicit_layer.d"
installed_lib="$sober_vulkan_dir/libVkLayer_VD_capture.so"
installed_manifest="$manifest_dir/VkLayer_VD_capture.json"

echo "=== Uninstalling VD Vulkan Implicit Layer from Sober Flatpak ==="

if [[ -f "$installed_manifest" ]]; then
    rm -f "$installed_manifest"
    echo "[UNINSTALL] Removed manifest $installed_manifest"
fi

if [[ -f "$installed_lib" ]]; then
    rm -f "$installed_lib"
    echo "[UNINSTALL] Removed shared library $installed_lib"
fi

if [[ "${1:-}" == "--clean-logs" ]]; then
    rm -f "$sober_vulkan_dir/vd_layer.log" \
          "$sober_vulkan_dir/vd_layer_status.json" \
          "$sober_vulkan_dir/vd_layer_status.json.tmp" \
          "$sober_vulkan_dir/vd_layer_shm.dat"
    echo "[UNINSTALL] Cleaned up logs and status files"
fi

echo "=== VD Vulkan Implicit Layer Uninstalled Successfully ==="
