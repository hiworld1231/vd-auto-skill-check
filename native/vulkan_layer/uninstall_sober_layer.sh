#!/usr/bin/env bash
set -euo pipefail

app_id="org.vinegarhq.Sober"
ext_base="${XDG_DATA_HOME:-$HOME/.local/share}/flatpak/extension/org.freedesktop.Platform.VulkanLayer.VDCapture"
sober_vulkan="$HOME/.var/app/$app_id/data/vulkan"

rm -rf "$ext_base"
rm -f "$sober_vulkan/libVkLayer_VD_capture.so" "$sober_vulkan/implicit_layer.d/VkLayer_VD_capture.json"

if [[ "${1:-}" == "--clean-logs" ]]; then
    rm -f "$sober_vulkan/vd_layer.log" \
          "$sober_vulkan/vd_layer_status.json" \
          "$sober_vulkan"/vd_layer_status.json.*.tmp \
          "$sober_vulkan/vd_layer_shm.dat"
fi

echo "[UNINSTALL] VD Vulkan layer removed"
