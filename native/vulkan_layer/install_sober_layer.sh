#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
app_id="org.vinegarhq.Sober"
ext_name="VDCapture"

runtime="$(flatpak info --show-runtime "$app_id")"
runtime_id="${runtime%%/*}"
runtime_arch_branch="${runtime#*/}"
runtime_arch="${runtime_arch_branch%%/*}"
runtime_branch="${runtime##*/}"

meta="$(flatpak info -m "$runtime_id//$runtime_branch")"
ext_version="$(awk '
    /^\[Extension org\.freedesktop\.Platform\.VulkanLayer\]$/ {in_section=1; next}
    /^\[/ {if (in_section) exit}
    in_section && /^[[:space:]]*version[[:space:]]*=/ {
        sub(/^[^=]*=[[:space:]]*/, ""); gsub(/[[:space:]]+$/, ""); print; exit
    }
' <<<"$meta")"

if [[ -z "$ext_version" ]]; then
    echo "ERROR: could not determine org.freedesktop.Platform.VulkanLayer extension version" >&2
    exit 1
fi

ext_root="${XDG_DATA_HOME:-$HOME/.local/share}/flatpak/extension/org.freedesktop.Platform.VulkanLayer.${ext_name}/${runtime_arch}/${ext_version}"
lib_dir="$ext_root/lib"
manifest_dir="$ext_root/share/vulkan/implicit_layer.d"
internal_lib="/usr/lib/extensions/vulkan/${ext_name}/lib/libVkLayer_VD_capture.so"

printf '[INSTALL] Sober runtime: %s\n' "$runtime"
printf '[INSTALL] VulkanLayer extension version: %s\n' "$ext_version"
printf '[INSTALL] Extension root: %s\n' "$ext_root"

"$script_dir/build_layer.sh"
mkdir -p "$lib_dir" "$manifest_dir"
cp -f "$root/build/vulkan-layer/libVkLayer_VD_capture.so" "$lib_dir/libVkLayer_VD_capture.so"
chmod 755 "$lib_dir/libVkLayer_VD_capture.so"
sed "s|@LAYER_LIBRARY_PATH@|$internal_lib|g" "$script_dir/VkLayer_VD_capture.json.in" > "$manifest_dir/VkLayer_VD_capture.json"
chmod 644 "$manifest_dir/VkLayer_VD_capture.json"

# Remove the obsolete app-data DSO/manifest. Sober accepts the VulkanLayer extension mount,
# while arbitrary user DSO paths can stop at the loader's 'Loading layer library' stage.
sober_vulkan="$HOME/.var/app/$app_id/data/vulkan"
rm -f "$sober_vulkan/libVkLayer_VD_capture.so" "$sober_vulkan/implicit_layer.d/VkLayer_VD_capture.json"

flatpak run --command=sh "$app_id" -c '
set -eu
test -r /usr/lib/extensions/vulkan/VDCapture/lib/libVkLayer_VD_capture.so
test -r /usr/share/vulkan/implicit_layer.d/VkLayer_VD_capture.json
echo "[VERIFY] extension library + merged manifest visible inside Sober"
'

echo "[INSTALL] Done"
