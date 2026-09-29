#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
build_dir="$root/build/vulkan-layer"

mkdir -p "$build_dir"

src="$script_dir/vd_capture_layer.cpp"
header="$script_dir/vd_capture_layer.h"
out_lib="$build_dir/libVkLayer_VD_capture.so"
out_manifest="$build_dir/VkLayer_VD_capture.json"

echo "[BUILD] Compiling Vulkan Implicit Layer..."
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC -shared \
    -fvisibility=hidden \
    -Wl,-Bsymbolic \
    -Wl,-soname,libVkLayer_VD_capture.so \
    -I"$script_dir" \
    "$src" -o "$out_lib"

echo "[BUILD] Shared library compiled at: $out_lib"

# Generate local manifest pointing to build dir
sed "s|@LAYER_LIBRARY_PATH@|$out_lib|g" "$script_dir/VkLayer_VD_capture.json.in" > "$out_manifest"
echo "[BUILD] Manifest generated at: $out_manifest"

echo "[BUILD] Success!"
