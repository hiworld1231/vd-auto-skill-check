#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$root/build/kwin-region"
mkdir -p "$build_dir"

protocol_dir="$(pkg-config --variable=pkgdatadir wayland-protocols)"
kwin_protocol="/usr/share/plasma-wayland-protocols/zkde-screencast-unstable-v1.xml"
xdg_protocol="$protocol_dir/stable/xdg-shell/xdg-shell.xml"
for file in "$kwin_protocol" "$xdg_protocol"; do
    if [[ ! -r "$file" ]]; then
        echo "Required Wayland protocol file is missing: $file" >&2
        exit 1
    fi
done

wayland-scanner client-header "$xdg_protocol" "$build_dir/xdg-shell-client-protocol.h"
wayland-scanner private-code "$xdg_protocol" "$build_dir/xdg-shell-protocol.c"
wayland-scanner client-header "$kwin_protocol" "$build_dir/zkde-screencast-client-protocol.h"
wayland-scanner private-code "$kwin_protocol" "$build_dir/zkde-screencast-protocol.c"

pkg_cflags="$(pkg-config --cflags Qt6Core Qt6Gui libpipewire-0.3 wayland-client)"
pkg_libs="$(pkg-config --libs Qt6Core Qt6Gui libpipewire-0.3 wayland-client)"
read -r -a cflags <<< "$pkg_cflags"
read -r -a libs <<< "$pkg_libs"

cc -fPIC "${cflags[@]}" -c "$build_dir/xdg-shell-protocol.c" \
    -o "$build_dir/xdg-shell-protocol.o"
cc -fPIC "${cflags[@]}" -c "$build_dir/zkde-screencast-protocol.c" \
    -o "$build_dir/zkde-screencast-protocol.o"
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC \
    -I/usr/include/KPipeWire -I"$build_dir" "${cflags[@]}" \
    "$root/native/kwin_region_worker.cpp" \
    "$build_dir/xdg-shell-protocol.o" "$build_dir/zkde-screencast-protocol.o" \
    -L/usr/lib -lKPipeWire "${libs[@]}" -o "$build_dir/kwin-region-worker"
echo "$build_dir/kwin-region-worker"
