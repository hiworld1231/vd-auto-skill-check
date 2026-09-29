#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_id="org.vinegarhq.Sober"
vulkan_dir="$HOME/.var/app/$app_id/data/vulkan"

# Capture env must exist in the Sober process itself, so restart it here.
flatpak kill "$app_id" >/dev/null 2>&1 || true
rm -f "$vulkan_dir/vd_layer_shm.dat" "$vulkan_dir/vd_layer_status.json" "$vulkan_dir/vd_layer.log"

flatpak run \
  --env=VD_CAPTURE_ENABLE=1 \
  --env=VD_CAPTURE_INTERVAL_MS=0 \
  "$app_id" \
  > /tmp/sober-vd-vulkan-solver.log 2>&1 &

printf '%s\n' 'Sober запущен с Vulkan capture. Зайди в игру; solver ждёт первый игровой кадр до 65 секунд.'
exec python "$project_dir/run-vulkan.py" "$@"
