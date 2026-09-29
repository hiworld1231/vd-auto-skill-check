#!/usr/bin/env bash
set -euo pipefail

app_id="org.vinegarhq.Sober"
vulkan_dir="$HOME/.var/app/$app_id/data/vulkan"
log_file="/tmp/sober-vd-vulkan.log"

if flatpak ps --columns=application 2>/dev/null | grep -Fxq "$app_id"; then
  printf '%s\n' 'Sober уже открыт. Не перезапускаю его.'
  printf '%s\n' 'Если он был запущен без Vulkan capture, закрой Sober вручную и снова запусти ./start-vulkan-sober.sh.'
  exit 0
fi

mkdir -p "$vulkan_dir"
rm -f "$vulkan_dir/vd_layer_shm.dat" "$vulkan_dir/vd_layer_status.json" "$vulkan_dir/vd_layer.log"

flatpak run \
  --env=VD_CAPTURE_ENABLE=1 \
  --env=VD_CAPTURE_INTERVAL_MS=0 \
  "$app_id" \
  >"$log_file" 2>&1 &

printf '%s\n' 'Sober запущен с Vulkan capture и останется работать отдельно от solver.'
printf 'Лог: %s\n' "$log_file"
