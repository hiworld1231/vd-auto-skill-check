# Violence District Auto Skill Check — CLEAN V5

Внешний CV-бот для Roblox **Violence District**. Один runtime обрабатывает обычные и ускоренные skill-check'и: он не определяет "perk mode", а измеряет фактическое движение стрелки в текущем check.

## Быстрый старт

```bash
git clone https://github.com/hiworld1231/vd-auto-skill-check.git
cd vd-auto-skill-check
python -m pip install -r requirements.txt
python tools/preflight.py
python run.py
```

Прямой запуск:

```bash
python skillcheck_bot.py --gen-rush
```

`--gen-rush` оставлен для совместимости со старыми командами; CLEAN V5 автоматически работает и с normal, и с perk-speed checks.

## Архитектура

```text
VFR capture
  -> fixed Roblox geometry / HYBRID needle detector
  -> continuous angle-vs-time predictor
  -> absolute predicted target crossing
  -> session delivery lead
  -> precise scheduler
  -> evdev/UInput Space
  -> post-hit outcome observer
  -> clean-only lead update + async replay recorder
```

Ключевые свойства:

- VFR `gpu-screen-recorder` capture; без искусственного CFR-дублирования кадров.
- Target — физический центр GREAT-зоны.
- Continuous measured angular speed; нет BASE/PERK tiers и `speed_profiles.json`.
- Predictive deadline пересчитывается по свежим unique-frame observations.
- Unsafe late direct-fire блокируется (`NO_FIRE`) вместо случайного MISS.
- Input fail-closed: отсутствие рабочего UInput backend — ошибка, а не скрытый dry-run.
- Capture worker death/stale-frame не маскируется последним кадром.
- Session lead обучается только на чистых landing samples.
- Frenzy имеет отдельный generation lifecycle; переходные samples не обучают lead.
- Recorder работает вне critical fire path и корректно drain'ится при shutdown.

## Конфиг

`config.json`:

- `fps` — верхний предел VFR capture (по умолчанию 120).
- `region` — ROI интерфейса skill check.
- `require_lmb` — принимать skill check только при удержании LMB.
- `genrush_seed_lead_ms` — cold-start delivery lead, не постоянная latency.
- `session_base_speed` — diagnostic/reset prior; FIRE использует measured speed.
- `space_hold_ms` — длительность удержания Space.
- `detector` — `hybrid` (production) или `baseline` (diagnostic).
- `allow_pynput_fallback` — по умолчанию `false`; Wayland production path — evdev/UInput.

## Linux / Wayland

Предпочтительно установить:

- `gpu-screen-recorder`
- `ffmpeg`
- доступ к `/dev/uinput`

Python-зависимости:

```bash
python -m pip install -r requirements.txt
```

Проверка окружения:

```bash
python tools/preflight.py
```

## Тесты

```bash
python -m pytest -q
```

Чистое GitHub-дерево содержит только focused release-tests. Перед публикацией этот exact tree проходит их без ошибок.

## Обновление

После первого клонирования дальше достаточно:

```bash
git pull
python -m pip install -r requirements.txt
python tools/preflight.py
```

## Важно

Offline/unit/replay проверки подтверждают логику и отсутствие известных regression-багов, но не гарантируют процент GREAT в реальной игре. Финальное доказательство качества — LIVE-сессии Violence District и сохранённые replay telemetry.
