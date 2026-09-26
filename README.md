# VD rewrite

Independent rewrite of the skill-check path under `vd_rewrite/`. It does not import the legacy root `core` implementation.

## Current state

The rewrite contains independent PipeWire capture, per-frame CV, robust motion fitting, generation lifecycle, scheduling, recording and replay. The planner is **GREAT-first**: it targets the narrow GREAT arc whenever the full timing uncertainty fits there. If GREAT is too narrow but the uncertainty fits inside the paired GREAT+GOOD success interval, it emits an explicit `GOOD_FALLBACK` plan. If neither interval is safe, it keeps `TARGET_UNCERTAIN` and does not claim a press.

The uploaded GitHub branch supports `preflight`, `capture-probe`, `dry-run`, and `replay`. The local rewrite also contains the physical evdev/UInput component used by `run`, but that single file could not be published through the connected GitHub write path. Consequently this branch deliberately lazy-loads physical input: dry-run/replay remain usable, while `run` reports a clear error instead of breaking module import.

## Setup

```bash
cd vd_rewrite
./setup.sh
python run.py preflight
python run.py capture-probe --synthetic --seconds 5
python run.py dry-run --seconds 60
```

Live capture opens KDE's standard screen-sharing selector. Select the monitor containing the game. The current ROI is fixed for the tested 1920x1080 layout: `800,420,320,240`, with ring radius 66 px.

Dry-run treats LMB as held but opens no input device. Every run records processed ROI frames losslessly under `recordings/` unless another output directory is supplied.

## Replay

```bash
python run.py replay --recording recordings/<session> --report reports/replay.json
```

Replay reruns CV, motion and engine state from the recorded PNG frames. Timer wakes are idealized, and images after a virtual claim are counterfactual because no physical Space was sent during replay. Therefore replay claims are scheduler/CV evidence, not proof of an in-game hit.

## GOOD fallback found from the live dry-run

The recording `20260926_154417_693878` repeatedly produced `TARGET_UNCERTAIN`: GREAT was about 10.7 degrees wide while the modeled uncertainty could be wider than half of GREAT. The paired GOOD arc immediately after it was about 42 degrees wide.

After the planner change, replay of that recording produced 3 virtual claims, all explicitly marked `GOOD_FALLBACK`. Frames ending in `TARGET_UNCERTAIN` dropped from 48 to 7. GREAT is still preferred whenever its own uncertainty envelope fits, and a remote/unpaired GOOD arc is never allowed to widen the target.

## Tests performed on the uploaded source snapshot

The local source snapshot from which this branch was uploaded passed 55 tests covering the rewrite logic; the only separately collected mouse-monitor test could not be collected in the execution environment because the `evdev` package was unavailable there. Python compile checks passed. The exact GOOD-fallback geometry from the live recording is also covered by `tests/test_planning.py`.

These checks validate program behavior. They do not establish Roblox-side hit accuracy or long-session stability.
