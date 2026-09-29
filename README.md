# VD screen solver

`./start-solver.sh` captures the selected monitor, watches a centered 320×240
skill-check area, and sends Space while LMB is held. By default it captures the
monitor and uses the `(800, 420)` area for 1920×1080. You can instead capture
just the game window with `--capture-source window`; that mode centers the same
area inside the selected window.
It aims at the white GREAT arc on every detected check, including successive
checks on a continuous Frenzy ring. It does not target GOOD. If the needle is
visible but its speed is not ready, the solver waits up to 300 ms for reliable
movement before a blind GREAT attempt. If measured movement later stalls, it
waits 150 ms after the last reliable motion. A capture gap discards the old
speed fit so the next press is planned only after motion is reacquired. No
screen-only solver can guarantee a hit when the target crosses between
captured frames.

## Start

```bash
./setup.sh
python run.py preflight
./start-solver.sh
```

Select the game monitor in KDE's screen-sharing dialog. With
`--capture-source window`, select the game window instead. Ctrl+C stops the
solver. Decision and timing logs are saved under `recordings/` unless an
explicit `--recording DIR` is supplied. Frame video is off by default; add
`--video` only when you need replay footage. The solver neither reads nor writes
the game process. It analyzes every captured frame while LMB is held and skips
CV while released. The capture worker runs at lower CPU priority than the game.
The default lead is 35 ms, based on recent CV freeze estimates; two consistent
physical observations can refine it for the current session.

## Compare capture profiles

The default is `quiet`: 60 FPS, capture worker nice 10, and an event-only
recording. The saved `baseline` remains available at 60 FPS and nice 5. To
compare load and responsiveness, stop the current solver before starting
another profile:

```bash
./start-solver.sh --variant baseline
./start-solver.sh --variant responsive
./start-solver.sh --variant quiet
./start-solver.sh --variant fps50
./start-solver.sh --variant fps45
./start-solver.sh --variant fps30
./start-solver.sh --variant deep-quiet
```

`baseline` requests 60 FPS and nice 5 for the capture worker. `responsive`
keeps 60 FPS and requests nice 0; `quiet` keeps 60 FPS and requests nice 10.
`fps50`, `fps45`, and `fps30` lower the capture rate while requesting nice 5;
`deep-quiet` uses 30 FPS and requests nice 10. A higher nice value lets the game
get CPU sooner but can delay frames. Lower FPS reduces capture work but can miss
a fast check between frames. All profiles use the same solver and continue
trying on each check it detects. The chosen profile, requested FPS, and worker
priority are recorded in each session's `manifest.json`.

You can also mix settings for an extra test, for example
`./start-solver.sh --variant quiet --fps 50 --capture-priority 5`.
Explicit `--fps` and `--capture-priority` values override the selected profile.
To compare monitor capture with a smaller game-window stream while keeping the
same solver profile, use `./start-solver.sh --variant quiet --capture-source window`.
Window capture can reduce PipeWire traffic when the game window is smaller than
the monitor; a fullscreen window may have the same cost as monitor capture.
For an opt-in lower-resolution test, run the game windowed at 16:9 and use
`--capture-source window --normalize-window-scale`. The worker crops a region
scaled to the window and resizes it to the solver's 320×240 input. This
normalization happens after PipeWire supplies the window image; it reduces
capture traffic only when the selected game window itself is smaller than the
monitor. A fullscreen 1920×1080 window still arrives as 1920×1080. The source
must be at least 960×540; upscaling does not restore detail, so verify actual
skill checks before relying on this mode. The stop summary reports both the
selected source dimensions and the 320×240 frame dimensions passed to the
detector.

On KDE Wayland, `--capture-source region` is a separate experimental mode that
asks KWin to crop a 180×250 strip around the two observed check positions before
it enters PipeWire:

```bash
./start-solver.sh --variant quiet --capture-source region
```

This avoids sending the full monitor image to the solver and cuts another 41%
from the 320×240 solver input. The detector offsets its two known search points
into the cropped strip. It builds a small
native helper on first use and registers its hidden KDE desktop entry. It needs
Qt 6, KPipeWire, PipeWire, Wayland client headers, and the Plasma screencast
protocol development file. It uses KWin's private unstable region protocol, so
it is limited to KDE Wayland. In multi-monitor layouts it captures the display
at workspace position `(0, 0)` when available. The requested ROI is scaled from
the 1920×1080 reference layout. Use `--capture-source window` if that does not
match the display running the game. Keep the solver output from a real run: a
faster capture cadence does not by itself prove that a skill check was hit.
`capture_source_size` is the full display mode; `capture_frame_size` is the
image sent to the detector (320×240 for monitor/window capture, 180×250 for
region capture).

Frame recording is off by default to avoid the frame copies and FFmpeg encoder
during play. Decision events and performance statistics are still recorded.
Use `--video` when you need a video for replay or inspection, for example:

```bash
./start-solver.sh --variant quiet
./start-solver.sh --variant baseline --video
```

`--no-video` is accepted as an explicit way to disable frame recording. Without
`--video`, sessions cannot be replayed or inspected frame by frame.

## Browser test bench

Run `./start-practice.sh` (or `./start-practice.sh --solver-test`). Open the
browser at a 1920×1080 viewport; F11 may be needed. The synthetic red pointer
and white/black zones appear at the exact
screen coordinates the solver captures. Hold LMB to activate the solver; press
F to start or stop a Frenzy sequence. The bench is for testing screen detection
and input timing, not proof of an in-game hit. Its Space response delay defaults
to 35 ms to match the solver's default `--lead-ms 35`; adjust the slider to the
same value if you override `--lead-ms`.

## Replay and diagnostics

```bash
python run.py replay --recording recordings/SESSION
python run.py summary --recording recordings/SESSION
python tools/audit_solver_coverage.py recordings/SESSION
node --test simulator/*.test.cjs
.venv/bin/python -m pytest -q
```

Replay reprocesses recorded video and timestamps, but virtual presses change
what would happen next, so later video is counterfactual. A CV landing label
only describes where the visible needle appeared to stop; it is not a
confirmed game result. The coverage audit separates predicted attempts from
`BLIND_NO_NEEDLE` and `BLIND_NO_MOTION` attempts. Performance output includes
frame age, delivery gaps, processing time, CPU shares for the main process and
PipeWire capture worker, capture skips, and recording drops. It excludes the
optional video encoder. Percentiles use the last 600 frames; `lifetime_max`
keeps the largest spike from the full run. A real-game run with visible results
is required to establish GREAT rate and game lag.
