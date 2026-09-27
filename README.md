# VD screen solver

`./start-solver.sh` captures the selected 1920×1080 monitor, watches a fixed
320×240 skill-check area at `(800, 420)`, and sends Space while LMB is held.
It aims at the white GREAT arc on every detected check, including successive
checks on a continuous Frenzy ring. It does not target GOOD. A missed timing
prediction triggers an immediate late attempt while the check remains visible;
missing ring, target, or needle pixels trigger reacquisition. No screen-only
solver can guarantee a hit when the target crosses between captured frames.

## Start

```bash
./setup.sh
python run.py preflight
./start-solver.sh
```

Select the game monitor in KDE's screen-sharing dialog. Ctrl+C stops the
solver. Video and timing logs are always saved under `recordings/` unless an
explicit `--recording DIR` is supplied. The recorder runs on a bounded queue;
frame drops and capture gaps are counted. The solver neither reads nor writes
the game process. Video is full-rate during a detected check and sampled once
per second otherwise. The solver still analyzes every captured frame while LMB
is held; while released, it skips CV. Capture and video encoding run at lower
CPU priority than the game.

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
to 60 ms to match the solver's default `--lead-ms 60`; adjust the slider to the
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
frame age, delivery gaps, processing time, main-process CPU share, capture
skips, and recording drops.
The CPU share excludes the capture and video-encoder processes. A real-game run with
visible results is required to establish GREAT rate and game lag.
