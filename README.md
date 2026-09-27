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
the game process. While LMB is released, it skips CV and records one idle
frame per second; full-rate solving and recording resume when LMB is held.

## Browser test bench

Run `./start-practice.sh`. Open the browser at a 1920×1080 viewport; F11 may
be needed. The synthetic red pointer and white/black zones appear at the exact
screen coordinates the solver captures. Hold LMB to activate the solver; press
F to start or stop a Frenzy sequence. The bench is for testing screen detection
and input timing, not proof of an in-game hit.

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
confirmed game result. Performance output includes frame age, delivery gaps,
processing time, main-process CPU share, capture skips, and recording drops.
The CPU share excludes the capture and video-encoder processes. A real-game run with
visible results is required to establish GREAT rate and game lag.
