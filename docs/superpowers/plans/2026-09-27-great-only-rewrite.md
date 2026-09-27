# GREAT-only Solver Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every screen-visible check receive one attempt aimed at GREAT, including continuous Frenzy, while keeping the live path responsive.

**Architecture:** Preserve the PipeWire capture and evdev output boundaries. Simplify the current detector → tracker → planner → dispatch path, removing GOOD scheduling and refusal on timing uncertainty. Keep bounded asynchronous recording, replay, and a browser bench that exercises the same screen geometry.

**Tech Stack:** Python 3, OpenCV, NumPy, GStreamer/PipeWire, evdev, pytest, browser JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-27-great-only-rewrite-design.md`

## Global Constraints

- The solver runs only while LMB is held and presses at most once per detected check.
- GOOD is never a target. Timing uncertainty alone never suppresses a visible check.
- Capture and input remain screen-only; no game-process integration.
- Recording is always enabled but never blocks dispatch; dropped frames are counted.
- A real-game GREAT rate and lag claim requires game-visible evidence, not replay labels.

## Review Focus

- Same ring through Frenzy target transition: a changed white arc gets a new generation and press.
- Consecutive targets with almost identical geometry: a new check cannot be silently merged; use disappearance, motion/reset, or outcome evidence where geometry is insufficient.
- Brief prompt or target occlusion: avoid both blind presses and permanent suppression.
- Fast needle crossing between frames: attempt timing must use source timestamps, and late attempts must be explicit.
- Saturated recorder queue: CV and Space continue, while drops are counted and replay remains explicit about gaps.

---

### Task 1: Establish check-coverage evidence

**Files:** Modify `tests/test_engine.py`, `tests/test_vision_solver_bench.py`; create `tools/audit_solver_coverage.py`.

**Interfaces:** Read recording `frames.jsonl` and `events.jsonl`; output counts of visible target changes, BEGIN events, KEYDOWN events, and observed landing labels per recording. No game-ground-truth claim.

- [ ] Add a test with a continuous ring whose GREAT arc changes after a press, and one with a short occlusion; assert one generation and one claim per visible check.
- [ ] Run targeted tests and confirm failure on the current engine.
- [ ] Implement the minimal audit script and validate it against the six existing physical recordings with KEYDOWN events.
- [ ] Record baseline coverage and reasons for missing claims in a short report under `reports/`.

### Task 2: GREAT-only lifecycle and scheduling

**Files:** Modify `vd/engine.py`, `vd/planning.py`, `vd/motion.py`, `vd/vision.py`, `vd/dispatch.py`; tests in `tests/test_engine.py`, `tests/test_planning.py`, `tests/test_vision_solver_bench.py`.

**Interfaces:** Keep `Engine.observe(m, *, now, held=True)`, `Engine.poll(now, *, held=True, capture_alive=True)`, `Planner.update(motion, *, frame_at, now)`, and `dispatch(...)`. Plans retain `target_grade='GREAT'`; diagnostics may mark late attempts.

- [ ] Add failing tests for uncertainty without GOOD fallback, fast crossing, fresh-frame late attempt, repeated geometry after check restart, continuous Frenzy, and no press when target pixels vanish.
- [ ] Run targeted tests and confirm each fails for the intended reason.
- [ ] Remove GOOD fallback and uncertainty refusal; schedule the best estimated GREAT-center time or an immediate late attempt for the active visible check.
- [ ] Make check transition detection work without whole-ring disappearance, preserving the one-press invariant and avoiding spurious generations during needle occlusion.
- [ ] Run targeted tests, then the full Python suite.

### Task 3: Input-delay adaptation and live performance

**Files:** Modify `vd/calibration.py`, `vd/runtime.py`, `native/pipewire_worker.py`, `vd/capture.py`, `vd/recording.py`, `run.py`; tests in `tests/test_calibration.py`, `tests/test_capture_roi_pipeline.py`, `tests/test_physical_runtime_mock.py`, `tests/test_recording_interrupt.py`.

**Interfaces:** Preserve `run_session(...)` and current CLI entry points; `run` always records. Performance summary reports CPU time plus frame processing, frame age/gap, capture skips, recorder drops, and dispatch lag.

- [ ] Add tests for fresh consistent delay estimates, capped update frequency, saturated recorder queue, and clean interrupt.
- [ ] Measure synthetic 10-second baseline and profile the live loop without touching game input.
- [ ] Remove avoidable per-frame work and ensure encoding/disk activity never runs under dispatch locks; keep one-frame capture mailbox.
- [ ] Run synthetic 10-second session with recording enabled, inspect CPU, frame gaps, drops, and interrupt behavior; compare to baseline without claiming game performance.

### Task 4: Solver browser bench and replay

**Files:** Modify `simulator/skillcheck.html`, `start-practice.sh`, `vd/runtime.py`, `README.md`; tests in `simulator/ui-smoke.test.cjs`, `tests/test_vision_solver_bench.py`.

**Interfaces:** `./start-practice.sh` opens only a solver-oriented bench; replay retains `python run.py replay --recording DIR`.

- [ ] Add failing bench tests for ordinary checks, rapid continuous Frenzy, matching ROI, one response per check, and visible GREAT/GOOD/MISS result.
- [ ] Remove unrelated training/perk controls and data loads while retaining repeatable target geometry and speed controls needed to stress the solver.
- [ ] Run browser tests, replay existing recordings, and compare detected check coverage with Task 1 baseline.
- [ ] Update README with startup, bench, recording, replay, and the exact verification limits.

### Task 5: Final verification

**Files:** `reports/`, `README.md`, relevant tests.

- [ ] Run `node --test simulator/*.test.cjs` and `.venv/bin/python -m pytest -q`.
- [ ] Run `python run.py preflight`, synthetic capture/solver checks, and a recorded bench session where UI access permits.
- [ ] Inspect actual recording and event evidence for visible check coverage, GREAT targeting, dispatch lag, and dropped frames.
- [ ] Report live-game hit rate and lag only if a real run with game-visible outcomes is available; otherwise state the exact unverified boundary.
