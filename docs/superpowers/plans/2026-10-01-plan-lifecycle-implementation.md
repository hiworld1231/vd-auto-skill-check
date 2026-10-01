# Plan Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a valid GREAT plan across ordinary fresh observations so it can reach dispatch instead of being erased and rebuilt as GOOD.

**Architecture:** `Engine.observe()` stops treating every frame as an invalidation event. `Planner.update()` evaluates a fresh candidate against the current live plan, preserving compatible GREAT commitments until they are replaced by a better GREAT, become physically unreachable, or a real lifecycle event invalidates them.

**Tech Stack:** Python 3, pytest, existing `vd.engine`, `vd.planning`, motion/vision contracts.

**Spec:** `docs/superpowers/specs/2026-10-01-plan-lifecycle-design.md`

## Global Constraints

- One generation owns one immutable unwrapped `SuccessOccurrence`.
- Never rearm another revolution or add +360 retry behavior.
- Fresh observation alone is not an invalidation reason.
- LMB release, capture loss, ring end, confirmed geometry change, occurrence pass, and observation expiry remain authoritative invalidation events.
- GOOD remains a fallback when GREAT was never safely committed or is no longer physically reachable.

## Review Focus

- Slightly noisier fresh frame before `press_at` must not erase a valid GREAT plan.
- Fresh frame may update timing for the same GREAT occurrence without changing generation.
- Once GREAT is physically unreachable, GOOD may replace it if GOOD remains reachable.
- Geometry change must still invalidate the old plan and start the proper generation/chain path.
- Expired/current-plan identity checks must still prevent stale dispatch.

---

### Task 1: Engine-level RED regression for plan survival

**Files:**
- Modify: `tests/test_engine.py`

**Interfaces:**
- Consumes: `Engine.observe(measurement, now=..., held=True)`, `Engine.poll(now, held=True, capture_alive=True)`.
- Produces: regression coverage for a GREAT plan surviving an ordinary fresh observation.

- [ ] **Step 1: Write a failing engine test**

Add `test_fresh_observation_does_not_cancel_pending_great_before_dispatch()` that drives one generation to a GREAT `planner.current`, feeds a compatible but slightly noisier fresh observation before `press_at`, asserts the same GREAT commitment remains schedulable, then asserts `poll()` claims a GREAT plan.

- [ ] **Step 2: Run the targeted test and verify RED**

Run: `./.venv/bin/pytest -q tests/test_engine.py::test_fresh_observation_does_not_cancel_pending_great_before_dispatch`

Expected: FAIL because `Engine.observe()` currently invalidates on every fresh observation.

- [ ] **Step 3: Commit the RED test**

Commit message: `test: preserve pending great across fresh observations`

### Task 2: Stop frame-arrival invalidation and preserve compatible GREAT

**Files:**
- Modify: `vd/engine.py`
- Modify: `vd/planning.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_planning_success_tail.py`

**Interfaces:**
- Consumes: current `Planner.current`, immutable occurrence, fresh `Motion`.
- Produces: `Planner.update()` that can return/preserve a compatible GREAT without requiring a frame-arrival invalidation.

- [ ] **Step 1: Remove unconditional frame invalidation**

Delete the unconditional `self.planner.invalidate('NEW_OBSERVATION')` from `Engine.observe()`.

- [ ] **Step 2: Make `Planner.update()` evaluate before replacing**

Do not begin `Planner.update()` by clearing `current`. Compute fresh reachability and candidate grade first. Preserve an existing GREAT for the same generation/occurrence while GREAT remains reachable and the existing plan has not expired; allow a fresh GREAT to replace/update timing. Replace GREAT with GOOD only after GREAT becomes physically unreachable or a real severe fallback condition is met.

- [ ] **Step 3: Keep versioning meaningful**

Advance `version` only when a plan is intentionally replaced/invalidated, not merely because a frame arrived. Ensure `claim()` identity checks still reject stale plan objects.

- [ ] **Step 4: Run targeted planner/engine tests**

Run: `./.venv/bin/pytest -q tests/test_engine.py tests/test_planning_success_tail.py`

Expected: PASS.

- [ ] **Step 5: Commit production fix**

Commit message: `fix: preserve live plans across observations`

### Task 3: Lifecycle regressions and full verification

**Files:**
- Modify if needed: `tests/test_engine.py`

**Interfaces:**
- Consumes: final engine/planner lifecycle semantics.
- Produces: coverage for genuine invalidation paths and no-rearm behavior.

- [ ] **Step 1: Add/confirm regression assertions**

Cover: geometry change invalidates old plan; LMB release/capture loss prevent claim; observation expiry prevents claim; unreachable GREAT can fall back to GOOD without another revolution.

- [ ] **Step 2: Run focused suites**

Run: `./.venv/bin/pytest -q tests/test_engine.py tests/test_planning.py tests/test_planning_success_tail.py tests/test_vision_solver_bench.py tests/test_physical_runtime_mock.py`

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run: `./.venv/bin/pytest -q`

Expected: PASS.

- [ ] **Step 4: Real Vulkan acceptance**

Run: `bash ./start-vulkan-solver.sh --variant quiet --video`

Acceptance: a GREAT plan created before `press_at` survives compatible fresh observations to `PRESS_CLAIM`/`KEYDOWN`; ordinary frame arrival no longer causes GREAT→GOOD churn. Synthetic green tests alone do not prove live success.
