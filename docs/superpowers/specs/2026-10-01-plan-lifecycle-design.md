# Plan Lifecycle Redesign

## Context

The live solver can produce a valid GREAT plan and then lose it before dispatch because every fresh observation currently calls `planner.invalidate('NEW_OBSERVATION')`. That clears `planner.current` unconditionally, so a subsequent frame must rebuild the plan from scratch. Small frame-to-frame changes in motion uncertainty can therefore replace a valid GREAT plan with GOOD before its scheduled `press_at` arrives.

This behavior is visible in live recordings as GREAT/GOOD churn within a single generation, while unit tests that call `Planner.update()` directly stay green because they do not model the full `Engine.observe()` lifecycle.

## Goal

Make a valid plan survive normal fresh observations until it is dispatched, superseded by a strictly better compatible plan, or invalidated by a real lifecycle event.

The solver should prefer GREAT when it has a safe committed GREAT plan, without letting one noisy frame erase that plan. GOOD remains the fallback when GREAT was never safely committed or becomes physically unreachable.

## Non-goals

- Do not weaken capture-liveness checks.
- Do not change Vulkan capture behavior.
- Do not change motion fitting or calibration in this change.
- Do not reintroduce +360 rearming or future-revolution retries.
- Do not make GREAT mandatory when only GOOD is safely reachable.

## Invariants

1. One generation owns one immutable unwrapped `SuccessOccurrence`.
2. A generation never schedules another revolution.
3. A claimed plan is consumed once.
4. A fresh observation alone is not an invalidation reason.
5. Real lifecycle changes still invalidate immediately: LMB release, capture loss, ring end, confirmed geometry change/new generation, observation expiry, or an occurrence becoming physically passed.
6. A pending GREAT plan may survive noisier frames as long as the same occurrence remains valid and the plan's window is still reachable.
7. GOOD remains available as a fallback, but a noisy observation must not overwrite a still-valid committed GREAT plan merely because the newest uncertainty estimate crossed the GREAT threshold by a small amount.

## Design

### 1. Stop invalidating on every observation

Remove the unconditional `self.planner.invalidate('NEW_OBSERVATION')` at the start of `Engine.observe()`.

`Engine.observe()` will instead let `Planner.update()` evaluate the fresh motion estimate against the existing plan.

### 2. Treat `Planner.current` as a live scheduled commitment

`Planner.update()` should not begin by invalidating the current plan. It should compute a candidate from the fresh observation, then decide whether the candidate should replace the current plan.

Replacement policy:

- New GREAT may replace GOOD when GREAT is safe and reachable.
- New GREAT may replace existing GREAT when it is the same occurrence and provides an updated timing estimate.
- New GOOD must not replace an existing still-reachable GREAT solely because the newest uncertainty estimate became slightly worse.
- If GREAT is no longer physically reachable, GOOD may replace it if GOOD remains reachable.
- If the whole occurrence is passed, clear the current plan and return `TARGET_PASSED`.

### 3. Preserve deadline safety

A surviving plan is still subject to `valid_until` and `latest_press_at`.

`Engine.poll()` continues to invalidate expired observations before dispatch. `Planner.claim()` continues to require generation/version/current identity, held input, live capture, `now >= press_at`, and `now <= valid_until`.

### 4. Versioning semantics

`version` should advance only when plan identity is intentionally invalidated or replaced, not just because a frame arrived.

This keeps `claim()` useful as a race-safety guard without turning every fresh observation into cancellation.

### 5. Geometry changes remain authoritative

The existing confirmed geometry-change path still starts a new generation and invalidates the old plan. Chained/frenzy occurrence anchoring remains unchanged by this redesign.

## Tests

Add engine-level regression tests, because planner-only tests cannot reproduce this bug.

Required RED cases before production changes:

1. `Engine.observe()` creates a GREAT plan, then a slightly noisier fresh observation arrives before `press_at`; the original GREAT commitment must remain schedulable and `poll()` must still claim it.
2. A fresh observation that genuinely makes GREAT unreachable but leaves GOOD reachable may replace the GREAT plan with GOOD.
3. A confirmed geometry change invalidates the old plan and starts a new generation.
4. LMB release/capture loss/observation expiry still prevent dispatch.
5. Existing no-rearm, frenzy, runtime timer-dispatch, and physical mock tests remain green.

## Verification

After the targeted RED/GREEN cycle:

- Run engine/planner targeted tests.
- Run frenzy bench tests.
- Run physical runtime mock tests.
- Run the full pytest suite.
- Then run a real Vulkan session with `--video` and inspect whether GREAT plans survive fresh observations to `PRESS_CLAIM`/`KEYDOWN` instead of being replaced by GOOD churn.

Synthetic green tests are not sufficient to claim the live issue is solved; the real Vulkan recording is the acceptance check.
