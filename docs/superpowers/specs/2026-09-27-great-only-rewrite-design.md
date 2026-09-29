# GREAT-only solver rewrite

## Intent and success criteria

The solver captures the selected 1920×1080 monitor's fixed skill-check ROI, runs only while LMB is held, and sends one Space press for each detected check. It aims only at the white GREAT arc, including during continuous Frenzy. Poor timing confidence does not suppress a press; absent target or needle pixels cause continued acquisition. The user wants this to stop overlooking new checks and to avoid noticeable game, cursor, or solver lag. Absolute hit guarantees are not possible from screen frames alone, so verification must report observed coverage, timing, and actual game outcomes separately.

## Scope

Keep screen capture, input gating, replay, always-on video and event recording, adaptive delay estimation, a short preflight, and performance statistics. Reduce the browser practice UI to a solver test bench with ordinary and fast Frenzy checks. Preserve the current screen-only boundary; no game-process reads or writes.

## Live pipeline

1. A PipeWire worker captures the selected monitor and crops the fixed ROI. A one-frame mailbox replaces old frames. Every frame carries a monotonic source timestamp.
2. CV detects the prompt, white arc, and red needle independently. It retains a recent confirmed arc only through brief needle occlusion. Measurements explicitly expose ambiguity and gaps.
3. A single check tracker owns a generation and one-press flag. A changed target starts a new generation immediately, including while the ring remains visible in Frenzy. Short visual dropouts retain the current generation without mistaking it for a new one.
4. Motion estimation fits speed and phase from recent real movements. It reacquires after discontinuities. The scheduler predicts when the needle reaches the GREAT center and accounts for observed input response delay. Uncertainty informs timing and telemetry, but is never a reason by itself to refuse a visible check. The planner never targets GOOD or waits for another full revolution after the current target passes.
5. The dispatch gate verifies LMB, capture health, current generation, current frame, and one-press state immediately before Space. If a timing estimate is late, it presses immediately while the check remains visible and records that timing as late; it does not invent a future full revolution. If visual evidence disappears before dispatch, it reacquires instead of pressing blindly. This cannot guarantee GREAT after the white arc has passed, but it avoids silently abandoning a visible check.
6. Post-press video can estimate a frozen needle position and input delay. Only consistent fresh observations adjust delay. Video-derived GREAT/GOOD/MISS labels are diagnostic, never represented as game-confirmed outcomes.

## Performance and recording

CV and dispatch run without video encoding or disk I/O under their locks. Recording receives copied frames through a bounded queue and must never block Space dispatch. When its queue fills, it drops video frames with an explicit count while preserving an event log when possible. Record CPU, frame delivery gap, frame age, CV time, skipped capture frames, recording drops, and dispatch lag. Investigate the cost of full-frame template matching and unnecessary work when no check is visible. The startup path should not spawn other UI components or active loops beyond capture, mouse monitoring, output, and the bounded recorder.

## Browser bench and replay

The browser bench renders the same ROI geometry and red needle that CV expects, at 1920×1080, with both ordinary checks and continuous fast Frenzy. It exposes check ID, target, and outcome for test evidence. Replay reprocesses recorded frames with their original timestamps and documents that its virtual presses are counterfactual after the first press. It measures how many visible check generations receive one claim, GREAT-target timing error, late dispatches, and missed transitions. Bench tests include close consecutive targets, unchanged ring across Frenzy transitions, frame gaps, brief occlusion, high speed, and repeated target geometry.

## Verification limits

Unit and browser tests prove algorithms and rendering only. Synthetic capture tests measure local load without the game. Saved recordings expose frame timing and screen-visible behavior, but their annotated outcomes are not authoritative game results. Final claims about live GREAT rate and absence of game lag require a user-observed real run or trustworthy game-visible result capture.
