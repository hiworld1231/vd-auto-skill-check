# Solver rewrite verification — 2026-09-27

## Baseline from existing physical recordings

`tools/audit_solver_coverage.py` counted 59 solver-detected check generations and 39 physical KEYDOWN events across the six recordings that contain KEYDOWN. No generation had duplicate KEYDOWN. Twenty of those generations had no KEYDOWN. Of the 39 KEYDOWN plans, 22 targeted GOOD_FALLBACK and 17 targeted GREAT. These are solver annotations, not a complete inventory of game checks or confirmed game results. One saved MP4 cannot be replayed because its `moov` atom is missing.

## Revised logic on saved frames

Five decodable recordings yielded 29 virtual replay claims with the revised logic, compared with 19 physical KEYDOWNs in those same historical runs. Per-recording virtual counts were 3, 9, 13, 2, and 2. Replay is counterfactual after a virtual claim and uses idealized timer wakes, so this is evidence that the old scheduler refused attempts, not a forecast of the live GREAT rate.

## Browser bench

Chromium rendered the new 1920×1080 bench. The production `Detector` on the `(800,420,320,240)` crop found the prompt at `(160,162.5)`, a 10.33° white GREAT arc, a 42.32° black GOOD arc, and one red needle. Browser simulation tests confirm Frenzy keeps phase through target transitions and a miss ends Frenzy.

## Synthetic performance

Ten seconds of 60 FPS synthetic capture with always-on video recording processed 600 frames, skipped one capture sequence, and dropped zero recording frames. Main-process CPU share was 25.2%; it excludes the capture worker and video encoder. Frame processing p95 was 5.998 ms, frame delivery gap p95 18.708 ms, and capture-to-receipt age p95 3.14 ms. This measures local synthetic transport, not game frame rate or game responsiveness.

## Remaining live verification

No game-visible outcome trace was collected after the rewrite. A real run with LMB held must verify one Space event per actual normal/Frenzy check, white GREAT outcome rate, and whether the game or cursor lags while capture and recording run. Fixed ROI and monitor resolution must match the live display.
