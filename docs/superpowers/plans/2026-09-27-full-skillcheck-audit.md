# Full Skill Check Audit and Repair Plan

> **For agentic workers:** This plan is being executed inline in the current task because the user explicitly requested autonomous end-to-end work and testing. Track each step with checkboxes; do not stop at a passing unit suite.

**Goal:** Reconstruct the ordinary and continuous Frenzy skill-check behavior from the full replay corpus, correct the local practice and screen-based solver, and measure latency and performance against the actual available evidence.

**Architecture:** Keep game behavior inference, the standalone practice simulator, and the screen-capture solver as separate components. Build route data only from ordered, attributable replay evidence; keep detector telemetry, program annotations, and game-confirmed observations explicitly distinct. Verify from corpus audits upward to UI behavior and runtime measurements.

**Tech Stack:** Python 3, OpenCV, pytest, Node.js, browser HTML/SVG/JavaScript, PipeWire/GStreamer capture.

**Spec:** The active user request in this Codex task: inspect all replays and the full local program, faithfully model ordinary and Frenzy checks, and test timing, responsiveness, and performance.

## Global Constraints

- Frenzy is one continuing check with the needle moving through 360 degrees and beyond; a miss ends it immediately.
- Never generate random target angles. Finite playback keeps recorded route boundaries; an infinite practice loop may cycle complete recorded routes, but its unobserved joins must be deterministic, constrained by replay geometry, and called out as synthetic.
- Distinguish game-visible evidence from legacy bot annotations and telemetry labels.
- Keep the solver screen-based and independent of direct game integration.
- Treat tests as evidence for only the behavior they actually cover; require runtime evidence for claims about live performance or game outcomes.
- Do not use or retain the sudo password supplied in chat.

## Review Focus

- Inferred legacy chain labels can be false positives; compare their geometry and timing with recorded frames or videos wherever available.
- Archive copies may be duplicate or conflicting versions; deduplicate by content and preserve provenance.
- Recorded video overlays can contaminate detected zones, arrows, and outcomes; never call them raw game pixels without verifying the source.
- Narrow hit windows plus solver frame cadence and dispatch delay can turn a mathematically valid plan into a late physical input.
- Firefox and the live game may be running without being exposed to the current UI-control surface; report the exact boundary and continue offline validation.

---

### Task 1: Complete replay corpus and video evidence audit

**Files:** `tools/audit_replay_corpus.py`, `tools/audit_replay_archives.py`, `tools/audit_replay_videos.py`; reports under `/home/oae/skillcheck_replays_archive_reports/`.

- [ ] Finish reconciling raw path counts for all JSON, MP4, MKV, and ZIP sources; current unique JSON hashes/geometry match the prior 2,803 records, but the fresh scan has 93 fewer raw JSON paths than the older inventory.
- [x] Decode every unique replay MP4 and every available screen-session MKV; retain per-file failures and source provenance.
- [x] Separate annotated legacy clips, user screen captures, and current solver captures in the report.
- [x] Report the exact number of unique decodable assets, labels, transitions, and unverified/ambiguous records.
- [x] Verify that each reused corpus report corresponds to current on-disk sources by content hash; preserve the unresolved 93-path JSON copy discrepancy separately.

**Acceptance evidence:** full-corpus totals reconcile to an independent file/hash inventory; each unique video is decoded or has an explicit error row; no report claims game-ground-truth status from program overlays.

### Task 2: Reconstruct ordinary and Frenzy sequences

**Files:** `tools/audit_replay_sequences.py`, `tools/audit_replay_manifest_order.py`, `tools/build_default_sequences.py`, `tools/build_frenzy_sequences.py`, `simulator/default-zone-sequences.js`, `simulator/frenzy-sequences.js`, corresponding tests.

- [x] Re-evaluate ordering using IDs, timestamps, prior duration/end time, session boundaries, outcomes, and frame/video evidence; quantify how many pairs each rule includes or rejects.
- [x] Test the suspicious close legacy chain records against their source media and identify whether `chain_count` was recorded by a bot heuristic or by game-visible evidence.
- [x] Compare ordinary and Frenzy angle distributions, repeated motifs, separation, zone geometry, and transition timing using all attributable records; preserve low-confidence data separately. Added a strict normal cohort (572 measured adjacent pairs, including seven under 1°) and leave-one-session-out Frenzy angle prediction; later positions are noisy.
- [x] Build exact contiguous routes with captured coordinates; infinite practice cycles complete routes with a deterministic, explicitly unobserved join order and generates no angular positions.
- [x] Add regressions for close ordinary zones, 360-degree continuous rotation, chain continuation after hits, immediate stop on miss, and route/session boundaries.

**Acceptance evidence:** reproducible reports identify every observed transition with provenance; sequence output reproduces observed positions, and the audit lists the synthetic infinite-loop joins and their angular separation.

### Task 3: Verify and repair the practice simulator

**Files:** `simulator/skillcheck.html`, `simulator/perk-model.js`, generated replay data, `simulator/*.test.cjs`.

- [x] Verify one centered check, compact controls, normal-check frequency control, and visible GREAT/GOOD/MISS result labels.
- [x] Verify F and menu keybinds start/cancel Frenzy, the infinite checkbox continues until a miss, and the needle/check does not reset between successful Frenzy segments.
- [x] Verify recorded routes and continuous continuation use the reviewed evidence model rather than independent random samples.
- [ ] Exercise Firefox at desktop and narrow viewports; process inspection confirms `simulator/skillcheck.html` is open there, but Codex does not expose Firefox through CUA. The local file is also blocked by the IAB URL policy. Earlier narrow-pane checks and DOM tests are not desktop Firefox evidence.

**Acceptance evidence:** browser interaction trace and screenshots for ordinary hit/miss, finite Frenzy, infinite Frenzy through at least one full revolution, and stop-on-miss.

### Task 4: Verify the screen-based solver and timing path

**Files:** `run.py`, `vd/`, `native/pipewire_worker.py`, `start-solver.sh`, solver tests and session recordings.

- [x] Trace capture timestamp → detector → motion fit → plan → key dispatch → post-input observation, including each rejection reason and measured delay.
- [ ] Replay coverage is 13/15 local solver recordings, 15,996 frames and 15 virtual claims; two lack a valid `frames.mp4`. Claims are counterfactual. Visual freeze and game labels have not been compared in Firefox.
- [x] Verify continuous LMB arming, Space press/release, immediate Ctrl+C cleanup, and no forced `--seconds` runtime.
- [ ] Separate calibration reliability failures from planner misses; correct the data-quality or estimator cause before enabling calibration changes.
- [x] Verify the solver never attaches to or writes into the game process; input remains explicit and user-controlled.

**Acceptance evidence:** replay coverage report, timing distributions and rejection breakdown, plus a verified live capture/input run when the active window is available.

### Task 5: Measure performance and end-to-end behavior

**Files:** runtime capture/recording code, `tests/test_vision_solver_bench.py`, new reproducible benchmark report if needed.

- [x] Measure capture FPS, dropped frames, processing p50/p95/max, frame age, input dispatch delay, and CPU use. One paired synthetic 10-second run covers recording on/off; a matched live monitor run is still missing.
- [x] Compare measurements to the available source frame cadence and check for multi-second stalls or sustained missed deadlines; the live sample had 10 skipped sequence numbers and a 259 ms maximum frame gap, while game outcomes remain unmeasured.
- [ ] Re-run the practice and solver workflows independently and together only through supported UI access; record each configuration and outcome.
- [ ] Keep the task open while Firefox interaction, calibration reliability, two damaged recordings, and simultaneous live practice/solver behavior remain unverified.

**Acceptance evidence:** repeatable benchmark outputs and run manifests; end-to-end outcome claims require visible game feedback, not internal solver events.
