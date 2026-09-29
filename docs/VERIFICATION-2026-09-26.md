# Verification 2026-09-26

## Live dry-run failure reproduced

Recording `20260926_154417_693878` contained 3976 frames. The old planner repeatedly reached `TARGET_UNCERTAIN` because the measured GREAT arc was around 10.7 degrees while the modeled trajectory/lead uncertainty could exceed the available GREAT half-width.

Representative observed geometry:

- GREAT start: `120.87349400366959`
- GREAT width: `10.728950934823914`
- paired GOOD start: `132.41010851560065`
- paired GOOD width: `42.14130537921591`
- motion speed: `251.15077255039893 deg/s`
- residual: `1.7121848764034837 deg`
- speed scatter: `44.415726752027744 deg/s`
- lead: `60 ms`
- lead uncertainty: `15 ms`

The prior policy rejected the entire generation whenever the uncertainty envelope did not fit inside GREAT.

## Planner change

The planner now evaluates targets in this order:

1. GREAT center with the original uncertainty calculation.
2. If GREAT is too narrow, the combined paired GREAT+GOOD success interval.
3. If the uncertainty does not fit there either, retain `TARGET_UNCERTAIN` and do not claim.

A fallback plan is explicitly labeled `target_grade=GOOD_FALLBACK`. GREAT remains labeled `GREAT`. A GOOD arc that is not geometrically paired with the current GREAT cannot expand permission.

## Replay result

Replay of the same recording after the change produced:

- frames: `3976`
- virtual claims: `3`
- all 3 claims: `GOOD_FALLBACK`
- `TARGET_UNCERTAIN` frames: `48 -> 7`

The three fallback plans had a combined success window of about 53.7 degrees and modeled uncertainty values of approximately 8.30, 7.48 and 5.91 degrees.

Replay uses idealized timer wakes and does not send physical input. It therefore verifies planner/CV behavior, not an actual Roblox hit.

## Regression checks

The uploaded local snapshot passed 55 rewrite tests in the available environment. One separately collected mouse-monitor test could not be collected there because `evdev` was not installed in that execution environment. Python compile checks passed.
