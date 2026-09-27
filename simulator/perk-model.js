(function (root, factory) {
  const model = factory();
  if (typeof module === 'object' && module.exports) module.exports = model;
  root.VDPerkModel = model;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';

  const tiers = Object.freeze({
    1: Object.freeze({ boost: 3, extra: 1, step: 0.05, window: 8 }),
    2: Object.freeze({ boost: 4, extra: 2, step: 0.075, window: 10 }),
    3: Object.freeze({ boost: 5, extra: 3, step: 0.1, window: 12 })
  });
  function createState() {
    return { greatStreak: 0, hitsSinceMiss: 0, effectEarned: false, bonusEarned: false };
  }

  function reset(state) {
    state.greatStreak = 0;
    state.hitsSinceMiss = 0;
    state.effectEarned = false;
    state.bonusEarned = false;
    return state;
  }

  function resolve(state, outcome, streakEligible, durationEligible, tier) {
    if (!tiers[tier]) throw new RangeError('Unknown Flawless Execution tier');
    if (!['GREAT', 'GOOD', 'MISS'].includes(outcome))
      throw new RangeError('Outcome must be GREAT, GOOD, or MISS');
    // The repair boost needs consecutive GREATs while alone. The duration
    // penalty applies to every successful repair check while the perk is on.
    if (!streakEligible && !durationEligible) return state;
    if (outcome === 'MISS') return reset(state);

    if (durationEligible) state.hitsSinceMiss += 1;
    if (streakEligible) {
      if (outcome === 'GREAT') {
        state.effectEarned = true;
        state.greatStreak += 1;
        if (state.greatStreak >= 3) state.bonusEarned = true;
      } else {
        state.greatStreak = 0;
      }
    }
    return state;
  }

  function currentBonus(state, eligible, tier) {
    const t = tiers[tier];
    if (!t) throw new RangeError('Unknown Flawless Execution tier');
    return eligible && state.effectEarned ? t.boost + (state.bonusEarned ? t.extra : 0) : 0;
  }

  function nextDuration(state, eligible, tier, baseDuration) {
    const t = tiers[tier];
    if (!t) throw new RangeError('Unknown Flawless Execution tier');
    if (!(baseDuration > 0)) throw new RangeError('Base duration must be positive');
    const reduction = eligible ? state.hitsSinceMiss * t.step : 0;
    return Math.max(0, baseDuration - reduction);
  }

  return Object.freeze({ tiers, createState, reset, resolve, currentBonus, nextDuration });
});
