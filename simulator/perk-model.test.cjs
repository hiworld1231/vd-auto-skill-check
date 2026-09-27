const assert = require('node:assert/strict');
const { test } = require('node:test');
const model = require('./perk-model.js');

test('perk tiers match the listed boosts, added boosts, speed steps, and return windows', () => {
  assert.deepEqual(Object.values(model.tiers).map(({boost, extra, step, window}) =>
    [boost, extra, step, window]), [
    [3, 1, 0.05, 8], [4, 2, 0.075, 10], [5, 3, 0.1, 12]
  ]);
});

test('three consecutive GREATs earn the tier III bonus and shorten each next check', () => {
  const state = model.createState();
  const base = 1.28;
  for (let hit = 1; hit <= 3; hit += 1) {
    model.resolve(state, 'GREAT', true, true, 3);
    assert.equal(model.currentBonus(state, true, 3), hit < 3 ? 5 : 8);
    assert.ok(Math.abs(model.nextDuration(state, true, 3, base) - (base - hit * 0.1)) < 1e-10);
  }
});

test('GOOD breaks consecutive GREATs but the earned effect lasts until a MISS', () => {
  const state = model.createState();
  for (let hit = 0; hit < 3; hit += 1) model.resolve(state, 'GREAT', true, true, 1);
  model.resolve(state, 'GOOD', true, true, 1);
  assert.equal(state.greatStreak, 0);
  assert.equal(model.currentBonus(state, true, 1), 4);
  assert.equal(model.nextDuration(state, true, 1, 1.28), 1.08);
  model.resolve(state, 'MISS', true, true, 1);
  assert.equal(model.currentBonus(state, true, 1), 0);
  assert.equal(model.nextDuration(state, true, 1, 1.28), 1.28);
});

test('GOOD before the first GREAT reduces the next check without earning repair boost', () => {
  const state = model.createState();
  model.resolve(state, 'GOOD', true, true, 2);
  assert.equal(state.greatStreak, 0);
  assert.equal(state.effectEarned, false);
  assert.equal(model.currentBonus(state, true, 2), 0);
  assert.ok(Math.abs(model.nextDuration(state, true, 2, 1.28) - 1.205) < 1e-10);
  model.resolve(state, 'MISS', true, true, 2);
  assert.equal(model.nextDuration(state, true, 2, 1.28), 1.28);
});

test('GOOD after the perk is earned keeps reducing each next duration until a miss', () => {
  const state = model.createState();
  model.resolve(state, 'GREAT', true, true, 2);
  model.resolve(state, 'GOOD', true, true, 2);
  assert.equal(state.greatStreak, 0);
  assert.equal(state.hitsSinceMiss, 2);
  assert.equal(model.currentBonus(state, true, 2), 4);
  assert.ok(Math.abs(model.nextDuration(state, true, 2, 1.28) - 1.13) < 1e-10);
});

test('not being alone blocks GREAT streak progress but successful checks still shorten duration', () => {
  const state = model.createState();
  model.resolve(state, 'GREAT', false, true, 2);
  assert.equal(state.greatStreak, 0);
  assert.equal(model.currentBonus(state, false, 2), 0);
  assert.equal(state.hitsSinceMiss, 1);
  assert.ok(Math.abs(model.nextDuration(state, true, 2, 1.28) - 1.205) < 1e-10);
  model.resolve(state, 'MISS', false, true, 2);
  assert.equal(state.hitsSinceMiss, 0);
  assert.equal(model.currentBonus(state, true, 2), 0);
  model.resolve(state, 'GREAT', false, false, 2);
  assert.equal(state.hitsSinceMiss, 0);
});

test('leaving repair pauses the effect; off-repair outcomes do not erase it before the return window', () => {
  const state = model.createState();
  for (let hit = 0; hit < 3; hit += 1) model.resolve(state, 'GREAT', true, true, 3);
  model.resolve(state, 'MISS', false, false, 3);
  assert.equal(state.greatStreak, 3);
  assert.equal(model.currentBonus(state, false, 3), 0);
  model.resolve(state, 'GREAT', true, true, 3);
  assert.equal(state.greatStreak, 4);
  assert.equal(model.currentBonus(state, true, 3), 8);
  model.resolve(state, 'MISS', true, true, 3);
  assert.equal(state.greatStreak, 0);
  assert.equal(model.currentBonus(state, true, 3), 0);
});

test('duration never becomes negative after an exceptionally long streak', () => {
  const state = model.createState();
  for (let hit = 0; hit < 20; hit += 1) model.resolve(state, 'GREAT', true, true, 3);
  assert.equal(model.nextDuration(state, true, 3, 0.87), 0);
});
