const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const perkModel = require('./perk-model.js');

const html = fs.readFileSync(__dirname + '/skillcheck.html', 'utf8');
const inlineScripts = [...html.matchAll(/<script>\s*([\s\S]*?)<\/script>/g)];
const inlineScript = inlineScripts[0][1];
const sceneListScript = fs.readFileSync(__dirname + '/scene-backgrounds.js', 'utf8');
const scenePaths = JSON.parse(sceneListScript.match(/=\s*(\[[\s\S]*\]);/)[1]);
const zoneSampleScript = fs.readFileSync(__dirname + '/zone-samples.js', 'utf8');
const zoneSamples = JSON.parse(zoneSampleScript.match(/=\s*(\[[\s\S]*\]);/)[1]);
const defaultZoneSequencePath = path.join(__dirname, 'default-zone-sequences.js');
const defaultZoneSequenceScript = fs.existsSync(defaultZoneSequencePath)
  ? fs.readFileSync(defaultZoneSequencePath, 'utf8') : '';
const defaultZoneSequences = defaultZoneSequenceScript
  ? JSON.parse(defaultZoneSequenceScript.match(/VDDefaultZoneSequences = (\[[\s\S]*?\]);/)[1]) : [];
const defaultZoneSequenceMeta = defaultZoneSequenceScript
  ? JSON.parse(defaultZoneSequenceScript.match(/VDDefaultZoneSequenceMeta = (\{[^\n]*\})/)[1]) : {};
const anchorScript = fs.readFileSync(__dirname + '/check-anchors.js', 'utf8');
const checkAnchors = JSON.parse(anchorScript.match(/=\s*(\[[\s\S]*\]);/)[1]);
const frenzySequencePath = path.join(__dirname, 'frenzy-sequences.js');
const frenzySequenceScript = fs.existsSync(frenzySequencePath) ? fs.readFileSync(frenzySequencePath, 'utf8') : '';
const frenzySequences = frenzySequenceScript ? JSON.parse(frenzySequenceScript.match(/VDFrenzySequences = (\[[\s\S]*?\]);/)[1]) : [];
const firstFrenzyRoute = frenzySequences[0];
const frenzySpeeds = frenzySequenceScript.match(/VDFrenzySpeeds = (\[[^\n]*\])/)
  ? JSON.parse(frenzySequenceScript.match(/VDFrenzySpeeds = (\[[^\n]*\])/)[1]) : [];
const frenzyTransitions = frenzySequenceScript.match(/VDFrenzyTransitions = (\[[^\n]*\])/)
  ? JSON.parse(frenzySequenceScript.match(/VDFrenzyTransitions = (\[[^\n]*\])/)[1]) : [];
const frenzyTiming = frenzySequenceScript.match(/VDFrenzyTiming = (\{[^\n]*\})/)
  ? JSON.parse(frenzySequenceScript.match(/VDFrenzyTiming = (\{[^\n]*\})/)[1]) : {};
const intervalSamplePath = path.join(__dirname, 'check-interval-samples.js');
const intervalSampleScript = fs.existsSync(intervalSamplePath) ? fs.readFileSync(intervalSamplePath, 'utf8') : '';
const intervalSamples = intervalSampleScript ? JSON.parse(intervalSampleScript.match(/=\s*(\[[\s\S]*\]);/)[1]) : [];

test('Frenzy data contains exact replay traces and measured adjacent transitions', () => {
  assert.ok(fs.existsSync(frenzySequencePath), 'replay-derived Frenzy routes are bundled');
  const sequenceScript = fs.readFileSync(frenzySequencePath, 'utf8');
  const sequences = JSON.parse(sequenceScript.match(/VDFrenzySequences = (\[[\s\S]*?\]);/)[1]);
  assert.equal(sequences.length, 26);
  assert.ok(sequences.every((route) => route.checks.length >= 2));
  assert.equal(sequences.reduce((count, route) => count + route.checks.length, 0), 203);
  assert.equal(Math.max(...sequences.map(route => route.checks.length)), 15);
  assert.equal(sequences.reduce((count, route) => count + route.checks.length - 1, 0), 177);
  assert.equal(frenzyTransitions.length, 177);
  assert.equal(frenzyTiming.count, 177);
  assert.ok(frenzyTiming.median_ms >= 125 && frenzyTiming.median_ms <= 175);
  for (const [greatWidth, goodWidth, gap, delta, sourceChain, sourceStart] of frenzyTransitions) {
    assert.ok(greatWidth >= 9 && greatWidth <= 12);
    assert.ok(goodWidth >= 35 && goodWidth <= 45);
    assert.ok(gap >= 0 && gap <= 8);
    assert.ok(Number.isInteger(sourceChain) && sourceChain >= 1);
    assert.ok(sourceStart >= 0 && sourceStart < 360);
    const distance = Math.min(delta, 360 - delta);
    assert.ok(distance >= 107.8, `measured transition too close: ${distance.toFixed(2)}°`);
  }
  for (const sequence of sequences) {
    assert.ok(sequence.last_observed_outcome || sequence.truncated);
    for (let index = 1; index < sequence.checks.length; index += 1) {
      const distance = Math.abs((sequence.checks[index][3] - sequence.checks[index - 1][3] + 540) % 360 - 180);
      assert.ok(distance >= 107.8, `replay transition too close: ${distance.toFixed(2)}°`);
    }
  }
  const firstCapturedTransition = sequences.find(route =>
    Math.abs(route.checks[0][3] - 126.699) < .001);
  assert.ok(firstCapturedTransition);
  assert.ok(Math.abs(firstCapturedTransition.checks[1][3] - 291.768) < .001);
  assert.equal(firstCapturedTransition.last_observed_outcome, 'MISS');
  assert.doesNotMatch(html, /id="frenzyTier"/);
  assert.doesNotMatch(inlineScript, /frenzyRemaining/);
  assert.match(inlineScript, /VDFrenzySequences/);
  assert.match(inlineScript, /VDFrenzySpeeds/);
  assert.match(inlineScript, /VDFrenzyTransitions/);
  assert.ok(frenzySpeeds[0] >= 250 && frenzySpeeds[0] <= 350);
  assert.ok(frenzySpeeds[1] >= 290 && frenzySpeeds[1] <= 340);
  assert.ok(frenzySpeeds[2] >= 350 && frenzySpeeds[2] <= 410);
});

test('infinite Frenzy has measured relative transitions and no absolute route-order cycle', () => {
  assert.equal(frenzyTransitions.length, 177);
  assert.doesNotMatch(frenzySequenceScript, /VDFrenzyRouteOrder/);
  for (const step of frenzyTransitions) {
    const distance = Math.min(step[3], 360 - step[3]);
    assert.ok(distance >= 108.0, `measured transition too close: ${distance.toFixed(2)}°`);
  }
});

test('Frenzy plays a recorded path across 360° then stops without fabricating a route seam', () => {
  const stand = createStand();
  stand.frame(100);
  stand.click('frenzyStart');
  let totalRotation = 0;
  let previousGreatStart = null;
  const route = firstFrenzyRoute;
  for (let index = 0; index < route.checks.length; index += 1) {
    const {travel, speed, greatStart} = pressGreatForCurrentCheck(stand);
    totalRotation += travel;
    if (index + 1 < route.checks.length) waitForFrenzyCheck(stand);
    if (previousGreatStart !== null) {
      const separation = Math.abs((greatStart - previousGreatStart + 540) % 360 - 180);
      assert.ok(separation >= 107.8, `successive Frenzy checks too close: ${separation.toFixed(2)}°`);
    }
    previousGreatStart = greatStart;
    const chain = route.checks[index][4];
    assert.ok(Math.abs(speed - frenzySpeeds[Math.min(chain - 1, frenzySpeeds.length - 1)]) < 1,
      `rapid-chain speed ${index + 1}: ${speed.toFixed(1)}°/s`);
    assert.match(stand.elements.get('message').textContent, /GREAT/);
    assert.equal(stand.elements.get('checkSvg').style.opacity,
      index + 1 < route.checks.length ? '1' : '0');
  }
  assert.ok(totalRotation > 360, `expected >360° in one observed path, got ${totalRotation}`);
  assert.match(stand.elements.get('frenzyStatus').textContent, /trace complete/);
  assert.doesNotMatch(stand.elements.get('frenzyStatus').textContent, /Frenzy active/);
});

test('Frenzy uses captured absolute zone angles after ordinary checks', () => {
  const stand = createStand();
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  stand.click('frenzyStart');

  const expectedStart = firstFrenzyRoute.checks[0][3];
  assert.ok(Math.abs(arcStart(stand.elements.get('greatArc').attributes.d) - expectedStart) < .02);
});

test('infinite Frenzy continues by measured angle deltas and ends immediately on a miss', () => {
  const stand = createStand();
  stand.frame(900);
  stand.elements.set('infiniteFrenzy', {checked:true});
  stand.click('frenzyStart');
  const firstRoute = firstFrenzyRoute;
  let totalRotation = 0;
  for (let index = 0; index < firstRoute.checks.length; index += 1) {
    totalRotation += pressGreatForCurrentCheck(stand).travel;
    if (index + 1 < firstRoute.checks.length) waitForFrenzyCheck(stand);
  }
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  assert.match(stand.elements.get('frenzyStatus').textContent, /continuous checks/);
  waitForFrenzyCheck(stand);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  let priorStart = firstRoute.checks.at(-1)[3];
  assert.match(stand.elements.get('frenzyStatus').textContent, /measured transitions/);
  for (let index = 0; index < 5; index += 1) {
    const [greatWidth, goodWidth, gap, delta] = frenzyTransitions[index];
    const expectedStart = (priorStart + delta) % 360;
    const great = stand.elements.get('greatArc').attributes.d;
    const actualStart = arcStart(great);
    assert.ok(Math.abs(actualStart - expectedStart) < .02);
    assert.ok(Math.abs(arcWidth(great) - greatWidth) < .02);
    const expectedGoodStart = (expectedStart + greatWidth + gap) % 360;
    assert.ok(Math.abs(arcStart(stand.elements.get('goodArc').attributes.d) - expectedGoodStart) < .02);
    assert.ok(Math.abs(arcWidth(stand.elements.get('goodArc').attributes.d) - goodWidth) < .02);
    const played = pressGreatForCurrentCheck(stand);
    totalRotation += played.travel;
    const speedIndex = firstRoute.checks.length + index;
    assert.ok(Math.abs(played.speed - frenzySpeeds[Math.min(speedIndex, frenzySpeeds.length - 1)]) < 1,
      'infinite Frenzy speed progression continues across captured route boundaries');
    priorStart = expectedStart;
    if (index < 4) waitForFrenzyCheck(stand);
  }
  assert.ok(totalRotation > 360, `infinite Frenzy reset before one turn: ${totalRotation.toFixed(2)}°`);

  stand.frame(stand.getTime() + 2000);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  stand.frame(stand.getTime() + 2000);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  assert.doesNotMatch(stand.elements.get('frenzyStatus').textContent, /measured transitions/);
  assert.match(stand.elements.get('message').textContent, /MISS/);
});

test('infinite Frenzy translates only replay-measured transitions through 100 checks', () => {
  const stand = createStand();
  stand.elements.set('infiniteFrenzy', {checked:true});
  stand.frame(100);
  stand.click('frenzyStart');
  const route = firstFrenzyRoute;
  for (let index = 0; index < route.checks.length; index += 1) {
    pressGreatForCurrentCheck(stand);
    if (index + 1 < route.checks.length) waitForFrenzyCheck(stand);
  }

  const minimumObservedSeparation = Math.min(...frenzyTransitions.map(step =>
    Math.min(step[3], 360 - step[3])));
  let previousStart = route.checks.at(-1)[3];
  waitForFrenzyCheck(stand);
  const generatedStarts = [];
  for (let index = 0; index < 100; index += 1) {
    const [greatWidth, goodWidth, gap, delta] = frenzyTransitions[index % frenzyTransitions.length];
    const expectedStart = (previousStart + delta) % 360;
    const currentStart = arcStart(stand.elements.get('greatArc').attributes.d);
    assert.ok(Math.abs(currentStart - expectedStart) < .02,
      `transition ${index + 1} should start at ${expectedStart.toFixed(3)}°, got ${currentStart.toFixed(3)}°`);
    assert.ok(Math.abs(arcWidth(stand.elements.get('greatArc').attributes.d) - greatWidth) < .02);
    assert.ok(Math.abs(arcWidth(stand.elements.get('goodArc').attributes.d) - goodWidth) < .02);
    assert.ok(Math.abs(arcStart(stand.elements.get('goodArc').attributes.d) -
      (expectedStart + greatWidth + gap) % 360) < .02);
    generatedStarts.push(currentStart.toFixed(3));
    previousStart = currentStart;
    pressGreatForCurrentCheck(stand);
    if (index + 1 < 100) waitForFrenzyCheck(stand);
  }
  assert.ok(new Set(generatedStarts).size >= 80,
    `continuation used only ${new Set(generatedStarts).size} distinct captured positions`);
});

test('practice cadence has a visible adjustable interval control', () => {
  assert.match(html, /id="checkDelay"/);
  assert.match(html, /Normal interval/);
  assert.match(inlineScript, /checkDelay/);
  const stand = createStand();
  stand.elements.get('checkDelay').value = '1';
  stand.elements.get('checkDelay').listeners.input();
  assert.equal(stand.elements.get('checkDelayValue').textContent, '1 s');
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  const afterCheck = stand.getTime();
  const delay = Math.max(.5, intervalSamples[0] / 6.5) * 1000;
  stand.frame(afterCheck + delay - 2);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  stand.frame(afterCheck + delay);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
});

test('Frenzy checks use their own adjustable interval before the next check', () => {
  assert.match(html, /id="frenzyDelay"/);
  assert.match(html, /Frenzy interval/);
  const stand = createStand();
  stand.elements.set('infiniteFrenzy', {checked:true});
  assert.ok(Math.abs(Number(stand.elements.get('frenzyDelay').value) - frenzyTiming.median_ms / 1000) < .005);
  stand.elements.get('frenzyDelay').value = '0.8';
  stand.elements.get('frenzyDelay').listeners.input();
  assert.equal(stand.elements.get('frenzyDelayValue').textContent, '0.8 s');
  stand.frame(900);
  stand.click('frenzyStart');
  pressGreatForCurrentCheck(stand);
  assert.match(stand.elements.get('message').textContent, /GREAT/);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  const hitAt = stand.getTime();
  stand.frame(hitAt + 799);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  stand.frame(hitAt + 800);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
});

test('ordinary check cadence uses a median-scaled practice version of measured intervals', () => {
  assert.equal(intervalSamples.length, 659);
  assert.ok(Math.min(...intervalSamples) > 1.5);
  assert.ok(Math.max(...intervalSamples) < 60);
  const sorted = [...intervalSamples].sort((a, b) => a - b);
  assert.ok(Math.abs(sorted[Math.floor(sorted.length / 2)] - 6.50) < .03);
  assert.ok(Math.abs(sorted[Math.floor((sorted.length - 1) * .9)] - 24.76) < .3);
  assert.match(inlineScript, /VDDefaultCheckIntervals/);
  assert.doesNotMatch(inlineScript, /function nextInterval\(\) \{ return 2\.65/);
  const stand = createStand();
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  assert.ok(stand.getTimerDelay() <= 5000, `practice wait is ${stand.getTimerDelay()} ms`);
});

test('practice waits on a timer while idle instead of running a full animation loop', () => {
  const stand = createStand();
  stand.frame(16);
  assert.equal(stand.getFrameCount(), 1);
  assert.ok(stand.getTimerDelay() > 800);
});

function arcStart(pathData) {
  const [x, y] = pathData.match(/^M ([\d.-]+) ([\d.-]+)/).slice(1).map(Number);
  return (Math.atan2(y - 162.5, x - 160) * 180 / Math.PI + 360) % 360;
}

function arcWidth(pathData) {
  const matches = [...pathData.matchAll(/M ([\d.-]+) ([\d.-]+)|66\.5 66\.5 0 0 1 ([\d.-]+) ([\d.-]+)/g)];
  const [, x1, y1] = matches[0];
  const [, , , x2, y2] = matches[1];
  const start = Math.atan2(Number(y1) - 162.5, Number(x1) - 160);
  const end = Math.atan2(Number(y2) - 162.5, Number(x2) - 160);
  return (end - start + 2 * Math.PI) % (2 * Math.PI) * 180 / Math.PI;
}

function waitForFrenzyCheck(stand) {
  stand.frame(stand.getTime() + Number(stand.elements.get('frenzyDelay').value || .5) * 1000 + 5);
}

function frameUntilCheck(stand) {
  for (let attempt = 0; attempt < 20 && stand.elements.get('checkSvg').style.opacity !== '1'; attempt += 1) {
    stand.frame(stand.getTime() + Math.min(stand.getTimerDelay() || 1000, 1000) + 1);
  }
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1', 'the scheduled check begins');
}

function frameUntilMessage(stand, pattern) {
  for (let attempt = 0; attempt < 20 && !pattern.test(stand.elements.get('message').textContent); attempt += 1) {
    stand.frame(stand.getTime() + Math.min(stand.getTimerDelay() || 1000, 1000) + 1);
  }
  assert.match(stand.elements.get('message').textContent, pattern);
}

function pressGreatForCurrentCheck(stand) {
  const start = arcStart(stand.elements.get('greatArc').attributes.d);
  const initialTravel = (start + 5 - needleAngle(stand) + 360) % 360;
  const beforeProbe = needleAngle(stand);
  stand.frame(stand.getTime() + 10);
  const speed = ((needleAngle(stand) - beforeProbe + 360) % 360) * 100;
  const remaining = (start + 5 - needleAngle(stand) + 360) % 360;
  stand.setTime(stand.getTime() + remaining / speed * 1000);
  stand.pressSpace();
  stand.frame(stand.getTime() + 1);
  return {travel: initialTravel, speed, greatStart: start};
}

test('practice stand starts exactly one animation loop', () => {
  assert.equal(inlineScripts.length, 1);
});

test('practice controls fit in a centered compact dock with help collapsed by default', () => {
  assert.match(html, /<header class="toolbar">/);
  assert.match(html, /<details class="help">\s*<summary/);
  assert.match(html, /\.layout\{[^}]*height:100dvh/);
  assert.match(html, /\.panel\{[^}]*max-width:820px/);
  assert.match(html, /@media\(max-width:780px\)[\s\S]*?\.control-grid\{grid-template-columns:1fr 1fr\}/);
  assert.doesNotMatch(html, /class="sub"/);
});

test('every DOM selector in the practice stand resolves to an element', () => {
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
  const selectors = new Set([...inlineScript.matchAll(/\$\('([^']+)'\)/g)].map((match) => match[1]));
  assert.deepEqual([...selectors].filter((id) => !ids.has(id)), []);
});

test('skill check center uses the original Space prompt art', () => {
  assert.match(html, /href="\.\.\/assets\/space_template\.png" x="125" y="145" width="70" height="35"/);
  const template = fs.readFileSync(path.resolve(__dirname, '../assets/space_template.png'));
  assert.equal(template.readUInt32BE(16), 70);
  assert.equal(template.readUInt32BE(20), 35);
});

test('skill check stays fixed at the center instead of moving between recording crops', () => {
  assert.deepEqual(checkAnchors, [[160, 162.5], [170, 82.5]]);
  const stand = createStand();
  stand.frame(900);
  assert.equal(stand.elements.get('ring').attributes.cx, 160);
  assert.equal(stand.elements.get('ring').attributes.cy, 162.5);
  assert.equal(stand.elements.get('keyPrompt').attributes.x, 125);
  assert.equal(stand.elements.get('keyPrompt').attributes.y, 145);

  stand.click('new');
  assert.equal(stand.elements.get('ring').attributes.cx, 160);
  assert.equal(stand.elements.get('ring').attributes.cy, 162.5);
  assert.equal(stand.elements.get('keyPrompt').attributes.x, 125);
  assert.equal(stand.elements.get('keyPrompt').attributes.y, 145);
  assert.equal(stand.elements.get('needle').attributes.x1, 160);
  assert.equal(stand.elements.get('needle').attributes.y1, 162.5);
});

test('every selected scene background exists in the local bundle', () => {
  assert.equal(scenePaths.length, 18);
  for (const scene of scenePaths) {
    assert.ok(fs.existsSync(path.resolve(__dirname, scene)), scene);
  }
});

test('target geometry comes from paired plausible replay measurements', () => {
  assert.equal(zoneSamples.length, 2473);
  for (const [great, good, gap, start] of zoneSamples) {
    assert.ok(great >= 9.5 && great <= 11.5);
    assert.ok(good >= 40 && good <= 44);
    assert.ok(gap >= 0 && gap <= 2);
    assert.ok(start >= 0 && start < 360);
  }
});

test('ordinary replay bundle contains measured routes from the normal chain', () => {
  assert.ok(fs.existsSync(defaultZoneSequencePath), 'ordered normal-mode routes are bundled');
  assert.equal(defaultZoneSequences.length, 258);
  assert.equal(defaultZoneSequences.reduce((sum, route) => sum + route.length, 0), 1262);
  assert.deepEqual(defaultZoneSequenceMeta, {
    explicit_session_measured_routes: 102,
    explicit_session_measured_checks: 625,
    manifest_routes: 156,
    manifest_checks: 637,
  });
  for (const route of defaultZoneSequences) {
    assert.ok(route.length >= 2, 'routes contain observed adjacent checks');
    for (const [great, good, gap, start] of route) {
      assert.ok(great >= 9 && great <= 12);
      assert.ok(good >= 40 && good <= 44);
      assert.ok(gap >= 0 && gap <= 2);
      assert.ok(start >= 0 && start < 360);
    }
  }
  assert.match(defaultZoneSequenceScript, /explicit-session measured traces first, then manifest traces/);
  assert.match(inlineScript, /VDDefaultZoneSequences/);
});

test('ordinary checks play their captured route in order', () => {
  assert.ok(fs.existsSync(defaultZoneSequencePath), 'ordered ordinary routes are bundled');
  assert.ok(defaultZoneSequenceScript.includes('window.VDDefaultZoneSequences ='), 'ordered route bundle is valid');
  assert.ok(html.includes('<script src="./default-zone-sequences.js"></script>'), 'practice loads ordered routes');
  assert.equal(defaultZoneSequences.length, 258);
  assert.equal(defaultZoneSequences.reduce((sum, route) => sum + route.length, 0), 1262);
  const stand = createStand();
  const expected = defaultZoneSequences[0].slice(0, 3);
  stand.frame(900);
  for (let index = 0; index < expected.length; index += 1) {
    if (index) stand.click('new');
    assert.ok(Math.abs(arcStart(stand.elements.get('greatArc').attributes.d) - expected[index][3]) < .02,
      `ordinary replay position ${index + 1} follows its captured route`);
  }
});

test('ordinary playback stops at a recorded route boundary instead of joining unrelated sessions', () => {
  const stand = createStand();
  const firstRoute = defaultZoneSequences[0];
  stand.frame(900);

  for (let index = 0; index < firstRoute.length; index += 1) {
    if (index > 0) frameUntilCheck(stand);
    assert.ok(Math.abs(arcStart(stand.elements.get('greatArc').attributes.d) - firstRoute[index][3]) < .02,
      `ordinary route check ${index + 1}: expected ${firstRoute[index][3]}, got ${arcStart(stand.elements.get('greatArc').attributes.d)}`);
    pressGreatForCurrentCheck(stand);
    assert.match(stand.elements.get('message').textContent, /GREAT/);
  }

  for (let attempt = 0; attempt < 20 && stand.elements.get('checkSvg').style.opacity !== '1'
       && !/ROUTE COMPLETE/.test(stand.elements.get('message').textContent); attempt += 1) {
    stand.frame(stand.getTime() + Math.min(stand.getTimerDelay() || 1000, 1000) + 1);
  }
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  assert.match(stand.elements.get('message').textContent, /ROUTE COMPLETE/);

  stand.click('new');
  assert.ok(Math.abs(arcStart(stand.elements.get('greatArc').attributes.d) - defaultZoneSequences[1][0][3]) < .02,
    'an explicit New check begins the next captured route');
});

test('missing replay geometry never falls back to invented random targets', () => {
  const stand = createStand({routes:[]});
  stand.frame(900);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  assert.match(stand.elements.get('message').textContent, /REPLAY DATA MISSING/);
  assert.doesNotMatch(stand.elements.get('frenzyStatus').textContent, /trace complete/);
});

test('every paired zone sample from the replay corpus resolves GREAT and GOOD at its arc centers', () => {
  const stand = createStand({routes:zoneSamples.map(sample => [sample, sample])});
  stand.frame(900);
  const pressAt = angle => {
    const current = needleAngle(stand);
    const initialTravel = (angle - current + 360) % 360;
    // Avoid the exact end-of-turn timeout boundary when the sampled zone is
    // at the same phase as the last check; move a couple degrees into it.
    const target = initialTravel > 359.7 ? (angle + 2) % 360 : angle;
    const clockwiseTravel = (target - current + 360) % 360;
    stand.setTime(stand.getTime() + clockwiseTravel / 278 * 1000);
    stand.pressSpace();
    stand.frame(stand.getTime() + 1);
  };

  for (let index = 0; index < zoneSamples.length; index += 1) {
    const [greatWidth, goodWidth, gap, start] = zoneSamples[index];
    if (index > 0) {
      stand.click('new');
    }
    pressAt((start + greatWidth / 2) % 360);
    assert.match(stand.elements.get('message').textContent, /GREAT/, `sample ${index} GREAT`);

    stand.click('new');
    pressAt((start + greatWidth + gap + goodWidth / 2) % 360);
    assert.match(stand.elements.get('message').textContent, /GOOD/, `sample ${index} GOOD`);
  }
});

test('needle starts at 270 degrees and GREAT zones use replay-recorded orientations', () => {
  assert.match(inlineScript, /\[greatWidth, goodWidth, gap, greatStart\] = sample/);
  assert.match(inlineScript, /let angle = 270/);
  assert.match(inlineScript, /checkStartAngle = frenzyActive \? angle : 270/);
  assert.match(inlineScript, /const needleSpeed = 278/);
});

test('solver-only bench matches the capture ROI and keeps its synthetic red pointer isolated', () => {
  assert.match(html, /solver-test/);
  assert.match(html, /\.solver-bench \.stage\{[^}]*left:800px;top:420px;width:320px;height:240px/);
  assert.match(html, /\.solver-bench \.needle\{stroke:#e83d43\}/);
  assert.match(html, /\.needle\{stroke:#d9dee0/);
  assert.match(html, /innerWidth===1920&&innerHeight===1080/);
  assert.match(html, /SOLVER TEST BENCH/);
});

test('F always starts Frenzy, even when a custom key is saved', () => {
  const stand = createStand({savedFrenzyKey:'KeyG'});
  stand.frame(100);
  stand.key('KeyF', 'f');
  assert.match(stand.elements.get('frenzyStatus').textContent, /Frenzy active · continuous checks/);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  const first = pressGreatForCurrentCheck(stand);
  assert.ok(first.travel > 0);
  assert.ok(Math.abs(first.speed - frenzySpeeds[0]) < 1);
  stand.key('KeyF', 'f');
  assert.match(stand.elements.get('frenzyStatus').textContent, /Inactive/);
  stand.key('KeyG', 'g');
  assert.match(stand.elements.get('frenzyStatus').textContent, /Frenzy active/);
  pressForOutcome(stand, 'MISS');
  assert.match(stand.elements.get('frenzyStatus').textContent, /Inactive/);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
});

test('Frenzy carries the needle through more than one full 360° turn', () => {
  const stand = createStand();
  stand.frame(100);
  stand.click('frenzyStart');
  let totalRotation = 0;
  for (let index = 0; index < 4; index += 1) {
    totalRotation += pressGreatForCurrentCheck(stand).travel;
    if (index < 3) waitForFrenzyCheck(stand);
    assert.match(stand.elements.get('message').textContent, /GREAT/);
    assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  }
  assert.ok(totalRotation > 360, `expected >360° total rotation, got ${totalRotation}`);
  assert.match(stand.elements.get('frenzyStatus').textContent, /Frenzy active/);
});

test('Frenzy keeps the needle moving through the recorded post-hit transition', () => {
  const stand = createStand();
  stand.elements.set('infiniteFrenzy', {checked:true});
  stand.elements.get('frenzyDelay').value = '0.15';
  stand.frame(100);
  stand.click('frenzyStart');
  const first = pressGreatForCurrentCheck(stand);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1',
    'the recorded ring remains on screen until the next check begins');
  const hitAt = stand.getTime();
  const angleAtHit = needleAngle(stand);
  stand.frame(hitAt + 75);
  const halfwayAngle = needleAngle(stand);
  assert.ok(((halfwayAngle - angleAtHit + 360) % 360) > 10,
    'needle continues to rotate after a successful Space input');
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  stand.frame(hitAt + 155);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  assert.ok(Math.abs(arcStart(stand.elements.get('greatArc').attributes.d) - firstFrenzyRoute.checks[1][3]) < .02,
    'the next recorded zone begins after the measured transition delay');
  const expectedAdvance = first.speed * .155;
  const actualAdvance = (needleAngle(stand) - angleAtHit + 360) % 360;
  assert.ok(Math.abs(actualAdvance - expectedAdvance) < 3,
    `needle should advance through the hit tail: expected ${expectedAdvance.toFixed(1)}°, got ${actualAdvance.toFixed(1)}°`);
});

test('a Space miss hides the check immediately and freezes its needle', () => {
  const stand = createStand();
  stand.frame(900);
  pressForOutcome(stand, 'MISS');
  assert.match(stand.elements.get('message').textContent, /MISS/);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  const needleAtMiss = {...stand.elements.get('needle').attributes};
  stand.frame(stand.getTime() + 100);
  assert.deepEqual(stand.elements.get('needle').attributes, needleAtMiss);
});

test('the visible GREAT-to-GOOD seam remains a GOOD hit like modern replay labels', () => {
  const sample = defaultZoneSequences.flat().find(([, , gap]) => gap >= .5);
  assert.ok(sample);
  const [great, , gap, start] = sample;
  const stand = createStand({routes:[[sample]]});
  stand.frame(900);

  const displayedGoodStart = (start + great + gap) % 360;
  const radians = displayedGoodStart * Math.PI / 180;
  const pathStart = stand.elements.get('goodArc').attributes.d.match(/^M ([\d.-]+) ([\d.-]+)/);
  assert.ok(Math.abs(Number(pathStart[1]) - (160 + 66.5 * Math.cos(radians))) < 1e-8);
  assert.ok(Math.abs(Number(pathStart[2]) - (162.5 + 66.5 * Math.sin(radians))) < 1e-8);

  const seamAngle = (start + great + gap / 2) % 360;
  const travel = (seamAngle - 270 + 360) % 360;
  stand.setTime(900 + travel / 278 * 1000);
  stand.pressSpace();
  stand.frame(stand.getTime() + 1);
  assert.match(stand.elements.get('message').textContent, /GOOD/);
});

test('an untouched check remains visible for one rotation before it misses', () => {
  const stand = createStand();
  stand.frame(900);
  assert.equal(stand.elements.get('needle').attributes.x2, 160);
  assert.equal(stand.elements.get('needle').attributes.y2, 88.5);
  stand.frame(2234);
  assert.match(stand.elements.get('message').textContent, /MISS/);
});

function createStand({routes = defaultZoneSequences, savedFrenzyKey = null} = {}) {
  let now = 0;
  let frameCallback;
  let frameCount = 0;
  let timerDelay = null;
  let randomValues = [0];
  let randomIndex = 0;
  const elements = new Map();
  const windowEvents = {};
  const storage = new Map();
  if (savedFrenzyKey) storage.set('vd-frenzy-key', savedFrenzyKey);
  const document = { getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      checked: id === 'alone' || id === 'repair',
      value: id === 'tier' ? '3' : id === 'checkDelay' ? '3.5' : id === 'frenzyDelay' ? '0.15' : '',
      style: {}, listeners: {}, attributes: {},
      classList: {toggle() {}},
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute(name, value) { this.attributes[name] = value; }
    });
    return elements.get(id);
  }};
  const fakeMath = new Proxy(Math, { get(target, name) {
    if (name === 'random') return () => {
      const result = randomValues[Math.min(randomIndex, randomValues.length - 1)];
      randomIndex += 1;
      return result;
    };
    return target[name];
  }});
  const window = { VDPerkModel: perkModel,
    VDSceneBackgrounds: scenePaths,
    VDDefaultZoneSequences: routes,
    VDFrenzySequences: frenzySequences,
    VDFrenzySpeeds: frenzySpeeds,
    VDFrenzyTransitions: frenzyTransitions,
    VDFrenzyTiming: frenzyTiming,
    VDDefaultCheckIntervals: intervalSamples,
    VDCheckAnchors: checkAnchors,
    setTimeout(callback, delay) { timerDelay = delay; return 1; },
    clearTimeout() { timerDelay = null; },
    addEventListener(name, callback) {
    windowEvents[name] = callback;
  }};
  vm.runInNewContext(inlineScript, {
    document, window, Math: fakeMath,
    localStorage: {
      getItem(key) { return storage.has(key) ? storage.get(key) : null; },
      setItem(key, value) { storage.set(key, String(value)); }
    },
    performance: { now: () => now },
    requestAnimationFrame(callback) { frameCallback = callback; frameCount += 1; }
  });
  return {
    elements, windowEvents,
    storage,
    setTime(value) { now = value; },
    getTime() { return now; },
    getFrameCount() { return frameCount; },
    getTimerDelay() { return timerDelay; },
    setRandom(values) { randomValues = values; randomIndex = 0; },
    frame(value = now) { now = value; frameCallback(now); },
    pressSpace(target = null) { windowEvents.keydown({code:'Space', target, preventDefault(){}}); },
    click(id) {
      const element = elements.get(id);
      if (element.onclick) element.onclick();
      if (element.listeners.click) element.listeners.click();
    },
    key(code, key) { windowEvents.keydown({code, key, target:null, repeat:false, preventDefault(){}}); },
    change(id, value) {
      elements.get(id).checked = value;
      elements.get(id).listeners.change();
    }
  };
}

function pressForOutcome(stand, outcome, { startCheck = false } = {}) {
  if (startCheck) stand.click('new');
  const great = visibleArc(stand, 'greatArc');
  const good = visibleArc(stand, 'goodArc');
  const angle = outcome === 'GREAT' ? great.start + great.width / 2 :
    outcome === 'GOOD' ? good.start + good.width / 2 :
    Array.from({length:360}, (_, index) => index).find(value =>
      !arcContains(value, great.start, great.width) &&
      !arcContains(value, great.start + great.width, (good.start + good.width - great.start - great.width + 360) % 360));
  const clockwiseTravel = ((angle - needleAngle(stand)) % 360 + 360) % 360;
  stand.setTime(stand.getTime() + clockwiseTravel / 278 * 1000);
  stand.pressSpace();
  stand.frame(stand.getTime() + 1);
}

function visibleArc(stand, id) {
  const d = stand.elements.get(id).attributes.d;
  const match = d.match(/^M ([\d.-]+) ([\d.-]+) A [^ ]+ [^ ]+ 0 0 1 ([\d.-]+) ([\d.-]+)$/);
  assert.ok(match, `${id} has a visible arc path`);
  const degrees = (x, y) => (Math.atan2(Number(y) - 162.5, Number(x) - 160) * 180 / Math.PI + 360) % 360;
  const start = degrees(match[1], match[2]);
  const end = degrees(match[3], match[4]);
  return {start, width:(end - start + 360) % 360};
}

function arcContains(angle, start, width) {
  return ((angle - start + 360) % 360) <= width;
}

function needleAngle(stand) {
  const needle = stand.elements.get('needle').attributes;
  return (Math.atan2(Number(needle.y2)-162.5, Number(needle.x2)-160) * 180/Math.PI + 360) % 360;
}

test('practice check is hidden while idle, appears, and resolves with Space', () => {
  const stand = createStand();
  stand.frame(899);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  stand.frame(900);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  stand.pressSpace();
  stand.frame(901);
  assert.match(stand.elements.get('message').textContent, /GREAT|GOOD|MISS/);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
});

test('result appears below the game field with the outcome color', () => {
  const stand = createStand();
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  assert.equal(stand.elements.get('result').textContent, 'GREAT');
  assert.equal(stand.elements.get('result').className, 'result great');
  pressForOutcome(stand, 'GOOD', {startCheck:true});
  assert.equal(stand.elements.get('result').textContent, 'GOOD');
  assert.equal(stand.elements.get('result').className, 'result good');
  pressForOutcome(stand, 'MISS', {startCheck:true});
  assert.equal(stand.elements.get('result').textContent, 'MISS');
  assert.equal(stand.elements.get('result').className, 'result miss');
});

test('Space remains available to focused controls instead of resolving the check', () => {
  const stand = createStand();
  stand.frame(900);
  let prevented = false;
  stand.pressSpace({closest() { return {}; }});
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  // The document handler leaves native keyboard controls untouched.
  const controlEvent = {code:'Space', target:{closest() { return {}; }}, preventDefault(){ prevented = true; }};
  stand.windowEvents.keydown(controlEvent);
  assert.equal(prevented, false);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
});

test('background stays still for 64 checks, then changes scene', () => {
  const stand = createStand();
  stand.frame(900);
  const first = stand.elements.get('stage').style.backgroundImage;
  for (let index = 0; index < 63; index += 1) stand.click('new');
  assert.equal(stand.elements.get('stage').style.backgroundImage, first);
  stand.click('new');
  const second = stand.elements.get('stage').style.backgroundImage;
  assert.match(first, /verified-scenes/);
  assert.match(second, /verified-scenes/);
  assert.notEqual(second, first);
});

test('UI applies Flawless Execution tiers across consecutive checks and miss', () => {
  const fastCapturedSample = defaultZoneSequences.flat().find(([, , , start]) => start <= 80);
  assert.ok(fastCapturedSample);
  const stand = createStand({routes:[Array(6).fill(fastCapturedSample)]});
  stand.change('fe', true);
  stand.setRandom([0, 0, 0, 0]);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>5%/);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>5%/);
  for (let count = 0; count < 2; count += 1) {
    pressForOutcome(stand, 'GREAT', {startCheck:true});
  }
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>8%/);
  assert.match(stand.elements.get('perkState').innerHTML, /duration −0\.300 s/);

  // Place the next pointer in GOOD: the Great streak breaks, but the earned
  // effect remains until a miss.
  stand.setRandom([0, 0, 0, .05]);
  pressForOutcome(stand, 'GOOD', {startCheck:true});
  assert.match(stand.elements.get('message').textContent, /GOOD/);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>8%/);
  assert.match(stand.elements.get('perkState').innerHTML, /duration −0\.400 s/);

  stand.setRandom([0, 0, 0, .5]);
  pressForOutcome(stand, 'MISS', {startCheck:true});
  assert.match(stand.elements.get('message').textContent, /MISS/);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>0%/);
});

test('UI renders all three FE tiers and their third-GREAT bonus and duration reduction', () => {
  const tiers = [
    {tier:'1', base:3, extra:4, reduction:'0.150', window:8},
    {tier:'2', base:4, extra:6, reduction:'0.225', window:10},
    {tier:'3', base:5, extra:8, reduction:'0.300', window:12}
  ];
  for (const expected of tiers) {
    const fastCapturedSample = defaultZoneSequences.flat().find(([, , , start]) => start <= 80);
    const stand = createStand({routes:[Array(6).fill(fastCapturedSample)]});
    stand.change('fe', true);
    stand.elements.get('tier').value = expected.tier;
    stand.elements.get('tier').listeners.change();
    stand.setRandom([0]);
    stand.frame(900);
    pressForOutcome(stand, 'GREAT');
    assert.match(stand.elements.get('perkState').innerHTML,
      new RegExp('Repair boost <b>' + expected.base + '%'));
    assert.match(stand.elements.get('perkState').innerHTML,
      new RegExp('Return window ' + expected.window + ' s'));
    for (let hit = 1; hit < 3; hit += 1)
      pressForOutcome(stand, 'GREAT', {startCheck:true});
    assert.match(stand.elements.get('perkState').innerHTML,
      new RegExp('Repair boost <b>' + expected.extra + '%'));
    assert.match(stand.elements.get('perkState').innerHTML,
      new RegExp('duration −' + expected.reduction + ' s'));
  }
});

test('turning FE off clears its earned boost and accumulated duration reduction', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>5%/);
  stand.change('fe', false);
  stand.frame(stand.getTime() + 1);
  assert.match(stand.elements.get('perkState').innerHTML, /Perk off/);
  assert.equal(stand.elements.get('meter').style.width, '0%');
  stand.change('fe', true);
  stand.frame(stand.getTime() + 1);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>0%/);
  assert.match(stand.elements.get('perkState').innerHTML, /duration −0\.000 s/);
});

test('successful GOOD before the first GREAT shortens the next check without earning a boost', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.setRandom([0, 0, 0, .05]);
  stand.frame(900);
  pressForOutcome(stand, 'GOOD');
  assert.match(stand.elements.get('message').textContent, /GOOD/);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>0%/);
  assert.match(stand.elements.get('perkState').innerHTML, /duration −0\.100 s/);
});

test('duration reduction applies while not alone, and a MISS there resets the perk', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>5%/);

  stand.change('alone', false);
  pressForOutcome(stand, 'GOOD', {startCheck:true});
  assert.match(stand.elements.get('perkState').innerHTML, /Paused/);
  assert.match(stand.elements.get('perkState').innerHTML, /duration −0\.200 s/);

  pressForOutcome(stand, 'MISS', {startCheck:true});
  assert.match(stand.elements.get('perkState').innerHTML, /duration −0\.000 s/);
  assert.equal(stand.elements.get('meter').style.width, '0%');
});

test('leaving pauses the earned effect, returning in time preserves it, expiry clears it', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  stand.click('leave');
  stand.frame(902);
  assert.match(stand.elements.get('perkState').innerHTML, /Paused/);
  stand.setTime(10000);
  stand.click('return');
  stand.frame(10001);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>5%/);

  stand.click('leave');
  stand.frame(10002);
  stand.frame(23003);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>0%/);
  assert.equal(stand.elements.get('repair').checked, false);
});

test('becoming not alone pauses an earned effect without erasing it', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  const earned = stand.elements.get('perkState').innerHTML.match(/Repair boost <b>(\d+)%/)[1];
  const streak = stand.elements.get('perkState').innerHTML.match(/Consecutive GREAT checks/);
  stand.change('alone', false);
  stand.frame(902);
  assert.match(stand.elements.get('perkState').innerHTML, /<b>Paused<\/b>/);
  assert.equal(stand.elements.get('perkState').innerHTML.match(/Repair boost <b>(\d+)%/)[1], earned);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  stand.setTime(10000);
  stand.click('new');
  stand.frame(10000);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  stand.change('alone', true);
  stand.frame(10001);
  assert.match(stand.elements.get('perkState').innerHTML, /<b>Active<\/b> · Repair boost <b>5%/);
});

test('return deadline uses the tier at departure, even if the selected tier changes', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  stand.click('leave');
  stand.elements.get('tier').value = '1';
  stand.elements.get('tier').listeners.change();
  stand.setTime(12000);
  stand.click('return');
  stand.frame(12001);
  assert.equal(stand.elements.get('repair').checked, true);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>3%/);
});

test('repair toggle is equivalent to leave and expires the preserved perk state at its saved deadline', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  stand.change('repair', false);
  stand.frame(902);
  assert.match(stand.elements.get('perkState').innerHTML, /Paused/);
  assert.equal(stand.elements.get('repair').checked, false);
  stand.setTime(10000);
  stand.frame(10001);
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>0%/);
  assert.match(stand.elements.get('perkState').innerHTML, /12 s/);
});

test('repair toggle can return during the tier window and resumes the retained boost', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  pressForOutcome(stand, 'GREAT');
  const boost = stand.elements.get('perkState').innerHTML.match(/Repair boost <b>(\d+)%/)[1];
  stand.change('repair', false);
  stand.setTime(8000);
  stand.change('repair', true);
  assert.match(stand.elements.get('perkState').innerHTML, /Active/);
  assert.equal(stand.elements.get('perkState').innerHTML.match(/Repair boost <b>(\d+)%/)[1], boost);
});

test('return status keeps the departure tier and missed deadline wins over a late Space before the next frame', () => {
  const sampleWithout180 = defaultZoneSequences.flat().find(([great, good, gap, start]) => {
    const angle = 180;
    const inside = (a, from, width) => ((a - from + 360) % 360) <= width;
    return !inside(angle, start, great) &&
      !inside(angle, (start + great + gap) % 360, good);
  });
  assert.ok(sampleWithout180);
  const stand = createStand({routes:[[sampleWithout180, sampleWithout180]]});
  stand.change('fe', true);
  stand.frame(900);
  stand.click('leave');
  stand.elements.get('tier').value = '1';
  stand.elements.get('tier').listeners.change();
  stand.frame(9000);
  assert.match(stand.elements.get('perkState').innerHTML, /Return window 12 s · 3\.9 s left/);

  stand.click('return');
  stand.setTime(9600);
  stand.frame(9601);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '1');
  stand.setTime(10600);
  stand.pressSpace();
  assert.match(stand.elements.get('message').textContent, /MISS/);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
});

test('a missed active timer resolves as MISS without pressing Space', () => {
  const stand = createStand();
  stand.change('fe', true);
  stand.frame(900);
  stand.frame(2234);
  assert.match(stand.elements.get('message').textContent, /MISS/);
  assert.equal(stand.elements.get('checkSvg').style.opacity, '0');
  assert.match(stand.elements.get('perkState').innerHTML, /Repair boost <b>0%/);
});
