import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import path from "node:path";
import ts from "typescript";
import * as tf from "@tensorflow/tfjs";

// Compile only the simulation modules, leaving the application and dependencies untouched.
const root = path.resolve(import.meta.dirname, "..");
await mkdir(path.join(root, ".sites-runtime"), { recursive: true });
const temp = await mkdtemp(path.join(root, ".sites-runtime/qat-check-"));
const simulations = [];
try {
  for (const name of ["dataset", "architecture", "quantization", "target-schedule", "hidden-means", "simulator"]) {
    const source = await readFile(path.join(root, "lib", `${name}.ts`), "utf8");
    const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    await writeFile(path.join(temp, `${name}.mjs`), compiled.replace('from "./dataset"', 'from "./dataset.mjs"').replace('from "./architecture"', 'from "./architecture.mjs"').replace('from "./quantization"', 'from "./quantization.mjs"').replace('from "./target-schedule"', 'from "./target-schedule.mjs"').replace('from "./hidden-means"', 'from "./hidden-means.mjs"'));
  }
  const { TransformerSphereSimulation: Simulation, fakeQuantize, RADIUS, probeAt } = await import(pathToFileURL(path.join(temp, "simulator.mjs")));
  const { QUANTIZATION_FORMATS, DEFAULT_QAT, blendAt, quantizationScale, quantizeValue } = await import(pathToFileURL(path.join(temp, "quantization.mjs")));
  const { includedTargets, dutyOnSteps } = await import(pathToFileURL(path.join(temp, "target-schedule.mjs")));
  const { averageHiddenByTarget } = await import(pathToFileURL(path.join(temp, "hidden-means.mjs")));
  await tf.setBackend("cpu");
  const baseline = tf.memory().numTensors;
  const config = { seed: 17, optimizer: "adamw", learningRate: 0.018, weightDecay: 0.05, targetedTokens: 4, untargetedTokens: 3, batchSize: 2 };
  const create = async (settings) => { const sim = await Simulation.create({ ...config, ...settings }); simulations.push(sim); return sim; };
  const close = (a, b, epsilon = 1e-6) => assert.ok(Math.abs(a - b) < epsilon, `${a} != ${b}`);
  // Unequal vector lengths catch projecting samples before averaging. Noncontiguous
  // next-token labels catch accidentally grouping by input IDs or array position.
  const means = averageHiddenByTarget([1,0,0, 4,0,0, -1,0,0, 0,2,0], [2,0,2,0], 4, RADIUS);
  assert.deepEqual(means.map(item => item.target), [0,2]);
  assert.deepEqual(means[0].mean, [2,1,0]);
  close(means[0].projected[0], RADIUS * 2 / Math.sqrt(5));
  close(means[0].projected[1], RADIUS / Math.sqrt(5));
  assert.ok(Math.hypot(...means[0].mean) > RADIUS);
  assert.deepEqual(means[1].mean, [0,0,0]);
  assert.equal(means[1].projected, null);
  assert.deepEqual(averageHiddenByTarget([], [], 4, RADIUS), []);
  const compare = (a, b) => {
    close(a.latest.loss, b.latest.loss, 1e-7);
    for (const name of Object.keys(a.variables)) assert.deepEqual(Array.from(a.variables[name].dataSync()), Array.from(b.variables[name].dataSync()));
    assert.deepEqual(a.latest.rawGradients, b.latest.rawGradients);
  };
  for (const format of QUANTIZATION_FORMATS.filter((item) => item.value !== "fp32")) {
    const values = Float32Array.of(format.min, -0.49, 0, 0.49, format.max);
    close(quantizationScale(values, format.value), 1);
    close(quantizeValue(format.min, format.value, 1), format.min);
    close(quantizeValue(format.max, format.value, 1), format.max);
    assert.equal(quantizeValue(1000, format.value, 1), format.max);
    assert.equal(quantizeValue(-1000, format.value, 1), format.min);
    assert.ok(Number.isFinite(quantizationScale([0, 0], format.value)));
    tf.tidy(() => {
      const weights = tf.tensor1d(values);
      const settings = { ...DEFAULT_QAT, format: format.value, schedule: "immediate" };
      const result = fakeQuantize(weights, settings, 1).dataSync();
      for (const value of result) assert.ok(Number.isInteger(value) && value >= format.min && value <= format.max);
      assert.deepEqual(Array.from(tf.grad((w) => fakeQuantize(w, settings, 1).sum())(weights).dataSync()), [1, 1, 1, 1, 1]);
      assert.deepEqual(Array.from(tf.grad((w) => fakeQuantize(w, settings, 0.5).sum())(weights).dataSync()), [1, 1, 1, 1, 1]);
    });
    const sim = await create({ qat: { ...DEFAULT_QAT, format: format.value, schedule: "immediate" } });
    for (let i = 0; i < 3; i++) sim.trainOne();
    assert.ok(Number.isFinite(sim.latest.loss));
    assert.ok(sim.latest.maxNormError < 1e-6);
    assert.ok(sim.latest.unusedGradientError < 1e-6);
    for (const value of sim.latest.effectivePositions) close(value / sim.latest.embeddingScale, Math.round(value / sim.latest.embeddingScale));
    assert.ok(Number.isFinite(probeAt(sim.latest, [RADIUS, 0, 0]).magnitude));
  }
  const schedule = { format: "sym-int3", schedule: "linear", start: 3, duration: 4 };
  assert.deepEqual([2, 3, 5, 7, 9].map((step) => blendAt(schedule, step)), [0, 0, 0.5, 1, 1]);
  close(blendAt({ ...schedule, schedule: "cosine" }, 5), 0.5);
  assert.deepEqual([2, 3].map((step) => blendAt({ ...schedule, schedule: "immediate" }, step)), [0, 1]);
  const planned = await create({ qat: schedule });
  const resumed = await create({});
  for (let i = 0; i < 3; i++) { compare(planned, resumed); planned.trainOne(); resumed.trainOne(); }
  planned.insertLetter([0, RADIUS, 0]);
  resumed.insertLetter([0, RADIUS, 0]);
  const saved = resumed.history.slice();
  const savedEvents = JSON.stringify(resumed.events);
  resumed.applyQat(schedule);
  assert.equal(resumed.step, 3);
  assert.equal(resumed.history.length, saved.length + 1);
  for (let i = 0; i < saved.length; i++) assert.equal(resumed.history[i], saved[i]);
  assert.equal(saved[0].qat.format, "fp32");
  assert.equal(JSON.stringify(resumed.events), savedEvents);
  for (let i = 0; i < 7; i++) { compare(planned, resumed); planned.trainOne(); resumed.trainOne(); }
  // Full immediate switch verifies cached gradients were replaced before the next update.
  const instant = await create({ qat: { ...schedule, schedule: "immediate", start: 0 } });
  const attached = await create({});
  attached.applyQat({ ...schedule, schedule: "immediate", start: 0 });
  compare(instant, attached); instant.trainOne(); attached.trainOne(); compare(instant, attached);
  const single = await create({ targetedTokens: 1, untargetedTokens: 0, batchSize: 1, optimizer: "rmsprop", qat: { ...DEFAULT_QAT, format: "ternary", schedule: "immediate" } });
  single.trainOne();
  assert.equal(single.latest.loss, 0);
  assert.equal(single.latest.accuracy, 1);
  assert.throws(() => single.applyQat({ ...schedule, duration: 0 }));
  assert.throws(() => single.applyQat({ ...schedule, start: 49_999, duration: 2 }));
  const transition = single.applyQat({ ...schedule, start: single.step });
  const transitionFrame = single.history.length - 1;
  const insertion = single.insertLetter([RADIUS, 0, 0]);
  assert.equal(single.history[transitionFrame], transition);
  assert.equal(single.history[transitionFrame].qatChanged, true);
  assert.equal(insertion.frame, transitionFrame + 1);
  assert.equal(single.history[insertion.frame].activeVocab, 2);

  const policy = (token, start, mode, period = 4, percent = 50) => ({ token, start, mode, period, percent });
  const state = (sim) => ({
    weights: Object.fromEntries(Object.entries(sim.variables).map(([name, value]) => [name, Array.from(value.dataSync())])),
    first: structuredClone(sim.firstMoment), second: structuredClone(sim.secondMoment), updates: sim.latest.optimizerStep,
  });
  for (const qat of [DEFAULT_QAT, { ...DEFAULT_QAT, format: "sym-int3", schedule: "immediate" }]) {
    const drop = await create({ qat });
    for (let i = 0; i < 3; i++) drop.trainOne();
    const before = state(drop);
    const historical = drop.history.slice();
    drop.scheduleTarget(policy(2, drop.step, "exclude"));
    assert.deepEqual(state(drop), before, "Membership change must retain all weights and optimizer state");
    assert.deepEqual(Array.from(drop.inputs.dataSync()), [0, 1, 3, 0, 1, 3, 0, 1]);
    assert.deepEqual(Array.from(drop.targets.dataSync()), [1, 3, 0, 1, 3, 0, 1, 3]);
    assert.deepEqual(drop.latest.hiddenMeans.map(item => item.target), [0,1,3]);
    for (const item of drop.latest.hiddenMeans) {
      const indices = [1,3,0,1,3,0,1,3].flatMap((target, index) => target === item.target ? [index] : []);
      assert.equal(item.count, indices.length);
      for (let d = 0; d < 3; d++) close(item.mean[d], indices.reduce((sum, i) => sum + drop.latest.hidden[i * 3 + d], 0) / indices.length);
      close(Math.hypot(...item.projected), RADIUS);
    }
    assert.deepEqual(historical.at(-1).hiddenMeans.map(item => item.target), [0,1,2,3]);
    assert.deepEqual(Array.from(drop.latest.targetMask), [1, 1, 0, 1, 0, 0, 0]);
    assert.ok(drop.latest.unusedGradientError < 1e-6, "Dropped numeric rows must satisfy the denominator-only gradient");
    for (let i = 0; i < historical.length; i++) assert.equal(drop.history[i], historical[i]);
    assert.equal(historical.at(-1).targetMask[2], 1);
    drop.trainOne();
    const trained = state(drop);
    drop.scheduleTarget(policy(2, drop.step, "include"));
    assert.deepEqual(state(drop), trained);
    assert.equal(drop.latest.includedTargetCount, 4);
    assert.deepEqual(drop.latest.hiddenMeans.map(item => item.target), [0,1,2,3]);
    assert.equal(drop.latest.targetCount, 4);
    assert.equal(drop.activeVocab, 7);
  }
  const timed = await create({});
  const manual = await create({});
  timed.scheduleTarget(policy(2, 2, "exclude"));
  timed.scheduleTarget(policy(2, 5, "include"));
  for (let step = 0; step < 8; step++) {
    if (step === 2) manual.scheduleTarget(policy(2, step, "exclude"));
    if (step === 5) manual.scheduleTarget(policy(2, step, "include"));
    compare(timed, manual);
    timed.trainOne(); manual.trainOne();
  }
  for (const optimizer of ["adamw", "rmsprop"]) {
    const duty = await create({ targetedTokens: 1, untargetedTokens: 1, batchSize: 1, optimizer, qat: { ...DEFAULT_QAT, format: "ternary", duration: 4 } });
    duty.scheduleTarget(policy(0, 0, "duty"));
    assert.ok(duty.latest.loss > 0, "One target is not zero loss when the softmax has other rows");
    duty.trainOne(); duty.trainOne();
    assert.equal(duty.latest.includedTargetCount, 0);
    const idle = state(duty);
    assert.ok(Number.isNaN(duty.latest.loss) && Number.isNaN(duty.latest.accuracy));
    assert.equal(duty.latest.hidden.length, 0);
    assert.deepEqual(duty.latest.hiddenMeans, []);
    assert.equal(probeAt(duty.latest, [RADIUS, 0, 0]).magnitude, 0);
    duty.trainOne();
    assert.deepEqual(state(duty), idle);
    assert.ok(duty.latest.optimizerMoves.every((v) => v === 0));
    duty.trainOne();
    assert.deepEqual(state(duty), idle, "Idle intervals must not decay momentum or weights");
    assert.equal(duty.step, 4);
    assert.equal(duty.latest.qatBlend, 1);
    assert.equal(duty.latest.includedTargetCount, 1);
    duty.trainOne();
    assert.equal(duty.latest.optimizerStep, 3);
    for (let i = 0; i < 15; i++) duty.trainOne();
  }
  assert.equal(dutyOnSteps(3, 50), 2);
  for (const percent of [0, 25, 50, 100]) {
    const rule = { ...policy(0, 2, "duty", 4, percent), id: 1 };
    const count = Array.from({ length: 12 }, (_, i) => includedTargets([rule], 1, i + 2).length).reduce((a, b) => a + b, 0);
    assert.equal(count, dutyOnSteps(4, percent) * 3);
  }
  const priority = await create({});
  const future = priority.scheduleTarget(policy(1, 4, "exclude"));
  const ruleFrame = priority.latest;
  priority.cancelTargetRule(future.id);
  assert.equal(ruleFrame.targetRules.length, 1);
  assert.equal(priority.latest.targetRules.length, 0);
  priority.scheduleTarget(policy(1, 0, "exclude"));
  priority.scheduleTarget(policy(1, 0, "include"));
  assert.equal(priority.latest.targetMask[1], 1);
  priority.trainOne();
  assert.throws(() => priority.scheduleTarget(policy(1, 0, "exclude")));
  assert.throws(() => priority.scheduleTarget(policy(1, 1, "duty", 0)));
  assert.throws(() => priority.scheduleTarget(policy(1, 1, "duty", 4, 101)));
  const wide = await create({ targetedTokens: 100, untargetedTokens: 100, batchSize: 1 });
  wide.scheduleTarget(policy(99, 0, "exclude"));
  assert.equal(wide.activeUntargetedCount, 100);
  assert.equal(wide.latest.includedTargetCount, 99);
  assert.equal(wide.insertLetter([RADIUS, 0, 0]), null);
  wide.trainOne();
  assert.ok(wide.latest.unusedGradientError < 1e-6);
  for (const sim of simulations) sim.dispose();
  simulations.length = 0;
  assert.equal(tf.memory().numTensors, baseline, "All simulator tensors must be disposed");
  console.log("PASS: target-conditioned hidden means/projection/empty means/replay; QAT formats/STE/resume; dataset exclusion/restoration; exact schedule boundaries; duty-cycle counts; empty-step optimizer preservation; QAT during idle steps; history; precedence/cancellation; 100-token capacity; tensor cleanup.");
} finally {
  for (const sim of simulations) sim.dispose();
  await rm(temp, { recursive: true, force: true });
}
