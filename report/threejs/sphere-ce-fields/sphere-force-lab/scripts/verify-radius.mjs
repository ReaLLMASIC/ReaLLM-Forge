import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import path from "node:path";
import ts from "typescript";
import * as tf from "@tensorflow/tfjs";

const root = path.resolve(import.meta.dirname, "..");
await mkdir(path.join(root, ".tmp"), { recursive: true });
const temp = await mkdtemp(path.join(root, ".tmp/radius-check-"));
const simulations = [];
const close = (actual, expected, tolerance = 2e-6) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const assign = (variable, values) => tf.tidy(() => variable.assign(tf.tensor(values, variable.shape)));
try {
  const modules = ["dataset", "architecture", "quantization", "target-schedule", "hidden-means", "view-projection", "simulator"];
  for (const name of modules) {
    const source = await readFile(path.join(root, "lib", `${name}.ts`), "utf8");
    let output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    for (const module of modules) output = output.replaceAll(`from "./${module}"`, `from "./${module}.mjs"`);
    await writeFile(path.join(temp, `${name}.mjs`), output);
  }
  const { TransformerSphereSimulation: Simulation, probeAt } = await import(pathToFileURL(path.join(temp, "simulator.mjs")));
  const { toDisplaySnapshot, displayExtent, displayedRowForce, supportsSphericalProbe } = await import(pathToFileURL(path.join(temp, "view-projection.mjs")));
  const { uniformMatrix } = await import(pathToFileURL(path.join(temp, "dataset.mjs")));
  await tf.setBackend("cpu");
  const baseline = tf.memory().numTensors;
  const config = { seed: 17, optimizer: "adamw", learningRate: 0.018, weightDecay: 0.05, targetedTokens: 4, untargetedTokens: 2, batchSize: 2, architecture: { modelDim: 5, maxContextLength: 6 } };
  const create = async (settings = {}) => {
    const sim = await Simulation.create({ ...config, ...settings });
    simulations.push(sim); return sim;
  };
  const state = sim => Object.fromEntries(Object.entries(sim.variables).map(([name, v]) => [name, Array.from(v.dataSync())]));
  const policy = (token, start, mode) => ({ token, start, mode, period: 4, percent: 50 });

  // Omitted mode preserves fixed-sphere training exactly.
  const implicit = await create(), fixed = await create({ radiusMode: "fixed" });
  const free = await create({ radiusMode: "free" });
  assert.deepEqual(free.latest.positions, fixed.latest.positions, "Matched initialization");
  for (let i = 0; i < 4; i++) {
    implicit.trainOne(); fixed.trainOne();
    assert.deepEqual(state(implicit), state(fixed));
    assert.ok(fixed.latest.maxNormError < 3e-7);
  }
  assert.equal(fixed.latest.radiusMode, "fixed");
  assert.equal(supportsSphericalProbe(fixed.latest), true);
  assert.equal(supportsSphericalProbe(free.latest), false);
  assert.throws(() => probeAt(free.latest, [free.radius, 0, 0]), /unavailable/);

  // Independent AdamW oracle, including moments, bias correction, and decay.
  // Float32 rounding matches storage of moments/parameters, not the test logic.
  const count = free.activeVocab * free.modelDim;
  const first = new Float32Array(count), second = new Float32Array(count);
  const reserved = Array.from(free.variables.wte.dataSync().slice(count));
  const initial = free.latest, initialPositions = Array.from(initial.positions);
  for (let step = 1; step <= 5; step++) {
    const before = free.latest, expected = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const g = before.rawGradients[i];
      first[i] = 0.9 * first[i] + 0.1 * g;
      second[i] = 0.99 * second[i] + 0.01 * g * g;
      expected[i] = before.positions[i] * (1 - config.learningRate * config.weightDecay)
        - config.learningRate * (first[i] / (1 - 0.9 ** step)) / (Math.sqrt(second[i] / (1 - 0.99 ** step)) + 1e-8);
    }
    free.trainOne();
    for (let i = 0; i < count; i++) {
      close(free.latest.positions[i], expected[i], 2e-7);
      close(free.latest.optimizerMoves[i], free.latest.positions[i] - before.positions[i], 2e-7);
    }
    assert.equal(free.latest.projectionCorrection, 0);
    assert.ok(Number.isNaN(free.latest.maxNormError));
    assert.deepEqual(Array.from(free.variables.wte.dataSync().slice(count)), reserved);
  }
  assert.deepEqual(Array.from(initial.positions), initialPositions, "History is immutable");
  assert.equal(initial.radiusMode, "free");
  assert.equal(initial.optimizer, "adamw");
  assert.equal(initial.weightDecay, config.weightDecay);
  assert.ok(Array.from(free.latest.rowNorms).some(n => Math.abs(n - free.radius) > 0.01));

  // Current-radius radial/tangential split and full-D norms, even off the sphere.
  let sumForce = 0;
  for (let row = 0; row < free.activeVocab; row++) {
    const o = row * free.modelDim;
    const w = Array.from(free.latest.positions.slice(o, o + free.modelDim));
    const g = Array.from(free.latest.rawGradients.slice(o, o + free.modelDim));
    const norm = Math.hypot(...w), dot = w.reduce((s, x, i) => s + x * g[i], 0);
    close(free.latest.rowNorms[row], norm);
    sumForce += Math.hypot(...g);
    let tangentDot = 0;
    for (let i = 0; i < w.length; i++) {
      close(free.latest.tangentForces[o + i], -g[i] + dot / norm ** 2 * w[i], 2e-7);
      tangentDot += w[i] * free.latest.tangentForces[o + i];
    }
    close(tangentDot, 0, 2e-7);
  }
  close(free.latest.meanForce, sumForce / free.activeVocab);
  close(free.latest.minRowNorm, Math.min(...free.latest.rowNorms));
  close(free.latest.maxRowNorm, Math.max(...free.latest.rowNorms));
  assert.ok(free.latest.unusedGradientError < 1e-5);

  // One-class CE is zero: isolate the decay term without optimizer gradients.
  for (const optimizer of ["adamw", "rmsprop"]) {
    const settings = { radiusMode: "free", optimizer, targetedTokens: 1, untargetedTokens: 0, batchSize: 1, learningRate: 0.1, weightDecay: 0.5, architecture: { modelDim: 3, maxContextLength: 2 } };
    const decay = await create(settings), none = await create({ ...settings, weightDecay: 0 });
    const bounded = await create({ ...settings, radiusMode: "fixed" });
    const original = Array.from(decay.latest.positions);
    for (let step = 1; step <= 4; step++) {
      decay.trainOne(); none.trainOne(); bounded.trainOne();
      assert.equal(decay.latest.loss, 0);
      close(decay.latest.meanRowNorm, Math.sqrt(3) * 0.95 ** step);
      close(bounded.latest.meanRowNorm, Math.sqrt(3));
      assert.deepEqual(Array.from(none.latest.positions), original);
    }
  }

  // Free mode retains QAT, insertion, Markov sampling, removal/restoration,
  // duty schedules, and the all-excluded optimizer pause.
  const integrated = await create({ radiusMode: "free", dataset: { mode: "markov", matrix: uniformMatrix(4), sampling: "per-step", seed: 24 } });
  integrated.trainOne();
  const prior = Array.from(integrated.latest.positions);
  const row = integrated.activeVocab;
  integrated.insertLetter([1, 2, 3]);
  assert.deepEqual(Array.from(integrated.latest.positions.slice(0, prior.length)), prior);
  close(integrated.latest.rowNorms[row], integrated.radius);
  assert.ok(integrated.firstMoment.wte.slice(row * 5, (row + 1) * 5).every(v => v === 0));
  integrated.applyQat({ format: "int3", schedule: "immediate", start: integrated.step, duration: 1 });
  integrated.scheduleTarget(policy(1, integrated.step, "exclude"));
  integrated.scheduleTarget(policy(1, integrated.step + 2, "include"));
  integrated.scheduleTarget(policy(2, integrated.step, "duty"));
  assert.equal(integrated.latest.targetMask[1], 0);
  for (let i = 0; i < 3; i++) integrated.trainOne();
  assert.equal(integrated.latest.targetMask[1], 1);
  assert.equal(integrated.latest.qatBlend, 1);
  assert.ok(Number.isFinite(integrated.latest.loss));
  assert.ok(integrated.latest.unusedGradientError < 1e-5);
  for (let token = 0; token < 4; token++) integrated.scheduleTarget(policy(token, integrated.step, "exclude"));
  const idleState = state(integrated), idleStep = integrated.optimizerStep;
  integrated.trainOne();
  assert.deepEqual(state(integrated), idleState);
  assert.equal(integrated.optimizerStep, idleStep);
  assert.equal(integrated.latest.hiddenMeans.length, 0);
  integrated.scheduleTarget(policy(0, integrated.step, "include"));
  integrated.trainOne();
  assert.ok(Number.isFinite(integrated.latest.loss));
  assert.equal(integrated.latest.projectionCorrection, 0);

  // Renderers share force selection and extent fitting for rows and trails.
  const display = toDisplaySnapshot(free.latest);
  assert.deepEqual(displayedRowForce(display, 0), Array.from(free.latest.rawGradients.slice(0, 3), x => -x));
  assert.deepEqual(displayedRowForce(toDisplaySnapshot(fixed.latest), 0), Array.from(fixed.latest.tangentForces.slice(0, 3)));
  const far = { ...display, positions: new Float32Array(display.positions) };
  far.positions[0] = 20; far.positions[1] = 0; far.positions[2] = 0;
  close(displayExtent(display, [far, display], 1, true, false), 20);
  assert.ok(displayExtent(display, [far, display], 1, false, false) < 20);
  close(displayExtent({ ...display, hiddenMeans: [{ target: 0, count: 1, mean: [30, 0, 0], projected: null }] }, [], 0, false, true), 30);
  const hidden = free.latest.hiddenMeans[0];
  if (hidden.projected) close(Math.hypot(...hidden.projected), free.radius);
  const planar = await create({ radiusMode: "free", architecture: { modelDim: 2, maxContextLength: 3 } });
  planar.trainOne();
  const planarDisplay = toDisplaySnapshot(planar.latest);
  assert.equal(supportsSphericalProbe(planar.latest), false);
  for (let i = 2; i < planarDisplay.positions.length; i += 3) assert.equal(planarDisplay.positions[i], 0);

  // Origin convention is finite; no fictitious radial direction is introduced.
  const zero = Float32Array.from(planar.variables.wte.dataSync());
  zero.fill(0, 0, 2); assign(planar.variables.wte, zero);
  const atOrigin = planar.evaluate().snapshot;
  assert.equal(atOrigin.rowNorms[0], 0);
  assert.ok(atOrigin.tangentForces.every(Number.isFinite));
  close(atOrigin.tangentForces[0], -atOrigin.rawGradients[0]);
  close(atOrigin.tangentForces[1], -atOrigin.rawGradients[1]);

  const beforeInvalid = tf.memory().numTensors;
  for (const invalid of [{radiusMode:"bad"}, {weightDecay:-1}, {weightDecay:NaN}, {weightDecay:Infinity}, {learningRate:0}, {learningRate:NaN}, {optimizer:"bad"}]) {
    await assert.rejects(Simulation.create({ ...config, ...invalid }));
  }
  assert.equal(tf.memory().numTensors, beforeInvalid);
  for (const sim of simulations.splice(0)) sim.dispose();
  assert.equal(tf.memory().numTensors, baseline);
  console.log("PASS: fixed-mode regression; matched initialization; free AdamW oracle; decay-only AdamW/RMSProp; radial/tangential split; norm diagnostics; QAT/Markov/insertion/schedules/idle; history; 2D/higher-D display; trail fitting; origin; validation; tensor cleanup.");
} finally {
  for (const sim of simulations) sim.dispose();
  await rm(temp, { recursive: true, force: true });
}
