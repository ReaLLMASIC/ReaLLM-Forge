import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import path from "node:path";
import ts from "typescript";
import * as tf from "@tensorflow/tfjs";

const root = path.resolve(import.meta.dirname, "..");
await mkdir(path.join(root, ".sites-runtime"), { recursive: true });
const temp = await mkdtemp(path.join(root, ".sites-runtime/architecture-check-"));
const simulations = [];
try {
  const modules = ["dataset", "architecture", "quantization", "target-schedule", "hidden-means", "view-projection", "simulator"];
  for (const name of modules) {
    const source = await readFile(path.join(root, "lib", `${name}.ts`), "utf8");
    let output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    for (const module of modules) output = output.replaceAll(`from "./${module}"`, `from "./${module}.mjs"`);
    await writeFile(path.join(temp, `${name}.mjs`), output);
  }
  const { TransformerSphereSimulation: Simulation, rotaryPositions, probeAt } = await import(pathToFileURL(path.join(temp, "simulator.mjs")));
  const { validateArchitecture, architectureParameterCount } = await import(pathToFileURL(path.join(temp, "architecture.mjs")));
  const { toDisplaySnapshot } = await import(pathToFileURL(path.join(temp, "view-projection.mjs")));
  const { DEFAULT_QAT } = await import(pathToFileURL(path.join(temp, "quantization.mjs")));
  await tf.setBackend("cpu");
  const baseline = tf.memory().numTensors;
  const close = (a, b, tolerance = 2e-5) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
  const create = async (architecture, settings = {}) => {
    const sim = await Simulation.create({ seed: 17, optimizer: "adamw", learningRate: 0.008, weightDecay: 0.05, targetedTokens: 4, untargetedTokens: 2, batchSize: 2, architecture, ...settings });
    simulations.push(sim); return sim;
  };
  const forward = sim => tf.tidy(() => {
    const result = sim.forward();
    return { loss: result.loss.dataSync()[0], logits: Array.from(result.logits.dataSync()) };
  });
  const assign = (variable, values) => tf.tidy(() => { variable.assign(tf.tensor(values, variable.shape)); });
  const cases = [
    { modelDim: 5, layers: 2, heads: 2, qkHeadDim: 4, valueHeadDim: 3, mlpDim: 11, maxContextLength: 7 },
    { modelDim: 5, layers: 2, heads: 3, qkHeadDim: 2, valueHeadDim: 4, mlpDim: 13, activation: "relu2", positionEncoding: "rope", maxContextLength: 6 },
    { modelDim: 4, layers: 2, heads: 2, qkHeadDim: 4, valueHeadDim: 1, blockMode: "attention", positionEncoding: "rope", maxContextLength: 5 },
    { modelDim: 6, layers: 3, mlpDim: 9, activation: "relu", blockMode: "mlp", maxContextLength: 7 },
    { modelDim: 2, layers: 1, heads: 1, qkHeadDim: 3, valueHeadDim: 1, maxContextLength: 1 },
  ];
  for (const config of cases) {
    const sim = await create(config), copy = await create(config), a = sim.architecture, d = a.modelDim;
    assert.deepEqual(sim.latest.positions, copy.latest.positions);
    assert.deepEqual(sim.latest.rawGradients, copy.latest.rawGradients);
    close(sim.latest.loss, copy.latest.loss);
    assert.equal(sim.latest.hidden.length, 2 * a.maxContextLength * d);
    assert.equal(Object.values(sim.variables).reduce((sum, variable) => sum + variable.size, 0), architectureParameterCount(a, 104));
    assert.equal(Boolean(sim.variables.block0_wq), a.blockMode !== "mlp");
    assert.equal(Boolean(sim.variables.block0_w1), a.blockMode !== "attention");
    assert.equal(Boolean(sim.variables.wpe), a.positionEncoding === "absolute");
    for (const gradient of Object.values(sim.currentEval.allGradients)) assert.ok(gradient.every(Number.isFinite));
    // Causality and MLP position independence, tested using input intervention.
    if (a.maxContextLength > 1) {
      const before = forward(sim).logits, savedInputs = sim.inputs;
      const values = Array.from(savedInputs.dataSync());
      values[a.maxContextLength - 1] = (values[a.maxContextLength - 1] + 1) % 4;
      sim.inputs = tf.tensor2d(values, savedInputs.shape, "int32");
      const after = forward(sim).logits;
      for (let i = 0; i < (a.maxContextLength - 1) * sim.activeVocab; i++) close(after[i], before[i], 1e-6);
      sim.inputs.dispose(); sim.inputs = savedInputs;
      if (a.blockMode === "mlp") {
        values[0] = (values[0] + 1) % 4;
        sim.inputs = tf.tensor2d(values, savedInputs.shape, "int32");
        const changed = forward(sim).logits;
        for (let i = sim.activeVocab; i < (a.maxContextLength - 1) * sim.activeVocab; i++) close(changed[i], before[i], 1e-6);
        sim.inputs.dispose(); sim.inputs = savedInputs;
      }
    }
    // Compare the largest-magnitude element in each weight gradient to an
    // independent central difference, including the second decoder block.
    if (a.positionEncoding === "rope" && a.blockMode === "full") {
      for (const name of ["block0_wq", "block0_wk", "block0_wv", "block0_wo", "block1_wq", "block1_w1", "block1_w2"]) {
        const variable = sim.variables[name], values = Array.from(variable.dataSync()), gradient = sim.currentEval.allGradients[name];
        const index = gradient.reduce((best, value, i) => Math.abs(value) > Math.abs(gradient[best]) ? i : best, 0);
        const original = values[index], epsilon = 0.002;
        values[index] = original + epsilon; assign(variable, values); const plus = forward(sim).loss;
        values[index] = original - epsilon; assign(variable, values); const minus = forward(sim).loss;
        values[index] = original; assign(variable, values);
        close((plus - minus) / (2 * epsilon), gradient[index], 0.002 + Math.abs(gradient[index]) * 0.025);
      }
    }
    const old = sim.latest, oldMean = structuredClone(old.hiddenMeans), oldPositions = new Float32Array(old.positions);
    sim.trainOne();
    assert.ok(Number.isFinite(sim.latest.loss));
    assert.ok(sim.latest.maxNormError < 1e-6);
    assert.ok(sim.latest.maxTangencyError < 2e-6);
    assert.ok(sim.latest.unusedGradientError < 2e-6);
    assert.deepEqual(old.hiddenMeans, oldMean); assert.deepEqual(old.positions, oldPositions);
    for (const item of sim.latest.hiddenMeans) {
      assert.equal(item.mean.length, d);
      close(Math.hypot(...item.projected), Math.sqrt(d));
      const labels = Array.from(sim.targets.dataSync());
      assert.equal(item.count, labels.filter(x => x === item.target).length);
    }
    const display = toDisplaySnapshot(sim.latest);
    assert.equal(display.positions.length, sim.activeVocab * 3);
    assert.equal(display.hidden, sim.latest.hidden);
    for (let row = 0; row < sim.activeVocab; row++) for (let c = 0; c < 3; c++) close(display.positions[row * 3 + c], c < d ? sim.latest.positions[row * d + c] : 0);
    if (d > 3) assert.ok(Math.hypot(...display.positions.slice(0, 3)) < sim.radius);
    sim.insertLetter([1, 2, 3]);
    const inserted = sim.latest.positions.slice(-d);
    close(Math.hypot(...inserted), sim.radius);
    for (let c = 3; c < d; c++) assert.equal(inserted[c], 0);
    sim.applyQat({ ...DEFAULT_QAT, format: "sym-int3", schedule: "immediate" });
    sim.trainOne(); assert.ok(sim.latest.unusedGradientError < 2e-6);
    assert.ok(Number.isFinite(probeAt(sim.latest, [sim.radius, 0, 0]).potential));
  }
  // RoPE identity, per-position norms, and shared-shift invariance.
  tf.tidy(() => {
    const values = [1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4];
    const q = rotaryPositions(tf.tensor4d(values, [1,1,4,4])).dataSync();
    const k = rotaryPositions(tf.tensor4d(values.map(x => 5-x), [1,1,4,4])).dataSync();
    assert.deepEqual(Array.from(q.slice(0,4)), values.slice(0,4));
    for (let p = 0; p < 4; p++) close(Math.hypot(...q.slice(p*4,p*4+4)), Math.sqrt(30));
    const dot = (i,j) => Array.from({length:4},(_,d)=>q[i*4+d]*k[j*4+d]).reduce((x,y)=>x+y,0);
    close(dot(0,1), dot(2,3));
  });
  const fixed = await create({ modelDim: 5, maxContextLength: 7 });
  fixed.scheduleTarget({ token: 2, start: 0, mode: "exclude", period: 4, percent: 50 });
  assert.equal(fixed.latest.sequenceLength, 7);
  assert.deepEqual(Array.from(fixed.inputs.dataSync()), [0,1,3,0,1,3,0, 1,3,0,1,3,0,1]);
  assert.deepEqual(Array.from(fixed.targets.dataSync()), [1,3,0,1,3,0,1, 3,0,1,3,0,1,3]);
  fixed.scheduleTarget({ token: 2, start: 0, mode: "include", period: 4, percent: 50 });
  assert.equal(fixed.latest.sequenceLength, 7);
  assert.equal(fixed.latest.hidden.length, 70);
  const short = await create({ maxContextLength: 1 }, { targetedTokens: 10, batchSize: 1 });
  assert.deepEqual(short.latest.hiddenMeans.map(x => x.target), [1]);
  const longest = await create({ modelDim: 2, blockMode: "mlp", mlpDim: 2, maxContextLength: 256 }, { targetedTokens: 1, untargetedTokens: 0, batchSize: 1 });
  longest.trainOne(); assert.equal(longest.latest.loss, 0); assert.equal(longest.latest.sequenceLength, 256);
  const fullD = fixed.latest;
  // Slice potential derivative: off-slice hidden coordinates do not enter the
  // inserted candidate's logit; partition still includes full-D existing rows.
  const u = [fullD.radius,0,0], epsilon = 0.001;
  const plus = [Math.sqrt(fullD.radius ** 2 - epsilon ** 2),epsilon,0], minus = [plus[0],-epsilon,0];
  close(-(probeAt(fullD,plus).potential-probeAt(fullD,minus).potential)/(2*epsilon), probeAt(fullD,u).force[1], 2e-5);
  for (const invalid of [{modelDim:1},{layers:0},{maxContextLength:257},{positionEncoding:"rope",qkHeadDim:3},{blockMode:"mlp",positionEncoding:"rope"},{heads:2,valueHeadDim:3,attentionDim:5}]) assert.throws(()=>validateArchitecture(invalid));
  assert.equal(validateArchitecture({modelDim:3,heads:2,qkHeadDim:4,valueHeadDim:3,positionEncoding:"rope"}).attentionDim,6);
  for (const sim of simulations.splice(0)) sim.dispose();
  assert.equal(tf.memory().numTensors, baseline, "No tensor leaks across architectures");
  console.log("PASS: configurable architecture matrix; finite-difference gradients; causal and MLP position independence; RoPE invariants; full-dimensional norms, means, diagnostics and display; QAT/insertion; fixed context cycles; bounds; tensor cleanup.");
} finally {
  for (const sim of simulations) sim.dispose();
  await rm(temp, { recursive: true, force: true });
}
