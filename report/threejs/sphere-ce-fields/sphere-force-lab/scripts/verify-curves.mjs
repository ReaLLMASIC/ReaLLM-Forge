import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import path from "node:path";
import ts from "typescript";
import * as tf from "@tensorflow/tfjs";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

const root = path.resolve(import.meta.dirname, "..");
await mkdir(path.join(root, ".tmp"), { recursive: true });
const temp = await mkdtemp(path.join(root, ".tmp/curves-check-"));
const simulations = [];
const close = (a, b, tolerance = 3e-6) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);
try {
  const names = ["dataset", "architecture", "quantization", "target-schedule", "hidden-means", "simulator", "metric-history", "utils"];
  const modules = [...names.map(name => `lib/${name}.ts`), "components/training-curves.tsx", "components/ui/button.tsx", "components/ui/slider.tsx"];
  for (const file of modules) {
    const relative = file.replace(/\.tsx?$/, ".mjs");
    const destination = path.join(temp, relative);
    await mkdir(path.dirname(destination), { recursive: true });
    let output = ts.transpileModule(await readFile(path.join(root, file), "utf8"), {
      compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
      fileName: file,
    }).outputText;
    for (const name of names) output = output.replaceAll(`from "./${name}"`, `from "./${name}.mjs"`);
    output = output.replace(/from "@\/([^"\n]+)"/g, (_, name) => `from "${pathToFileURL(path.join(temp, `${name}.mjs`)).href}"`);
    await writeFile(destination, output);
  }
  const { TransformerSphereSimulation: Simulation } = await import(pathToFileURL(path.join(temp, "lib/simulator.mjs")));
  const { uniformMatrix, compileMarkov, markovBatch } = await import(pathToFileURL(path.join(temp, "lib/dataset.mjs")));
  const { frameAtIteration, sampleMetric } = await import(pathToFileURL(path.join(temp, "lib/metric-history.mjs")));
  const { TrainingCurves } = await import(pathToFileURL(path.join(temp, "components/training-curves.mjs")));
  await tf.setBackend("cpu");
  const baseline = tf.memory().numTensors;
  const config = { seed: 17, optimizer: "adamw", learningRate: 0.01, weightDecay: 0.05, radiusMode: "free", targetedTokens: 4, untargetedTokens: 2, batchSize: 3, architecture: { modelDim: 5, maxContextLength: 9 } };
  const dataset = { mode: "markov", matrix: uniformMatrix(4), seed: 99, sampling: "per-step" };
  const create = async (settings = {}) => { const sim = await Simulation.create({ ...config, ...settings }); simulations.push(sim); return sim; };
  const state = sim => Object.fromEntries(Object.entries(sim.variables).map(([key, variable]) => [key, Array.from(variable.dataSync())]));
  const policy = (token, mode) => ({ token, mode, start: 0, period: 2, percent: 50 });

  // Validation CE and accuracy checked from logits, independent of the loss helper.
  const verifyReference = sim => {
    const { logits, labels } = tf.tidy(() => ({
      logits: Array.from(sim.forward(sim.validationInputs, sim.validationTargets).logits.dataSync()),
      labels: Array.from(sim.validationTargets.dataSync()),
    }));
    let ce = 0, correct = 0;
    for (let sample = 0; sample < labels.length; sample++) {
      const row = logits.slice(sample * sim.activeVocab, (sample + 1) * sim.activeVocab);
      const max = Math.max(...row);
      ce += Math.log(row.reduce((sum, value) => sum + Math.exp(value - max), 0)) + max - row[labels[sample]];
      correct += Number(row.indexOf(max) === labels[sample]);
    }
    close(sim.latest.validation.loss, ce / labels.length);
    close(sim.latest.validation.accuracy, correct / labels.length);
  };

  const direct = await create();
  assert.equal(direct.validationInputs, undefined, "No extra batch on the direct path");
  for (let step = 0; step < 3; step++) {
    assert.equal(direct.latest.validation.kind, "same-data");
    assert.equal(direct.latest.validation.loss, direct.latest.loss);
    assert.equal(direct.latest.validation.accuracy, direct.latest.accuracy);
    direct.trainOne();
  }
  const fresh = await create({ dataset }), fixed = await create({ dataset: { ...dataset, sampling: "fixed" } });
  const validationInput = fresh.validationInputs;
  const savedBatch = Array.from(validationInput.dataSync());
  assert.deepEqual(savedBatch, Array.from(fixed.validationInputs.dataSync()));
  assert.notDeepEqual(savedBatch, Array.from(fresh.inputs.dataSync()));
  const expected = markovBatch(compileMarkov(dataset.matrix, [0,1,2,3]), [0,1,2,3], 3, 9, (99 ^ 0xa511e9b3) >>> 0, 0);
  assert.deepEqual(savedBatch, Array.from(expected.inputs));
  assert.equal(fresh.latest.validation.kind, "independent-markov");
  assert.equal(fresh.latest.validation.sampleCount, 27);
  const historical = fresh.latest, historicalMetrics = { ...fresh.latest.validation };
  const before = state(fresh), moments = structuredClone(fresh.firstMoment), trainBatch = Array.from(fresh.inputs.dataSync());
  fresh.evaluate(); fresh.evaluate();
  assert.deepEqual(state(fresh), before, "Evaluation must not train");
  assert.deepEqual(fresh.firstMoment, moments);
  assert.deepEqual(Array.from(fresh.inputs.dataSync()), trainBatch);
  for (let i = 0; i < 5; i++) {
    verifyReference(fresh);
    fresh.trainOne();
    assert.equal(fresh.validationInputs, validationInput, "Validation batch stays fixed");
  }
  assert.deepEqual(historical.validation, historicalMetrics, "Saved metrics are immutable");
  fresh.applyQat({ format: "int3", schedule: "immediate", start: fresh.step, duration: 1 });
  fresh.insertLetter([1,0,0]);
  verifyReference(fresh);
  assert.equal(fresh.validationInputs, validationInput);

  // Membership changes rebuild the same fixed stream; restoration restores it.
  const scheduled = await create({ dataset });
  const originalValidation = Array.from(scheduled.validationInputs.dataSync());
  scheduled.scheduleTarget(policy(3, "exclude"));
  assert.ok(!Array.from(scheduled.validationInputs.dataSync()).includes(3));
  assert.ok(!Array.from(scheduled.validationTargets.dataSync()).includes(3));
  verifyReference(scheduled);
  scheduled.scheduleTarget(policy(3, "include"));
  assert.deepEqual(Array.from(scheduled.validationInputs.dataSync()), originalValidation);
  for (let i = 0; i < 4; i++) scheduled.scheduleTarget(policy(i, "exclude"));
  assert.equal(scheduled.validationInputs, undefined);
  assert.equal(scheduled.latest.validation.kind, "empty");
  assert.ok(Number.isNaN(scheduled.latest.validation.loss));
  assert.ok(Number.isNaN(scheduled.latest.validation.accuracy));
  assert.equal(scheduled.latest.validation.sampleCount, 0);
  scheduled.scheduleTarget(policy(0, "include"));
  assert.equal(scheduled.latest.validation.kind, "same-data");
  scheduled.scheduleTarget(policy(1, "include"));
  assert.equal(scheduled.latest.validation.kind, "independent-markov");
  verifyReference(scheduled);

  // Actual iteration mapping, including edits at the same step and boundaries.
  const steps = [0,0,1,1,2,8,8].map(step => ({ step }));
  for (const [iteration, frame] of [[-5,1], [0,1], [1,3], [1.6,4], [7,4], [8,6], [99,6]]) assert.equal(frameAtIteration(steps, iteration), frame);
  assert.equal(frameAtIteration([], 0), 0);
  const fake = (step, loss) => ({ ...direct.latest, step, loss, accuracy: step % 2, validation: { ...direct.latest.validation, loss, accuracy: 0.5 } });
  const long = Array.from({ length: 50_001 }, (_, i) => fake(i, i === 1111 ? 20 : i === 2222 ? 0 : 1));
  const compressed = sampleMetric(long, "loss");
  assert.ok(compressed.length <= 4 * 240);
  assert.equal(compressed[0].frame, 0); assert.equal(compressed.at(-1).frame, 50_000);
  assert.ok(compressed.some(point => point.frame === 1111 && point.value === 20), "Narrow spikes survive sampling");
  assert.ok(compressed.some(point => point.frame === 2222 && point.value === 0));
  const gaps = sampleMetric([fake(0,1),fake(1,NaN),fake(2,2),fake(3,3),fake(4,NaN),fake(5,4)], "loss", 1);
  assert.deepEqual(gaps.filter(point => point.move).map(point => point.step), [0,2,5]);
  assert.deepEqual(sampleMetric([fake(0,NaN)], "loss"), []);
  assert.equal(sampleMetric([fake(0,1)], "accuracy")[0].value, 0);

  // Render the real component: ordered panels, a real accessible slider,
  // and stable full-run paths/scales when the cursor moves backward.
  const chartHistory = [fake(0,3),fake(1,2),fake(2,NaN),fake(3,1),fake(3,0.8),fake(4,0.7)];
  const render = frame => renderToStaticMarkup(React.createElement(TrainingCurves, { history: chartHistory, frame, revision: 1, disabled: false, onSeek() {} }));
  const firstHtml = render(0), lastHtml = render(5);
  assert.ok(firstHtml.indexOf('data-metric-chart="loss"') < firstHtml.indexOf('data-metric-chart="validationLoss"'));
  assert.ok(firstHtml.indexOf('data-metric-chart="validationLoss"') < firstHtml.indexOf('data-metric-chart="accuracy"'));
  assert.ok(firstHtml.indexOf('data-metric-chart="accuracy"') < firstHtml.indexOf('aria-label="Metric iteration"'));
  assert.match(firstHtml, /role="slider"/);
  // Radix assigns aria-valuenow after mounting its thumb collection in the DOM.
  assert.match(firstHtml, /aria-valuetext="Iteration 0 of 4"/); assert.match(lastHtml, /aria-valuetext="Iteration 4 of 4"/);
  assert.equal((lastHtml.match(/data-selected-iteration="4"/g) ?? []).length, 3);
  const paths = html => [...html.matchAll(/<path d="([^"]*)" fill="none" stroke=/g)].map(match => match[1]);
  assert.equal(paths(firstHtml).length, 4);
  assert.deepEqual(paths(firstHtml), paths(lastHtml), "Scrubbing never truncates or rescales the recorded curves");
  assert.match(firstHtml, /no separate held-out validation set/);

  for (const sim of simulations.splice(0)) sim.dispose();
  assert.equal(tf.memory().numTensors, baseline);
  console.log("PASS: independent fixed validation CE/accuracy; read-only evaluation; deterministic fast path; QAT/insertion/membership/empty phases; history; exact iteration mapping; gap/extrema preservation; stacked chart rendering and accessible slider; stable full-run curves; tensor cleanup.");
} finally {
  for (const sim of simulations) sim.dispose();
  await rm(temp, { recursive: true, force: true });
}
