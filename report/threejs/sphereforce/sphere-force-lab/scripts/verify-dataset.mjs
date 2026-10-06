import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile, rm, mkdir } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import path from "node:path";
import ts from "typescript";
import * as tf from "@tensorflow/tfjs";

const root = path.resolve(import.meta.dirname, "..");
await mkdir(path.join(root, ".sites-runtime"), { recursive: true });
const temp = await mkdtemp(path.join(root, ".sites-runtime/dataset-check-"));
const simulations = [];
try {
  const modules = ["dataset", "architecture", "quantization", "target-schedule", "hidden-means", "simulator"];
  for (const name of modules) {
    const source = await readFile(path.join(root, "lib", `${name}.ts`), "utf8");
    let output = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    for (const module of modules) output = output.replaceAll(`from "./${module}"`, `from "./${module}.mjs"`);
    await writeFile(path.join(temp, `${name}.mjs`), output);
  }
  const { TransformerSphereSimulation: Simulation } = await import(pathToFileURL(path.join(temp, "simulator.mjs")));
  const { cycleMatrix, uniformMatrix, normalizeMatrix, validateDataset, parseMatrix, compileMarkov, markovBatch } = await import(pathToFileURL(path.join(temp, "dataset.mjs")));
  await tf.setBackend("cpu");
  const baseline = tf.memory().numTensors;
  const config = { seed: 17, optimizer: "adamw", learningRate: .01, weightDecay: .05, targetedTokens: 4, untargetedTokens: 2, batchSize: 2, architecture: { modelDim: 5, maxContextLength: 7 } };
  const dataset = { mode: "markov", matrix: uniformMatrix(4), sampling: "per-step", seed: 99 };
  const create = async (settings = {}) => { const sim = await Simulation.create({ ...config, ...settings }); simulations.push(sim); return sim; };
  const batch = sim => ({ inputs: Array.from(sim.inputs.dataSync()), targets: Array.from(sim.targets.dataSync()) });
  const weights = sim => Object.fromEntries(Object.entries(sim.variables).map(([name, variable]) => [name, Array.from(variable.dataSync())]));
  const policy = (token, start, mode, period = 2, percent = 50) => ({ token, start, mode, period, percent });

  assert.deepEqual(parseMatrix("0, 1\n1, 0", 2), [[0,1],[1,0]]);
  assert.deepEqual(parseMatrix("[[0,1],[1,0]]", 2), [[0,1],[1,0]]);
  assert.deepEqual(normalizeMatrix([[2,6],[4,0]], 2), [[.25,.75],[1,0]]);
  for (const text of ["0,,1\n1,0", "0,\n1,0", "[[0,1],[1]]", "[[0,\"1\"],[1,0]]", "[[0,null],[1,0]]", "bad"]) assert.throws(() => parseMatrix(text,2));
  for (const matrix of [[[0,0],[1,0]], [[-.1,1.1],[1,0]], [[NaN,1],[1,0]], [[0,1],[Infinity,0]], [[.2,.2],[1,0]]]) assert.throws(() => validateDataset({mode:"markov",matrix},2,17));
  assert.throws(() => normalizeMatrix([[0,0],[1,0]],2));
  assert.throws(() => validateDataset({...dataset, seed:-1},4,17));
  assert.throws(() => validateDataset({...dataset, sampling:"invalid"},4,17));
  const immutable = validateDataset(dataset,4,17);
  assert.ok(Object.isFrozen(immutable.matrix[0]));
  assert.notEqual(immutable.matrix, dataset.matrix);
  assert.deepEqual(validateDataset({mode:"cycle",matrix:[[NaN]]},4,17),{mode:"cycle"});

  // Orientation and empirical conditional probabilities on an asymmetric chain.
  const asymmetric = compileMarkov([[0,.2,.8],[1,0,0],[1,0,0]],[0,1,2]);
  const sampled = markovBatch(asymmetric,[0,1,2],100,1000,12,0);
  const from0 = [], from1 = [], from2 = [];
  for (let i = 0; i < sampled.inputs.length; i++) [from0,from1,from2][sampled.inputs[i]].push(sampled.targets[i]);
  assert.ok(Math.abs(from0.filter(x=>x===1).length/from0.length - .2) < .01);
  assert.ok(from1.every(x=>x===0) && from2.every(x=>x===0));
  for (let b=0;b<100;b++) for(let t=0;t<999;t++) assert.equal(sampled.targets[b*1000+t],sampled.inputs[b*1000+t+1]);
  assert.deepEqual(sampled,markovBatch(asymmetric,[0,1,2],100,1000,12,0));
  assert.notDeepEqual(sampled,markovBatch(asymmetric,[0,1,2],100,1000,12,1));
  const conditioned = compileMarkov(uniformMatrix(4),[0,1,3]);
  assert.deepEqual(Array.from(conditioned.rows[0].destinations),[0,1,3]);
  assert.ok(Math.abs(conditioned.rows[0].cdf[0]-1/3)<1e-12);
  const absorbing = compileMarkov(cycleMatrix(4),[0,1,3]);
  assert.deepEqual(absorbing.fallbackRows,[1]);
  assert.deepEqual(Array.from(absorbing.rows[1].destinations),[1]);

  // Fast-path correctness: one-hot cycle matches direct training exactly.
  const direct = await create(), deterministic = await create({dataset:{...dataset,matrix:cycleMatrix(4)}});
  const directInput = direct.inputs, directTarget = direct.targets, directMask = direct.causalMask, deterministicInput = deterministic.inputs;
  for(let i=0;i<4;i++) {
    assert.deepEqual(batch(direct),batch(deterministic));
    assert.deepEqual(weights(direct),weights(deterministic));
    assert.deepEqual(direct.latest.rawGradients,deterministic.latest.rawGradients);
    direct.trainOne(); deterministic.trainOne();
  }
  assert.equal(direct.compiledMarkov,undefined);
  assert.equal(direct.inputs,directInput); assert.equal(direct.targets,directTarget); assert.equal(direct.causalMask,directMask);
  assert.equal(deterministic.inputs,deterministicInput);

  const fresh = await create({dataset}), twin = await create({dataset}), fixed = await create({dataset:{...dataset,sampling:"fixed"}});
  const fixedInput = fixed.inputs, oldBatch = batch(fresh), historical = fresh.latest, savedPreview = [...fresh.latest.dataset.preview];
  for(let i=0;i<3;i++) {
    assert.deepEqual(batch(fresh),batch(twin)); assert.deepEqual(weights(fresh),weights(twin));
    const previousMask = fresh.causalMask;
    fresh.trainOne(); twin.trainOne(); fixed.trainOne();
    assert.equal(fresh.causalMask,previousMask);
    assert.equal(fresh.latest.dataset.batchStep,i+1);
    assert.ok(Number.isFinite(fresh.latest.loss));
  }
  assert.notDeepEqual(batch(fresh),oldBatch); assert.equal(fixed.inputs,fixedInput);
  assert.deepEqual(historical.dataset.preview,savedPreview);
  const beforeEdit = batch(fresh), inputAtStep = fresh.inputs;
  fresh.applyQat({format:"sym-int3",schedule:"immediate",start:fresh.step,duration:1});
  fresh.insertLetter([1,0,0]);
  fresh.scheduleTarget(policy(3,fresh.step+2,"exclude"));
  assert.deepEqual(batch(fresh),beforeEdit); assert.equal(fresh.inputs,inputAtStep);
  for(const item of fresh.latest.hiddenMeans) {
    const labels = batch(fresh).targets;
    const indices = labels.flatMap((target,i)=>target===item.target?[i]:[]);
    assert.equal(item.count,indices.length);
    for(let d=0;d<5;d++) assert.ok(Math.abs(item.mean[d]-indices.reduce((sum,i)=>sum+fresh.latest.hidden[i*5+d],0)/indices.length)<1e-7);
  }
  fresh.trainOne(); fresh.trainOne();
  assert.equal(fresh.latest.targetMask[3],0);
  assert.ok(!batch(fresh).inputs.includes(3) && !batch(fresh).targets.includes(3));
  assert.ok(fresh.latest.unusedGradientError<1e-6);
  fresh.scheduleTarget(policy(3,fresh.step,"include"));
  assert.equal(fresh.latest.targetMask[3],1);
  assert.deepEqual(fresh.datasetConfig.matrix,dataset.matrix);

  const duty = await create({targetedTokens:1,untargetedTokens:0,batchSize:1,dataset:{...dataset,matrix:[[1]]}});
  duty.scheduleTarget(policy(0,0,"duty"));
  duty.trainOne(); assert.equal(duty.latest.includedTargetCount,0);
  const pausedWeights = weights(duty), pausedUpdates = duty.latest.optimizerStep;
  duty.trainOne(); assert.equal(duty.latest.includedTargetCount,1);
  assert.deepEqual(weights(duty),pausedWeights); assert.equal(duty.latest.optimizerStep,pausedUpdates);
  assert.equal(duty.latest.loss,0);
  const fallbackSim = await create({dataset:{...dataset,matrix:cycleMatrix(4)}});
  fallbackSim.scheduleTarget(policy(2,0,"exclude"));
  assert.deepEqual(fallbackSim.latest.dataset.fallbackRows,[1]);
  fallbackSim.scheduleTarget(policy(2,0,"include"));
  assert.deepEqual(fallbackSim.latest.dataset.fallbackRows,[]);
  const savedTensors = tf.memory().numTensors;
  await assert.rejects(()=>Simulation.create({...config,dataset:{...dataset,matrix:[[0]]}}));
  assert.equal(tf.memory().numTensors,savedTensors);
  for(const sim of simulations.splice(0)) sim.dispose();
  assert.equal(tf.memory().numTensors,baseline);
  console.log("PASS: matrix validation/parsing; conditional probabilities and chain alignment; reproducible sampling; direct/one-hot cache equivalence; per-step/fixed batches; same-step QAT/insertion stability; target dropout/restoration/duty cycles; hidden means; immutable replay; tensor cleanup.");
} finally {
  for(const sim of simulations) sim.dispose();
  await rm(temp,{recursive:true,force:true});
}
