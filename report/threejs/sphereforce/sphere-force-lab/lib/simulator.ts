"use client";

import * as tf from "@tensorflow/tfjs";
import { blendAt, DEFAULT_QAT, QatConfig, quantizationScale, quantizeValue, validateQat } from "./quantization";
import { includedTargets, TargetRule, TargetRuleDraft, validateTargetRule, describeTargetRule } from "./target-schedule";
import { averageHiddenByTarget, type HiddenMean } from "./hidden-means";

import { ArchitectureConfig, validateArchitecture, architectureParameterCount } from "./architecture";
import { CompiledMarkov, DatasetConfig, DatasetSnapshot, compileMarkov, markovBatch, validateDataset } from "./dataset";

export const D_MODEL = 3;
export const RADIUS = Math.sqrt(D_MODEL);
export const MAX_TARGET_TOKENS = 100;
export const MAX_UNTARGETED_TOKENS = 100;

export function tokenLabel(index: number, targetCount: number) {
  if (index < targetCount) return String(index);
  const unusedIndex = index - targetCount;
  return unusedIndex < 26 ? String.fromCharCode(97 + unusedIndex) : `u${unusedIndex + 1}`;
}

export type OptimizerKind = "adamw" | "rmsprop";

export type SimConfig = {
  seed: number;
  optimizer: OptimizerKind;
  learningRate: number;
  weightDecay: number;
  targetedTokens: number;
  untargetedTokens: number;
  batchSize: number;
  qat?: QatConfig;
  architecture?: Partial<ArchitectureConfig>;
  dataset?: DatasetConfig;
};

export type InsertionEvent = {
  token: string;
  step: number;
  frame: number;
  direction: [number, number, number];
};

export type Snapshot = {
  modelDim: number;
  radius: number;
  architecture: ArchitectureConfig;
  sequenceLength: number;
  parameterCount: number;
  dataset: DatasetSnapshot;
  step: number;
  optimizerStep: number;
  includedTargetCount: number;
  targetMask: Uint8Array;
  datasetChanged?: string;
  targetRules: readonly TargetRule[];
  loss: number;
  accuracy: number;
  targetCount: number;
  batchSize: number;
  activeVocab: number;
  positions: Float32Array;
  effectivePositions: Float32Array;
  qat: QatConfig;
  qatBlend: number;
  embeddingScale: number;
  quantizationRmse: number;
  qatChanged?: boolean;
  rawGradients: Float32Array;
  tangentForces: Float32Array;
  optimizerMoves: Float32Array;
  hidden: Float32Array;
  hiddenMeans: HiddenMean[];
  logPartition: Float32Array;
  unusedMass: number;
  meanTangentForce: number;
  maxTangentForce: number;
  meanLetterDistance: number;
  maxNormError: number;
  projectionCorrection: number;
  optimizerForceCosine: number;
  maxTangencyError: number;
  unusedGradientError: number;
};

type EvalState = {
  snapshot: Snapshot;
  allGradients: Record<string, Float32Array>;
};

type NamedVars = Record<string, tf.Variable>;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rng: () => number) {
  const u = Math.max(rng(), 1e-7);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randomArray(length: number, scale: number, rng: () => number) {
  return Float32Array.from({ length }, () => normal(rng) * scale);
}

function normalizeRows(data: Float32Array, rows: number, dimension: number) {
  const out = new Float32Array(data);
  const radius = Math.sqrt(dimension);
  for (let i = 0; i < rows; i += 1) {
    const o = i * dimension;
    let n = 0;
    for (let d = 0; d < dimension; d++) n += out[o + d] ** 2;
    n = Math.sqrt(n);
    for (let d = 0; d < dimension; d++) out[o + d] = n > 1e-12 ? out[o + d] * radius / n : (d === 0 ? radius : 0);
  }
  return out;
}

/** Adjacent-pair RoPE on [batch, heads, positions, even head width]. */
export function rotaryPositions(x: tf.Tensor4D) {
  const [batch, heads, length, width] = x.shape;
  const pairs = width / 2;
  const angles = Float32Array.from({ length: length * pairs }, (_, i) => Math.floor(i / pairs) * 10000 ** (-2 * (i % pairs) / width));
  const angle = tf.tensor(angles, [1, 1, length, pairs]);
  const paired = x.reshape([batch, heads, length, pairs, 2]);
  const even = paired.slice([0, 0, 0, 0, 0], [batch, heads, length, pairs, 1]).squeeze([4]);
  const odd = paired.slice([0, 0, 0, 0, 1], [batch, heads, length, pairs, 1]).squeeze([4]);
  const c = angle.cos(), s = angle.sin();
  return tf.stack([even.mul(c).sub(odd.mul(s)), even.mul(s).add(odd.mul(c))], -1).reshape(x.shape) as tf.Tensor4D;
}

function rmsNorm(x: tf.Tensor, gain: tf.Tensor1D) {
  return x.mul(x.square().mean(-1, true).add(1e-5).rsqrt()).mul(gain);
}

function gelu(x: tf.Tensor) {
  const c = Math.sqrt(2 / Math.PI);
  return x.mul(0.5).mul(x.add(x.pow(tf.scalar(3)).mul(0.044715)).mul(c).tanh().add(1));
}

function cosine(a: ArrayLike<number>, b: ArrayLike<number>) {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa > 0 && bb > 0 ? dot / Math.sqrt(aa * bb) : 0;
}

// Weight-only fake quantization with an identity straight-through estimator.
// The quantizer and its scale are detached; d[(1-a)w + a Q(w)]/dw := 1.
export function fakeQuantize(weight: tf.Tensor, config: QatConfig, alpha: number) {
  if (alpha === 0 || config.format === "fp32") return weight;
  const op = tf.customGrad((...inputs) => {
    const input = inputs[0] as tf.Tensor;
    const values = input.dataSync();
    const scale = quantizationScale(values, config.format);
    const effective = Float32Array.from(values, (value) =>
      (1 - alpha) * value + alpha * quantizeValue(value, config.format, scale));
    return { value: tf.tensor(effective, input.shape), gradFunc: (dy: tf.Tensor) => dy.clone() };
  });
  return op(weight);
}

export type ProbeMode = "force" | "potential" | "probability" | "radial";

export function probeAt(snapshot: Snapshot, point: ArrayLike<number>) {
  const u = point;
  const effective = Array.from(point, (value) => (1 - snapshot.qatBlend) * value + snapshot.qatBlend * quantizeValue(value, snapshot.qat.format, snapshot.embeddingScale));
  const nSamples = snapshot.logPartition.length;
  if (nSamples === 0) return { force: [0, 0, 0] as [number, number, number], magnitude: 0, potential: 0, probability: 0, radial: 0 };
  let gx = 0;
  let gy = 0;
  let gz = 0;
  let pressure = 0;
  let potential = 0;
  for (let s = 0; s < nSamples; s += 1) {
    const o = s * snapshot.modelDim;
    const dot =
      effective[0] * snapshot.hidden[o] +
      effective[1] * snapshot.hidden[o + 1] +
      effective[2] * (snapshot.modelDim > 2 ? snapshot.hidden[o + 2] : 0);
    const q = dot - snapshot.logPartition[s];
    const p = q >= 0 ? 1 / (1 + Math.exp(-q)) : Math.exp(q) / (1 + Math.exp(q));
    pressure += p;
    potential += q > 30 ? q : Math.log1p(Math.exp(q));
    gx += p * snapshot.hidden[o];
    gy += p * snapshot.hidden[o + 1];
    gz += p * (snapshot.modelDim > 2 ? snapshot.hidden[o + 2] : 0);
  }
  const inv = 1 / nSamples;
  gx *= inv;
  gy *= inv;
  gz *= inv;
  pressure *= inv;
  potential *= inv;
  const radialCoefficient = (u[0] * gx + u[1] * gy + u[2] * gz) / (snapshot.radius * snapshot.radius);
  const fx = -gx + radialCoefficient * u[0];
  const fy = -gy + radialCoefficient * u[1];
  const fz = -gz + radialCoefficient * u[2];
  return {
    force: [fx, fy, fz] as [number, number, number],
    magnitude: Math.hypot(fx, fy, fz),
    potential,
    probability: pressure,
    radial: -(u[0] * gx + u[1] * gy + u[2] * gz) / snapshot.radius,
  };
}

export class TransformerSphereSimulation {
  readonly config: SimConfig;
  readonly architecture: ArchitectureConfig;
  readonly modelDim: number;
  readonly radius: number;
  readonly variables: NamedVars;
  readonly events: InsertionEvent[] = [];
  readonly history: Snapshot[] = [];
  private inputs!: tf.Tensor2D;
  private targets!: tf.Tensor2D;
  private causalMask!: tf.Tensor2D;
  private included: number[] = [];
  readonly datasetConfig: DatasetConfig;
  private compiledMarkov?: CompiledMarkov;
  private datasetSnapshot!: DatasetSnapshot;
  private targetRules: readonly TargetRule[] = [];
  private nextRuleId = 1;
  private readonly firstMoment: Record<string, Float32Array> = {};
  private readonly secondMoment: Record<string, Float32Array> = {};
  private currentEval!: EvalState;
  private lastMoves: Float32Array;
  private lastProjectionCorrection = 0;
  private optimizerStep = 0;
  private qatConfig: QatConfig;
  step = 0;
  activeVocab: number;

  private constructor(config: SimConfig, variables: NamedVars) {
    this.architecture = validateArchitecture(config.architecture ?? { maxContextLength: config.targetedTokens }, config.batchSize, config.targetedTokens + MAX_UNTARGETED_TOKENS);
    this.modelDim = this.architecture.modelDim;
    this.radius = Math.sqrt(this.modelDim);
    this.datasetConfig = validateDataset(config.dataset, config.targetedTokens, config.seed);
    this.config = { ...config, architecture: this.architecture, dataset: this.datasetConfig };
    this.qatConfig = validateQat(config.qat ?? DEFAULT_QAT);
    this.activeVocab = config.targetedTokens + config.untargetedTokens;
    this.lastMoves = new Float32Array((config.targetedTokens + MAX_UNTARGETED_TOKENS) * this.modelDim);
    this.variables = variables;
    this.refreshDataset();
    for (const [name, variable] of Object.entries(this.variables)) {
      this.firstMoment[name] = new Float32Array(variable.size);
      this.secondMoment[name] = new Float32Array(variable.size);
    }
  }

  private refreshDataset() {
    const included = includedTargets(this.targetRules, this.config.targetedTokens, this.step);
    const membershipChanged = !this.inputs || included.length !== this.included.length || included.some((token, index) => token !== this.included[index]);
    const markov = this.datasetConfig.mode === "markov";
    if (markov && membershipChanged) this.compiledMarkov = compileMarkov(this.datasetConfig.matrix!, included);
    const freshBatch = markov && this.datasetConfig.sampling === "per-step" && !this.compiledMarkov!.deterministic;
    const batchStep = freshBatch ? this.step : 0;
    // Direct cycles and deterministic matrices keep their tensors. A QAT edit,
    // insertion, or reevaluation at the same step never redraws a random batch.
    if (!membershipChanged && (!freshBatch || this.datasetSnapshot.batchStep === batchStep)) return undefined;
    const removed = this.included.filter((token) => !included.includes(token));
    const restored = included.filter((token) => !this.included.includes(token));
    this.inputs?.dispose();
    this.targets?.dispose();
    this.included = included;
    const length = included.length ? this.architecture.maxContextLength : 0;
    let inputs: Int32Array, targets: Int32Array;
    if (markov) {
      ({ inputs, targets } = markovBatch(this.compiledMarkov!, included, this.config.batchSize, length, this.datasetConfig.seed!, batchStep));
    } else {
      // No transition matrix, CDF construction, or RNG on the direct path.
      inputs = new Int32Array(this.config.batchSize * length);
      targets = new Int32Array(this.config.batchSize * length);
      for (let b = 0; b < this.config.batchSize; b += 1) {
        for (let t = 0; t < length; t += 1) {
          inputs[b * length + t] = included[(b + t) % included.length];
          targets[b * length + t] = included[(b + t + 1) % included.length];
        }
      }
    }
    this.inputs = tf.tensor2d(inputs, [this.config.batchSize, length], "int32");
    this.targets = tf.tensor2d(targets, [this.config.batchSize, length], "int32");
    if (!this.causalMask || this.causalMask.shape[0] !== length) {
      this.causalMask?.dispose();
      const mask = new Float32Array(length * length);
      for (let q = 0; q < length; q += 1) {
        for (let k = 0; k < length; k += 1) mask[q * length + k] = k > q ? -1e9 : 0;
      }
      this.causalMask = tf.tensor2d(mask, [length, length]);
    }
    this.datasetSnapshot = Object.freeze({
      mode: this.datasetConfig.mode, sampling: this.datasetConfig.sampling ?? "fixed",
      seed: this.datasetConfig.seed ?? this.config.seed, batchStep,
      deterministic: this.compiledMarkov?.deterministic ?? true,
      fallbackRows: this.compiledMarkov?.fallbackRows ?? Object.freeze([]),
      preview: Object.freeze(length ? [...inputs.slice(0, length), targets[length - 1]] : []),
    });
    return [removed.length ? `Excluded ${removed.join(", ")}` : "", restored.length ? `Included ${restored.join(", ")}` : ""].filter(Boolean).join(" · ");
  }

  static async create(config: SimConfig) {
    validateQat(config.qat ?? DEFAULT_QAT);
    if (!Number.isInteger(config.targetedTokens) || config.targetedTokens < 1 || config.targetedTokens > MAX_TARGET_TOKENS) {
      throw new Error(`Targeted tokens must be an integer from 1 to ${MAX_TARGET_TOKENS}.`);
    }
    if (!Number.isInteger(config.untargetedTokens) || config.untargetedTokens < 0 || config.untargetedTokens > MAX_UNTARGETED_TOKENS) {
      throw new Error(`Untargeted tokens must be an integer from 0 to ${MAX_UNTARGETED_TOKENS}.`);
    }
    if (!Number.isInteger(config.batchSize) || config.batchSize < 1 || config.batchSize > 100) {
      throw new Error("Batch size must be an integer from 1 to 100.");
    }
    validateDataset(config.dataset, config.targetedTokens, config.seed);
    const architecture = validateArchitecture(config.architecture ?? { maxContextLength: config.targetedTokens }, config.batchSize, config.targetedTokens + MAX_UNTARGETED_TOKENS);
    const { modelDim: d, heads, qkHeadDim, attentionDim, mlpDim, layers, blockMode } = architecture;
    await tf.setBackend("cpu");
    await tf.ready();
    const rng = mulberry32(config.seed);
    const variableNamespace = `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 9)}`;
    const make = (name: string, values: Float32Array, shape: number[]) =>
      tf.tidy(() => tf.variable(tf.tensor(values, shape), true, `${name}_${variableNamespace}`));
    const totalRows = config.targetedTokens + MAX_UNTARGETED_TOKENS;
    const embedding = normalizeRows(randomArray(totalRows * d, 1, rng), totalRows, d);
    const variables: NamedVars = { wte: make("wte", embedding, [totalRows, d]) };
    const matrix = (name: string, rows: number, cols: number, scale: number) => { variables[name] = make(name, randomArray(rows * cols, scale, rng), [rows, cols]); };
    const gain = (name: string) => { variables[name] = make(name, new Float32Array(d).fill(1), [d]); };
    const bias = (name: string, width: number) => { variables[name] = make(name, new Float32Array(width), [width]); };
    if (architecture.positionEncoding === "absolute") matrix("wpe", architecture.maxContextLength, d, 0.08);
    for (let layer = 0; layer < layers; layer++) {
      const prefix = `block${layer}_`;
      if (blockMode !== "mlp") {
        matrix(prefix + "wq", d, heads * qkHeadDim, 0.42 * Math.sqrt(3 / d));
        matrix(prefix + "wk", d, heads * qkHeadDim, 0.42 * Math.sqrt(3 / d));
        matrix(prefix + "wv", d, attentionDim, 0.42 * Math.sqrt(3 / d));
        matrix(prefix + "wo", attentionDim, d, 0.3 * Math.sqrt(3 / attentionDim));
        gain(prefix + "g1");
      }
      if (blockMode !== "attention") {
        matrix(prefix + "w1", d, mlpDim, 0.35 * Math.sqrt(3 / d));
        bias(prefix + "b1", mlpDim);
        matrix(prefix + "w2", mlpDim, d, 0.24 * Math.sqrt(12 / mlpDim));
        bias(prefix + "b2", d);
        gain(prefix + "g2");
      }
    }
    gain("gf");
    const sim = new TransformerSphereSimulation({ ...config, architecture }, variables);
    try {
      sim.currentEval = sim.evaluate();
      sim.history.push(sim.currentEval.snapshot);
      return sim;
    } catch (cause) {
      sim.dispose();
      throw cause;
    }
  }

  private forward() {
    const alpha = blendAt(this.qatConfig, this.step);
    const v: Record<string, tf.Tensor> = {};
    for (const [name, variable] of Object.entries(this.variables)) {
      // Reserved, inactive token rows never set the embedding quantizer scale.
      const weight = name === "wte" ? variable.slice([0, 0], [this.activeVocab, this.modelDim]) : variable;
      v[name] = variable.rank === 2
        ? fakeQuantize(weight, this.qatConfig, alpha) : weight;
    }
    const batchSize = this.config.batchSize;
    const sequenceLength = this.inputs.shape[1];
    const sampleCount = batchSize * sequenceLength;
    const linear = (input: tf.Tensor, weight: tf.Tensor2D, outputWidth: number) =>
      tf.matMul(input.reshape([sampleCount, weight.shape[0]]), weight).reshape([batchSize, sequenceLength, outputWidth]);
    const a = this.architecture;
    let x = tf.gather(v.wte, this.inputs);
    if (a.positionEncoding === "absolute") x = x.add(v.wpe.slice([0, 0], [sequenceLength, this.modelDim]).expandDims(0));
    for (let layer = 0; layer < a.layers; layer++) {
      const p = `block${layer}_`;
      if (a.blockMode !== "mlp") {
        const xn = rmsNorm(x, v[p + "g1"] as tf.Tensor1D);
        const heads = (name: string, width: number) => linear(xn, v[p + name] as tf.Tensor2D, a.heads * width)
          .reshape([batchSize, sequenceLength, a.heads, width]).transpose([0, 2, 1, 3]) as tf.Tensor4D;
        let q = heads("wq", a.qkHeadDim), k = heads("wk", a.qkHeadDim);
        if (a.positionEncoding === "rope") { q = rotaryPositions(q); k = rotaryPositions(k); }
        const values = heads("wv", a.valueHeadDim);
        const scores = tf.matMul(q, k, false, true).div(Math.sqrt(a.qkHeadDim)).add(this.causalMask);
        const attended = tf.matMul(tf.softmax(scores, -1), values).transpose([0, 2, 1, 3])
          .reshape([batchSize, sequenceLength, a.attentionDim]);
        x = x.add(linear(attended, v[p + "wo"] as tf.Tensor2D, this.modelDim));
      }
      if (a.blockMode !== "attention") {
        const xn = rmsNorm(x, v[p + "g2"] as tf.Tensor1D);
        const pre = linear(xn, v[p + "w1"] as tf.Tensor2D, a.mlpDim).add(v[p + "b1"]);
        const activated = a.activation === "gelu" ? gelu(pre) : a.activation === "relu" ? pre.relu() : pre.relu().square();
        x = x.add(linear(activated, v[p + "w2"] as tf.Tensor2D, this.modelDim).add(v[p + "b2"]));
      }
    }
    const hidden = rmsNorm(x, v.gf as tf.Tensor1D);
    const activeRows = v.wte.slice([0, 0], [this.activeVocab, this.modelDim]);
    const logits = tf.matMul(hidden.reshape([sampleCount, this.modelDim]), activeRows, false, true)
      .reshape([batchSize, sequenceLength, this.activeVocab]);
    const loss = this.activeVocab === 1
      ? logits.sum().mul(0)
      : tf.losses.softmaxCrossEntropy(tf.oneHot(this.targets, this.activeVocab), logits).mean();
    return { hidden, logits, loss, effectiveRows: activeRows };
  }

  private evaluate(): EvalState {
    const targetMask = new Uint8Array(this.activeVocab);
    this.included.forEach((token) => { targetMask[token] = 1; });
    const hasData = this.included.length > 0;
    const sampleCount = this.config.batchSize * this.inputs.shape[1];
    const variableList = Object.values(this.variables);
    const allGradients: Record<string, Float32Array> = {};
    let loss = Number.NaN;
    if (hasData) {
      const vg = tf.variableGrads(() => tf.tidy(() => this.forward().loss as tf.Scalar), variableList);
      for (const [name, variable] of Object.entries(this.variables)) allGradients[name] = vg.grads[variable.name] ? Float32Array.from(vg.grads[variable.name].dataSync()) : new Float32Array(variable.size);
      loss = vg.value.dataSync()[0];
      vg.value.dispose();
      Object.values(vg.grads).forEach((g) => g.dispose());
    } else {
      for (const [name, variable] of Object.entries(this.variables)) allGradients[name] = new Float32Array(variable.size);
    }

    const diagnostics = tf.tidy(() => {
      if (!hasData) return {
        effectivePositions: Float32Array.from(fakeQuantize(this.variables.wte.slice([0, 0], [this.activeVocab, this.modelDim]), this.qatConfig, blendAt(this.qatConfig, this.step)).dataSync()),
        hidden: new Float32Array(0), logPartition: new Float32Array(0), accuracy: Number.NaN, unusedMass: Number.NaN,
      };
      const result = this.forward();
      const predictions = result.logits.argMax(-1);
      const accuracy = predictions.equal(this.targets).mean().dataSync()[0];
      const probabilities = tf.softmax(result.logits, -1);
      const unusedMass = probabilities.mul(tf.tensor1d(Array.from(targetMask, (value) => 1 - value))).sum(-1).mean().dataSync()[0];
      return {
        effectivePositions: Float32Array.from(result.effectiveRows.dataSync()),
        hidden: Float32Array.from(result.hidden.reshape([sampleCount, this.modelDim]).dataSync()),
        logPartition: Float32Array.from(tf.logSumExp(result.logits, -1).reshape([sampleCount]).dataSync()),
        accuracy,
        unusedMass,
      };
    });

    const positions = Float32Array.from(this.variables.wte.dataSync().slice(0, this.activeVocab * this.modelDim));
    const effectivePositions = diagnostics.effectivePositions;
    let quantizationError = 0;
    for (let i = 0; i < positions.length; i += 1) quantizationError += (positions[i] - effectivePositions[i]) ** 2;
    const raw = Float32Array.from(allGradients.wte.slice(0, this.activeVocab * this.modelDim));
    const tangent = new Float32Array(this.activeVocab * this.modelDim);
    let forceSum = 0;
    let forceMax = 0;
    let normError = 0;
    let tangencyError = 0;
    for (let row = 0; row < this.activeVocab; row += 1) {
      const o = row * this.modelDim;
      let dot = 0, norm2 = 0, force2 = 0, tangentDot = 0;
      for (let d = 0; d < this.modelDim; d++) { dot += positions[o + d] * raw[o + d]; norm2 += positions[o + d] ** 2; }
      const c = dot / (this.radius * this.radius);
      for (let d = 0; d < this.modelDim; d++) {
        const f = -raw[o + d] + c * positions[o + d];
        tangent[o + d] = f; force2 += f * f; tangentDot += positions[o + d] * f;
      }
      const fn = Math.sqrt(force2);
      tangencyError = Math.max(tangencyError, Math.abs(tangentDot) / (this.radius * Math.max(fn, 1e-12)));
      forceSum += fn;
      forceMax = Math.max(forceMax, fn);
      normError = Math.max(normError, Math.abs(Math.sqrt(norm2) - this.radius));
    }
    let distanceSum = 0;
    let pairs = 0;
    for (let i = 0; i < this.activeVocab; i += 1) {
      if (targetMask[i]) continue;
      for (let j = i + 1; j < this.activeVocab; j += 1) {
        if (targetMask[j]) continue;
        let distance2 = 0;
        for (let d = 0; d < this.modelDim; d++) distance2 += (positions[i * this.modelDim + d] - positions[j * this.modelDim + d]) ** 2;
        distanceSum += Math.sqrt(distance2);
        pairs += 1;
      }
    }
    const moves = Float32Array.from(this.lastMoves.slice(0, this.activeVocab * this.modelDim));
    let unusedGradientError = hasData ? 0 : Number.NaN;
    for (let row = 0; hasData && row < this.activeVocab; row += 1) {
      if (targetMask[row]) continue;
      const ro = row * this.modelDim;
      const expected = new Float64Array(this.modelDim);
      for (let sample = 0; sample < diagnostics.logPartition.length; sample++) {
        const ho = sample * this.modelDim;
        let logit = 0;
        for (let d = 0; d < this.modelDim; d++) logit += effectivePositions[ro + d] * diagnostics.hidden[ho + d];
        const probability = Math.exp(logit - diagnostics.logPartition[sample]);
        for (let d = 0; d < this.modelDim; d++) expected[d] += probability * diagnostics.hidden[ho + d];
      }
      let error2 = 0;
      for (let d = 0; d < this.modelDim; d++) error2 += (expected[d] / diagnostics.logPartition.length - raw[ro + d]) ** 2;
      unusedGradientError = Math.max(unusedGradientError, Math.sqrt(error2));
    }
    const snapshot: Snapshot = {
      modelDim: this.modelDim, radius: this.radius, architecture: this.architecture,
      sequenceLength: this.inputs.shape[1],
      parameterCount: architectureParameterCount(this.architecture, this.activeVocab),
      dataset: this.datasetSnapshot,
      step: this.step,
      optimizerStep: this.optimizerStep,
      includedTargetCount: this.included.length,
      targetMask,
      targetRules: this.targetRules,
      loss,
      accuracy: diagnostics.accuracy,
      targetCount: this.config.targetedTokens,
      batchSize: this.config.batchSize,
      activeVocab: this.activeVocab,
      positions,
      effectivePositions,
      qat: { ...this.qatConfig },
      qatBlend: blendAt(this.qatConfig, this.step),
      embeddingScale: quantizationScale(positions, this.qatConfig.format),
      quantizationRmse: Math.sqrt(quantizationError / positions.length),
      rawGradients: raw,
      tangentForces: tangent,
      optimizerMoves: moves,
      hidden: diagnostics.hidden,
      // Each valid position has equal CE weight. Read actual next-token labels
      // so removals, restorations, and future cycle changes cannot shift groups.
      hiddenMeans: averageHiddenByTarget(diagnostics.hidden, this.targets.dataSync(), this.config.targetedTokens, this.radius, this.modelDim),
      logPartition: diagnostics.logPartition,
      unusedMass: diagnostics.unusedMass,
      meanTangentForce: forceSum / this.activeVocab,
      maxTangentForce: forceMax,
      meanLetterDistance: pairs ? distanceSum / pairs : 0,
      maxNormError: normError,
      projectionCorrection: this.lastProjectionCorrection,
      optimizerForceCosine: cosine(moves, tangent),
      maxTangencyError: tangencyError,
      unusedGradientError,
    };
    return { snapshot, allGradients };
  }

  trainOne() {
    if (this.included.length === 0) {
      this.lastMoves.fill(0);
      this.lastProjectionCorrection = 0;
      return this.advanceClock();
    }
    const gradients = this.currentEval.allGradients;
    const previousRows = Float32Array.from(this.variables.wte.dataSync());
    this.optimizerStep += 1;
    let projectionCorrectionSquared = 0;
    for (const [name, variable] of Object.entries(this.variables)) {
      const values = Float32Array.from(variable.dataSync());
      const grad = gradients[name];
      const m = this.firstMoment[name];
      const s = this.secondMoment[name];
      for (let i = 0; i < (name === "wte" ? this.activeVocab * this.modelDim : values.length); i += 1) {
        const old = values[i];
        if (this.config.optimizer === "adamw") {
          m[i] = 0.9 * m[i] + 0.1 * grad[i];
          s[i] = 0.99 * s[i] + 0.01 * grad[i] * grad[i];
          const mHat = m[i] / (1 - 0.9 ** this.optimizerStep);
          const sHat = s[i] / (1 - 0.99 ** this.optimizerStep);
          values[i] = old * (1 - this.config.learningRate * this.config.weightDecay) -
            this.config.learningRate * mHat / (Math.sqrt(sHat) + 1e-8);
        } else {
          s[i] = 0.99 * s[i] + 0.01 * grad[i] * grad[i];
          values[i] = old * (1 - this.config.learningRate * this.config.weightDecay) -
            this.config.learningRate * grad[i] / (Math.sqrt(s[i]) + 1e-8);
        }
      }
      if (name === "wte") {
        const proposed = new Float32Array(values);
        const projected = normalizeRows(values, this.activeVocab, this.modelDim);
        for (let i = 0; i < this.activeVocab * this.modelDim; i += 1) {
          const d = projected[i] - proposed[i];
          projectionCorrectionSquared += d * d;
          values[i] = projected[i];
        }
      }
      const assigned = tf.tensor(values, variable.shape);
      variable.assign(assigned);
      assigned.dispose();
    }
    const currentRows = this.variables.wte.dataSync();
    this.lastMoves.fill(0);
    for (let i = 0; i < this.activeVocab * this.modelDim; i += 1) this.lastMoves[i] = currentRows[i] - previousRows[i];
    this.lastProjectionCorrection = Math.sqrt(projectionCorrectionSquared);
    return this.advanceClock();
  }

  private advanceClock() {
    this.step += 1;
    const changed = this.refreshDataset();
    this.currentEval = this.evaluate();
    this.currentEval.snapshot.datasetChanged = changed;
    this.history.push(this.currentEval.snapshot);
    return this.currentEval.snapshot;
  }

  scheduleTarget(draft: TargetRuleDraft) {
    const rule = { ...validateTargetRule(draft, this.config.targetedTokens, this.step), id: this.nextRuleId++ };
    this.targetRules = [...this.targetRules, rule];
    this.refreshDataset();
    this.currentEval = this.evaluate();
    this.currentEval.snapshot.datasetChanged = `Token ${rule.token}: ${describeTargetRule(rule)} at ${rule.start}${rule.start > this.step ? " (scheduled)" : ""}`;
    this.history.push(this.currentEval.snapshot);
    return rule;
  }

  cancelTargetRule(id: number) {
    const rule = this.targetRules.find((item) => item.id === id);
    if (!rule || rule.start <= this.step) throw new Error("Only future rules can be cancelled. Apply a new rule now to override an active rule.");
    this.targetRules = this.targetRules.filter((item) => item.id !== id);
    this.currentEval = this.evaluate();
    this.currentEval.snapshot.datasetChanged = `Cancelled token ${rule.token} rule at ${rule.start}`;
    this.history.push(this.currentEval.snapshot);
  }

  applyQat(config: QatConfig) {
    const previous = this.qatConfig;
    this.qatConfig = validateQat(config);
    try {
      const evaluation = this.evaluate();
      evaluation.snapshot.qatChanged = true;
      this.currentEval = evaluation;
      // Preserve the pre-switch frame as well as all optimizer moments and weights.
      this.history.push(evaluation.snapshot);
      return evaluation.snapshot;
    } catch (cause) {
      this.qatConfig = previous;
      throw cause;
    }
  }

  insertLetter(direction: [number, number, number]) {
    if (this.activeVocab - this.config.targetedTokens >= MAX_UNTARGETED_TOKENS) return null;
    const row = this.activeVocab;
    const visible = this.modelDim === 2 ? [direction[0], direction[1], 0] : direction;
    const norm = Math.hypot(...visible);
    if (norm < 1e-9) return null;
    const point: [number, number, number] = [
      (visible[0] / norm) * this.radius,
      (visible[1] / norm) * this.radius,
      (visible[2] / norm) * this.radius,
    ];
    const values = Float32Array.from(this.variables.wte.dataSync());
    values.fill(0, row * this.modelDim, (row + 1) * this.modelDim);
    for (let d = 0; d < Math.min(3, this.modelDim); d++) values[row * this.modelDim + d] = point[d];
    const assigned = tf.tensor(values, this.variables.wte.shape);
    this.variables.wte.assign(assigned);
    assigned.dispose();
    this.firstMoment.wte.fill(0, row * this.modelDim, row * this.modelDim + this.modelDim);
    this.secondMoment.wte.fill(0, row * this.modelDim, row * this.modelDim + this.modelDim);
    this.lastMoves.fill(0, row * this.modelDim, row * this.modelDim + this.modelDim);
    const event = { token: tokenLabel(row, this.config.targetedTokens), step: this.step, frame: this.history.length, direction: point } satisfies InsertionEvent;
    this.events.push(event);
    this.activeVocab += 1;
    this.currentEval = this.evaluate();
    this.history.push(this.currentEval.snapshot);
    return event;
  }

  get latest() {
    return this.currentEval.snapshot;
  }

  get activeUntargetedCount() {
    return this.activeVocab - this.config.targetedTokens;
  }

  dispose() {
    Object.values(this.variables).forEach((v) => v.dispose());
    this.inputs.dispose();
    this.targets.dispose();
    this.causalMask.dispose();
  }
}
