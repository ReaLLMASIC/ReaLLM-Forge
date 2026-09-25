"use client";

import * as tf from "@tensorflow/tfjs";

export const D_MODEL = 3;
export const RADIUS = Math.sqrt(D_MODEL);
export const DIGITS = Array.from({ length: 10 }, (_, i) => String(i));
export const LETTERS = Array.from({ length: 10 }, (_, i) => String.fromCharCode(97 + i));
export const TOKENS = [...DIGITS, ...LETTERS];

export type OptimizerKind = "adamw" | "rmsprop";

export type SimConfig = {
  seed: number;
  optimizer: OptimizerKind;
  learningRate: number;
  weightDecay: number;
};

export type InsertionEvent = {
  token: string;
  step: number;
  direction: [number, number, number];
};

export type Snapshot = {
  step: number;
  loss: number;
  accuracy: number;
  activeVocab: number;
  positions: number[];
  rawGradients: number[];
  tangentForces: number[];
  optimizerMoves: number[];
  hidden: number[];
  logPartition: number[];
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

function normalizeRows(data: Float32Array, rows: number, radius = RADIUS) {
  const out = new Float32Array(data);
  for (let i = 0; i < rows; i += 1) {
    const o = i * D_MODEL;
    const n = Math.hypot(out[o], out[o + 1], out[o + 2]);
    const s = n > 1e-12 ? radius / n : 0;
    if (s === 0) {
      out[o] = radius;
      out[o + 1] = 0;
      out[o + 2] = 0;
    } else {
      out[o] *= s;
      out[o + 1] *= s;
      out[o + 2] *= s;
    }
  }
  return out;
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

export type ProbeMode = "force" | "potential" | "probability" | "radial";

export function probeAt(snapshot: Snapshot, point: ArrayLike<number>) {
  const u = point;
  const nSamples = snapshot.logPartition.length;
  let gx = 0;
  let gy = 0;
  let gz = 0;
  let pressure = 0;
  let potential = 0;
  for (let s = 0; s < nSamples; s += 1) {
    const o = s * D_MODEL;
    const dot =
      u[0] * snapshot.hidden[o] +
      u[1] * snapshot.hidden[o + 1] +
      u[2] * snapshot.hidden[o + 2];
    const q = dot - snapshot.logPartition[s];
    const p = q >= 0 ? 1 / (1 + Math.exp(-q)) : Math.exp(q) / (1 + Math.exp(q));
    pressure += p;
    potential += q > 30 ? q : Math.log1p(Math.exp(q));
    gx += p * snapshot.hidden[o];
    gy += p * snapshot.hidden[o + 1];
    gz += p * snapshot.hidden[o + 2];
  }
  const inv = 1 / nSamples;
  gx *= inv;
  gy *= inv;
  gz *= inv;
  pressure *= inv;
  potential *= inv;
  const radialCoefficient = (u[0] * gx + u[1] * gy + u[2] * gz) / (RADIUS * RADIUS);
  const fx = -gx + radialCoefficient * u[0];
  const fy = -gy + radialCoefficient * u[1];
  const fz = -gz + radialCoefficient * u[2];
  return {
    force: [fx, fy, fz] as [number, number, number],
    magnitude: Math.hypot(fx, fy, fz),
    potential,
    probability: pressure,
    radial: -(u[0] * gx + u[1] * gy + u[2] * gz) / RADIUS,
  };
}

export class TransformerSphereSimulation {
  readonly config: SimConfig;
  readonly variables: NamedVars;
  readonly events: InsertionEvent[] = [];
  readonly history: Snapshot[] = [];
  private readonly inputs: tf.Tensor2D;
  private readonly targets: tf.Tensor2D;
  private readonly causalMask: tf.Tensor2D;
  private readonly firstMoment: Record<string, Float32Array> = {};
  private readonly secondMoment: Record<string, Float32Array> = {};
  private currentEval!: EvalState;
  private lastMoves = new Float32Array(20 * D_MODEL);
  private lastProjectionCorrection = 0;
  private optimizerStep = 0;
  step = 0;
  activeVocab = 10;

  private constructor(config: SimConfig, variables: NamedVars) {
    this.config = { ...config };
    this.variables = variables;
    const inputs: number[] = [];
    const targets: number[] = [];
    for (let b = 0; b < 10; b += 1) {
      for (let t = 0; t < 10; t += 1) {
        inputs.push((b + t) % 10);
        targets.push((b + t + 1) % 10);
      }
    }
    this.inputs = tf.tensor2d(inputs, [10, 10], "int32");
    this.targets = tf.tensor2d(targets, [10, 10], "int32");
    const mask = new Float32Array(100);
    for (let q = 0; q < 10; q += 1) {
      for (let k = 0; k < 10; k += 1) mask[q * 10 + k] = k > q ? -1e9 : 0;
    }
    this.causalMask = tf.tensor2d(mask, [10, 10]);
    for (const [name, variable] of Object.entries(this.variables)) {
      this.firstMoment[name] = new Float32Array(variable.size);
      this.secondMoment[name] = new Float32Array(variable.size);
    }
  }

  static async create(config: SimConfig) {
    await tf.setBackend("cpu");
    await tf.ready();
    const rng = mulberry32(config.seed);
    const make = (name: string, values: Float32Array, shape: number[]) =>
      tf.variable(tf.tensor(values, shape), true, name);
    const embedding = normalizeRows(randomArray(60, 1, rng), 20);
    const variables: NamedVars = {
      wte: make("wte", embedding, [20, D_MODEL]),
      wpe: make("wpe", randomArray(30, 0.08, rng), [10, D_MODEL]),
      wq: make("wq", randomArray(9, 0.42, rng), [D_MODEL, D_MODEL]),
      wk: make("wk", randomArray(9, 0.42, rng), [D_MODEL, D_MODEL]),
      wv: make("wv", randomArray(9, 0.42, rng), [D_MODEL, D_MODEL]),
      wo: make("wo", randomArray(9, 0.3, rng), [D_MODEL, D_MODEL]),
      w1: make("w1", randomArray(36, 0.35, rng), [D_MODEL, 12]),
      b1: make("b1", new Float32Array(12), [12]),
      w2: make("w2", randomArray(36, 0.24, rng), [12, D_MODEL]),
      b2: make("b2", new Float32Array(D_MODEL), [D_MODEL]),
      g1: make("g1", Float32Array.of(1, 1, 1), [D_MODEL]),
      g2: make("g2", Float32Array.of(1, 1, 1), [D_MODEL]),
      gf: make("gf", Float32Array.of(1, 1, 1), [D_MODEL]),
    };
    const sim = new TransformerSphereSimulation(config, variables);
    sim.currentEval = sim.evaluate();
    sim.history.push(sim.currentEval.snapshot);
    return sim;
  }

  private forward() {
    const v = this.variables;
    const linear = (input: tf.Tensor, weight: tf.Tensor2D, outputWidth: number) =>
      tf.matMul(input.reshape([100, weight.shape[0]]), weight).reshape([10, 10, outputWidth]);
    const x = tf.gather(v.wte, this.inputs).add(v.wpe.expandDims(0));
    const xn = rmsNorm(x, v.g1 as tf.Tensor1D);
    const q = linear(xn, v.wq as tf.Tensor2D, D_MODEL);
    const k = linear(xn, v.wk as tf.Tensor2D, D_MODEL);
    const values = linear(xn, v.wv as tf.Tensor2D, D_MODEL);
    const scores = tf.matMul(q, k, false, true).div(Math.sqrt(D_MODEL)).add(this.causalMask);
    const attention = tf.softmax(scores, -1);
    const h1 = x.add(linear(attention.matMul(values), v.wo as tf.Tensor2D, D_MODEL));
    const h1n = rmsNorm(h1, v.g2 as tf.Tensor1D);
    const ffHidden = gelu(linear(h1n, v.w1 as tf.Tensor2D, 12).add(v.b1));
    const ff = linear(ffHidden, v.w2 as tf.Tensor2D, D_MODEL).add(v.b2);
    const h2 = h1.add(ff);
    const hidden = rmsNorm(h2, v.gf as tf.Tensor1D);
    const activeRows = v.wte.slice([0, 0], [this.activeVocab, D_MODEL]);
    const logits = tf.matMul(hidden.reshape([100, D_MODEL]), activeRows, false, true)
      .reshape([10, 10, this.activeVocab]);
    const labels = tf.oneHot(this.targets, this.activeVocab);
    const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean() as tf.Scalar;
    return { hidden, logits, loss };
  }

  private evaluate(): EvalState {
    const variableList = Object.values(this.variables);
    const vg = tf.variableGrads(() => tf.tidy(() => this.forward().loss), variableList);
    const allGradients: Record<string, Float32Array> = {};
    for (const [name, variable] of Object.entries(this.variables)) {
      const gradient = vg.grads[variable.name];
      allGradients[name] = Float32Array.from(gradient.dataSync());
    }
    const loss = vg.value.dataSync()[0];
    vg.value.dispose();
    Object.values(vg.grads).forEach((g) => g.dispose());

    const diagnostics = tf.tidy(() => {
      const result = this.forward();
      const predictions = result.logits.argMax(-1);
      const accuracy = predictions.equal(this.targets).mean().dataSync()[0];
      const probabilities = tf.softmax(result.logits, -1);
      const unusedMass =
        this.activeVocab > 10
          ? probabilities.slice([0, 0, 10], [10, 10, this.activeVocab - 10]).sum(-1).mean().dataSync()[0]
          : 0;
      return {
        hidden: Array.from(result.hidden.reshape([100, D_MODEL]).dataSync()),
        logPartition: Array.from(tf.logSumExp(result.logits, -1).reshape([100]).dataSync()),
        accuracy,
        unusedMass,
      };
    });

    const positions = Array.from(this.variables.wte.dataSync().slice(0, this.activeVocab * D_MODEL));
    const raw = Array.from(allGradients.wte.slice(0, this.activeVocab * D_MODEL));
    const tangent: number[] = [];
    let forceSum = 0;
    let forceMax = 0;
    let normError = 0;
    let tangencyError = 0;
    for (let row = 0; row < this.activeVocab; row += 1) {
      const o = row * D_MODEL;
      const dot = positions[o] * raw[o] + positions[o + 1] * raw[o + 1] + positions[o + 2] * raw[o + 2];
      const c = dot / (RADIUS * RADIUS);
      const fx = -raw[o] + c * positions[o];
      const fy = -raw[o + 1] + c * positions[o + 1];
      const fz = -raw[o + 2] + c * positions[o + 2];
      tangent.push(fx, fy, fz);
      const fn = Math.hypot(fx, fy, fz);
      tangencyError = Math.max(tangencyError, Math.abs(positions[o] * fx + positions[o + 1] * fy + positions[o + 2] * fz) / (RADIUS * Math.max(fn, 1e-12)));
      forceSum += fn;
      forceMax = Math.max(forceMax, fn);
      normError = Math.max(normError, Math.abs(Math.hypot(positions[o], positions[o + 1], positions[o + 2]) - RADIUS));
    }
    let distanceSum = 0;
    let pairs = 0;
    for (let i = 10; i < this.activeVocab; i += 1) {
      for (let j = i + 1; j < this.activeVocab; j += 1) {
        const a = i * 3;
        const b = j * 3;
        distanceSum += Math.hypot(
          positions[a] - positions[b],
          positions[a + 1] - positions[b + 1],
          positions[a + 2] - positions[b + 2],
        );
        pairs += 1;
      }
    }
    const moves = Array.from(this.lastMoves.slice(0, this.activeVocab * D_MODEL));
    let unusedGradientError = 0;
    for (let row = 10; row < this.activeVocab; row += 1) {
      const ro = row * D_MODEL;
      let gx = 0, gy = 0, gz = 0;
      for (let sample = 0; sample < diagnostics.logPartition.length; sample += 1) {
        const ho = sample * D_MODEL;
        const logit = positions[ro] * diagnostics.hidden[ho] + positions[ro + 1] * diagnostics.hidden[ho + 1] + positions[ro + 2] * diagnostics.hidden[ho + 2];
        const probability = Math.exp(logit - diagnostics.logPartition[sample]);
        gx += probability * diagnostics.hidden[ho];
        gy += probability * diagnostics.hidden[ho + 1];
        gz += probability * diagnostics.hidden[ho + 2];
      }
      gx /= diagnostics.logPartition.length;
      gy /= diagnostics.logPartition.length;
      gz /= diagnostics.logPartition.length;
      unusedGradientError = Math.max(unusedGradientError, Math.hypot(gx - raw[ro], gy - raw[ro + 1], gz - raw[ro + 2]));
    }
    const snapshot: Snapshot = {
      step: this.step,
      loss,
      accuracy: diagnostics.accuracy,
      activeVocab: this.activeVocab,
      positions,
      rawGradients: raw,
      tangentForces: tangent,
      optimizerMoves: moves,
      hidden: diagnostics.hidden,
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
    const gradients = this.currentEval.allGradients;
    const previousRows = Float32Array.from(this.variables.wte.dataSync());
    this.optimizerStep += 1;
    let projectionCorrectionSquared = 0;
    for (const [name, variable] of Object.entries(this.variables)) {
      const values = Float32Array.from(variable.dataSync());
      const grad = gradients[name];
      const m = this.firstMoment[name];
      const s = this.secondMoment[name];
      for (let i = 0; i < values.length; i += 1) {
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
        const projected = normalizeRows(values, this.activeVocab);
        for (let i = 0; i < this.activeVocab * D_MODEL; i += 1) {
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
    for (let i = 0; i < this.activeVocab * D_MODEL; i += 1) this.lastMoves[i] = currentRows[i] - previousRows[i];
    this.lastProjectionCorrection = Math.sqrt(projectionCorrectionSquared);
    this.step += 1;
    this.currentEval = this.evaluate();
    this.history.push(this.currentEval.snapshot);
    return this.currentEval.snapshot;
  }

  insertLetter(direction: [number, number, number]) {
    if (this.activeVocab >= TOKENS.length) return null;
    const row = this.activeVocab;
    const norm = Math.hypot(...direction);
    if (norm < 1e-9) return null;
    const point: [number, number, number] = [
      (direction[0] / norm) * RADIUS,
      (direction[1] / norm) * RADIUS,
      (direction[2] / norm) * RADIUS,
    ];
    const values = Float32Array.from(this.variables.wte.dataSync());
    values.set(point, row * D_MODEL);
    const assigned = tf.tensor(values, this.variables.wte.shape);
    this.variables.wte.assign(assigned);
    assigned.dispose();
    this.firstMoment.wte.fill(0, row * D_MODEL, row * D_MODEL + D_MODEL);
    this.secondMoment.wte.fill(0, row * D_MODEL, row * D_MODEL + D_MODEL);
    this.lastMoves.fill(0, row * D_MODEL, row * D_MODEL + D_MODEL);
    const event = { token: TOKENS[row], step: this.step, direction: point } satisfies InsertionEvent;
    this.events.push(event);
    this.activeVocab += 1;
    this.currentEval = this.evaluate();
    this.history[this.history.length - 1] = this.currentEval.snapshot;
    return event;
  }

  get latest() {
    return this.currentEval.snapshot;
  }

  dispose() {
    Object.values(this.variables).forEach((v) => v.dispose());
    this.inputs.dispose();
    this.targets.dispose();
    this.causalMask.dispose();
  }
}
