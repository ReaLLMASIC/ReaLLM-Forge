export type DatasetConfig = {
  mode: "cycle" | "markov";
  matrix?: readonly (readonly number[])[];
  sampling?: "per-step" | "fixed";
  seed?: number;
};

export type DatasetSnapshot = {
  mode: "cycle" | "markov";
  sampling: "per-step" | "fixed";
  seed: number;
  batchStep: number;
  deterministic: boolean;
  fallbackRows: readonly number[];
  /** First training sequence, including its final next-token label. */
  preview: readonly number[];
};

export function cycleMatrix(size: number): number[][] {
  return Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => Number(j === (i + 1) % size)));
}

export function uniformMatrix(size: number): number[][] {
  return Array.from({ length: size }, () => Array(size).fill(1 / size));
}

function checkShape(matrix: unknown, size: number): asserts matrix is number[][] {
  if (!Array.isArray(matrix) || matrix.length !== size || matrix.some(row => !Array.isArray(row) || row.length !== size)) {
    throw new Error(`The transition matrix must have ${size} rows and ${size} columns, ordered by token ID 0–${size - 1}.`);
  }
  for (let i = 0; i < size; i++) {
    if (matrix[i].some((p: unknown) => typeof p !== "number" || !Number.isFinite(p) || p < 0)) throw new Error(`Row ${i}: use finite, nonnegative probabilities.`);
  }
}

export function normalizeMatrix(matrix: unknown, size: number): number[][] {
  checkShape(matrix, size);
  return matrix.map((row, i) => {
    const sum = row.reduce((a, b) => a + b, 0);
    if (!Number.isFinite(sum) || sum <= 0) throw new Error(`Row ${i} has no positive outgoing probability. Add a transition before normalizing.`);
    return row.map(p => p / sum);
  });
}

export function validateDataset(input: DatasetConfig | undefined, size: number, modelSeed: number): DatasetConfig {
  if (!input || input.mode === "cycle") return Object.freeze({ mode: "cycle" });
  if (input.mode !== "markov") throw new Error("Choose direct cycle or Markov matrix for the dataset.");
  const sampling = input.sampling ?? "per-step", seed = input.seed ?? modelSeed;
  if (sampling !== "per-step" && sampling !== "fixed") throw new Error("Invalid Markov sampling mode.");
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error("Dataset seed must be an integer from 0 to 4294967295.");
  checkShape(input.matrix, size);
  for (let i = 0; i < size; i++) {
    const sum = input.matrix[i].reduce((a, b) => a + b, 0);
    if (!Number.isFinite(sum) || Math.abs(sum - 1) > 1e-6) throw new Error(`Row ${i} sums to ${sum.toPrecision(6)}, not 1. Edit its probabilities or choose Normalize rows.`);
  }
  // Only rounding-sized row-sum discrepancies are corrected automatically.
  const matrix = normalizeMatrix(input.matrix, size).map(row => Object.freeze(row));
  return Object.freeze({ mode: "markov", matrix: Object.freeze(matrix), sampling, seed });
}

/** Accept a JSON nested array or comma/whitespace-separated numeric rows. */
export function parseMatrix(text: string, size: number): number[][] {
  const trimmed = text.trim();
  let matrix: unknown;
  if (trimmed.startsWith("[")) {
    try { matrix = JSON.parse(trimmed); } catch { throw new Error("Invalid JSON matrix. Use a nested array of numeric probabilities."); }
  } else {
    matrix = trimmed.split(/\r?\n/).filter(line => line.trim()).map(line => {
      if (line.includes(",") && line.split(",").some(cell => !cell.trim())) throw new Error("Empty matrix cell. Enter an explicit 0 for impossible transitions.");
      return line.trim().split(/[,\s]+/).map(value => Number(value));
    });
  }
  checkShape(matrix, size);
  return matrix;
}

type TransitionRow = { destinations: Int32Array; cdf: Float64Array };
export type CompiledMarkov = {
  rows: (TransitionRow | undefined)[];
  deterministic: boolean;
  fallbackRows: readonly number[];
};

/** Mask excluded destinations, condition each row, then compile sparse CDFs. */
export function compileMarkov(matrix: readonly (readonly number[])[], included: readonly number[]): CompiledMarkov {
  const rows: (TransitionRow | undefined)[] = Array(matrix.length);
  const fallbackRows: number[] = [];
  let deterministic = true;
  for (const source of included) {
    let destinations = included.filter(target => matrix[source][target] > 0);
    let total = destinations.reduce((sum, target) => sum + matrix[source][target], 0);
    const fallback = destinations.length === 0;
    if (fallback) { destinations = [source]; total = 1; fallbackRows.push(source); }
    if (destinations.length > 1) deterministic = false;
    let cumulative = 0;
    const cdf = Float64Array.from(destinations, target => {
      cumulative += fallback ? 1 : matrix[source][target] / total;
      return cumulative;
    });
    cdf[cdf.length - 1] = 1;
    rows[source] = { destinations: Int32Array.from(destinations), cdf };
  }
  return { rows, deterministic, fallbackRows: Object.freeze(fallbackRows) };
}

function seededRandom(seed: number, step: number) {
  // Dataset-local counter seed: never consumes the model initialization RNG.
  let state = ((seed >>> 0) ^ Math.imul(step + 1, 0x9e3779b1)) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function markovBatch(compiled: CompiledMarkov, included: readonly number[], batchSize: number, length: number, seed: number, step: number) {
  const inputs = new Int32Array(batchSize * length), targets = new Int32Array(batchSize * length);
  if (!included.length) return { inputs, targets };
  const random = seededRandom(seed, step);
  for (let b = 0; b < batchSize; b++) {
    // Same round-robin starts as direct cycles; not a stationary-distribution draw.
    let current = included[b % included.length];
    for (let t = 0; t < length; t++) {
      const row = compiled.rows[current]!;
      let index = 0;
      if (row.destinations.length > 1) {
        const u = random();
        let high = row.cdf.length - 1;
        while (index < high) {
          const mid = (index + high) >>> 1;
          if (u < row.cdf[mid]) high = mid;
          else index = mid + 1;
        }
      }
      const next = row.destinations[index];
      inputs[b * length + t] = current;
      targets[b * length + t] = next;
      current = next;
    }
  }
  return { inputs, targets };
}
