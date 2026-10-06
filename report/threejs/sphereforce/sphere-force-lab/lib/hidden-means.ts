export type HiddenMean = {
  target: number;
  count: number;
  mean: number[];
  projected: number[] | null;
};

/** Average actual LM-head inputs by next-token target, then project the mean. */
export function averageHiddenByTarget(
  hidden: ArrayLike<number>,
  targets: ArrayLike<number>,
  targetCount: number,
  radius: number,
  dimension = 3,
): HiddenMean[] {
  if (hidden.length !== targets.length * dimension) throw new Error("Hidden states and target labels must align.");
  const sums = new Float64Array(targetCount * dimension);
  const counts = new Uint32Array(targetCount);
  for (let i = 0; i < targets.length; i += 1) {
    const target = targets[i];
    if (!Number.isInteger(target) || target < 0 || target >= targetCount) throw new Error("Invalid hidden-state target.");
    counts[target] += 1;
    for (let d = 0; d < dimension; d += 1) sums[target * dimension + d] += hidden[i * dimension + d];
  }
  const result: HiddenMean[] = [];
  for (let target = 0; target < targetCount; target += 1) {
    const count = counts[target];
    if (!count) continue;
    const mean = Array.from({ length: dimension }, (_, d) => sums[target * dimension + d] / count);
    const norm = Math.hypot(...mean);
    if (!Number.isFinite(norm)) continue;
    result.push({ target, count, mean, projected: norm > 1e-10 ? mean.map(value => radius * value / norm) as number[] : null });
  }
  return result;
}
