import type { Snapshot } from "./simulator";

type Vec3 = [number, number, number];
export type DisplaySnapshot = Omit<Snapshot, "hiddenMeans"> & {
  hiddenMeans: { target: number; count: number; mean: Vec3; projected: Vec3 | null }[];
};
const cache = new WeakMap<Snapshot, DisplaySnapshot>();
const xyz = (v: ArrayLike<number>): Vec3 => [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];

/** Crop only display vectors. Hidden states and scalar diagnostics stay full-D. */
export function toDisplaySnapshot(snapshot: Snapshot): DisplaySnapshot {
  const saved = cache.get(snapshot);
  if (saved) return saved;
  const crop = (values: Float32Array) => {
    const result = new Float32Array(snapshot.activeVocab * 3);
    for (let row = 0; row < snapshot.activeVocab; row++) {
      for (let d = 0; d < Math.min(3, snapshot.modelDim); d++) result[row * 3 + d] = values[row * snapshot.modelDim + d];
    }
    return result;
  };
  const display: DisplaySnapshot = {
    ...snapshot,
    positions: crop(snapshot.positions), effectivePositions: crop(snapshot.effectivePositions),
    rawGradients: crop(snapshot.rawGradients), tangentForces: crop(snapshot.tangentForces), optimizerMoves: crop(snapshot.optimizerMoves),
    hiddenMeans: snapshot.hiddenMeans.map(item => ({ ...item, mean: xyz(item.mean), projected: item.projected ? xyz(item.projected) : null })),
  };
  cache.set(snapshot, display);
  return display;
}
