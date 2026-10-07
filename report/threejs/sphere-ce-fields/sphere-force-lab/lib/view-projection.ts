import type { Snapshot } from "./simulator";

type Vec3 = [number, number, number];
export type DisplaySnapshot = Omit<Snapshot, "hiddenMeans"> & {
  hiddenMeans: { target: number; count: number; mean: Vec3; projected: Vec3 | null }[];
};
const cache = new WeakMap<Snapshot, DisplaySnapshot>();
const xyz = (v: ArrayLike<number>): Vec3 => [v[0] ?? 0, v[1] ?? 0, v[2] ?? 0];

export function supportsSphericalProbe(snapshot: Pick<Snapshot, "radiusMode" | "modelDim">) {
  return snapshot.radiusMode !== "free" && snapshot.modelDim >= 3;
}

/** CE force only: optimizer preconditioning, momentum, and decay are separate. */
export function displayedRowForce(snapshot: DisplaySnapshot, row: number): Vec3 {
  const o = row * 3;
  return snapshot.radiusMode === "free"
    ? [-snapshot.rawGradients[o], -snapshot.rawGradients[o + 1], -snapshot.rawGradients[o + 2]]
    : [snapshot.tangentForces[o], snapshot.tangentForces[o + 1], snapshot.tangentForces[o + 2]];
}

/** Fit actual coordinates, including visible trail history, without normalizing rows. */
export function displayExtent(snapshot: DisplaySnapshot, history: readonly DisplaySnapshot[], frameIndex: number, showTrails: boolean, showHiddenInSpace: boolean) {
  let extent = snapshot.radius;
  const include = (values: Float32Array) => {
    for (let i = 0; i < values.length; i += 3) {
      const norm = Math.hypot(values[i], values[i + 1], values[i + 2]);
      if (Number.isFinite(norm)) extent = Math.max(extent, norm);
    }
  };
  include(snapshot.positions);
  if (showTrails && snapshot.radiusMode === "free") {
    for (let h = Math.max(0, frameIndex - 160); h <= frameIndex; h++) {
      if (history[h]) include(history[h].positions);
    }
  }
  if (showHiddenInSpace) for (const item of snapshot.hiddenMeans) extent = Math.max(extent, Math.hypot(...item.mean));
  return extent;
}

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
