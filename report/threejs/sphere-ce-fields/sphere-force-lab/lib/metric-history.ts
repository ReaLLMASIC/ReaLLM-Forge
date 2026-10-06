import type { Snapshot } from "./simulator";

export type MetricFrame = Pick<Snapshot, "step" | "loss" | "accuracy" | "validation">;
export type MetricKey = "loss" | "validationLoss" | "accuracy" | "validationAccuracy";
export type CurvePoint = { frame: number; step: number; value: number; move: boolean };

export function metricValue(frame: MetricFrame, key: MetricKey) {
  if (key === "validationLoss") return frame.validation.loss;
  if (key === "validationAccuracy") return frame.validation.accuracy;
  return frame[key];
}

/** Latest frame at an iteration, including same-step policy/QAT/insertion edits. */
export function frameAtIteration(history: readonly Pick<Snapshot, "step">[], iteration: number) {
  if (!history.length || !Number.isFinite(iteration)) return 0;
  const step = Math.max(history[0].step, Math.min(history[history.length - 1].step, Math.round(iteration)));
  let low = 0, high = history.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (history[middle].step <= step) low = middle + 1;
    else high = middle;
  }
  return Math.max(0, low - 1);
}

/** Keep first/last and both extrema per horizontal bin; never bridge missing data. */
export function sampleMetric(history: readonly MetricFrame[], key: MetricKey, bins = 240): CurvePoint[] {
  const points: CurvePoint[] = [];
  const start = history[0]?.step ?? 0;
  const span = Math.max(1, (history[history.length - 1]?.step ?? start) - start);
  const count = Math.max(1, Math.floor(bins));
  let bucket = -1, first = -1, last = -1, min = -1, max = -1, move = true;
  const flush = () => {
    if (first < 0) return;
    const indices = [...new Set([first, min, max, last])].sort((a, b) => a - b);
    for (const frame of indices) {
      points.push({ frame, step: history[frame].step, value: metricValue(history[frame], key), move });
      move = false;
    }
    first = last = min = max = -1;
  };
  for (let i = 0; i < history.length; i++) {
    const value = metricValue(history[i], key);
    if (!Number.isFinite(value)) { flush(); move = true; bucket = -1; continue; }
    const nextBucket = Math.min(count - 1, Math.floor((history[i].step - start) / span * count));
    if (nextBucket !== bucket) { flush(); bucket = nextBucket; }
    if (first < 0) first = min = max = i;
    last = i;
    if (value < metricValue(history[min], key)) min = i;
    if (value > metricValue(history[max], key)) max = i;
  }
  flush();
  return points;
}
