"use client";

import { useMemo } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { CurvePoint, frameAtIteration, MetricKey, metricValue, sampleMetric } from "@/lib/metric-history";
import type { Snapshot } from "@/lib/simulator";

const WIDTH = 320, HEIGHT = 126, LEFT = 43, RIGHT = 12, TOP = 12, BOTTOM = 28;
const PLOT_WIDTH = WIDTH - LEFT - RIGHT, PLOT_HEIGHT = HEIGHT - TOP - BOTTOM;
const definitions: { key: MetricKey; label: string; color: string; dashed?: boolean }[] = [
  { key: "loss", label: "Train", color: "#67e8f9" },
  { key: "validationLoss", label: "Validation / reference", color: "#f0abfc" },
  { key: "accuracy", label: "Train", color: "#6ee7b7" },
  { key: "validationAccuracy", label: "Validation / reference", color: "#fcd34d", dashed: true },
];

function number(value: number, percent = false) {
  if (!Number.isFinite(value)) return "—";
  if (percent) return `${(value * 100).toFixed(1)}%`;
  if (value > 0 && value < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

type Series = typeof definitions[number] & { points: CurvePoint[] };
function MetricPlot({ title, series, history, frame, max, percent = false, disabled, onSeek }: {
  title: string; series: Series[]; history: readonly Snapshot[]; frame: number;
  max: number; percent?: boolean; disabled: boolean; onSeek: (frame: number) => void;
}) {
  const selected = history[frame];
  const first = history[0].step, last = history[history.length - 1].step;
  const x = (step: number) => LEFT + (step - first) / Math.max(1, last - first) * PLOT_WIDTH;
  const y = (value: number) => TOP + (1 - Math.max(0, Math.min(max, value)) / max) * PLOT_HEIGHT;
  const cursor = x(selected.step);
  const ticks = [...new Set([first, Math.round((first + last) / 2), last])];
  // Paths depend on the recorded run, never on the selected cursor.
  const paths = useMemo(() => series.map(item => item.points.map(point => {
    const px = LEFT + (point.step - first) / Math.max(1, last - first) * PLOT_WIDTH;
    const py = TOP + (1 - Math.max(0, Math.min(max, point.value)) / max) * PLOT_HEIGHT;
    const pair = `${px.toFixed(2)},${py.toFixed(2)}`;
    return `${point.move ? "M" : "L"}${pair}${point.move ? `L${pair}` : ""}`;
  }).join(" ")), [series, first, last, max]);
  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 pt-3 pb-1" data-metric-chart={percent ? "accuracy" : series[0].key}>
      <h3 className="text-xs font-semibold text-slate-200">{title}</h3>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px]" aria-live="off">
        {series.map(item => <span key={item.key} style={{ color: item.color }}>
          {item.dashed ? "┄" : "—"} {item.label}: <span className="font-mono tabular-nums">{number(metricValue(selected, item.key), percent)}</span>
        </span>)}
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className={`block h-auto w-full ${disabled ? "" : "cursor-crosshair"}`} role="img" aria-label={`${title}, iterations ${first} to ${last}; selected iteration ${selected.step}`} onClick={event => {
        if (disabled || last === first) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const ratio = ((event.clientX - rect.left) / rect.width * WIDTH - LEFT) / PLOT_WIDTH;
        onSeek(frameAtIteration(history, first + Math.max(0, Math.min(1, ratio)) * (last - first)));
      }}>
        <title>{`${title}. Click or tap to select an iteration, or use the slider below.`}</title>
        {[0, 0.5, 1].map(fraction => <g key={fraction}>
          <line x1={LEFT} x2={WIDTH - RIGHT} y1={y(fraction * max)} y2={y(fraction * max)} stroke="#ffffff12" />
          <text x={LEFT - 7} y={y(fraction * max) + 3} textAnchor="end" fill="#94a3b8" fontSize="10">{percent ? `${fraction * 100}%` : (fraction * max).toPrecision(2)}</text>
        </g>)}
        {series.map((item, index) => <path key={item.key} d={paths[index]} fill="none" stroke={item.color} strokeWidth="1.8" strokeDasharray={item.dashed ? "4 3" : undefined} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />)}
        <line data-selected-iteration={selected.step} x1={cursor} x2={cursor} y1={TOP} y2={HEIGHT - BOTTOM} stroke="#e2e8f0" strokeDasharray="3 3" opacity="0.8" />
        {series.map(item => {
          const value = metricValue(selected, item.key);
          return Number.isFinite(value) ? <circle key={item.key} cx={cursor} cy={y(value)} r="3" fill={item.color} stroke="#07101e" strokeWidth="1.5" /> : null;
        })}
        {ticks.map(step => <text key={step} x={x(step)} y={HEIGHT - 10} textAnchor={step === first ? "start" : step === last ? "end" : "middle"} fill="#94a3b8" fontSize="10">{step}</text>)}
        {!series.some(item => item.points.length) && <text x={LEFT + PLOT_WIDTH / 2} y={TOP + PLOT_HEIGHT / 2} textAnchor="middle" fill="#94a3b8" fontSize="11">No data in this run</text>}
      </svg>
    </div>
  );
}

export function TrainingCurves({ history, frame, revision, disabled, onSeek }: {
  history: readonly Snapshot[]; frame: number; revision: number; disabled: boolean; onSeek: (frame: number) => void;
}) {
  const series = useMemo(() => definitions.map(item => ({ ...item, points: sampleMetric(history, item.key) })), [history, revision]);
  const groups = useMemo(() => [series.slice(0, 1), series.slice(1, 2), series.slice(2)], [series]);
  const lossMax = useMemo(() => {
    let largest = 0;
    for (const item of series.slice(0, 2)) for (const point of item.points) largest = Math.max(largest, point.value);
    return largest > 0 ? largest * 1.05 : 1;
  }, [series]);
  const snapshot = history[frame];
  const lastFrame = history.length - 1, first = history[0].step, last = history[lastFrame].step;
  return (
    <section className="training-curves mt-4 space-y-2.5" aria-label="Training curves and iteration navigation">
      <div className="flex items-center justify-between text-xs text-slate-400"><span>Recorded run</span><span className="font-mono">{first} → {last}</span></div>
      <MetricPlot title="Training cross-entropy loss" series={groups[0]} history={history} frame={frame} max={lossMax} disabled={disabled} onSeek={onSeek} />
      <MetricPlot title="Validation / reference cross-entropy loss" series={groups[1]} history={history} frame={frame} max={lossMax} disabled={disabled} onSeek={onSeek} />
      <MetricPlot title="Target accuracy · next-token top-1" series={groups[2]} history={history} frame={frame} max={1} percent disabled={disabled} onSeek={onSeek} />
      <div className="rounded-xl border border-cyan-300/15 bg-cyan-300/[0.035] p-3">
        <div className="flex items-center justify-between gap-2 text-xs">
          <span className="font-semibold text-cyan-100">Iteration <span className="font-mono">{snapshot.step}</span> / {last}</span>
          <Button size="xs" variant="ghost" disabled={disabled || frame === lastFrame} onClick={() => onSeek(lastFrame)}>Latest</Button>
        </div>
        <div className="mt-2" style={{ marginLeft: `${LEFT / WIDTH * 100}%`, marginRight: `${RIGHT / WIDTH * 100}%` }}>
          <Slider aria-label="Metric iteration" aria-valuetext={`Iteration ${snapshot.step} of ${last}`} min={first} max={Math.max(first + 1, last)} step={1} disabled={disabled || first === last} value={[snapshot.step]}
            onPointerDownCapture={() => { if (!disabled) onSeek(frame); }}
            onKeyDownCapture={event => { if (!disabled && ["Home", "End", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown"].includes(event.key)) onSeek(frame); }}
            onValueChange={([step]) => onSeek(frameAtIteration(history, step))} className="min-h-8" />
        </div>
        <div className="mt-1 flex items-center justify-between gap-2 text-[11px] text-slate-400">
          <span>Frame {frame} / {lastFrame}</span>
          <div className="flex gap-1">
            <Button size="icon-xs" variant="ghost" aria-label="Previous recorded frame" disabled={disabled || frame === 0} onClick={() => onSeek(frame - 1)}><ChevronLeft /></Button>
            <Button size="icon-xs" variant="ghost" aria-label="Next recorded frame" disabled={disabled || frame === lastFrame} onClick={() => onSeek(frame + 1)}><ChevronRight /></Button>
          </div>
        </div>
        <p className="mt-1 text-[11px] leading-relaxed text-slate-400">Click a plot or drag to inspect. Both sliders and all views stay linked. Arrow buttons visit individual frames, including edits at the same iteration.</p>
      </div>
      <p className="px-1 text-[11px] leading-relaxed text-slate-400">
        {snapshot.validation.kind === "independent-markov" ? `Validation uses a fixed independent Markov sample (${snapshot.validation.sampleCount} positions; seed ${snapshot.validation.seed}). Membership changes rebuild that sample; it is never used for updates. Sampled sequences can overlap training.` : snapshot.validation.kind === "same-data" ? "Deterministic data: reference and training metrics use the same batch and coincide. There is no separate held-out validation set." : "No targets included: loss and accuracy are unavailable. Gaps preserve these empty phases."}
        {" "}Accuracy counts correctly predicted next-token labels over all batch positions. Loss plots share a scale; accuracy spans 0–100%. Curves retain the full run while you scrub. Dataset or vocabulary changes can change the objective.
      </p>
    </section>
  );
}
