"use client";

import { useState } from "react";
import { Braces } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { DatasetConfig, cycleMatrix, uniformMatrix, normalizeMatrix, parseMatrix, validateDataset } from "@/lib/dataset";

type Props = { value: DatasetConfig; onChange: (value: DatasetConfig) => void; onApply: () => void; targetCount: number; disabled: boolean };
const selectStyle = "mt-1 w-full border-white/10 bg-[#091527] text-slate-200";
const buttonStyle = "border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]";

export function DatasetControls({ value, onChange, onApply, targetCount, disabled }: Props) {
  const [row, setRow] = useState(0);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const size = Number.isInteger(targetCount) && targetCount >= 1 && targetCount <= 100 ? targetCount : 10;
  const matrix = value.matrix;
  const selected = Math.min(row, size - 1);
  const shapeMatches = matrix?.length === size && matrix.every(row => row.length === size);
  let issue = "";
  try { validateDataset(value, size, 17); } catch (cause) { issue = (cause as Error).message; }
  const set = (patch: Partial<DatasetConfig>) => { setError(null); onChange({ ...value, ...patch }); };
  const editMatrix = (next: number[][]) => set({ matrix: next });
  const act = (fn: () => void) => { try { fn(); setError(null); } catch (cause) { setError((cause as Error).message); } };
  const sum = shapeMatches ? matrix[selected].reduce((a, b) => a + b, 0) : 0;

  return <section aria-label="Dataset source" className="space-y-3 border-t border-white/[0.07] pt-4">
    <div className="section-label"><Braces className="size-3.5" /> Dataset source</div>
    <label className="block text-sm text-slate-300">Training dataset
      <Select value={value.mode} disabled={disabled} onValueChange={mode => set({ mode: mode as DatasetConfig["mode"], matrix: matrix ?? cycleMatrix(size) })}>
        <SelectTrigger aria-label="Training dataset" className={selectStyle}><SelectValue /></SelectTrigger>
        <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
          <SelectItem value="cycle">Direct cycle · fast</SelectItem>
          <SelectItem value="markov">Markov transition matrix</SelectItem>
        </SelectContent>
      </Select>
    </label>
    {value.mode === "cycle" ? <p className="text-sm leading-relaxed text-slate-400">Repeats the included numeric IDs in order. Cached sequences; no probability sampling or matrix work. This remains the default.</p> : <>
      <p className="text-sm leading-relaxed text-slate-400">P[i,j] = probability of token j after token i. Rows and columns use original target IDs 0–{size - 1}; each row must sum to 1.</p>
      <div className="grid grid-cols-2 gap-2">
        <Button type="button" variant="outline" className={buttonStyle} disabled={disabled} onClick={() => editMatrix(cycleMatrix(size))}>Cycle matrix</Button>
        <Button type="button" variant="outline" className={buttonStyle} disabled={disabled} onClick={() => editMatrix(uniformMatrix(size))}>Uniform matrix</Button>
      </div>
      {shapeMatches ? <>
        <label className="block text-sm text-slate-300">Current token (row)
          <Select value={String(selected)} disabled={disabled} onValueChange={value => setRow(Number(value))}>
            <SelectTrigger aria-label="Current token row" className={selectStyle}><SelectValue /></SelectTrigger>
            <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">{matrix.map((_, i) => <SelectItem key={i} value={String(i)}>From token {i}</SelectItem>)}</SelectContent>
          </Select>
        </label>
        <div className="grid max-h-64 grid-cols-2 gap-2 overflow-y-auto rounded-lg border border-white/10 p-2" aria-label={`Transition probabilities from token ${selected}`}>
          {matrix[selected].map((probability, column) => <label key={column} className="text-sm text-slate-300">To {column}
            <input className="lab-input mt-1" aria-label={`Probability ${selected} to ${column}`} type="number" min={0} max={1} step="0.05" value={probability} disabled={disabled} onChange={event => {
              const next = matrix.map(row => [...row]); next[selected][column] = Number(event.target.value); editMatrix(next);
            }} />
          </label>)}
        </div>
        <p className={`font-mono text-sm ${Math.abs(sum - 1) <= 1e-6 ? "text-emerald-300" : "text-amber-200"}`}>Row {selected} sum: {Number.isFinite(sum) ? Number(sum.toPrecision(7)) : "invalid"}</p>
      </> : <p className="text-sm text-amber-200">Target count changed. Load a {size}×{size} matrix or choose a preset; your existing entries have not been resized or discarded.</p>}
      <Button type="button" variant="outline" className={`${buttonStyle} w-full`} disabled={disabled || !shapeMatches} onClick={() => act(() => editMatrix(normalizeMatrix(matrix, size)))}>Normalize rows</Button>
      <label className="block text-sm text-slate-300">Paste full matrix
        <textarea aria-label="Full transition matrix" className="lab-input mt-1 min-h-24 resize-y font-mono text-xs" value={text} disabled={disabled} onChange={event => setText(event.target.value)} placeholder={`${size}×${size} JSON array or one numeric row per line`} />
      </label>
      <div className="grid grid-cols-2 gap-2">
        <Button type="button" variant="outline" className={buttonStyle} disabled={disabled} onClick={() => act(() => editMatrix(parseMatrix(text, size)))}>Load into editor</Button>
        <Button type="button" variant="outline" className={buttonStyle} disabled={disabled || !shapeMatches} onClick={() => setText(matrix!.map(row => row.join(", ")).join("\n"))}>Show full matrix</Button>
      </div>
      <p className="text-xs leading-relaxed text-slate-400">Comma- or space-separated rows, or a JSON array. Load first, then validate below. Normalize explicitly if using relative weights.</p>
      <label className="block text-sm text-slate-300">Markov sampling
        <Select value={value.sampling ?? "per-step"} disabled={disabled} onValueChange={sampling => set({ sampling: sampling as DatasetConfig["sampling"] })}>
          <SelectTrigger aria-label="Markov sampling" className={selectStyle}><SelectValue /></SelectTrigger>
          <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100"><SelectItem value="per-step">Fresh batch each iteration</SelectItem><SelectItem value="fixed">Fixed seeded batch</SelectItem></SelectContent>
        </Select>
      </label>
      <label className="block text-sm text-slate-300">Dataset seed
        <input aria-label="Dataset seed" className="lab-input mt-1" type="number" min={0} max={4294967295} step={1} value={value.seed ?? 17} disabled={disabled} onChange={event => set({ seed: Number(event.target.value) })} />
      </label>
      <p className="text-sm leading-relaxed text-slate-400">Sequences start round-robin across included IDs, not at stationarity. Each transition samples a next-token label. Deterministic rows skip sampling; fully deterministic matrices reuse their batch.</p>
      <p className="text-xs leading-relaxed text-slate-400">Exclusion removes a target from both starts and destinations. Remaining row probabilities are renormalized; a row with no remaining destination becomes a self-loop. Restoration reuses your original matrix.</p>
      {issue && <p role="status" className="text-sm text-amber-200">{issue}</p>}
      {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
    </>}
    <Button type="button" variant="outline" className={`${buttonStyle} w-full`} disabled={disabled} onClick={onApply}>Apply dataset + reset</Button>
    <p className="text-xs text-slate-400">Resets the entire run using all current dataset, architecture, experiment, and QAT drafts. Matrix edits do not change a running model until reset.</p>
  </section>;
}
