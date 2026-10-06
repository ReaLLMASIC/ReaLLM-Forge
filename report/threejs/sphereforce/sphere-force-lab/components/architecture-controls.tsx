"use client";

import { Cpu } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ArchitectureConfig, architectureParameterCount, validateArchitecture } from "@/lib/architecture";

type Props = { value: ArchitectureConfig; onChange: (value: ArchitectureConfig) => void; batchSize: number; rows: number; disabled: boolean };
export function ArchitectureControls({ value: a, onChange, batchSize, rows, disabled }: Props) {
  const attention = a.blockMode !== "mlp", mlp = a.blockMode !== "attention";
  let issue = "";
  try { validateArchitecture(a, batchSize, rows); } catch (cause) { issue = (cause as Error).message; }
  const set = (patch: Partial<ArchitectureConfig>) => onChange({ ...a, ...patch });
  const number = (key: "modelDim" | "layers" | "mlpDim" | "heads" | "qkHeadDim" | "valueHeadDim" | "attentionDim" | "maxContextLength", label: string, min: number, max: number, inactive = false) => (
    <label className={`block text-sm text-slate-300 ${inactive ? "opacity-45" : ""}`} key={key}><span className="block min-h-10">{label}</span>
      <input aria-label={label} className="lab-input mt-1 text-sm" type="number" min={min} max={max} step={1} value={a[key]} disabled={disabled || inactive} onChange={event => {
        const value = Number(event.target.value);
        if (key === "heads") set({ heads: value, attentionDim: value * a.valueHeadDim });
        else if (key === "valueHeadDim") set({ valueHeadDim: value, attentionDim: value * a.heads });
        else if (key === "attentionDim") set({ attentionDim: value, valueHeadDim: value / a.heads });
        else set({ [key]: value });
      }} />
    </label>
  );
  const select = (key: "blockMode" | "positionEncoding" | "activation", label: string, options: [string, string][], inactive = false) => (
    <label className={`block text-sm text-slate-300 ${inactive ? "opacity-45" : ""}`}>{label}
      <Select value={a[key]} disabled={disabled || inactive} onValueChange={value => {
        const patch = { [key]: value } as Partial<ArchitectureConfig>;
        if (key === "blockMode" && value === "mlp") patch.positionEncoding = "absolute";
        set(patch);
      }}>
        <SelectTrigger aria-label={label} className="mt-1 w-full border-white/10 bg-[#091527] text-slate-200"><SelectValue /></SelectTrigger>
        <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">{options.map(([value, label]) => <SelectItem key={value} value={value}>{label}</SelectItem>)}</SelectContent>
      </Select>
    </label>
  );
  return <section className="space-y-3 border-t border-white/[0.07] pt-4" aria-label="Model architecture">
    <div className="section-label"><Cpu className="size-3.5" /> Model architecture</div>
    <p className="text-sm leading-relaxed text-slate-400">Applies with reset. Every decoder block uses the selected structure.</p>
    <div className="grid grid-cols-2 gap-3">{number("modelDim", "Model dimension", 2, 128)}{number("layers", "Decoder blocks", 1, 8)}</div>
    {select("blockMode", "Block structure", [["full", "Attention + MLP"], ["attention", "Attention only"], ["mlp", "MLP only"]])}
    {a.blockMode === "mlp" && <p className="text-sm text-amber-200">MLP-only blocks do not mix token positions. Absolute positions are used; RoPE needs attention.</p>}
    <div className="grid grid-cols-2 gap-3">{number("heads", "Attention heads", 1, 16, !attention)}{number("qkHeadDim", "Q/K width per head", 1, 128, !attention)}{number("valueHeadDim", "V/O width per head", 1, 128, !attention)}{number("attentionDim", "Attention hidden dimension", 1, 2048, !attention)}</div>
    <p className="text-xs leading-relaxed text-slate-400">Q/K widths are shared. Attention width = heads × V/O width. Editing either width updates the other; the total must divide evenly by heads. Wₒ maps the concatenated values back to the model dimension.</p>
    {number("mlpDim", "MLP hidden dimension", 1, 1024, !mlp)}
    {select("activation", "MLP activation", [["gelu", "GELU"], ["relu", "ReLU"], ["relu2", "ReLU²"]], !mlp)}
    {select("positionEncoding", "Position embeddings", [["absolute", "Learned absolute"], ["rope", "Rotary (RoPE)"]], !attention)}
    {a.positionEncoding === "rope" && <p className="text-xs text-slate-400">Rotates every Q/K coordinate pair, requiring an even Q/K width. Base 10,000.</p>}
    {number("maxContextLength", "Maximum context length", 1, 256)}
    <p className="text-xs leading-relaxed text-slate-400">Each training sequence fills this window using the selected dataset source. Removals change allowed tokens, not the window length. A short window and small batch may leave some included targets unsampled; see the coverage below the viewer.</p>
    <div className="rounded-lg border border-cyan-300/15 bg-cyan-300/5 p-3 text-sm text-cyan-100">{Number.isFinite(architectureParameterCount(a, rows)) ? architectureParameterCount(a, rows).toLocaleString() : "—"} active parameters at reset</div>
    {issue && <p role="status" className="text-sm text-amber-200">{issue}</p>}
  </section>;
}
