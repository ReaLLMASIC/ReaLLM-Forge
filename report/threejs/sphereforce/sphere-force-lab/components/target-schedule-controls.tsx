"use client";

import { useState } from "react";
import { Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Snapshot } from "@/lib/simulator";
import { activeTargetRule, describeTargetRule, dutyOnSteps, TargetMode, TargetRuleDraft } from "@/lib/target-schedule";

export function TargetScheduleControls({ live, disabled, maxIterations, onApply, onCancel, onSelect }: {
  live: Snapshot; disabled: boolean; maxIterations: number;
  onApply: (rule: TargetRuleDraft) => void; onCancel: (id: number) => void; onSelect: (token: number) => void;
}) {
  const [token, setToken] = useState(0);
  const [mode, setMode] = useState<TargetMode>("exclude");
  const [start, setStart] = useState(100);
  const [period, setPeriod] = useState(100);
  const [percent, setPercent] = useState(50);
  const selected = Math.min(token, live.targetCount - 1);
  const activeRule = activeTargetRule(live.targetRules, selected, live.step);
  const onSteps = dutyOnSteps(period, percent);
  const validDuty = Number.isInteger(period) && period > 0 && period <= 50_000 && Number.isFinite(percent) && percent >= 0 && percent <= 100;
  const tokenRules = live.targetRules.filter((rule) => rule.token === selected).sort((a, b) => a.start - b.start || a.id - b.id);
  const apply = (at: number) => onApply({ token: selected, mode, start: at, period, percent });

  return <section className="target-controls border-t border-white/[0.07] pt-4" aria-label="Target inclusion schedules">
    <div className="section-label"><Target className="size-3.5" /> Target inclusion</div>
    <p className="mt-2 text-sm text-slate-400">Edit the live run at iteration {live.step}. Amber tokens are excluded from inputs and labels, but remain in the output vocabulary.</p>
    <div className="mt-3 grid max-h-36 grid-cols-5 gap-1.5 overflow-y-auto" aria-label="Original target tokens">
      {Array.from({ length: live.targetCount }, (_, index) => <button type="button" key={index}
        aria-label={`Edit token ${index}: ${live.targetMask[index] ? "included" : "excluded"}`} aria-pressed={selected === index}
        onClick={() => { setToken(index); onSelect(index); }}
        className={`rounded-md border py-1.5 font-mono text-sm ${selected === index ? "ring-2 ring-white/60" : ""} ${live.targetMask[index] ? "border-cyan-300/20 bg-cyan-300/10 text-cyan-200" : "border-amber-300/25 bg-amber-300/10 text-amber-200"}`}>{index}</button>)}
    </div>
    <div className="mt-3 rounded-lg border border-white/10 bg-white/[0.025] p-2.5 text-sm">
      <div className={live.targetMask[selected] ? "text-cyan-200" : "text-amber-200"}>Token {selected} · {live.targetMask[selected] ? "included" : "untargeted"}</div>
      <div className="mt-1 text-slate-400">Active: {activeRule ? describeTargetRule(activeRule) : "include (default)"}</div>
    </div>
    <label className="control-label mt-3 block">New policy
      <Select value={mode} onValueChange={(value) => setMode(value as TargetMode)}>
        <SelectTrigger aria-label="Target policy" className="mt-1 w-full"><SelectValue /></SelectTrigger>
        <SelectContent><SelectItem value="exclude">Exclude target</SelectItem><SelectItem value="include">Restore target</SelectItem><SelectItem value="duty">Repeating duty cycle</SelectItem></SelectContent>
      </Select>
    </label>
    {mode === "duty" && <div className="mt-3 space-y-2">
      <div className="grid grid-cols-2 gap-2">
        <label className="control-label">Included %<input aria-label="Duty included percentage" className="lab-input mt-1" type="number" min="0" max="100" step="1" value={percent} onChange={(event) => setPercent(Number(event.target.value))} /></label>
        <label className="control-label">Period<input aria-label="Duty period iterations" className="lab-input mt-1" type="number" min="1" max="50000" step="1" value={period} onChange={(event) => setPeriod(Number(event.target.value))} /></label>
      </div>
      {validDuty && <div className="text-sm leading-relaxed text-slate-400">
        <div className="my-2 flex h-2 overflow-hidden rounded-full bg-amber-300/30" role="img" aria-label={`${onSteps} included then ${period - onSteps} excluded iterations per period`}><div className="h-full bg-cyan-300" style={{ width: `${onSteps / period * 100}%` }} /></div>
        First {onSteps} iterations included, then {period - onSteps} excluded; repeats every {period}. Actual inclusion: {(onSteps / period * 100).toFixed(2)}% per full period (rounded to whole iterations).
      </div>}
    </div>}
    <Button className="mt-3 w-full bg-cyan-200 text-slate-950 hover:bg-cyan-100" disabled={disabled} onClick={() => apply(live.step)}>Apply policy now</Button>
    <label className="control-label mt-3 block">Scheduled start iteration<input aria-label="Target scheduled start" className="lab-input mt-1" type="number" min={live.step} max="50000" step="1" value={start} onChange={(event) => setStart(Number(event.target.value))} /></label>
    <Button variant="outline" className="mt-2 w-full" disabled={disabled} onClick={() => apply(start)}>Schedule policy</Button>
    {start >= maxIterations && <p className="mt-2 text-sm text-amber-200">Raise the training limit above {start} to train under this policy.</p>}
    <p className="mt-2 text-sm leading-relaxed text-slate-400">Changes pause the run. Use Run to continue. A policy lasts until the next one for this token; schedule a restore to end a dropout or duty cycle. At equal start times, the newest rule wins.</p>
    {tokenRules.length > 0 && <div className="mt-3 max-h-48 space-y-2 overflow-y-auto" aria-label={`Policies for token ${selected}`}>
      {tokenRules.map((rule) => <div key={rule.id} className="rounded-lg border border-white/10 p-2 text-sm">
        <div className="text-slate-300">At {rule.start}: {describeTargetRule(rule)}</div>
        {rule.start > live.step ? <Button variant="ghost" size="sm" disabled={disabled} aria-label={`Cancel policy ${rule.id} for token ${selected}`} onClick={() => onCancel(rule.id)}>Cancel scheduled policy</Button> : <div className="mt-1 text-xs text-slate-500">{activeRule?.id === rule.id ? "Active policy" : "Past policy"}</div>}
      </div>)}
    </div>}
  </section>;
}
