"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
  Activity,
  ArrowRight,
  Atom,
  Braces,
  CircleDot,
  Cpu,
  Gauge,
  History,
  Pause,
  Play,
  Plus,
  RotateCcw,
  SkipForward,
  Sparkles,
  Target,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SphereScene, ViewMode } from "@/components/sphere-scene";
import {
  D_MODEL,
  LETTERS,
  OptimizerKind,
  ProbeMode,
  Snapshot,
  TOKENS,
  TransformerSphereSimulation,
} from "@/lib/simulator";

const MAX_STEPS = 1200;
const COMPACT_QUERY = "(max-width: 1279px)";
function subscribeToLayout(callback: () => void) {
  const query = window.matchMedia(COMPACT_QUERY);
  query.addEventListener("change", callback);
  return () => query.removeEventListener("change", callback);
}
const getCompactLayout = () => window.matchMedia(COMPACT_QUERY).matches;
const getServerLayout = () => false;

const modes: { id: ViewMode; label: string; icon: typeof Atom; description: string }[] = [
  { id: "rows", label: "Rows + trails", icon: Atom, description: "Tied token rows and their native 3D paths." },
  { id: "forces", label: "CE tangent force", icon: Zap, description: "Saved −P∇L arrows at every active row." },
  { id: "optimizer", label: "Optimizer move", icon: ArrowRight, description: "Observed post-projection displacement; not a gradient." },
  { id: "probe", label: "Probe field", icon: Target, description: "Counterfactual force on a new untargeted row." },
  { id: "decomposition", label: "Decomposition", icon: Braces, description: "Ambient force split into tangent and radial parts." },
];

const probeModes: { id: ProbeMode; label: string; units: string }[] = [
  { id: "force", label: "Tangent force magnitude", units: "‖−Pᵤg(u)‖" },
  { id: "potential", label: "Insertion loss potential", units: "mean softplus" },
  { id: "probability", label: "Mean probe probability", units: "probability" },
  { id: "radial", label: "Signed radial force", units: "outward component" },
];

function format(value: number, digits = 4) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) > 0 && Math.abs(value) < 1e-3) return value.toExponential(2);
  return value.toFixed(digits);
}

function Metric({ label, value, detail, accent = "cyan" }: { label: string; value: string; detail?: string; accent?: "cyan" | "pink" | "gold" | "green" }) {
  const colors = {
    cyan: "text-cyan-300",
    pink: "text-fuchsia-300",
    gold: "text-amber-300",
    green: "text-emerald-300",
  };
  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.035] px-3 py-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-500">{label}</div>
      <div className={`mt-1 font-mono text-[15px] font-semibold tabular-nums ${colors[accent]}`}>{value}</div>
      {detail && <div className="mt-0.5 text-[10px] text-slate-500">{detail}</div>}
    </div>
  );
}

function TinyLossChart({ history, frame }: { history: Snapshot[]; frame: number }) {
  const path = useMemo(() => {
    const points = history.slice(0, frame + 1);
    if (points.length < 2) return "";
    const values = points.map((p) => p.loss);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, 1e-8);
    return values.map((v, i) => {
      const x = (i / (values.length - 1)) * 260;
      const y = 58 - ((v - min) / span) * 48;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
  }, [history, frame]);
  return (
    <svg viewBox="0 0 260 68" className="h-[68px] w-full" role="img" aria-label="Loss history through the selected iteration">
      <defs>
        <linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#67e8f9" stopOpacity=".28" />
          <stop offset="1" stopColor="#67e8f9" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${path} L260,68 L0,68 Z`} fill="url(#loss-fill)" />
      <path d={path} fill="none" stroke="#67e8f9" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function VectorRow({ label, values, color }: { label: string; values: number[]; color: string }) {
  return (
    <tr className="border-t border-white/[0.06]">
      <th className="py-2 pr-3 text-left text-[11px] font-medium" style={{ color }}>{label}</th>
      {values.map((v, i) => <td key={i} className="px-1 py-2 text-right font-mono text-[11px] tabular-nums text-slate-300">{format(v, 3)}</td>)}
    </tr>
  );
}

export function SphereForceLab() {
  const compact = useSyncExternalStore(subscribeToLayout, getCompactLayout, getServerLayout);
  const [mobilePanel, setMobilePanel] = useState("controls");
  const [placingLetter, setPlacingLetter] = useState(false);
  const simulationRef = useRef<TransformerSphereSimulation | null>(null);
  const initCounter = useRef(0);
  const [ready, setReady] = useState(false);
  const [working, setWorking] = useState(false);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<Snapshot[]>([]);
  const [events, setEvents] = useState<{ token: string; step: number; direction: [number, number, number] }[]>([]);
  const [frame, setFrame] = useState(0);
  const [mode, setMode] = useState<ViewMode>("probe");
  const [probeMode, setProbeMode] = useState<ProbeMode>("force");
  const [selectedToken, setSelectedToken] = useState(0);
  const [showTrails, setShowTrails] = useState(true);
  const [showFieldArrows, setShowFieldArrows] = useState(false);
  const [normalizeArrows, setNormalizeArrows] = useState(false);
  const [frameAutoscale, setFrameAutoscale] = useState(false);
  const [scale, setScale] = useState({ min: 0, max: 0.18 });
  const [seed, setSeed] = useState(17);
  const [optimizer, setOptimizer] = useState<OptimizerKind>("adamw");
  const [learningRate, setLearningRate] = useState(0.018);
  const [weightDecay, setWeightDecay] = useState(0.05);
  const [error, setError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("Initializing the CPU simulation…");

  const initialize = useCallback(async (settings?: Partial<{ seed: number; optimizer: OptimizerKind; learningRate: number; weightDecay: number }>) => {
    const initId = ++initCounter.current;
    setRunning(false);
    setPlacingLetter(false);
    setReady(false);
    setError(null);
    setAnnouncement("Resetting model, optimizer state, and timeline…");
    simulationRef.current?.dispose();
    simulationRef.current = null;
    try {
      const sim = await TransformerSphereSimulation.create({
        seed: settings?.seed ?? seed,
        optimizer: settings?.optimizer ?? optimizer,
        learningRate: settings?.learningRate ?? learningRate,
        weightDecay: settings?.weightDecay ?? weightDecay,
      });
      if (initCounter.current !== initId) {
        sim.dispose();
        return;
      }
      simulationRef.current = sim;
      setHistory([...sim.history]);
      setEvents([]);
      setFrame(0);
      setSelectedToken(0);
      setReady(true);
      setAnnouncement("Ready at iteration 0. Click the sphere to insert letter a.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setAnnouncement("The simulation could not initialize.");
    }
  }, [learningRate, optimizer, seed, weightDecay]);

  useEffect(() => {
    const timer = window.setTimeout(() => void initialize(), 0);
    return () => {
      window.clearTimeout(timer);
      initCounter.current += 1;
      simulationRef.current?.dispose();
    };
    // This effect intentionally runs once; the Apply + reset button uses the latest settings.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const syncFromSimulation = useCallback((announce?: string) => {
    const sim = simulationRef.current;
    if (!sim) return;
    const next = [...sim.history];
    setHistory(next);
    setEvents([...sim.events]);
    setFrame(next.length - 1);
    if (announce) setAnnouncement(announce);
  }, []);

  const trainMany = useCallback(async (count: number) => {
    const sim = simulationRef.current;
    if (!sim || working || !ready) return;
    setWorking(true);
    setRunning(false);
    setError(null);
    try {
      const remaining = Math.min(count, MAX_STEPS - sim.step);
      for (let i = 0; i < remaining; i += 1) {
        sim.trainOne();
        if (i % 3 === 2) await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
      syncFromSimulation(`Advanced to iteration ${sim.step}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setWorking(false);
    }
  }, [ready, syncFromSimulation, working]);

  useEffect(() => {
    if (!running || !ready || working) return;
    let cancelled = false;
    let timer = 0;
    const tick = () => {
      if (cancelled) return;
      const sim = simulationRef.current;
      if (!sim || sim.step >= MAX_STEPS) {
        setRunning(false);
        return;
      }
      try {
        sim.trainOne();
        syncFromSimulation();
        timer = window.setTimeout(tick, 24);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
        setRunning(false);
      }
    };
    timer = window.setTimeout(tick, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [ready, running, syncFromSimulation, working]);

  const snapshot = history[Math.min(frame, history.length - 1)];
  const liveFrame = Math.max(0, history.length - 1);
  const isLive = frame === liveFrame;
  const activeMode = modes.find((item) => item.id === mode)!;
  const activeProbeMode = probeModes.find((item) => item.id === probeMode)!;
  const selected = snapshot && selectedToken < snapshot.activeVocab ? selectedToken : 0;

  const insertLetter = useCallback((point: [number, number, number]) => {
    const current = simulationRef.current;
    if (!current || frame !== current.history.length - 1 || current.activeVocab >= TOKENS.length) return;
    const event = current.insertLetter(point);
    if (!event) return;
    setPlacingLetter(false);
    syncFromSimulation(`Inserted ${event.token} at iteration ${event.step}; its optimizer state is zero and it is denominator-only.`);
    setSelectedToken(current.activeVocab - 1);
  }, [frame, syncFromSimulation]);

  const handleScale = useCallback((next: { min: number; max: number }) => {
    setScale((old) => Math.abs(old.min - next.min) < 1e-12 && Math.abs(old.max - next.max) < 1e-12 ? old : next);
  }, []);

  if (!snapshot) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-[#050914] text-slate-100">
        <div className="max-w-md text-center">
          <div className="mx-auto mb-5 grid size-14 place-items-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-300"><Atom className="size-7 animate-pulse" /></div>
          <h1 className="text-xl font-semibold">Building the 3D transformer state</h1>
          <p className="mt-2 text-sm text-slate-400">Loading TensorFlow.js on the CPU and evaluating the exact tied-row gradient at iteration 0.</p>
          {error && <p className="mt-4 rounded-lg border border-red-400/20 bg-red-400/10 p-3 text-sm text-red-300">{error}</p>}
        </div>
      </main>
    );
  }

  const vector = (source: number[], token = selected) => source.slice(token * D_MODEL, token * D_MODEL + D_MODEL);
  const rawForce = vector(snapshot.rawGradients).map((v) => -v);
  const tangentForce = vector(snapshot.tangentForces);
  const radialForce = rawForce.map((v, i) => v - tangentForce[i]);
  const nextLetter = LETTERS[snapshot.activeVocab - 10] ?? null;

  return (
    <main className="sphere-lab min-h-screen bg-[#050914] text-slate-100">
      <header className="lab-header border-b border-white/[0.07] bg-[#07101e]/95 px-4 py-3 backdrop-blur-xl lg:px-6">
        <div className="mx-auto flex max-w-[1700px] flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-300"><Atom className="size-5" /></div>
            <div>
              <h1 className="text-[17px] font-semibold tracking-tight">Sphere Force Lab</h1>
              <p className="lab-subtitle text-[11px] text-slate-500">Native 3D tied-row dynamics</p>
            </div>
          </div>
          <div className="lab-model-badges flex flex-wrap items-center gap-2 text-[11px]">
            <span className="chip"><Cpu className="size-3.5" /> CPU</span>
            <span className="chip">1 causal block</span>
            <span className="chip">d = 3</span>
            <span className="chip">R = √3</span>
            <span className="chip chip-live"><CircleDot className="size-3.5" /> tied WTE / LM head</span>
          </div>
        </div>
      </header>

      <Tabs value={mobilePanel} onValueChange={setMobilePanel} className="lab-workspace mx-auto max-w-[1700px] gap-4 p-4 lg:p-5">
        <TabsList aria-label="Experiment panels" className="mobile-panel-tabs">
          <TabsTrigger value="controls"><Gauge /> Controls</TabsTrigger>
          <TabsTrigger value="diagnostics"><Activity /> Diagnostics</TabsTrigger>
        </TabsList>
        <TabsContent value="controls" forceMount asChild>
        <aside className="lab-controls lab-detail-panel panel space-y-5 p-4">
          <section>
            <div className="section-label"><Gauge className="size-3.5" /> Training</div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Button className="h-10 bg-cyan-300 text-[#04101c] hover:bg-cyan-200" disabled={!ready || working} onClick={() => setRunning((value) => !value)}>
                {running ? <Pause /> : <Play />} {running ? "Pause" : "Run"}
              </Button>
              <Button className="h-10 border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]" variant="outline" disabled={!ready || working} onClick={() => void trainMany(1)}>
                <SkipForward /> Step
              </Button>
              <Button className="border-white/10 bg-white/[0.04] text-slate-300 hover:bg-white/[0.08]" variant="outline" size="sm" disabled={!ready || working} onClick={() => void trainMany(10)}>+10</Button>
              <Button className="border-white/10 bg-white/[0.04] text-slate-300 hover:bg-white/[0.08]" variant="outline" size="sm" disabled={!ready || working} onClick={() => void trainMany(50)}>+50</Button>
            </div>
            <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full bg-gradient-to-r from-cyan-400 to-fuchsia-400" style={{ width: `${Math.min(100, (snapshot.step / MAX_STEPS) * 100)}%` }} /></div>
            <div className="mt-1.5 flex justify-between font-mono text-[10px] text-slate-500"><span>iteration {snapshot.step}</span><span>cap {MAX_STEPS}</span></div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="section-label"><RotateCcw className="size-3.5" /> Configuration</div>
            <div className="mt-3 space-y-3">
              <label className="control-label">Optimizer
                <Select value={optimizer} onValueChange={(value) => setOptimizer(value as OptimizerKind)}>
                  <SelectTrigger className="mt-1 w-full border-white/10 bg-[#091527] text-slate-200"><SelectValue /></SelectTrigger>
                  <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
                    <SelectItem value="adamw">AdamW</SelectItem>
                    <SelectItem value="rmsprop">RMSProp</SelectItem>
                  </SelectContent>
                </Select>
              </label>
              <div className="grid grid-cols-2 gap-2">
                <label className="control-label">Learning rate
                  <input className="lab-input mt-1" type="number" min="0.0001" max="0.2" step="0.001" value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} />
                </label>
                <label className="control-label">Weight decay
                  <input className="lab-input mt-1" type="number" min="0" max="1" step="0.01" value={weightDecay} onChange={(e) => setWeightDecay(Number(e.target.value))} />
                </label>
              </div>
              <label className="control-label">Seed
                <input className="lab-input mt-1" type="number" step="1" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              </label>
              <Button variant="outline" className="w-full border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]" onClick={() => void initialize()} disabled={working}>
                <RotateCcw /> Apply + reset timeline
              </Button>
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="section-label"><Sparkles className="size-3.5" /> View</div>
            <div className="mt-3 space-y-1.5">
              {modes.map((item) => {
                const Icon = item.icon;
                const active = mode === item.id;
                return (
                  <button key={item.id} type="button" onClick={() => setMode(item.id)} className={`view-button ${active ? "view-button-active" : ""}`}>
                    <Icon className="size-4" /><span><strong>{item.label}</strong><small>{item.description}</small></span>
                  </button>
                );
              })}
            </div>
            {mode === "probe" && (
              <label className="control-label mt-3 block">Surface quantity
                <Select value={probeMode} onValueChange={(value) => setProbeMode(value as ProbeMode)}>
                  <SelectTrigger className="mt-1 w-full border-white/10 bg-[#091527] text-slate-200"><SelectValue /></SelectTrigger>
                  <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
                    {probeModes.map((item) => <SelectItem key={item.id} value={item.id}>{item.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </label>
            )}
            <div className="mt-3 space-y-2.5">
              <label className="toggle-row"><span>Trajectory trails</span><Switch checked={showTrails} onCheckedChange={setShowTrails} /></label>
              <label className="toggle-row"><span>Normalize arrow lengths</span><Switch checked={normalizeArrows} onCheckedChange={setNormalizeArrows} /></label>
              {mode === "probe" && <label className="toggle-row"><span>Sparse field arrows</span><Switch checked={showFieldArrows} onCheckedChange={setShowFieldArrows} /></label>}
              {mode === "probe" && <label className="toggle-row"><span>Per-frame color scale</span><Switch checked={frameAutoscale} onCheckedChange={setFrameAutoscale} /></label>}
            </div>
          </section>
        </aside>
        </TabsContent>

        <section className="lab-stage min-w-0">
          <div className="mobile-quick-controls">
            <div className="mobile-training-actions">
              <Button disabled={!ready || working} onClick={() => setRunning((value) => !value)}>{running ? <Pause /> : <Play />}{running ? "Pause" : "Run"}</Button>
              <Button variant="secondary" disabled={!ready || working} onClick={() => void trainMany(1)}><SkipForward /> Step</Button>
              <Button variant="secondary" disabled={!ready || working} onClick={() => void trainMany(10)}>+10</Button>
              <Button variant="outline" aria-pressed={placingLetter} disabled={!ready || !isLive || !nextLetter || working} onClick={() => setPlacingLetter((value) => !value)}>{placingLetter ? "Cancel" : `Place ${nextLetter ?? "—"}`}</Button>
            </div>
            <div className="mobile-view-row">
              <Select value={mode} onValueChange={(value) => setMode(value as ViewMode)}>
                <SelectTrigger aria-label="Visualization" className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{modes.map((item) => <SelectItem key={item.id} value={item.id}>{item.label}</SelectItem>)}</SelectContent>
              </Select>
              <div className="mobile-live-metrics"><span>Step <b>{snapshot.step}</b></span><span>Loss <b>{format(snapshot.loss, 3)}</b></span></div>
            </div>
          </div>
          <div className="sphere-viewport relative h-[min(68vh,760px)] min-h-[500px] overflow-hidden rounded-[24px] border border-white/[0.08] bg-[#050914] shadow-2xl shadow-black/25">
            <SphereScene
              snapshot={snapshot}
              history={history}
              frameIndex={frame}
              mode={mode}
              probeMode={probeMode}
              selectedToken={selected}
              showTrails={showTrails}
              showFieldArrows={showFieldArrows}
              normalizeArrows={normalizeArrows}
              frameAutoscale={frameAutoscale}
              insertionAllowed={isLive && snapshot.activeVocab < 20 && ready && !working && (!compact || placingLetter)}
              compact={compact}
              onInsert={insertLetter}
              onScale={handleScale}
            />
            <div className="sphere-caption pointer-events-none absolute bottom-4 left-4 right-4 flex flex-wrap items-end justify-between gap-3">
              <div className="max-w-[430px] rounded-xl border border-white/10 bg-[#07101e]/88 p-3 backdrop-blur-md">
                <div className="flex items-center gap-2 text-sm font-semibold text-white"><activeMode.icon className="size-4 text-cyan-300" />{activeMode.label}</div>
                <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{activeMode.description} Displayed gradient is evaluated after iteration {snapshot.step} projection and before the next update.</p>
                {mode === "decomposition" && <div className="mt-2 flex gap-3 text-[10px]"><span className="text-cyan-300">● ambient −∇L</span><span className="text-fuchsia-300">● tangent −P∇L</span><span className="text-amber-300">● radial removed</span></div>}
                {mode === "forces" && <div className="mt-2 text-[10px] text-fuchsia-300">Magenta arrows = actual tied-row CE tangent force</div>}
                {mode === "optimizer" && <div className="mt-2 text-[10px] text-emerald-300">Green arrows = observed projected move from the previous iteration</div>}
              </div>
              {mode === "probe" && (
                <div className="w-[220px] rounded-xl border border-white/10 bg-[#07101e]/88 p-3 backdrop-blur-md">
                  <div className="flex justify-between text-[10px] text-slate-400"><span>{activeProbeMode.label}</span><span>{frameAutoscale ? "frame" : "fixed"}</span></div>
                  <div className="heatbar mt-2 h-2.5 rounded-full" />
                  <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-300"><span>{format(scale.min, 3)}</span><span>{activeProbeMode.units}</span><span>{format(scale.max, 3)}</span></div>
                </div>
              )}
            </div>
          </div>

          <div className="lab-timeline panel mt-4 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="section-label"><History className="size-3.5" /> Iteration timeline</div>
                <p className="mt-1 text-[11px] text-slate-500">Every step is retained. Scrubbing changes rows, arrows, heatmap, and numbers together.</p>
              </div>
              {!isLive && <Button size="sm" className="bg-cyan-300 text-[#04101c] hover:bg-cyan-200" onClick={() => setFrame(liveFrame)}>Return to live · {liveFrame}</Button>}
            </div>
            <div className="relative mt-4 px-1">
              <Slider aria-label="Training iteration" min={0} max={Math.max(1, liveFrame)} disabled={liveFrame === 0 || working} step={1} value={[frame]} onValueChange={(value) => { setRunning(false); setPlacingLetter(false); setFrame(value[0]); }} />
              <div className="timeline-events pointer-events-none absolute inset-x-1 top-[5px] h-1.5">
                {events.map((event, index) => <span key={`${event.token}-${event.step}-${index}`} className="absolute top-0 size-1.5 -translate-x-1/2 rounded-full bg-amber-300 ring-2 ring-[#07101e]" style={{ left: `${liveFrame ? (event.step / liveFrame) * 100 : 0}%` }} title={`${event.token} inserted at step ${event.step}`} />)}
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px]">
              <div className="font-mono text-slate-300">frame {frame} / {liveFrame} · optimizer iteration {snapshot.step}</div>
              <div className="flex flex-wrap gap-1.5">
                {events.length ? events.map((event, i) => <span key={`${event.token}-${i}`} className="rounded-full border border-amber-300/20 bg-amber-300/10 px-2 py-0.5 text-amber-200">{event.token} @ {event.step}</span>) : <span className="text-slate-500">No letter insertions yet</span>}
              </div>
            </div>
          </div>
        </section>

        <TabsContent value="diagnostics" forceMount asChild>
        <aside className="lab-diagnostics lab-detail-panel panel space-y-5 p-4">
          <section>
            <div className="flex items-center justify-between gap-3">
              <div className="section-label"><Activity className="size-3.5" /> Selected iteration</div>
              <span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${isLive ? "bg-emerald-300/10 text-emerald-300" : "bg-amber-300/10 text-amber-300"}`}>{isLive ? "LIVE EDGE" : "HISTORY"}</span>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Metric label="Loss" value={format(snapshot.loss)} accent="cyan" />
              <Metric label="Next-digit acc." value={`${(snapshot.accuracy * 100).toFixed(1)}%`} accent="green" />
              <Metric label="Mean tangent force" value={format(snapshot.meanTangentForce)} accent="pink" />
              <Metric label="Unused prob. mass" value={format(snapshot.unusedMass)} accent="gold" />
              <Metric label="Letter pair distance" value={snapshot.activeVocab > 11 ? format(snapshot.meanLetterDistance) : "—"} detail="mean chord" />
              <Metric label="Max norm error" value={format(snapshot.maxNormError)} detail="from √3" />
            </div>
            <div className="mt-3 rounded-xl border border-white/[0.06] bg-white/[0.025] px-3 pt-2">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500"><span>Cross-entropy loss</span><span>0 → {snapshot.step}</span></div>
              <TinyLossChart history={history} frame={frame} />
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="flex items-center justify-between gap-3">
              <div className="section-label"><Plus className="size-3.5" /> Add untargeted row</div>
              <span className="font-mono text-[10px] text-slate-500">{snapshot.activeVocab - 10}/10</span>
            </div>
            <div className={`mt-3 rounded-xl border p-3 ${nextLetter && isLive ? "border-amber-300/20 bg-amber-300/[0.06]" : "border-white/[0.06] bg-white/[0.025]"}`}>
              {nextLetter ? (
                <>
                  <div className="flex items-center gap-3"><span className="grid size-9 place-items-center rounded-lg border border-amber-300/30 bg-amber-300/10 font-mono text-lg font-bold text-amber-200">{nextLetter}</span><div><div className="text-sm font-medium">{compact ? `Choose “Place ${nextLetter}”, then tap the sphere` : "Click any sphere point"}</div><div className="text-[10px] text-slate-500">{compact ? "Drag to rotate without inserting." : "Short click inserts; drag only rotates."}</div></div></div>
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-400">The new row is normalized to √3, receives zero optimizer moments, joins the softmax denominator, and never appears in inputs or targets.</p>
                  {!isLive && <p className="mt-2 text-[10px] font-medium text-amber-300">Return to the live edge before inserting.</p>}
                </>
              ) : <p className="text-[11px] text-slate-400">All ten letters a–j are active. Further sphere clicks do not mutate the model.</p>}
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="flex items-center justify-between gap-3">
              <div className="section-label"><CircleDot className="size-3.5" /> Row inspector</div>
              <Select value={String(selected)} onValueChange={(value) => setSelectedToken(Number(value))}>
                <SelectTrigger size="sm" className="w-[92px] border-white/10 bg-[#091527] font-mono text-slate-200"><SelectValue /></SelectTrigger>
                <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
                  {TOKENS.slice(0, snapshot.activeVocab).map((token, index) => <SelectItem key={token} value={String(index)}>{token} · {index < 10 ? "digit" : "letter"}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <table className="mt-3 w-full table-fixed" aria-label={`Coordinates and forces for token ${TOKENS[selected]}`}>
              <thead><tr className="text-[10px] uppercase tracking-wider text-slate-600"><th className="pb-1 text-left">vector</th><th className="pb-1 text-right">x</th><th className="pb-1 text-right">y</th><th className="pb-1 text-right">z</th></tr></thead>
              <tbody>
                <VectorRow label="row w" values={vector(snapshot.positions)} color="#67e8f9" />
                <VectorRow label="ambient −∇L" values={rawForce} color="#67e8f9" />
                <VectorRow label="tangent −P∇L" values={tangentForce} color="#f0abfc" />
                <VectorRow label="radial removed" values={radialForce} color="#fcd34d" />
                <VectorRow label="observed Δw" values={vector(snapshot.optimizerMoves)} color="#6ee7b7" />
              </tbody>
            </table>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Metric label="Projection correction" value={format(snapshot.projectionCorrection)} />
              <Metric label="Move · force cosine" value={format(snapshot.optimizerForceCosine)} />
              <Metric label="Tangency residual" value={format(snapshot.maxTangencyError)} detail="max normalized |w·F|" />
              <Metric label="Unused-gradient check" value={snapshot.activeVocab > 10 ? format(snapshot.unusedGradientError) : "—"} detail="autodiff − analytic" />
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="section-label"><Braces className="size-3.5" /> Exact field being shown</div>
            <div className="mt-3 rounded-xl border border-cyan-300/10 bg-cyan-300/[0.04] p-3 font-mono text-[11px] leading-relaxed text-cyan-100/80">
              <div>pᵤ(s) = σ(uᵀhₛ − log Zₛ)</div>
              <div>g(u) = meanₛ pᵤ(s)hₛ</div>
              <div className="text-fuchsia-200">F(u) = −(I − uuᵀ/R²)g(u)</div>
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-slate-500">The surface freezes the selected iteration’s 100 hidden states and active-vocabulary partition. It is an exact instantaneous counterfactual for one newly inserted untargeted row—not a forecast of its full future path.</p>
          </section>
        </aside>
        </TabsContent>
      </Tabs>

      <div className="sr-only" aria-live="polite">{announcement}</div>
      {error && <div className="fixed bottom-4 left-1/2 z-50 max-w-lg -translate-x-1/2 rounded-xl border border-red-400/20 bg-[#2a1018]/95 px-4 py-3 text-sm text-red-200 shadow-xl">{error}</div>}
    </main>
  );
}
