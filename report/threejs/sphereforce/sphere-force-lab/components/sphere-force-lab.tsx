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
  Rewind,
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
import { ArchitectureControls } from "@/components/architecture-controls";
import { DatasetControls } from "@/components/dataset-controls";
import { DatasetConfig, validateDataset } from "@/lib/dataset";
import { ArchitectureConfig, DEFAULT_ARCHITECTURE, validateArchitecture } from "@/lib/architecture";
import { toDisplaySnapshot } from "@/lib/view-projection";
import { SphereScene, ViewMode } from "@/components/sphere-scene";
import { TargetScheduleControls } from "@/components/target-schedule-controls";
import { TargetRuleDraft } from "@/lib/target-schedule";
import { BlendSchedule, DEFAULT_QAT, QatConfig, QUANTIZATION_FORMATS, QuantizationFormat, validateQat } from "@/lib/quantization";
import {
  InsertionEvent,
  MAX_TARGET_TOKENS,
  MAX_UNTARGETED_TOKENS,
  OptimizerKind,
  ProbeMode,
  SimConfig,
  Snapshot,
  TransformerSphereSimulation,
  tokenLabel,
} from "@/lib/simulator";

const DEFAULT_MAX_ITERATIONS = 1200;
const MAX_ITERATIONS = 50_000;
const REPLAY_FRAME_MS = 40;
const COMPACT_QUERY = "(max-width: 1279px)";
function subscribeToLayout(callback: () => void) {
  const query = window.matchMedia(COMPACT_QUERY);
  query.addEventListener("change", callback);
  return () => query.removeEventListener("change", callback);
}
const getCompactLayout = () => window.matchMedia(COMPACT_QUERY).matches;
const getServerLayout = () => false;

const modes: { id: ViewMode; label: string; icon: typeof Atom; description: string }[] = [
  { id: "rows", label: "Rows + trails", icon: Atom, description: "Tied token rows and their recorded paths." },
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

function boundedInteger(value: number, min: number, max: number, fallback: number) {
  return Number.isFinite(value) ? Math.min(max, Math.max(min, Math.round(value))) : fallback;
}

function formatMemory(bytes: number) {
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(bytes < 100 * 1024 ** 2 ? 1 : 0)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function estimateHistoryBytes(targeted: number, untargeted: number, batchSize: number, iterations: number, architecture: ArchitectureConfig) {
  const samples = architecture.maxContextLength * batchSize;
  const d = architecture.modelDim;
  const activeRows = targeted + untargeted;
  const bytesPerFrame = samples * (d + 1) * 4 + activeRows * (d + 3) * 5 * 4 + targeted * (16 * d + 240) + (architecture.maxContextLength + 1) * 8 + 800;
  return bytesPerFrame * (iterations + 1);
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
    const source = history.slice(0, frame + 1);
    const stride = Math.max(1, Math.ceil(source.length / 260));
    const points = source.map((point, index) => ({ point, index })).filter((_, index) => index % stride === 0 || index === source.length - 1);
    if (points.length < 2) return "";
    const values = points.map((p) => p.point.loss);
    const finite = values.filter(Number.isFinite);
    if (!finite.length) return "";
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    const span = Math.max(max - min, 1e-8);
    let drawing = false;
    let previousIndex = -1;
    return values.map((v, i) => {
      for (let index = previousIndex + 1; index <= points[i].index; index += 1) {
        if (!Number.isFinite(source[index].loss)) drawing = false;
      }
      previousIndex = points[i].index;
      if (!Number.isFinite(v)) { drawing = false; return ""; }
      const x = (points[i].index / Math.max(1, source.length - 1)) * 260;
      const y = 58 - ((v - min) / span) * 48;
      const command = drawing ? "L" : "M";
      drawing = true;
      return `${command}${x.toFixed(2)},${y.toFixed(2)}`;
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
  const [replaying, setReplaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [timeline, setTimeline] = useState<{ history: Snapshot[]; events: InsertionEvent[]; liveStep: number; revision: number }>({
    history: [],
    events: [],
    liveStep: 0,
    revision: 0,
  });
  const [frame, setFrame] = useState(0);
  const [mode, setMode] = useState<ViewMode>("probe");
  const [probeMode, setProbeMode] = useState<ProbeMode>("force");
  const [selectedToken, setSelectedToken] = useState(0);
  const [showTrails, setShowTrails] = useState(true);
  const [showFieldArrows, setShowFieldArrows] = useState(false);
  const [normalizeArrows, setNormalizeArrows] = useState(false);
  const [frameAutoscale, setFrameAutoscale] = useState(false);
  const [showHiddenOnSphere, setShowHiddenOnSphere] = useState(false);
  const [showHiddenInSpace, setShowHiddenInSpace] = useState(false);
  const [scale, setScale] = useState({ min: 0, max: 0.18 });
  const [seed, setSeed] = useState(17);
  const [optimizer, setOptimizer] = useState<OptimizerKind>("adamw");
  const [learningRate, setLearningRate] = useState(0.018);
  const [weightDecay, setWeightDecay] = useState(0.05);
  const [targetedTokens, setTargetedTokens] = useState(10);
  const [untargetedTokens, setUntargetedTokens] = useState(10);
  const [batchSize, setBatchSize] = useState(10);
  const [maxIterations, setMaxIterations] = useState(DEFAULT_MAX_ITERATIONS);
  const [maxIterationsDraft, setMaxIterationsDraft] = useState(DEFAULT_MAX_ITERATIONS);
  const [architectureDraft, setArchitectureDraft] = useState<ArchitectureConfig>({ ...DEFAULT_ARCHITECTURE });
  const [datasetDraft, setDatasetDraft] = useState<DatasetConfig>({ mode: "cycle", sampling: "per-step", seed: 17 });
  const [qatDraft, setQatDraft] = useState<QatConfig>({ ...DEFAULT_QAT });
  const [error, setError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("Initializing the CPU simulation…");

  const initialize = useCallback(async (settings?: Partial<SimConfig>) => {
    try {
      validateQat(settings?.qat ?? qatDraft);
      validateDataset(settings?.dataset ?? datasetDraft, settings?.targetedTokens ?? boundedInteger(targetedTokens, 1, MAX_TARGET_TOKENS, 10), settings?.seed ?? seed);
      const architecture = validateArchitecture(settings?.architecture ?? architectureDraft, settings?.batchSize ?? batchSize, (settings?.targetedTokens ?? targetedTokens) + MAX_UNTARGETED_TOKENS);
      if (estimateHistoryBytes(settings?.targetedTokens ?? targetedTokens, settings?.untargetedTokens ?? untargetedTokens, settings?.batchSize ?? batchSize, maxIterationsDraft, architecture) > 1024 ** 3) throw new Error("Recorded history would exceed 1 GB. Reduce iterations, batch size, context length, or model dimension before resetting.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return;
    }
    const initId = ++initCounter.current;
    setRunning(false);
    setReplaying(false);
    setPlacingLetter(false);
    setReady(false);
    setError(null);
    setAnnouncement("Resetting model, optimizer state, and timeline…");

    try {
      const sim = await TransformerSphereSimulation.create({
        seed: settings?.seed ?? seed,
        optimizer: settings?.optimizer ?? optimizer,
        learningRate: settings?.learningRate ?? learningRate,
        weightDecay: settings?.weightDecay ?? weightDecay,
        targetedTokens: settings?.targetedTokens ?? boundedInteger(targetedTokens, 1, MAX_TARGET_TOKENS, 10),
        untargetedTokens: settings?.untargetedTokens ?? boundedInteger(untargetedTokens, 0, MAX_UNTARGETED_TOKENS, 10),
        batchSize: settings?.batchSize ?? boundedInteger(batchSize, 1, 100, 10),
        qat: settings?.qat ?? qatDraft,
        architecture: settings?.architecture ?? architectureDraft,
        dataset: settings?.dataset ?? datasetDraft,
      });
      if (initCounter.current !== initId) {
        sim.dispose();
        return;
      }
      simulationRef.current?.dispose();
      simulationRef.current = sim;
      setArchitectureDraft({ ...sim.architecture });
      if (sim.modelDim === 2) setMode("rows");
      setTimeline((current) => ({ history: sim.history, events: sim.events, liveStep: sim.step, revision: current.revision + 1 }));
      setFrame(0);
      setSelectedToken(0);
      setTargetedTokens(sim.config.targetedTokens);
      setUntargetedTokens(sim.config.untargetedTokens);
      setBatchSize(sim.config.batchSize);
      const nextLimit = boundedInteger(maxIterationsDraft, 1, MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS);
      setMaxIterations(nextLimit);
      setMaxIterationsDraft(nextLimit);
      setReady(true);
      setAnnouncement(`Ready at iteration 0 with ${sim.config.targetedTokens} targeted and ${sim.config.untargetedTokens} random untargeted rows.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setReady(Boolean(simulationRef.current));
      setAnnouncement("The simulation could not initialize. The previous run is retained.");
    }
  }, [architectureDraft, datasetDraft, batchSize, learningRate, maxIterationsDraft, optimizer, qatDraft, seed, targetedTokens, untargetedTokens, weightDecay]);

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
    setTimeline((current) => ({ history: sim.history, events: sim.events, liveStep: sim.step, revision: current.revision + 1 }));
    setFrame(sim.history.length - 1);
    if (announce) setAnnouncement(announce);
  }, []);

  const history = timeline.history;
  const displayHistory = useMemo(() => history.map(toDisplaySnapshot), [history, timeline.revision]);
  const events = timeline.events;
  const liveFrame = Math.max(0, history.length - 1);

  const trainMany = useCallback(async (count: number) => {
    const sim = simulationRef.current;
    if (!sim || working || !ready) return;
    setWorking(true);
    setRunning(false);
    setReplaying(false);
    setError(null);
    try {
      const remaining = Math.max(0, Math.min(count, maxIterations - sim.step));
      for (let i = 0; i < remaining; i += 1) {
        sim.trainOne();
        if (i % 3 === 2) await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
      syncFromSimulation(remaining > 0 ? `Advanced to iteration ${sim.step}.` : `Iteration limit ${maxIterations} reached.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setWorking(false);
    }
  }, [maxIterations, ready, syncFromSimulation, working]);

  useEffect(() => {
    if (!running || !ready || working) return;
    let cancelled = false;
    let timer = 0;
    const tick = () => {
      if (cancelled) return;
      const sim = simulationRef.current;
      if (!sim || sim.step >= maxIterations) {
        setRunning(false);
        if (sim) setAnnouncement(`Paused at the iteration limit (${maxIterations}). Raise the limit to continue this run.`);
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
  }, [maxIterations, ready, running, syncFromSimulation, working]);

  useEffect(() => {
    if (!replaying || running || working || liveFrame === 0) return;
    const nextFrame = Math.min(liveFrame, frame + 1);
    const timer = window.setTimeout(
      () => {
        setFrame(nextFrame);
        if (nextFrame >= liveFrame) {
          setReplaying(false);
          setAnnouncement(`Replay complete at iteration ${history[liveFrame]?.step ?? liveFrame}.`);
        }
      },
      REPLAY_FRAME_MS / replaySpeed,
    );
    return () => window.clearTimeout(timer);
  }, [frame, history, liveFrame, replaySpeed, replaying, running, working]);

  const toggleReplay = useCallback(() => {
    if (liveFrame === 0) return;
    setRunning(false);
    setPlacingLetter(false);
    if (replaying) {
      setReplaying(false);
      setAnnouncement(`Replay paused at frame ${frame}.`);
      return;
    }
    if (frame >= liveFrame) setFrame(0);
    setReplaying(true);
    setAnnouncement(frame >= liveFrame ? "Replaying the recorded run from iteration 0." : `Resuming replay from frame ${frame}.`);
  }, [frame, liveFrame, replaying]);

  const applyLimitAndResume = useCallback(() => {
    const sim = simulationRef.current;
    if (!sim) return;
    const nextLimit = boundedInteger(maxIterationsDraft, 1, MAX_ITERATIONS, maxIterations);
    if (estimateHistoryBytes(sim.config.targetedTokens, sim.activeUntargetedCount, sim.config.batchSize, nextLimit, sim.architecture) > 1024 ** 3) {
      setError("Recorded history would exceed 1 GB. Choose a lower iteration limit.");
      return;
    }
    if (nextLimit < sim.step) {
      setError(`The new limit must be at least the current iteration (${sim.step}).`);
      return;
    }
    setError(null);
    setMaxIterations(nextLimit);
    setMaxIterationsDraft(nextLimit);
    setReplaying(false);
    setFrame(sim.history.length - 1);
    if (nextLimit > sim.step) {
      setRunning(true);
      setAnnouncement(`Training resumed toward the new limit of ${nextLimit} iterations.`);
    } else {
      setRunning(false);
      setAnnouncement(`Iteration limit is ${nextLimit}; raise it above ${sim.step} to continue.`);
    }
  }, [maxIterations, maxIterationsDraft]);

  const applyQat = (resume: boolean) => {
    const sim = simulationRef.current;
    if (!sim || working) return;
    setRunning(false);
    setReplaying(false);
    setPlacingLetter(false);
    setError(null);
    try {
      if (resume && sim.step >= MAX_ITERATIONS) throw new Error("The 50,000 iteration limit has been reached.");
      const config = { ...qatDraft, start: resume ? sim.step : qatDraft.start };
      sim.applyQat(config);
      setQatDraft(config);
      syncFromSimulation(`Applied ${config.format} at iteration ${sim.step}; weights and optimizer state retained.`);
      if (resume) {
        const requestedLimit = boundedInteger(maxIterationsDraft, 1, MAX_ITERATIONS, maxIterations);
        const nextLimit = requestedLimit > sim.step ? requestedLimit : Math.min(MAX_ITERATIONS, sim.step + (config.schedule === "immediate" ? 200 : config.duration) + 1);
        if (estimateHistoryBytes(sim.config.targetedTokens, sim.activeUntargetedCount, sim.config.batchSize, nextLimit, sim.architecture) > 1024 ** 3) throw new Error("QAT applied and paused: the requested history exceeds 1 GB. Lower the iteration limit to continue.");
        setMaxIterations(nextLimit);
        setMaxIterationsDraft(nextLimit);
        setRunning(true);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const changeTargetPolicy = (draft: TargetRuleDraft | number) => {
    const sim = simulationRef.current;
    if (!sim || working) return;
    setRunning(false);
    setReplaying(false);
    setPlacingLetter(false);
    setError(null);
    try {
      if (typeof draft === "number") sim.cancelTargetRule(draft);
      else sim.scheduleTarget(draft);
      syncFromSimulation(sim.latest.datasetChanged);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const snapshot = history[Math.min(frame, history.length - 1)];
  const isLive = frame === liveFrame;
  const activeMode = modes.find((item) => item.id === mode)!;
  const activeProbeMode = probeModes.find((item) => item.id === probeMode)!;
  const selected = snapshot && selectedToken < snapshot.activeVocab ? selectedToken : 0;

  const insertLetter = useCallback((point: [number, number, number]) => {
    const current = simulationRef.current;
    if (!current || frame !== current.history.length - 1 || current.activeUntargetedCount >= MAX_UNTARGETED_TOKENS) return;
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
          <h1 className="text-xl font-semibold">Building the transformer state</h1>
          <p className="mt-2 text-sm text-slate-400">Loading TensorFlow.js on the CPU and evaluating the exact tied-row gradient at iteration 0.</p>
          {error && <p className="mt-4 rounded-lg border border-red-400/20 bg-red-400/10 p-3 text-sm text-red-300">{error}</p>}
        </div>
      </main>
    );
  }

  const displaySnapshot = displayHistory[Math.min(frame, displayHistory.length - 1)];
  const vector = (source: Float32Array, token = selected) => Array.from({ length: 3 }, (_, d) => d < snapshot.modelDim ? source[token * snapshot.modelDim + d] : 0);
  const rawForce = vector(snapshot.rawGradients).map((v) => -v);
  const tangentForce = vector(snapshot.tangentForces);
  const radialForce = rawForce.map((v, i) => v - tangentForce[i]);
  const activeUntargeted = snapshot.activeVocab - snapshot.includedTargetCount;
  const addedUntargeted = snapshot.activeVocab - snapshot.targetCount;
  const qatActive = snapshot.qatBlend > 0;
  const precisionLabel = QUANTIZATION_FORMATS.find((item) => item.value === snapshot.qat.format)!.label;
  const qatEnd = qatDraft.start + (qatDraft.schedule === "immediate" ? 0 : qatDraft.duration);
  const nextUntargeted = addedUntargeted < MAX_UNTARGETED_TOKENS ? tokenLabel(snapshot.activeVocab, snapshot.targetCount) : null;
  const estimatedHistory = estimateHistoryBytes(
    boundedInteger(targetedTokens, 1, MAX_TARGET_TOKENS, 10),
    boundedInteger(untargetedTokens, 0, MAX_UNTARGETED_TOKENS, 10),
    boundedInteger(batchSize, 1, 100, 10),
    boundedInteger(maxIterationsDraft, 1, MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS),
    architectureDraft,
  );
  const liveStep = timeline.liveStep;
  const atLimit = liveStep >= maxIterations;
  const toggleTraining = () => {
    setReplaying(false);
    setPlacingLetter(false);
    setFrame(liveFrame);
    if (atLimit) {
      setAnnouncement(`Iteration limit ${maxIterations} reached. Enter a higher limit and choose “Set + resume”.`);
      return;
    }
    setRunning((value) => !value);
  };

  return (
    <main className="sphere-lab min-h-screen bg-[#050914] text-slate-100">
      <header className="lab-header border-b border-white/[0.07] bg-[#07101e]/95 px-4 py-3 backdrop-blur-xl lg:px-6">
        <div className="mx-auto flex max-w-[1700px] flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-300"><Atom className="size-5" /></div>
            <div>
              <h1 className="text-[17px] font-semibold tracking-tight">Sphere Force Lab</h1>
              <p className="lab-subtitle text-[11px] text-slate-500">{snapshot.modelDim === 3 ? "Native 3D tied-row dynamics" : `${snapshot.modelDim}D tied-row dynamics · first 3 coordinates`}</p>
            </div>
          </div>
          <div className="lab-model-badges flex flex-wrap items-center gap-2 text-[11px]">
            <span className="chip"><Cpu className="size-3.5" /> CPU</span>
            <span className="chip">{snapshot.architecture.layers} {snapshot.architecture.blockMode === "mlp" ? "MLP" : "decoder"} block{snapshot.architecture.layers === 1 ? "" : "s"}</span>
            <span className="chip">d = {snapshot.modelDim}</span>
            <span className="chip">R = √{snapshot.modelDim}</span>
            <span className="chip">{snapshot.sequenceLength} context · {snapshot.parameterCount.toLocaleString()} parameters</span>
            <span className="chip">{snapshot.dataset.mode === "cycle" ? "Direct cycle" : "Markov"}</span>
            <span className="chip">{snapshot.includedTargetCount}/{snapshot.targetCount} targets included</span>
            <span className="chip">{activeUntargeted} untargeted</span>
            <span className="chip chip-live"><CircleDot className="size-3.5" /> tied WTE / LM head</span>
            <span className="chip">{snapshot.qat.format === "fp32" ? "FP32" : `${snapshot.qat.format} · ${(snapshot.qatBlend * 100).toFixed(0)}% blend`}</span>
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
              <Button className="h-10 bg-cyan-300 text-[#04101c] hover:bg-cyan-200" disabled={!ready || working || atLimit} onClick={toggleTraining}>
                {running ? <Pause /> : <Play />} {running ? "Pause" : atLimit ? "At limit" : "Run"}
              </Button>
              <Button className="h-10 border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]" variant="outline" disabled={!ready || working || atLimit} onClick={() => void trainMany(1)}>
                <SkipForward /> Step
              </Button>
              <Button className="border-white/10 bg-white/[0.04] text-slate-300 hover:bg-white/[0.08]" variant="outline" size="sm" disabled={!ready || working || atLimit} onClick={() => void trainMany(10)}>+10</Button>
              <Button className="border-white/10 bg-white/[0.04] text-slate-300 hover:bg-white/[0.08]" variant="outline" size="sm" disabled={!ready || working || atLimit} onClick={() => void trainMany(50)}>+50</Button>
            </div>
            <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full bg-gradient-to-r from-cyan-400 to-fuchsia-400" style={{ width: `${Math.min(100, (snapshot.step / Math.max(1, maxIterations)) * 100)}%` }} /></div>
            <div className="mt-1.5 flex justify-between font-mono text-[10px] text-slate-500"><span>iteration {snapshot.step}</span><span>limit {maxIterations}</span></div>
            <div className="mt-3 grid grid-cols-[minmax(0,1fr)_auto] gap-2">
              <label className="control-label">Maximum iterations
                <input aria-label="Maximum iterations" className="lab-input mt-1" type="number" min="1" max={MAX_ITERATIONS} step="100" value={maxIterationsDraft} onChange={(event) => setMaxIterationsDraft(Number(event.target.value))} />
              </label>
              <Button className="mt-[18px] border-cyan-300/20 bg-cyan-300/10 text-cyan-200 hover:bg-cyan-300/15" variant="outline" disabled={!ready || working} onClick={applyLimitAndResume}>Set + resume</Button>
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-slate-500">Raise this limit and resume without resetting weights, optimizer moments, insertions, or history.</p>
          </section>

          <DatasetControls value={datasetDraft} onChange={setDatasetDraft} onApply={() => void initialize()} targetCount={targetedTokens} disabled={working} />
          <ArchitectureControls value={architectureDraft} onChange={setArchitectureDraft} batchSize={batchSize} rows={targetedTokens + untargetedTokens} disabled={working} />
              <Button variant="outline" className="w-full border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]" onClick={() => void initialize()} disabled={working}>
                <RotateCcw /> Apply + reset timeline
              </Button>
              <p className="text-[11px] text-slate-500">Reset applies dataset, architecture, experiment, and QAT settings; clears history and target schedules.</p>

          <TargetScheduleControls live={history[liveFrame]} disabled={!ready || working} maxIterations={maxIterationsDraft} onApply={changeTargetPolicy} onCancel={changeTargetPolicy} onSelect={setSelectedToken} />

          <section className="qat-controls border-t border-white/[0.07] pt-4" aria-label="Quantization-aware training">
            <div className="section-label"><Braces className="size-3.5" /> Quantization-aware training</div>
            <div className="mt-3 space-y-3">
              <label className="control-label block">Weight precision
                <Select value={qatDraft.format} onValueChange={(value) => setQatDraft((current) => ({ ...current, format: value as QuantizationFormat }))}>
                  <SelectTrigger aria-label="Weight precision" className="mt-1 w-full border-white/10 bg-[#091527] text-slate-200"><SelectValue /></SelectTrigger>
                  <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
                    {QUANTIZATION_FORMATS.map((item) => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </label>
              <label className="control-label block">Blend schedule
                <Select value={qatDraft.schedule} onValueChange={(value) => setQatDraft((current) => ({ ...current, schedule: value as BlendSchedule }))} disabled={qatDraft.format === "fp32"}>
                  <SelectTrigger aria-label="Blend schedule" className="mt-1 w-full border-white/10 bg-[#091527] text-slate-200"><SelectValue /></SelectTrigger>
                  <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
                    <SelectItem value="linear">Linear blend</SelectItem>
                    <SelectItem value="cosine">Cosine blend</SelectItem>
                    <SelectItem value="immediate">Full QAT at start</SelectItem>
                  </SelectContent>
                </Select>
              </label>
              <div className="grid grid-cols-2 gap-2">
                <label className="control-label">Start iteration
                  <input aria-label="QAT start iteration" className="lab-input mt-1" type="number" min="0" max={MAX_ITERATIONS} step="1" disabled={qatDraft.format === "fp32"} value={qatDraft.start} onChange={(event) => setQatDraft((current) => ({ ...current, start: Number(event.target.value) }))} />
                </label>
                <label className="control-label">Blend duration
                  <input aria-label="QAT blend duration" className="lab-input mt-1" type="number" min="1" max={MAX_ITERATIONS} step="1" disabled={qatDraft.format === "fp32" || qatDraft.schedule === "immediate"} value={qatDraft.duration} onChange={(event) => setQatDraft((current) => ({ ...current, duration: Number(event.target.value) }))} />
                </label>
              </div>
              <div className="rounded-lg border border-fuchsia-300/15 bg-fuchsia-300/[0.04] p-3 text-sm leading-relaxed text-slate-300">
                {qatDraft.format === "fp32" ? "Full-precision forward pass." : <>0 → 100% quantized weight blend{qatDraft.schedule === "immediate" ? ` at iteration ${qatDraft.start}.` : ` over iterations ${qatDraft.start}–${qatEnd}.`}</>}
                {qatDraft.format !== "fp32" && qatEnd > maxIterationsDraft && <p className="mt-1 text-amber-200">{qatEnd > MAX_ITERATIONS ? "The schedule exceeds 50,000 iterations. Choose an earlier start or shorter duration." : "The schedule ends beyond the maximum iterations. Raise the limit to finish it."}</p>}
              </div>
              <Button variant="outline" className="w-full border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]" disabled={!ready || working} onClick={() => applyQat(false)}>Apply QAT to current run</Button>
              <Button className="w-full bg-fuchsia-200 text-[#18091c] hover:bg-fuchsia-100" disabled={!ready || working || qatDraft.format === "fp32" || liveStep >= MAX_ITERATIONS} onClick={() => applyQat(true)}><Play /> Resume into QAT</Button>
              <p className="text-sm leading-relaxed text-slate-400">Apply uses the chosen start and pauses. Resume starts at the live iteration ({liveStep}) and trains toward the maximum above, extending it if already reached. Both retain weights, optimizer state, and history.</p>
              <details className="text-sm leading-relaxed text-slate-400">
                <summary className="cursor-pointer text-slate-300">Quantization details</summary>
                <p className="mt-2">Embeddings and attention/MLP matrices use per-tensor scaled integer codes with zero point 0. Activations, biases, and norm gains stay full precision. Ternary uses scaled −1, 0, 1 with nearest rounding. Scales are recalculated each forward pass.</p>
                <p className="mt-2">Effective weight = (1 − blend) × master + blend × quantized. The backward pass uses an identity straight-through estimator (STE). The sphere shows full-precision master rows; quantized forward rows are not renormalized.</p>
              </details>
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="section-label"><RotateCcw className="size-3.5" /> Experiment setup</div>
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <label className="control-label">Targeted tokens
                  <input aria-label="Targeted tokens" className="lab-input mt-1" type="number" min="1" max={MAX_TARGET_TOKENS} step="1" value={targetedTokens} onChange={(event) => setTargetedTokens(Number(event.target.value))} />
                </label>
                <label className="control-label">Initial untargeted
                  <input aria-label="Initial untargeted tokens" className="lab-input mt-1" type="number" min="0" max={MAX_UNTARGETED_TOKENS} step="1" value={untargetedTokens} onChange={(event) => setUntargetedTokens(Number(event.target.value))} />
                </label>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <label className="control-label">Batch size
                  <input aria-label="Batch size" className="lab-input mt-1" type="number" min="1" max="100" step="1" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
                </label>
                <label className="control-label">Seed
                  <input aria-label="Random seed" className="lab-input mt-1" type="number" step="1" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
                </label>
              </div>
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
              <div className={`rounded-lg border px-3 py-2 text-[10px] leading-relaxed ${estimatedHistory > 512 * 1024 ** 2 ? "border-amber-300/20 bg-amber-300/[0.06] text-amber-200" : "border-white/[0.06] bg-white/[0.025] text-slate-500"}`}>
                Estimated recorded history at this setup: <span className="font-mono">{formatMemory(estimatedHistory)}</span>. Every iteration remains replayable.
              </div>
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
            <fieldset className="mt-4 space-y-3 border-t border-white/10 pt-3">
              <legend className="px-1 text-sm font-medium text-slate-200">Average hidden states</legend>
              <label className="toggle-row"><span>Projected on sphere</span><Switch aria-label="Average hidden states projected on sphere" checked={showHiddenOnSphere} onCheckedChange={setShowHiddenOnSphere} /></label>
              <label className="toggle-row"><span>In free space</span><Switch aria-label="Average hidden states in free space" checked={showHiddenInSpace} onCheckedChange={setShowHiddenInSpace} /></label>
              <p className="text-sm leading-relaxed text-slate-400">Mean LM-head input for each predicted target at this frame. Enable either view or both. Projection follows averaging; free space preserves length. Targets absent from this frame are omitted.</p>
            </fieldset>
          </section>
        </aside>
        </TabsContent>

        <section className="lab-stage min-w-0">
          <div className="mobile-quick-controls">
            <div className="mobile-training-actions">
              <Button disabled={!ready || working || atLimit} onClick={toggleTraining}>{running ? <Pause /> : <Play />}{running ? "Pause" : atLimit ? "At limit" : "Run"}</Button>
              <Button variant="secondary" disabled={!ready || working || atLimit} onClick={() => void trainMany(1)}><SkipForward /> Step</Button>
              <Button variant="secondary" disabled={!ready || working || atLimit} onClick={() => void trainMany(10)}>+10</Button>
              <Button variant="outline" aria-pressed={placingLetter} disabled={!ready || !isLive || !nextUntargeted || working || replaying} onClick={() => setPlacingLetter((value) => !value)}>{placingLetter ? "Cancel" : `Place ${nextUntargeted ?? "—"}`}</Button>
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
              key={JSON.stringify(snapshot.architecture)}
              snapshot={displaySnapshot}
              history={displayHistory}
              frameIndex={frame}
              mode={mode}
              probeMode={probeMode}
              selectedToken={selected}
              showTrails={showTrails}
              showFieldArrows={showFieldArrows}
              normalizeArrows={normalizeArrows}
              frameAutoscale={frameAutoscale}
              showHiddenOnSphere={showHiddenOnSphere}
              showHiddenInSpace={showHiddenInSpace}
              insertionAllowed={isLive && addedUntargeted < MAX_UNTARGETED_TOKENS && ready && !working && !running && !replaying && (!compact || placingLetter)}
              compact={compact}
              onInsert={insertLetter}
              onScale={handleScale}
            />
            {snapshot.includedTargetCount === 0 && <div className="pointer-events-none absolute left-4 right-4 top-16 rounded-xl border border-amber-300/25 bg-[#151721]/95 p-3 text-sm text-amber-100">No targets included. The iteration clock and schedules continue; weights and optimizer state are paused. Loss and probe field are unavailable.</div>}
            <div className="sphere-caption pointer-events-none absolute bottom-4 left-4 right-4 flex flex-wrap items-end justify-between gap-3">
              <div className="max-w-[430px] rounded-xl border border-white/10 bg-[#07101e]/88 p-3 backdrop-blur-md">
                <div className="flex items-center gap-2 text-sm font-semibold text-white"><activeMode.icon className="size-4 text-cyan-300" />{activeMode.label}</div>
                <p className="mt-1 text-[11px] leading-relaxed text-slate-400">{activeMode.description} Displayed gradient is evaluated after iteration {snapshot.step} projection and before the next update.</p>
                {mode === "decomposition" && <div className="mt-2 flex gap-3 text-[10px]"><span className="text-cyan-300">● ambient −∇L</span><span className="text-fuchsia-300">● tangent −P∇L</span><span className="text-amber-300">● radial removed</span></div>}
                {qatActive && <p className="mt-1 text-[11px] text-fuchsia-200">QAT: arrows use the STE surrogate on master rows. Probe freezes the embedding scale.</p>}
                {mode === "forces" && <div className="mt-2 text-[10px] text-fuchsia-300">Magenta arrows = {qatActive ? "STE surrogate tangent force" : "actual tied-row CE tangent force"}</div>}
                {mode === "optimizer" && <div className="mt-2 text-[10px] text-emerald-300">Green arrows = observed projected move from the previous iteration</div>}
              </div>
              {mode === "probe" && snapshot.modelDim >= 3 && snapshot.includedTargetCount > 0 && (
                <div className="w-[220px] rounded-xl border border-white/10 bg-[#07101e]/88 p-3 backdrop-blur-md">
                  <div className="flex justify-between text-[10px] text-slate-400"><span>{activeProbeMode.label}</span><span>{frameAutoscale ? "frame" : "fixed"}</span></div>
                  <div className="heatbar mt-2 h-2.5 rounded-full" />
                  <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-300"><span>{format(scale.min, 3)}</span><span>{activeProbeMode.units}</span><span>{format(scale.max, 3)}</span></div>
                </div>
              )}
            </div>
          </div>

          {snapshot.hiddenMeans.length < snapshot.includedTargetCount && <p className="mt-2 rounded-lg border border-amber-300/20 bg-amber-300/5 p-3 text-sm text-amber-100">Batch coverage: {snapshot.hiddenMeans.length}/{snapshot.includedTargetCount} included targets appear as prediction labels. Observed IDs: {snapshot.hiddenMeans.map(item => item.target).join(", ")}. {snapshot.dataset.mode === "cycle" ? "Increase context length or batch size for full coverage. The direct batch does not rotate between steps." : "Markov probabilities and finite sampling determine exposure; included does not guarantee observed in this batch."}</p>}
          {snapshot.modelDim !== 3 && <p className="mt-2 rounded-lg border border-cyan-300/15 bg-cyan-300/5 p-3 text-sm leading-relaxed text-cyan-100">{snapshot.modelDim > 3 ? "Rows, arrows, and hidden means show coordinates 1–3 without rescaling. Sphere projection is computed in the full model space before display. The probe samples the slice x₄…=0 and shows only forces within that slice. Numeric diagnostics use all dimensions." : "Two-dimensional model: rows and hidden means lie in the XY plane. The spherical probe surface is unavailable; row forces and diagnostics remain exact."}</p>}
          {(showHiddenOnSphere || showHiddenInSpace) && <p className="mt-2 px-2 text-sm text-cyan-100">Hidden means: {showHiddenInSpace && "◆ μ = free space"}{showHiddenInSpace && showHiddenOnSphere && " · "}{showHiddenOnSphere && "◇ μˢ = sphere"}. Colors match targets; dashed lines connect both views.</p>}

          <div className="lab-timeline panel mt-4 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="section-label"><History className="size-3.5" /> Iteration timeline</div>
                <p className="mt-1 text-[11px] text-slate-500">Every step is retained. Replay and scrubbing update rows, hidden means, arrows, heatmap, and numbers together.</p>
              </div>
              <div className="timeline-playback flex flex-wrap items-center gap-2">
                <Select value={String(replaySpeed)} onValueChange={(value) => setReplaySpeed(Number(value))}>
                  <SelectTrigger aria-label="Replay speed" size="sm" className="w-[82px] border-white/10 bg-[#091527] font-mono text-slate-200"><SelectValue /></SelectTrigger>
                  <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
                    <SelectItem value="0.25">0.25×</SelectItem>
                    <SelectItem value="0.5">0.5×</SelectItem>
                    <SelectItem value="1">1×</SelectItem>
                    <SelectItem value="2">2×</SelectItem>
                    <SelectItem value="4">4×</SelectItem>
                  </SelectContent>
                </Select>
                <Button size="sm" variant="outline" className="border-white/10 bg-white/[0.04] text-slate-200 hover:bg-white/[0.08]" disabled={liveFrame === 0 || working} onClick={toggleReplay}>
                  {replaying ? <Pause /> : frame >= liveFrame ? <Rewind /> : <Play />} {replaying ? "Pause replay" : frame >= liveFrame ? "Replay from start" : "Resume replay"}
                </Button>
                {!isLive && !replaying && <Button size="sm" className="bg-cyan-300 text-[#04101c] hover:bg-cyan-200" onClick={() => setFrame(liveFrame)}>Return live · {liveStep}</Button>}
              </div>
            </div>
            <div className="relative mt-4 px-1">
              <Slider aria-label="Training iteration" min={0} max={Math.max(1, liveFrame)} disabled={liveFrame === 0 || working} step={1} value={[frame]} onValueChange={(value) => { setRunning(false); setReplaying(false); setPlacingLetter(false); setFrame(value[0]); }} />
              <div className="timeline-events pointer-events-none absolute inset-x-1 top-[5px] h-1.5">
                {events.map((event, index) => <span key={`${event.token}-${event.step}-${index}`} className="absolute top-0 size-1.5 -translate-x-1/2 rounded-full bg-amber-300 ring-2 ring-[#07101e]" style={{ left: `${liveFrame ? event.frame / liveFrame * 100 : 0}%` }} title={`${event.token} inserted at step ${event.step}`} />)}
                {history.map((item, index) => item.qatChanged ? <span key={`qat-${index}`} className="absolute top-0 size-1.5 -translate-x-1/2 rounded-full bg-fuchsia-300 ring-2 ring-[#07101e]" style={{ left: `${liveFrame ? index / liveFrame * 100 : 0}%` }} title={`${item.qat.format} applied at step ${item.step}`} /> : null)}
                {history.map((item, index) => item.datasetChanged ? <span key={`dataset-${index}`} className="absolute top-0 size-1.5 -translate-x-1/2 rounded-full bg-cyan-200 ring-2 ring-[#07101e]" style={{ left: `${liveFrame ? index / liveFrame * 100 : 0}%` }} title={`Iteration ${item.step}: ${item.datasetChanged}`} /> : null)}
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px]">
              <div className="font-mono text-slate-300">frame {frame} / {liveFrame} · iteration {snapshot.step} · optimizer updates {snapshot.optimizerStep}{replaying ? ` · replay ${replaySpeed}×` : ""}</div>
              <div className="flex flex-wrap gap-1.5">
                {events.length ? events.map((event, i) => <span key={`${event.token}-${i}`} className="rounded-full border border-amber-300/20 bg-amber-300/10 px-2 py-0.5 text-amber-200">{event.token} @ {event.step}</span>) : <span className="text-slate-500">No manual row placements yet</span>}
              </div>
            </div>
            {snapshot.datasetChanged && <p className="mt-2 text-sm text-cyan-200">{snapshot.datasetChanged}</p>}
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
              <Metric label="Next-token acc." value={Number.isFinite(snapshot.accuracy) ? `${(snapshot.accuracy * 100).toFixed(1)}%` : "—"} accent="green" />
              <Metric label="Mean tangent force" value={format(snapshot.meanTangentForce)} accent="pink" />
              <Metric label="Unused prob. mass" value={format(snapshot.unusedMass)} accent="gold" />
              <Metric label="Unused pair distance" value={activeUntargeted > 1 ? format(snapshot.meanLetterDistance) : "—"} detail="mean chord" />
              <Metric label="Max norm error" value={format(snapshot.maxNormError)} detail={`from √${snapshot.modelDim}`} />
            </div>
            <div className="mt-3 rounded-xl border border-cyan-300/15 bg-cyan-300/[0.04] p-3 text-sm" aria-label="Dataset at selected iteration">
              <div className="font-medium text-cyan-200">{snapshot.includedTargetCount} included · {activeUntargeted} untargeted</div>
              {snapshot.dataset.mode === "cycle" ? <p className="mt-1 break-words text-slate-300">Cycle: {Array.from(snapshot.targetMask).flatMap((included, index) => included ? [String(index)] : []).join(" → ") || "empty"}{snapshot.includedTargetCount > 0 ? " → repeat" : ""}</p> : <>
                <p className="mt-1 text-cyan-100">Markov · {snapshot.dataset.deterministic ? "deterministic cached batch" : snapshot.dataset.sampling === "fixed" ? "fixed seeded batch" : `batch for iteration ${snapshot.dataset.batchStep}`}</p>
                <p className="mt-1 text-slate-400">Dataset seed {snapshot.dataset.seed}. Loss and fields describe this frame’s sampled batch, not the exact expectation over the matrix.</p>
                {snapshot.dataset.fallbackRows.length > 0 && <p className="mt-1 text-amber-200">Self-loop fallback after exclusion: {snapshot.dataset.fallbackRows.join(", ")}.</p>}
              </>}
              <p className="mt-1 break-words text-slate-300">First sequence: {snapshot.dataset.preview.slice(0, 25).join(" → ") || "empty"}{snapshot.dataset.preview.length > 25 ? " → …" : ""}</p>
              <p className="mt-1 text-slate-400">{snapshot.logPartition.length} training positions · {snapshot.hiddenMeans.length} observed targets · {snapshot.optimizerStep} optimizer updates</p>
            </div>
            <div className="mt-3 rounded-xl border border-fuchsia-300/15 bg-fuchsia-300/[0.04] p-3 text-sm">
              <div className="font-medium text-fuchsia-200">{precisionLabel}</div>
              <div className="mt-1 text-slate-300">Quantized blend: <span className="font-mono">{(snapshot.qatBlend * 100).toFixed(1)}%</span></div>
              {snapshot.qat.format !== "fp32" && <p className="mt-1 text-slate-400">{snapshot.qat.schedule} · start {snapshot.qat.start}{snapshot.qat.schedule !== "immediate" && ` · duration ${snapshot.qat.duration}`}</p>}
              <div className="mt-1 text-slate-400">Row quantization RMSE: <span className="font-mono">{format(snapshot.quantizationRmse)}</span></div>
              {snapshot.qatChanged && <p className="mt-1 text-fuchsia-200">QAT configuration applied at this frame.</p>}
            </div>
            <div className="mt-3 rounded-xl border border-white/[0.06] bg-white/[0.025] px-3 pt-2">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-slate-500"><span>Cross-entropy loss</span><span>0 → {snapshot.step}</span></div>
              <TinyLossChart history={history} frame={frame} />
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="flex items-center justify-between gap-3">
              <div className="section-label"><Plus className="size-3.5" /> Add untargeted row</div>
              <span className="font-mono text-[10px] text-slate-500">{addedUntargeted}/{MAX_UNTARGETED_TOKENS} added</span>
            </div>
            <div className={`mt-3 rounded-xl border p-3 ${nextUntargeted && isLive ? "border-amber-300/20 bg-amber-300/[0.06]" : "border-white/[0.06] bg-white/[0.025]"}`}>
              {nextUntargeted ? (
                <>
                  <div className="flex items-center gap-3"><span className="grid size-9 place-items-center rounded-lg border border-amber-300/30 bg-amber-300/10 font-mono text-sm font-bold text-amber-200">{nextUntargeted}</span><div><div className="text-sm font-medium">{compact ? `Choose “Place ${nextUntargeted}”, then tap the sphere` : "Click any sphere point"}</div><div className="text-[10px] text-slate-500">{compact ? "Drag to rotate without inserting." : "Short click inserts; drag only rotates."}</div></div></div>
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-400">The new row is normalized to √{snapshot.modelDim}, receives zero optimizer moments, joins the softmax denominator, and never appears in inputs or targets. {snapshot.modelDim > 3 && "Coordinates beyond the third start at zero."}</p>
                  {!isLive && <p className="mt-2 text-[10px] font-medium text-amber-300">Return to the live edge before inserting.</p>}
                </>
              ) : <p className="text-[11px] text-slate-400">All {MAX_UNTARGETED_TOKENS} untargeted rows are active. Further sphere clicks do not mutate the model.</p>}
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="flex items-center justify-between gap-3">
              <div className="section-label"><CircleDot className="size-3.5" /> Row inspector</div>
              <Select value={String(selected)} onValueChange={(value) => setSelectedToken(Number(value))}>
                <SelectTrigger size="sm" className="w-[126px] border-white/10 bg-[#091527] font-mono text-slate-200"><SelectValue /></SelectTrigger>
                <SelectContent className="border-white/10 bg-[#0b1728] text-slate-100">
                  {Array.from({ length: snapshot.activeVocab }, (_, index) => (
                    <SelectItem key={index} value={String(index)}>{tokenLabel(index, snapshot.targetCount)} · {snapshot.targetMask[index] ? "target" : "untargeted"}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <p className="mt-2 text-xs text-slate-400">Vector components shown: coordinates 1–3{snapshot.modelDim === 2 ? " (z = 0)" : ""}.</p>
            <table className="mt-3 w-full table-fixed" aria-label={`Coordinates and forces for token ${tokenLabel(selected, snapshot.targetCount)}`}>
              <thead><tr className="text-[10px] uppercase tracking-wider text-slate-600"><th className="pb-1 text-left">vector</th><th className="pb-1 text-right">x</th><th className="pb-1 text-right">y</th><th className="pb-1 text-right">z</th></tr></thead>
              <tbody>
                <VectorRow label="master w" values={vector(snapshot.positions)} color="#67e8f9" />
                <VectorRow label="forward w" values={vector(snapshot.effectivePositions)} color="#f0abfc" />
                <VectorRow label="ambient −∇L" values={rawForce} color="#67e8f9" />
                <VectorRow label="tangent −P∇L" values={tangentForce} color="#f0abfc" />
                <VectorRow label="radial removed" values={radialForce} color="#fcd34d" />
                <VectorRow label="observed Δw" values={vector(snapshot.optimizerMoves)} color="#6ee7b7" />
              </tbody>
            </table>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Metric label="Projection correction" value={format(snapshot.projectionCorrection)} />
              <Metric label="Move · force cosine" value={format(snapshot.optimizerForceCosine)} detail="prior move vs current force; full-D" />
              <Metric label="Tangency residual" value={format(snapshot.maxTangencyError)} detail="full-D normalized |w·F|" />
              <Metric label="Unused-gradient check" value={activeUntargeted > 0 ? format(snapshot.unusedGradientError) : "—"} detail={qatActive ? "STE − analytic surrogate" : "autodiff − analytic"} />
            </div>
          </section>

          <section className="border-t border-white/[0.07] pt-4">
            <div className="section-label"><Braces className="size-3.5" /> {qatActive ? "QAT surrogate field" : "Exact field being shown"}</div>
            <div className="mt-3 rounded-xl border border-cyan-300/10 bg-cyan-300/[0.04] p-3 font-mono text-[11px] leading-relaxed text-cyan-100/80">
              <div>pᵤ(s) = σ({qatActive ? "u_eff" : "u"}ᵀhₛ − log Zₛ)</div>
              <div>g(u) = meanₛ pᵤ(s)hₛ</div>
              <div className="text-fuchsia-200">F(u) = −(I − uuᵀ/R²)g(u)</div>
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-400">The surface freezes the selected iteration’s {snapshot.logPartition.length} hidden states and active-vocabulary partition. {snapshot.includedTargetCount === 0 ? "No probe field is defined for an empty dataset." : qatActive ? "The candidate row uses the saved blend and embedding scale. Arrows show its STE surrogate; actual insertion may recalibrate the scale and change existing logits." : "It gives the instantaneous counterfactual for one newly inserted untargeted row."}</p>
          </section>
        </aside>
        </TabsContent>
      </Tabs>

      <div className="sr-only" aria-live="polite">{announcement}</div>
      {error && <div className="fixed bottom-4 left-1/2 z-50 max-w-lg -translate-x-1/2 rounded-xl border border-red-400/20 bg-[#2a1018]/95 px-4 py-3 text-sm text-red-200 shadow-xl">{error}</div>}
    </main>
  );
}
