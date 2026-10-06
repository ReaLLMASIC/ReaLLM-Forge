"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { SphereSceneProps } from "@/components/sphere-scene";
import { D_MODEL, probeAt, tokenLabel } from "@/lib/simulator";

type Vec3 = [number, number, number];

function rotate(v: Vec3, yaw: number, pitch: number): Vec3 {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x = cy * v[0] + sy * v[2];
  const z = -sy * v[0] + cy * v[2];
  return [x, cp * v[1] - sp * z, sp * v[1] + cp * z];
}

function inverseRotate(v: Vec3, yaw: number, pitch: number): Vec3 {
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const y = cp * v[1] + sp * v[2];
  const z1 = -sp * v[1] + cp * v[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return [cy * v[0] - sy * z1, y, sy * v[0] + cy * z1];
}

function rangeFor(mode: SphereSceneProps["probeMode"]) {
  if (mode === "force") return { min: 0, max: 0.035 };
  if (mode === "potential") return { min: 0, max: 0.2 };
  if (mode === "probability") return { min: 0, max: 0.16 };
  return { min: -0.6, max: 0.15 };
}

function valueAt(snapshot: SphereSceneProps["snapshot"], mode: SphereSceneProps["probeMode"], point: Vec3) {
  const probe = probeAt(snapshot, point);
  if (mode === "force") return probe.magnitude;
  if (mode === "potential") return probe.potential;
  if (mode === "probability") return probe.probability;
  return probe.radial;
}

function heatColor(t: number) {
  const stops = [[5,13,33],[20,59,97],[13,145,158],[122,214,133],[250,212,71]];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  return [
    Math.round(stops[i][0] * (1 - f) + stops[i + 1][0] * f),
    Math.round(stops[i][1] * (1 - f) + stops[i + 1][1] * f),
    Math.round(stops[i][2] * (1 - f) + stops[i + 1][2] * f),
  ];
}

function tokenColor(index: number, targetCount: number) {
  const digits = ["#68e7ff","#56d7ff","#4ec6ff","#5ab3ff","#7ea0ff","#a98bff","#d57cff","#f57add","#ff819f","#ffaa70"];
  if (index >= targetCount) return "#ffc95d";
  if (targetCount <= digits.length) return digits[index % digits.length];
  const hue = ((0.53 + index / Math.max(1, targetCount) * 0.72) % 1) * 360;
  return `hsl(${hue.toFixed(1)} 82% 68%)`;
}

export function CanvasSphereFallback(props: SphereSceneProps) {
  const {
    compact,
    frameAutoscale,
    frameIndex,
    history,
    insertionAllowed,
    mode,
    normalizeArrows,
    onInsert,
    onScale,
    probeMode,
    selectedToken,
    showFieldArrows,
    showTrails,
    showHiddenOnSphere,
    showHiddenInSpace,
    snapshot,
  } = props;
  const RADIUS = snapshot.radius;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState({ yaw: -0.58, pitch: 0.33 });
  const [sizeTick, setSizeTick] = useState(0);
  const dragRef = useRef({ down: false, moved: false, id: -1, x: 0, y: 0, yaw: 0, pitch: 0 });
  const [hover, setHover] = useState<Vec3 | null>(null);
  const extent = showHiddenInSpace ? Math.max(RADIUS, ...snapshot.hiddenMeans.map(item => Math.hypot(...item.mean))) : RADIUS;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => setSizeTick((value) => value + 1));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const spherePoint = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const radiusPx = Math.min(rect.width, rect.height) * 0.355 * RADIUS / extent;
    const x = ((clientX - rect.left) - rect.width / 2) / radiusPx * RADIUS;
    const y = -((clientY - rect.top) - rect.height / 2) / radiusPx * RADIUS;
    const q = RADIUS * RADIUS - x * x - y * y;
    if (q < 0) return null;
    if (snapshot.modelDim === 2) {
      const origin = inverseRotate([x, y, 0], rotation.yaw, rotation.pitch);
      const ray = inverseRotate([0, 0, 1], rotation.yaw, rotation.pitch);
      if (Math.abs(ray[2]) < 1e-6) return null;
      const t = -origin[2] / ray[2];
      const px = origin[0] + t * ray[0], py = origin[1] + t * ray[1];
      const norm = Math.hypot(px, py);
      return norm > 1e-9 && norm <= RADIUS * 1.1 ? [px * RADIUS / norm, py * RADIUS / norm, 0] as Vec3 : null;
    }
    return inverseRotate([x, y, Math.sqrt(q)], rotation.yaw, rotation.pitch);
  }, [rotation, extent, RADIUS, snapshot.modelDim]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const width = rect.width;
    const height = rect.height;
    const cx = width / 2;
    const cy = height / 2;
    const radiusPx = Math.min(width, height) * 0.355 * RADIUS / extent;
    ctx.fillStyle = "#050914";
    ctx.fillRect(0, 0, width, height);
    const project = (v: Vec3) => {
      const r = rotate(v, rotation.yaw, rotation.pitch);
      return { x: cx + (r[0] / RADIUS) * radiusPx, y: cy - (r[1] / RADIUS) * radiusPx, z: r[2] };
    };

    let range = rangeFor(probeMode);
    if (mode === "probe" && snapshot.modelDim >= 3 && snapshot.includedTargetCount > 0 && frameAutoscale) {
      let min = Infinity;
      let max = -Infinity;
      const scaleSteps = snapshot.logPartition.length > 5000 ? 6 : snapshot.logPartition.length > 2000 ? 8 : snapshot.logPartition.length > 500 ? 12 : 18;
      for (let iy = -scaleSteps; iy <= scaleSteps; iy += 1) {
        for (let ix = -scaleSteps; ix <= scaleSteps; ix += 1) {
          const x = (ix / scaleSteps) * RADIUS;
          const y = (iy / scaleSteps) * RADIUS;
          const q = RADIUS * RADIUS - x * x - y * y;
          if (q < 0) continue;
          const point = inverseRotate([x, y, Math.sqrt(q)], rotation.yaw, rotation.pitch);
          const value = valueAt(snapshot, probeMode, point);
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
      if (Number.isFinite(min) && Number.isFinite(max)) range = { min, max: max > min ? max : min + 1e-12 };
    }
    onScale(range);

    if (mode === "probe" && snapshot.modelDim >= 3 && snapshot.includedTargetCount > 0) {
      const sampleCount = snapshot.logPartition.length;
      const size = sampleCount > 5000 ? 48 : sampleCount > 2000 ? 64 : sampleCount > 500 ? 104 : 180;
      const offscreen = document.createElement("canvas");
      offscreen.width = size;
      offscreen.height = size;
      const octx = offscreen.getContext("2d")!;
      const image = octx.createImageData(size, size);
      for (let py = 0; py < size; py += 1) {
        for (let px = 0; px < size; px += 1) {
          const nx = ((px + 0.5) / size * 2 - 1) * RADIUS;
          const ny = -(((py + 0.5) / size * 2 - 1) * RADIUS);
          const q = RADIUS * RADIUS - nx * nx - ny * ny;
          const offset = (py * size + px) * 4;
          if (q < 0) { image.data[offset + 3] = 0; continue; }
          const point = inverseRotate([nx, ny, Math.sqrt(q)], rotation.yaw, rotation.pitch);
          const value = valueAt(snapshot, probeMode, point);
          const [r,g,b] = heatColor((value - range.min) / (range.max - range.min));
          const edge = Math.sqrt(q) / RADIUS;
          const shade = 0.62 + 0.38 * edge;
          image.data[offset] = r * shade;
          image.data[offset + 1] = g * shade;
          image.data[offset + 2] = b * shade;
          image.data[offset + 3] = 205;
        }
      }
      octx.putImageData(image, 0, 0);
      ctx.save();
      ctx.globalAlpha = 0.75;
      ctx.drawImage(offscreen, cx - radiusPx, cy - radiusPx, radiusPx * 2, radiusPx * 2);
      ctx.restore();
    } else if (snapshot.modelDim >= 3) {
      const glow = ctx.createRadialGradient(cx - radiusPx * .35, cy - radiusPx * .4, radiusPx * .1, cx, cy, radiusPx);
      glow.addColorStop(0, "rgba(58,117,160,.22)");
      glow.addColorStop(.72, "rgba(22,62,99,.09)");
      glow.addColorStop(1, "rgba(8,24,44,.03)");
      ctx.fillStyle = glow;
      ctx.beginPath(); ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2); ctx.fill();
    }

    const drawGreatCircle = (axis: 0 | 1 | 2) => {
      ctx.beginPath();
      for (let i = 0; i <= 100; i += 1) {
        const a = i / 100 * Math.PI * 2;
        const v: Vec3 = axis === 0 ? [0, Math.cos(a) * RADIUS, Math.sin(a) * RADIUS] : axis === 1 ? [Math.cos(a) * RADIUS, 0, Math.sin(a) * RADIUS] : [Math.cos(a) * RADIUS, Math.sin(a) * RADIUS, 0];
        const p = project(v);
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = "rgba(205,224,245,.20)";
      ctx.lineWidth = 1;
      ctx.stroke();
    };
    if (snapshot.modelDim >= 3) { drawGreatCircle(0); drawGreatCircle(1); }
    drawGreatCircle(2);
    if (snapshot.modelDim >= 3) { ctx.beginPath(); ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2); ctx.strokeStyle = "rgba(190,222,249,.34)"; ctx.lineWidth = 1.4; ctx.stroke(); }

    if (showTrails) {
      for (let token = 0; token < snapshot.activeVocab; token += 1) {
        ctx.beginPath();
        let started = false;
        for (let h = Math.max(0, frameIndex - 160); h <= frameIndex; h += 1) {
          const frame = history[h];
          if (!frame || frame.activeVocab <= token) continue;
          const o = token * D_MODEL;
          const p = project([frame.positions[o], frame.positions[o + 1], frame.positions[o + 2]]);
          if (!started) { ctx.moveTo(p.x, p.y); started = true; } else ctx.lineTo(p.x, p.y);
        }
        const targeted = Boolean(snapshot.targetMask[token]);
        const color = (snapshot.targetMask[token] ? tokenColor(token, snapshot.targetCount) : "#ffc95d");
        ctx.strokeStyle = color.startsWith("#") ? color + (targeted ? "66" : "aa") : color;
        ctx.globalAlpha = targeted ? 0.48 : 0.72;
        ctx.lineWidth = targeted ? 1.3 : 2;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    const arrow = (origin: Vec3, vector: Vec3, color: string, factor: number) => {
      const magnitude = Math.hypot(...vector);
      if (magnitude < 1e-10) return;
      const length = normalizeArrows ? 0.48 : Math.max(0.08, Math.min(0.8, magnitude * factor));
      const unit: Vec3 = [vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude];
      const end: Vec3 = [origin[0] + unit[0] * length, origin[1] + unit[1] * length, origin[2] + unit[2] * length];
      const a = project(origin), b = project(end);
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(b.x - 8 * Math.cos(angle - .45), b.y - 8 * Math.sin(angle - .45)); ctx.lineTo(b.x - 8 * Math.cos(angle + .45), b.y - 8 * Math.sin(angle + .45)); ctx.closePath(); ctx.fill();
    };

    if (mode === "probe" && snapshot.modelDim >= 3 && snapshot.includedTargetCount > 0 && showFieldArrows) {
      const golden = Math.PI * (3 - Math.sqrt(5));
      for (let i = 0; i < 42; i += 1) {
        const y = 1 - i / 41 * 2;
        const rr = Math.sqrt(1 - y * y);
        const p: Vec3 = [Math.cos(golden * i) * rr * RADIUS, y * RADIUS, Math.sin(golden * i) * rr * RADIUS];
        if (rotate(p, rotation.yaw, rotation.pitch)[2] < 0) continue;
        arrow(p, probeAt(snapshot, p).force, "rgba(245,250,255,.72)", 5.5);
      }
    }

    const tokenOrder = Array.from({ length: snapshot.activeVocab }, (_, i) => i).sort((a,b) => {
      const oa = a * 3, ob = b * 3;
      return rotate([snapshot.positions[oa],snapshot.positions[oa+1],snapshot.positions[oa+2]],rotation.yaw,rotation.pitch)[2] - rotate([snapshot.positions[ob],snapshot.positions[ob+1],snapshot.positions[ob+2]],rotation.yaw,rotation.pitch)[2];
    });
    for (const token of tokenOrder) {
      const o = token * 3;
      const world: Vec3 = [snapshot.positions[o], snapshot.positions[o + 1], snapshot.positions[o + 2]];
      const p = project(world);
      const front = p.z >= 0;
      ctx.globalAlpha = front ? 1 : .34;
      ctx.fillStyle = (snapshot.targetMask[token] ? tokenColor(token, snapshot.targetCount) : "#ffc95d");
      ctx.strokeStyle = "rgba(4,10,20,.9)";
      ctx.lineWidth = 2;
      const targeted = Boolean(snapshot.targetMask[token]);
      const crowdedScale = snapshot.activeVocab > 80 ? 0.72 : snapshot.activeVocab > 40 ? 0.84 : 1;
      const selectedScale = token === selectedToken ? 1.35 : 1;
      const markerRadius = 12 * crowdedScale * selectedScale;
      ctx.beginPath();
      if (targeted) ctx.arc(p.x, p.y, markerRadius, 0, Math.PI * 2);
      else ctx.rect(p.x - markerRadius * .84, p.y - markerRadius * .84, markerRadius * 1.68, markerRadius * 1.68);
      ctx.fill(); ctx.stroke();
      const labelStride = Math.max(1, Math.ceil(snapshot.activeVocab / 24));
      const showLabel = snapshot.activeVocab <= 32 || token === selectedToken || token % labelStride === 0;
      if (showLabel) {
        const label = tokenLabel(token, snapshot.targetCount);
        ctx.fillStyle = "#04101c";
        ctx.font = `700 ${label.length > 2 ? 8 : 11}px ui-monospace, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, p.x, p.y + .5);
      }
      ctx.globalAlpha = 1;
      if (mode === "forces") arrow(world, [snapshot.tangentForces[o],snapshot.tangentForces[o+1],snapshot.tangentForces[o+2]], "#ff5bd6", 8);
      if (mode === "optimizer") arrow(world, [snapshot.optimizerMoves[o],snapshot.optimizerMoves[o+1],snapshot.optimizerMoves[o+2]], "#73f7b5", 34);
    }
    if (showHiddenInSpace || showHiddenOnSphere) {
      for (const item of snapshot.hiddenMeans) {
        const color = tokenColor(item.target, snapshot.targetCount);
        const raw = project(item.mean);
        const projected = item.projected ? project(item.projected) : null;
        if (showHiddenInSpace && showHiddenOnSphere && projected) {
          ctx.save(); ctx.strokeStyle = color; ctx.globalAlpha = 0.65; ctx.setLineDash([4, 3]);
          ctx.beginPath(); ctx.moveTo(raw.x, raw.y); ctx.lineTo(projected.x, projected.y); ctx.stroke(); ctx.restore();
        }
        const drawMean = (p: typeof raw, onSphere: boolean) => {
          ctx.save(); ctx.globalAlpha = p.z >= 0 ? 1 : 0.6;
          ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.fillStyle = onSphere ? "#050914" : color;
          ctx.beginPath(); ctx.moveTo(p.x, p.y - 7); ctx.lineTo(p.x + 7, p.y); ctx.lineTo(p.x, p.y + 7); ctx.lineTo(p.x - 7, p.y); ctx.closePath(); ctx.fill(); ctx.stroke();
          if (snapshot.hiddenMeans.length <= 24 || item.target === selectedToken || item.target % Math.ceil(snapshot.hiddenMeans.length / 24) === 0) {
            const label = `μ${item.target}${onSphere ? "ˢ" : ""}`;
            const y = p.y + (onSphere ? -17 : 18);
            ctx.font = "600 14px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.lineWidth = 4; ctx.strokeStyle = "#050914"; ctx.strokeText(label, p.x, y);
            ctx.fillStyle = color; ctx.fillText(label, p.x, y);
          }
          ctx.restore();
        };
        if (showHiddenInSpace) drawMean(raw, false);
        if (showHiddenOnSphere && projected) drawMean(projected, true);
      }
    }
    if (mode === "decomposition" && selectedToken < snapshot.activeVocab) {
      const o = selectedToken * 3;
      const world: Vec3 = [snapshot.positions[o],snapshot.positions[o+1],snapshot.positions[o+2]];
      const raw: Vec3 = [-snapshot.rawGradients[o],-snapshot.rawGradients[o+1],-snapshot.rawGradients[o+2]];
      const tangent: Vec3 = [snapshot.tangentForces[o],snapshot.tangentForces[o+1],snapshot.tangentForces[o+2]];
      arrow(world, raw, "#6bdcff", 8);
      arrow(world, tangent, "#ff5bd6", 8);
      arrow(world, [raw[0]-tangent[0],raw[1]-tangent[1],raw[2]-tangent[2]], "#ffc95d", 8);
    }
    if (hover) {
      const p = project(hover);
      ctx.beginPath(); ctx.arc(p.x,p.y,7,0,Math.PI*2); ctx.fillStyle="#ffcf58"; ctx.fill(); ctx.strokeStyle="#fff4bd"; ctx.stroke();
    }
  }, [
    hover,
    frameAutoscale,
    showHiddenOnSphere,
    showHiddenInSpace,
    extent,
    frameIndex,
    history,
    mode,
    normalizeArrows,
    onScale,
    probeMode,
    selectedToken,
    showFieldArrows,
    showTrails,
    snapshot,
    rotation,
    sizeTick,
  ]);

  return (
    <div className="sphere-canvas relative h-full min-h-[430px] w-full overflow-hidden rounded-[22px] bg-[#050914]">
      <canvas
        ref={canvasRef}
        className="h-full w-full touch-none cursor-crosshair"
        aria-label="Interactive 3D sphere visualization rendered with a 2D compatibility canvas"
        onPointerDown={(event) => {
          if (!event.isPrimary || event.button !== 0) { dragRef.current.moved = true; return; }
          dragRef.current = { down: true, moved: false, id: event.pointerId, x: event.clientX, y: event.clientY, yaw: rotation.yaw, pitch: rotation.pitch };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (drag.down && event.pointerId === drag.id) {
            const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
            if (Math.hypot(dx,dy) > 4) drag.moved = true;
            setRotation({ yaw: drag.yaw + dx * .008, pitch: Math.max(-1.35, Math.min(1.35, drag.pitch + dy * .008)) });
          } else if (insertionAllowed) setHover(spherePoint(event.clientX,event.clientY));
        }}
        onPointerUp={(event) => {
          const drag = dragRef.current;
          if (!drag.down || drag.id !== event.pointerId) return;
          drag.down = false;
          if (!drag.moved && insertionAllowed) {
            const point = spherePoint(event.clientX,event.clientY);
            if (point) onInsert(point);
          }
        }}
        onPointerCancel={() => { dragRef.current.down = false; dragRef.current.moved = true; setHover(null); }}
        onLostPointerCapture={() => { dragRef.current.down = false; }}
        onPointerLeave={() => setHover(null)}
      />
      <div className="sphere-hint pointer-events-none absolute left-4 top-4 z-10 rounded-full border border-white/10 bg-[#07101e]/82 px-3 py-1.5 text-[11px] font-medium tracking-wide text-slate-300 backdrop-blur">{compact ? (insertionAllowed ? "Tap the surface to place an untargeted row" : "Drag to orbit · use Place row to insert") : "Compatibility canvas · drag to orbit · click sphere to add an untargeted row"}</div>
    </div>
  );
}
