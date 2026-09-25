"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { SphereSceneProps } from "@/components/sphere-scene";
import { D_MODEL, RADIUS, TOKENS, probeAt } from "@/lib/simulator";

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

function valueAt(props: SphereSceneProps, point: Vec3) {
  const probe = probeAt(props.snapshot, point);
  if (props.probeMode === "force") return probe.magnitude;
  if (props.probeMode === "potential") return probe.potential;
  if (props.probeMode === "probability") return probe.probability;
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

function tokenColor(index: number) {
  const digits = ["#68e7ff","#56d7ff","#4ec6ff","#5ab3ff","#7ea0ff","#a98bff","#d57cff","#f57add","#ff819f","#ffaa70"];
  return index < 10 ? digits[index] : "#ffc95d";
}

export function CanvasSphereFallback(props: SphereSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState({ yaw: -0.58, pitch: 0.33 });
  const [sizeTick, setSizeTick] = useState(0);
  const dragRef = useRef({ down: false, moved: false, id: -1, x: 0, y: 0, yaw: 0, pitch: 0 });
  const [hover, setHover] = useState<Vec3 | null>(null);

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
    const radiusPx = Math.min(rect.width, rect.height) * 0.355;
    const x = ((clientX - rect.left) - rect.width / 2) / radiusPx * RADIUS;
    const y = -((clientY - rect.top) - rect.height / 2) / radiusPx * RADIUS;
    const q = RADIUS * RADIUS - x * x - y * y;
    if (q < 0) return null;
    return inverseRotate([x, y, Math.sqrt(q)], rotation.yaw, rotation.pitch);
  }, [rotation]);

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
    const radiusPx = Math.min(width, height) * 0.355;
    ctx.fillStyle = "#050914";
    ctx.fillRect(0, 0, width, height);
    const project = (v: Vec3) => {
      const r = rotate(v, rotation.yaw, rotation.pitch);
      return { x: cx + (r[0] / RADIUS) * radiusPx, y: cy - (r[1] / RADIUS) * radiusPx, z: r[2] };
    };

    let range = rangeFor(props.probeMode);
    if (props.mode === "probe" && props.frameAutoscale) {
      let min = Infinity;
      let max = -Infinity;
      for (let iy = -18; iy <= 18; iy += 1) {
        for (let ix = -18; ix <= 18; ix += 1) {
          const x = (ix / 18) * RADIUS;
          const y = (iy / 18) * RADIUS;
          const q = RADIUS * RADIUS - x * x - y * y;
          if (q < 0) continue;
          const point = inverseRotate([x, y, Math.sqrt(q)], rotation.yaw, rotation.pitch);
          const value = valueAt(props, point);
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
      if (Number.isFinite(min) && Number.isFinite(max)) range = { min, max: max > min ? max : min + 1e-12 };
    }
    props.onScale(range);

    if (props.mode === "probe") {
      const size = 180;
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
          const value = valueAt(props, point);
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
    } else {
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
    drawGreatCircle(0); drawGreatCircle(1); drawGreatCircle(2);
    ctx.beginPath(); ctx.arc(cx, cy, radiusPx, 0, Math.PI * 2); ctx.strokeStyle = "rgba(190,222,249,.34)"; ctx.lineWidth = 1.4; ctx.stroke();

    if (props.showTrails) {
      for (let token = 0; token < props.snapshot.activeVocab; token += 1) {
        ctx.beginPath();
        let started = false;
        for (let h = Math.max(0, props.frameIndex - 160); h <= props.frameIndex; h += 1) {
          const frame = props.history[h];
          if (!frame || frame.activeVocab <= token) continue;
          const o = token * D_MODEL;
          const p = project([frame.positions[o], frame.positions[o + 1], frame.positions[o + 2]]);
          if (!started) { ctx.moveTo(p.x, p.y); started = true; } else ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = tokenColor(token) + (token < 10 ? "66" : "aa");
        ctx.lineWidth = token < 10 ? 1.3 : 2;
        ctx.stroke();
      }
    }

    const arrow = (origin: Vec3, vector: Vec3, color: string, factor: number) => {
      const magnitude = Math.hypot(...vector);
      if (magnitude < 1e-10) return;
      const length = props.normalizeArrows ? 0.48 : Math.max(0.08, Math.min(0.8, magnitude * factor));
      const unit: Vec3 = [vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude];
      const end: Vec3 = [origin[0] + unit[0] * length, origin[1] + unit[1] * length, origin[2] + unit[2] * length];
      const a = project(origin), b = project(end);
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.2;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(b.x - 8 * Math.cos(angle - .45), b.y - 8 * Math.sin(angle - .45)); ctx.lineTo(b.x - 8 * Math.cos(angle + .45), b.y - 8 * Math.sin(angle + .45)); ctx.closePath(); ctx.fill();
    };

    if (props.mode === "probe" && props.showFieldArrows) {
      const golden = Math.PI * (3 - Math.sqrt(5));
      for (let i = 0; i < 42; i += 1) {
        const y = 1 - i / 41 * 2;
        const rr = Math.sqrt(1 - y * y);
        const p: Vec3 = [Math.cos(golden * i) * rr * RADIUS, y * RADIUS, Math.sin(golden * i) * rr * RADIUS];
        if (rotate(p, rotation.yaw, rotation.pitch)[2] < 0) continue;
        arrow(p, probeAt(props.snapshot, p).force, "rgba(245,250,255,.72)", 5.5);
      }
    }

    const tokenOrder = Array.from({ length: props.snapshot.activeVocab }, (_, i) => i).sort((a,b) => {
      const oa = a * 3, ob = b * 3;
      return rotate([props.snapshot.positions[oa],props.snapshot.positions[oa+1],props.snapshot.positions[oa+2]],rotation.yaw,rotation.pitch)[2] - rotate([props.snapshot.positions[ob],props.snapshot.positions[ob+1],props.snapshot.positions[ob+2]],rotation.yaw,rotation.pitch)[2];
    });
    for (const token of tokenOrder) {
      const o = token * 3;
      const world: Vec3 = [props.snapshot.positions[o], props.snapshot.positions[o + 1], props.snapshot.positions[o + 2]];
      const p = project(world);
      const front = p.z >= 0;
      ctx.globalAlpha = front ? 1 : .34;
      ctx.fillStyle = tokenColor(token);
      ctx.strokeStyle = "rgba(4,10,20,.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (token < 10) ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
      else ctx.rect(p.x - 10, p.y - 10, 20, 20);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#04101c"; ctx.font = "700 11px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(TOKENS[token], p.x, p.y + .5);
      ctx.globalAlpha = 1;
      if (props.mode === "forces") arrow(world, [props.snapshot.tangentForces[o],props.snapshot.tangentForces[o+1],props.snapshot.tangentForces[o+2]], "#ff5bd6", 8);
      if (props.mode === "optimizer") arrow(world, [props.snapshot.optimizerMoves[o],props.snapshot.optimizerMoves[o+1],props.snapshot.optimizerMoves[o+2]], "#73f7b5", 34);
    }
    if (props.mode === "decomposition" && props.selectedToken < props.snapshot.activeVocab) {
      const o = props.selectedToken * 3;
      const world: Vec3 = [props.snapshot.positions[o],props.snapshot.positions[o+1],props.snapshot.positions[o+2]];
      const raw: Vec3 = [-props.snapshot.rawGradients[o],-props.snapshot.rawGradients[o+1],-props.snapshot.rawGradients[o+2]];
      const tangent: Vec3 = [props.snapshot.tangentForces[o],props.snapshot.tangentForces[o+1],props.snapshot.tangentForces[o+2]];
      arrow(world, raw, "#6bdcff", 8);
      arrow(world, tangent, "#ff5bd6", 8);
      arrow(world, [raw[0]-tangent[0],raw[1]-tangent[1],raw[2]-tangent[2]], "#ffc95d", 8);
    }
    if (hover) {
      const p = project(hover);
      ctx.beginPath(); ctx.arc(p.x,p.y,7,0,Math.PI*2); ctx.fillStyle="#ffcf58"; ctx.fill(); ctx.strokeStyle="#fff4bd"; ctx.stroke();
    }
  }, [props, rotation, hover, sizeTick]);

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
          } else if (props.insertionAllowed) setHover(spherePoint(event.clientX,event.clientY));
        }}
        onPointerUp={(event) => {
          const drag = dragRef.current;
          if (!drag.down || drag.id !== event.pointerId) return;
          drag.down = false;
          if (!drag.moved && props.insertionAllowed) {
            const point = spherePoint(event.clientX,event.clientY);
            if (point) props.onInsert(point);
          }
        }}
        onPointerCancel={() => { dragRef.current.down = false; dragRef.current.moved = true; setHover(null); }}
        onLostPointerCapture={() => { dragRef.current.down = false; }}
        onPointerLeave={() => setHover(null)}
      />
      <div className="sphere-hint pointer-events-none absolute left-4 top-4 z-10 rounded-full border border-white/10 bg-[#07101e]/82 px-3 py-1.5 text-[11px] font-medium tracking-wide text-slate-300 backdrop-blur">{props.compact ? (props.insertionAllowed ? "Tap the surface to place your letter" : "Drag to orbit · use Place letter to insert") : "Compatibility canvas · drag to orbit · click sphere to add a letter"}</div>
    </div>
  );
}
