"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CanvasSphereFallback } from "@/components/canvas-sphere-fallback";
import { D_MODEL, ProbeMode, RADIUS, Snapshot, TOKENS, probeAt } from "@/lib/simulator";

export type ViewMode = "rows" | "forces" | "optimizer" | "probe" | "decomposition";

export type SphereSceneProps = {
  compact?: boolean;
  snapshot: Snapshot;
  history: Snapshot[];
  frameIndex: number;
  mode: ViewMode;
  probeMode: ProbeMode;
  selectedToken: number;
  showTrails: boolean;
  showFieldArrows: boolean;
  normalizeArrows: boolean;
  frameAutoscale: boolean;
  insertionAllowed: boolean;
  onInsert: (point: [number, number, number]) => void;
  onScale: (scale: { min: number; max: number }) => void;
};

type SceneState = {
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  sphere: THREE.Mesh<THREE.SphereGeometry, THREE.MeshPhongMaterial>;
  dynamic: THREE.Group;
  hover: THREE.Mesh;
  raycaster: THREE.Raycaster;
  pointer: THREE.Vector2;
};

const tokenColors = [
  0x68e7ff, 0x56d7ff, 0x4ec6ff, 0x5ab3ff, 0x7ea0ff,
  0xa98bff, 0xd57cff, 0xf57add, 0xff819f, 0xffaa70,
];

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    const mesh = child as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(material)) material.forEach((m) => m.dispose());
    else material?.dispose?.();
    const sprite = child as THREE.Sprite;
    const spriteMaterial = sprite.material as THREE.SpriteMaterial | undefined;
    spriteMaterial?.map?.dispose();
  });
}

function clearGroup(group: THREE.Group) {
  while (group.children.length) {
    const child = group.children.pop()!;
    disposeObject(child);
  }
}

function labelSprite(text: string, color: string, isLetter: boolean) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, 128, 128);
  ctx.fillStyle = "rgba(3,8,18,.82)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.beginPath();
  if (isLetter) ctx.roundRect(18, 18, 92, 92, 20);
  else ctx.arc(64, 64, 46, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.font = "700 58px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#f8fbff";
  ctx.fillText(text, 64, 67);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  sprite.scale.setScalar(0.34);
  sprite.renderOrder = 10;
  return sprite;
}

function addGreatCircle(scene: THREE.Scene, plane: "xy" | "xz" | "yz") {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= 128; i += 1) {
    const a = (i / 128) * Math.PI * 2;
    const c = Math.cos(a) * RADIUS;
    const s = Math.sin(a) * RADIUS;
    points.push(
      plane === "xy" ? new THREE.Vector3(c, s, 0) :
      plane === "xz" ? new THREE.Vector3(c, 0, s) : new THREE.Vector3(0, c, s),
    );
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xb9c7df, transparent: true, opacity: 0.18 }));
  scene.add(line);
}

function viridisLike(t: number) {
  const stops = [
    [0.02, 0.05, 0.13],
    [0.08, 0.23, 0.38],
    [0.05, 0.57, 0.62],
    [0.48, 0.84, 0.52],
    [0.98, 0.83, 0.28],
  ];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  return new THREE.Color(
    stops[i][0] * (1 - f) + stops[i + 1][0] * f,
    stops[i][1] * (1 - f) + stops[i + 1][1] * f,
    stops[i][2] * (1 - f) + stops[i + 1][2] * f,
  );
}

function fixedRange(mode: ProbeMode) {
  if (mode === "force") return { min: 0, max: 0.035 };
  if (mode === "potential") return { min: 0, max: 0.2 };
  if (mode === "probability") return { min: 0, max: 0.16 };
  return { min: -0.6, max: 0.15 };
}

function fibonacciSphere(count: number) {
  const points: THREE.Vector3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (i / (count - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = golden * i;
    points.push(new THREE.Vector3(Math.cos(theta) * radius, y, Math.sin(theta) * radius).multiplyScalar(RADIUS));
  }
  return points;
}

export function SphereScene(props: SphereSceneProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<SceneState | null>(null);
  const propsRef = useRef(props);
  const [fallback, setFallback] = useState(false);
  useEffect(() => {
    propsRef.current = props;
  }, [props]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050914);
    scene.fog = new THREE.FogExp2(0x050914, 0.035);
    const camera = new THREE.OrthographicCamera(-2.75, 2.75, 2.75, -2.75, 0.01, 100);
    camera.position.set(4.6, 3.1, 5.2);
    camera.lookAt(0, 0, 0);
    const contextProbe = document.createElement("canvas");
    if (!contextProbe.getContext("webgl2") && !contextProbe.getContext("webgl")) {
      window.setTimeout(() => setFallback(true), 0);
      return;
    }
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    } catch {
      window.setTimeout(() => setFallback(true), 0);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minZoom = 0.75;
    controls.maxZoom = 2.5;
    controls.enablePan = false;
    scene.add(new THREE.HemisphereLight(0xbfe8ff, 0x12182c, 2.0));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(4, 5, 6);
    scene.add(key);
    addGreatCircle(scene, "xy");
    addGreatCircle(scene, "xz");
    addGreatCircle(scene, "yz");
    const sphereGeometry = new THREE.SphereGeometry(RADIUS, 64, 40);
    const colors = new Float32Array(sphereGeometry.attributes.position.count * 3);
    colors.fill(0.08);
    sphereGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const sphere = new THREE.Mesh(
      sphereGeometry,
      new THREE.MeshPhongMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        depthWrite: false,
        shininess: 25,
      }),
    );
    sphere.renderOrder = 0;
    scene.add(sphere);
    const dynamic = new THREE.Group();
    scene.add(dynamic);
    const hover = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 18, 12),
      new THREE.MeshBasicMaterial({ color: 0xffcf58, transparent: true, opacity: 0.9 }),
    );
    hover.visible = false;
    hover.renderOrder = 20;
    scene.add(hover);
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    stateRef.current = { scene, camera, renderer, controls, sphere, dynamic, hover, raycaster, pointer };

    const resize = () => {
      const { width, height } = mount.getBoundingClientRect();
      // Keep CSS size in logical pixels while Three.js scales the backing
      // buffer for Retina displays; otherwise the canvas is enlarged/clipped.
      renderer.setSize(width, height);
      const aspect = width / Math.max(height, 1);
      camera.left = -2.65 * aspect;
      camera.right = 2.65 * aspect;
      camera.top = 2.65;
      camera.bottom = -2.65;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();
    let down = { x: 0, y: 0, id: -1, moved: false };
    const pointers = new Set<number>();
    const projectPointer = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      return raycaster.intersectObject(sphere, false)[0]?.point ?? null;
    };
    const pointerDown = (event: PointerEvent) => {
      pointers.add(event.pointerId);
      if (pointers.size === 1 && event.button === 0) down = { x: event.clientX, y: event.clientY, id: event.pointerId, moved: false };
      else down.moved = true;
    };
    const pointerMove = (event: PointerEvent) => {
      if (pointers.has(event.pointerId) && Math.hypot(event.clientX - down.x, event.clientY - down.y) > 6) down.moved = true;
      if (!propsRef.current.insertionAllowed) {
        hover.visible = false;
        return;
      }
      const point = projectPointer(event);
      hover.visible = Boolean(point);
      if (point) hover.position.copy(point.clone().normalize().multiplyScalar(RADIUS * 1.012));
    };
    const pointerUp = (event: PointerEvent) => {
      pointers.delete(event.pointerId);
      if (down.moved || down.id !== event.pointerId || Math.hypot(event.clientX - down.x, event.clientY - down.y) > 6) return;
      down.id = -1;
      if (!propsRef.current.insertionAllowed) return;
      const point = projectPointer(event);
      if (point) propsRef.current.onInsert([point.x, point.y, point.z]);
    };
    const pointerCancel = (event: PointerEvent) => { pointers.delete(event.pointerId); down.moved = true; hover.visible = false; };
    renderer.domElement.addEventListener("pointerdown", pointerDown);
    renderer.domElement.addEventListener("pointermove", pointerMove);
    renderer.domElement.addEventListener("pointerup", pointerUp);
    renderer.domElement.addEventListener("pointercancel", pointerCancel);
    renderer.domElement.addEventListener("pointerleave", () => { hover.visible = false; });

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", pointerDown);
      renderer.domElement.removeEventListener("pointermove", pointerMove);
      renderer.domElement.removeEventListener("pointerup", pointerUp);
      renderer.domElement.removeEventListener("pointercancel", pointerCancel);
      controls.dispose();
      clearGroup(dynamic);
      sphereGeometry.dispose();
      sphere.material.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      stateRef.current = null;
    };
  }, []);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    const { snapshot, history, frameIndex, mode } = props;
    clearGroup(state.dynamic);
    const heatmapVisible = mode === "probe";
    state.sphere.material.opacity = heatmapVisible ? 0.58 : 0.075;
    const positions = state.sphere.geometry.attributes.position as THREE.BufferAttribute;
    const values: number[] = [];
    if (heatmapVisible) {
      for (let i = 0; i < positions.count; i += 1) {
        const point = [positions.getX(i), positions.getY(i), positions.getZ(i)];
        const probe = probeAt(snapshot, point);
        values.push(
          props.probeMode === "force" ? probe.magnitude :
          props.probeMode === "potential" ? probe.potential :
          props.probeMode === "probability" ? probe.probability : probe.radial,
        );
      }
    }
    let range = fixedRange(props.probeMode);
    if (heatmapVisible && props.frameAutoscale && values.length) {
      const min = Math.min(...values);
      let max = Math.max(...values);
      if (Math.abs(max - min) < 1e-12) max = min + 1e-12;
      range = { min, max };
    }
    props.onScale(range);
    const colorAttr = state.sphere.geometry.attributes.color as THREE.BufferAttribute;
    for (let i = 0; i < positions.count; i += 1) {
      const t = heatmapVisible ? (values[i] - range.min) / (range.max - range.min) : 0.1;
      const color = heatmapVisible ? viridisLike(t) : new THREE.Color(0x24456a);
      colorAttr.setXYZ(i, color.r, color.g, color.b);
    }
    colorAttr.needsUpdate = true;

    if (props.showTrails) {
      for (let token = 0; token < snapshot.activeVocab; token += 1) {
        const points: THREE.Vector3[] = [];
        const start = Math.max(0, frameIndex - 160);
        for (let h = start; h <= frameIndex; h += 1) {
          const frame = history[h];
          if (!frame || frame.activeVocab <= token) continue;
          const o = token * D_MODEL;
          points.push(new THREE.Vector3(frame.positions[o], frame.positions[o + 1], frame.positions[o + 2]).multiplyScalar(1.005));
        }
        if (points.length > 1) {
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          const color = token < 10 ? tokenColors[token] : 0xffc95d;
          const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: token < 10 ? 0.42 : 0.72 }));
          line.renderOrder = 3;
          state.dynamic.add(line);
        }
      }
    }

    const addArrow = (origin: THREE.Vector3, vector: THREE.Vector3, color: number, rawScale = 1) => {
      const magnitude = vector.length();
      if (magnitude < 1e-10) return;
      const length = props.normalizeArrows ? 0.48 : Math.min(0.8, Math.max(0.08, magnitude * rawScale));
      const arrow = new THREE.ArrowHelper(vector.clone().normalize(), origin.clone().multiplyScalar(1.02), length, color, 0.13, 0.07);
      arrow.renderOrder = 8;
      state.dynamic.add(arrow);
    };

    for (let token = 0; token < snapshot.activeVocab; token += 1) {
      const o = token * D_MODEL;
      const p = new THREE.Vector3(snapshot.positions[o], snapshot.positions[o + 1], snapshot.positions[o + 2]);
      const isLetter = token >= 10;
      const color = isLetter ? 0xffc95d : tokenColors[token];
      const geometry = isLetter ? new THREE.OctahedronGeometry(0.075, 0) : new THREE.SphereGeometry(0.072, 18, 12);
      const marker = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.32, roughness: 0.35 }),
      );
      marker.position.copy(p.clone().multiplyScalar(1.015));
      marker.renderOrder = 6;
      state.dynamic.add(marker);
      const label = labelSprite(TOKENS[token], isLetter ? "#ffc95d" : "#68e7ff", isLetter);
      label.position.copy(p.clone().multiplyScalar(1.16));
      state.dynamic.add(label);
      if (mode === "forces") {
        addArrow(p, new THREE.Vector3(snapshot.tangentForces[o], snapshot.tangentForces[o + 1], snapshot.tangentForces[o + 2]), 0xff5bd6, 8);
      }
      if (mode === "optimizer") {
        addArrow(p, new THREE.Vector3(snapshot.optimizerMoves[o], snapshot.optimizerMoves[o + 1], snapshot.optimizerMoves[o + 2]), 0x73f7b5, 34);
      }
    }

    if (mode === "decomposition" && snapshot.activeVocab > props.selectedToken) {
      const o = props.selectedToken * D_MODEL;
      const p = new THREE.Vector3(snapshot.positions[o], snapshot.positions[o + 1], snapshot.positions[o + 2]);
      const raw = new THREE.Vector3(-snapshot.rawGradients[o], -snapshot.rawGradients[o + 1], -snapshot.rawGradients[o + 2]);
      const tangent = new THREE.Vector3(snapshot.tangentForces[o], snapshot.tangentForces[o + 1], snapshot.tangentForces[o + 2]);
      const radial = raw.clone().sub(tangent);
      addArrow(p, raw, 0x6bdcff, 8);
      addArrow(p, tangent, 0xff5bd6, 8);
      addArrow(p, radial, 0xffc95d, 8);
    }

    if (mode === "probe" && props.showFieldArrows) {
      for (const point of fibonacciSphere(54)) {
        const probe = probeAt(snapshot, point.toArray());
        addArrow(point, new THREE.Vector3(...probe.force), 0xf6fbff, 5.5);
      }
    }
  }, [
    props.snapshot,
    props.history,
    props.frameIndex,
    props.mode,
    props.probeMode,
    props.selectedToken,
    props.showTrails,
    props.showFieldArrows,
    props.normalizeArrows,
    props.frameAutoscale,
    props,
  ]);

  if (fallback) return <CanvasSphereFallback {...props} />;

  return (
    <div className="sphere-canvas relative h-full min-h-[430px] w-full overflow-hidden rounded-[22px] bg-[#050914]" ref={mountRef}>
      <div className="sphere-hint pointer-events-none absolute left-4 top-4 z-10 rounded-full border border-white/10 bg-[#07101e]/82 px-3 py-1.5 text-[11px] font-medium tracking-wide text-slate-300 backdrop-blur">
        {props.compact ? (props.insertionAllowed ? "Tap the surface to place your letter" : "Drag to orbit · pinch to zoom") : "Drag to orbit · scroll to zoom · click surface to add a letter"}
      </div>
    </div>
  );
}
