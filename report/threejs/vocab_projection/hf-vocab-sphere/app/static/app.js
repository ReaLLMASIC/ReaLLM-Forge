import * as THREE from 'https://esm.sh/three@0.180.0';
import { OrbitControls } from 'https://esm.sh/three@0.180.0/examples/jsm/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const SPHERE_RADIUS = 2.18;
const BG = '#1b1b2f';
const COLORS = {
  anchor: '#ff7b72',
  near: '#7dd3fc',
  middle: '#c77dff',
  orthogonal: '#ffd166',
  far: '#ff7b72',
  edge: '#58c4dd',
  sphere: '#ffffff',
  grid: '#8d99ae',
};

const state = {
  status: null,
  methods: new Map(),
  searchResults: [],
  selected: new Map(),
  projection: null,
  pinnedIndex: null,
  hoveredIndex: null,
  autoRotate: false,
  showEdges: true,
  showLabels: false,
  colorMode: 'anchor',
  pointSize: 0.075,
};

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function formatCompact(value) {
  if (!Number.isFinite(Number(value))) return '—';
  return new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value));
}

function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes)) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = Number(bytes);
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[index]}`;
}

function showToast(message, type = 'info') {
  const container = $('toast-container');
  const element = document.createElement('div');
  element.className = `toast ${type === 'error' ? 'error' : ''}`;
  element.textContent = message;
  container.appendChild(element);
  setTimeout(() => {
    element.style.opacity = '0';
    setTimeout(() => element.remove(), 280);
  }, 2500);
}

function setLoading(active, title = 'Working', detail = 'Preparing vocabulary geometry…') {
  $('loadingOverlay').classList.toggle('hidden', !active);
  $('loadingTitle').textContent = title;
  $('loadingDetail').textContent = detail;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload?.detail || `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return payload;
}

// ---------------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------------

const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(BG, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
scene.fog = new THREE.FogExp2(BG, 0.042);

const camera = new THREE.PerspectiveCamera(47, 1, 0.1, 100);
const INITIAL_CAMERA = new THREE.Vector3(0.15, 0.3, 6.8);
camera.position.copy(INITIAL_CAMERA);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.enablePan = true;
controls.minDistance = 3.1;
controls.maxDistance = 16;
controls.target.set(0, 0, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(4, 5, 7);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.45);
rimLight.position.set(-5, -2, -4);
scene.add(rimLight);

const world = new THREE.Group();
scene.add(world);

const sphere = new THREE.Mesh(
  new THREE.SphereGeometry(SPHERE_RADIUS, 48, 32),
  new THREE.MeshBasicMaterial({ color: COLORS.sphere, wireframe: true, transparent: true, opacity: 0.055, depthWrite: false }),
);
world.add(sphere);

function createSphereGrid() {
  const group = new THREE.Group();
  const material = new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.11, depthWrite: false });
  for (let lat = -60; lat <= 60; lat += 30) {
    const phi = THREE.MathUtils.degToRad(90 - lat);
    const y = SPHERE_RADIUS * Math.cos(phi);
    const radius = SPHERE_RADIUS * Math.sin(phi);
    const points = [];
    for (let i = 0; i <= 96; i += 1) {
      const theta = (i / 96) * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(theta) * radius, y, Math.sin(theta) * radius));
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material.clone()));
  }
  for (let lon = 0; lon < 180; lon += 30) {
    const angle = THREE.MathUtils.degToRad(lon);
    const points = [];
    for (let i = 0; i <= 96; i += 1) {
      const phi = (i / 96) * Math.PI;
      const x = SPHERE_RADIUS * Math.sin(phi) * Math.cos(angle);
      const y = SPHERE_RADIUS * Math.cos(phi);
      const z = SPHERE_RADIUS * Math.sin(phi) * Math.sin(angle);
      points.push(new THREE.Vector3(x, y, z));
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material.clone()));
  }
  return group;
}
world.add(createSphereGrid());

function makeAxis(direction, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints([
    direction.clone().multiplyScalar(-SPHERE_RADIUS * 1.12),
    direction.clone().multiplyScalar(SPHERE_RADIUS * 1.12),
  ]);
  return new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.2, depthWrite: false }),
  );
}
world.add(makeAxis(new THREE.Vector3(1, 0, 0), '#ff7b72'));
world.add(makeAxis(new THREE.Vector3(0, 1, 0), '#52b788'));
world.add(makeAxis(new THREE.Vector3(0, 0, 1), '#7dd3fc'));

function makeStars() {
  const rng = mulberry32(7129);
  const positions = [];
  for (let i = 0; i < 520; i += 1) {
    const radius = 8 + rng() * 12;
    const theta = rng() * Math.PI * 2;
    const u = rng() * 2 - 1;
    const s = Math.sqrt(1 - u * u);
    positions.push(radius * s * Math.cos(theta), radius * u, radius * s * Math.sin(theta));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const points = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ color: '#ffffff', size: 0.018, transparent: true, opacity: 0.28, depthWrite: false }),
  );
  scene.add(points);
}
makeStars();

const pointGeometry = new THREE.BufferGeometry();
const pointMaterial = new THREE.PointsMaterial({
  size: state.pointSize,
  vertexColors: true,
  transparent: true,
  opacity: 0.95,
  sizeAttenuation: true,
  depthWrite: false,
});
const tokenPoints = new THREE.Points(pointGeometry, pointMaterial);
tokenPoints.renderOrder = 4;
world.add(tokenPoints);

const edgeGeometry = new THREE.BufferGeometry();
const edgeMaterial = new THREE.LineBasicMaterial({ color: COLORS.edge, transparent: true, opacity: 0.2, depthWrite: false });
const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
edgeLines.renderOrder = 1;
world.add(edgeLines);

const anchorMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.095, 20, 16),
  new THREE.MeshStandardMaterial({ color: COLORS.anchor, emissive: COLORS.anchor, emissiveIntensity: 0.45, roughness: 0.45 }),
);
anchorMesh.visible = false;
anchorMesh.renderOrder = 6;
world.add(anchorMesh);

const hoverRing = new THREE.Mesh(
  new THREE.TorusGeometry(0.12, 0.012, 10, 42),
  new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.9, depthWrite: false }),
);
hoverRing.visible = false;
hoverRing.renderOrder = 7;
world.add(hoverRing);

const pinnedRing = new THREE.Mesh(
  new THREE.TorusGeometry(0.145, 0.015, 10, 42),
  new THREE.MeshBasicMaterial({ color: '#ffd166', transparent: true, opacity: 0.95, depthWrite: false }),
);
pinnedRing.visible = false;
pinnedRing.renderOrder = 8;
world.add(pinnedRing);

const labelGroup = new THREE.Group();
world.add(labelGroup);

let scenePoints = [];
let edgePairs = [];
let transition = null;
let labelEntries = [];

function mulberry32(seed) {
  let value = seed >>> 0;
  return function random() {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeTextSprite(text, color = '#ffffff') {
  const canvasElement = document.createElement('canvas');
  canvasElement.width = 512;
  canvasElement.height = 128;
  const context = canvasElement.getContext('2d');
  context.clearRect(0, 0, canvasElement.width, canvasElement.height);
  context.font = '700 34px Inter, Arial, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.lineWidth = 8;
  context.strokeStyle = 'rgba(8,8,20,0.88)';
  context.strokeText(text, canvasElement.width / 2, canvasElement.height / 2);
  context.fillStyle = color;
  context.fillText(text, canvasElement.width / 2, canvasElement.height / 2);
  const texture = new THREE.CanvasTexture(canvasElement);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
  sprite.scale.set(1.45, 0.36, 1);
  sprite.renderOrder = 9;
  sprite.userData.texture = texture;
  return sprite;
}

function clearLabels() {
  for (const child of [...labelGroup.children]) {
    labelGroup.remove(child);
    child.material?.map?.dispose?.();
    child.material?.dispose?.();
  }
  labelEntries = [];
}

function rebuildLabels() {
  clearLabels();
  if (!state.projection) return;
  const points = state.projection.points;
  const candidates = [];
  const anchorIndex = points.findIndex((point) => point.is_anchor);
  if (anchorIndex >= 0) candidates.push(anchorIndex);
  if (state.showLabels) {
    const cap = points.length <= 80 ? points.length : 80;
    for (let index = 0; index < cap; index += 1) candidates.push(index);
  }
  if (state.pinnedIndex !== null) candidates.push(state.pinnedIndex);
  const unique = [...new Set(candidates)].filter((index) => index >= 0 && index < points.length);
  for (const index of unique) {
    const point = points[index];
    const label = makeTextSprite(`${point.display} · ${point.token_id}`, point.is_anchor ? COLORS.anchor : '#ffffff');
    labelGroup.add(label);
    labelEntries.push({ index, sprite: label });
  }
  updateLabelsFromPositions();
}

function updateLabelsFromPositions() {
  const attribute = tokenPoints.geometry.getAttribute('position');
  if (!attribute) return;
  for (const entry of labelEntries) {
    if (entry.index >= attribute.count) continue;
    const position = new THREE.Vector3().fromBufferAttribute(attribute, entry.index);
    entry.sprite.position.copy(position.clone().multiplyScalar(1.06));
  }
}

function gradientColor(t) {
  const stops = [
    [0.0, new THREE.Color(COLORS.near)],
    [0.42, new THREE.Color(COLORS.middle)],
    [0.68, new THREE.Color(COLORS.orthogonal)],
    [1.0, new THREE.Color(COLORS.far)],
  ];
  const value = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0.5));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [aT, aColor] = stops[i];
    const [bT, bColor] = stops[i + 1];
    if (value <= bT) {
      const local = (value - aT) / Math.max(1e-9, bT - aT);
      return aColor.clone().lerp(bColor, Math.max(0, Math.min(1, local)));
    }
  }
  return stops.at(-1)[1].clone();
}

function colorForPoint(point, ranges) {
  if (point.is_anchor) return new THREE.Color(COLORS.anchor);
  if (state.colorMode === 'id') {
    const vocab = Math.max(1, state.status?.vocab_size || 1);
    return new THREE.Color().setHSL((point.token_id / vocab + 0.54) % 1, 0.7, 0.64);
  }
  if (state.colorMode === 'norm') {
    const span = Math.max(1e-9, ranges.maxNorm - ranges.minNorm);
    return gradientColor((point.magnitude - ranges.minNorm) / span);
  }
  const angle = point.angle_to_anchor_deg;
  return gradientColor(Number.isFinite(angle) ? angle / 180 : point.index / Math.max(1, scenePoints.length - 1));
}

function recolorPoints() {
  if (!state.projection) return;
  const points = state.projection.points;
  const norms = points.map((point) => Number(point.magnitude)).filter(Number.isFinite);
  const ranges = {
    minNorm: norms.length ? Math.min(...norms) : 0,
    maxNorm: norms.length ? Math.max(...norms) : 1,
  };
  const colors = [];
  for (const point of points) {
    const color = colorForPoint(point, ranges);
    colors.push(color.r, color.g, color.b);
  }
  tokenPoints.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  tokenPoints.geometry.attributes.color.needsUpdate = true;
  $('legendCard').classList.toggle('hidden', state.colorMode !== 'anchor');
}

function updateEdgesFromPositions() {
  if (!state.projection || !state.showEdges || !edgePairs.length) {
    edgeLines.visible = false;
    return;
  }
  const positions = tokenPoints.geometry.getAttribute('position');
  if (!positions) return;
  const values = [];
  for (const edge of edgePairs) {
    if (edge.source_index >= positions.count || edge.target_index >= positions.count) continue;
    values.push(
      positions.getX(edge.source_index), positions.getY(edge.source_index), positions.getZ(edge.source_index),
      positions.getX(edge.target_index), positions.getY(edge.target_index), positions.getZ(edge.target_index),
    );
  }
  edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(values, 3));
  edgeGeometry.computeBoundingSphere();
  edgeLines.visible = true;
}

function updateMarkersFromPositions() {
  const attribute = tokenPoints.geometry.getAttribute('position');
  if (!attribute || !state.projection) {
    anchorMesh.visible = false;
    hoverRing.visible = false;
    pinnedRing.visible = false;
    return;
  }
  const anchorIndex = state.projection.points.findIndex((point) => point.is_anchor);
  if (anchorIndex >= 0 && anchorIndex < attribute.count) {
    anchorMesh.position.set(attribute.getX(anchorIndex), attribute.getY(anchorIndex), attribute.getZ(anchorIndex));
    anchorMesh.visible = true;
  } else {
    anchorMesh.visible = false;
  }
  if (state.hoveredIndex !== null && state.hoveredIndex < attribute.count) {
    hoverRing.position.set(attribute.getX(state.hoveredIndex), attribute.getY(state.hoveredIndex), attribute.getZ(state.hoveredIndex));
    hoverRing.lookAt(camera.position);
    hoverRing.visible = true;
  } else {
    hoverRing.visible = false;
  }
  if (state.pinnedIndex !== null && state.pinnedIndex < attribute.count) {
    pinnedRing.position.set(attribute.getX(state.pinnedIndex), attribute.getY(state.pinnedIndex), attribute.getZ(state.pinnedIndex));
    pinnedRing.lookAt(camera.position);
    pinnedRing.visible = true;
  } else {
    pinnedRing.visible = false;
  }
}

function renderProjection(data) {
  state.projection = data;
  scenePoints = data.points;
  edgePairs = data.edges || [];
  state.pinnedIndex = null;
  state.hoveredIndex = null;
  renderInspector();

  const targets = new Float32Array(data.points.length * 3);
  data.points.forEach((point, index) => {
    targets[index * 3] = point.x * SPHERE_RADIUS;
    targets[index * 3 + 1] = point.y * SPHERE_RADIUS;
    targets[index * 3 + 2] = point.z * SPHERE_RADIUS;
  });

  const old = tokenPoints.geometry.getAttribute('position');
  let starts;
  if (old && old.count === data.points.length) {
    starts = new Float32Array(old.array);
  } else {
    starts = new Float32Array(targets.length);
    for (let i = 0; i < data.points.length; i += 1) {
      starts[i * 3] = targets[i * 3] * 0.14;
      starts[i * 3 + 1] = targets[i * 3 + 1] * 0.14;
      starts[i * 3 + 2] = targets[i * 3 + 2] * 0.14;
    }
  }
  tokenPoints.geometry.setAttribute('position', new THREE.Float32BufferAttribute(starts, 3));
  tokenPoints.geometry.computeBoundingSphere();
  tokenPoints.visible = true;
  recolorPoints();
  transition = { starts, targets, startedAt: performance.now(), duration: 820 };
  rebuildLabels();
  updateEdgesFromPositions();
  updateMarkersFromPositions();
}

function renderDemo() {
  state.projection = null;
  state.pinnedIndex = null;
  state.hoveredIndex = null;
  scenePoints = [];
  edgePairs = [];
  clearLabels();
  const count = 180;
  const positions = [];
  const colors = [];
  for (let i = 0; i < count; i += 1) {
    const y = 1 - (i / (count - 1)) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = Math.PI * (3 - Math.sqrt(5)) * i;
    positions.push(Math.cos(theta) * radius * SPHERE_RADIUS, y * SPHERE_RADIUS, Math.sin(theta) * radius * SPHERE_RADIUS);
    const color = gradientColor(i / (count - 1));
    colors.push(color.r, color.g, color.b);
  }
  tokenPoints.geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  tokenPoints.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  tokenPoints.geometry.computeBoundingSphere();
  edgeLines.visible = false;
  anchorMesh.visible = false;
  hoverRing.visible = false;
  pinnedRing.visible = false;
  transition = null;
}

function resetCamera() {
  camera.position.copy(INITIAL_CAMERA);
  controls.target.set(0, 0, 0);
  controls.update();
  showToast('Camera reset');
}

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.11;
const pointer = new THREE.Vector2();
let pointerDown = null;

function pointerToNdc(event) {
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function setHovered(index, event = null) {
  if (state.hoveredIndex === index) {
    if (index !== null && event) moveTooltip(event);
    return;
  }
  state.hoveredIndex = index;
  const point = index === null ? null : state.projection?.points?.[index];
  const tooltip = $('tooltip');
  const inspector = $('sceneInspector');
  if (!point) {
    tooltip.classList.add('hidden');
    inspector.classList.add('hidden');
    canvas.style.cursor = 'grab';
  } else {
    tooltip.innerHTML = `<strong class="mono">${escapeHtml(point.display)}</strong><span>ID ${point.token_id}${Number.isFinite(point.angle_to_anchor_deg) ? ` · ${formatNumber(point.angle_to_anchor_deg, 2)}° from anchor` : ''}</span>`;
    tooltip.classList.remove('hidden');
    $('sceneInspectorToken').textContent = `${point.display} · ${point.token_id}`;
    $('sceneInspectorMeta').textContent = `norm ${formatNumber(point.magnitude, 4)}${Number.isFinite(point.angle_to_anchor_deg) ? ` · angle ${formatNumber(point.angle_to_anchor_deg, 2)}°` : ''}`;
    inspector.classList.remove('hidden');
    canvas.style.cursor = 'pointer';
    if (event) moveTooltip(event);
  }
  updateMarkersFromPositions();
}

function moveTooltip(event) {
  const rect = $('scene-wrap').getBoundingClientRect();
  const tooltip = $('tooltip');
  tooltip.style.left = `${Math.min(rect.width - 320, event.clientX - rect.left + 14)}px`;
  tooltip.style.top = `${Math.max(8, event.clientY - rect.top + 14)}px`;
}

canvas.addEventListener('pointermove', (event) => {
  if (!state.projection) return;
  pointerToNdc(event);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(tokenPoints, false);
  setHovered(hits.length ? hits[0].index : null, event);
});
canvas.addEventListener('pointerleave', () => setHovered(null));
canvas.addEventListener('pointerdown', (event) => { pointerDown = { x: event.clientX, y: event.clientY }; });
canvas.addEventListener('pointerup', (event) => {
  if (!pointerDown || state.hoveredIndex === null) return;
  const distance = Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y);
  pointerDown = null;
  if (distance < 5) pinPoint(state.hoveredIndex);
});

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
renderDemo();

let lastTimestamp = performance.now();
function animate(timestamp) {
  const delta = Math.min(0.05, (timestamp - lastTimestamp) / 1000);
  lastTimestamp = timestamp;

  if (transition) {
    const rawT = Math.min(1, (timestamp - transition.startedAt) / transition.duration);
    const t = rawT * rawT * (3 - 2 * rawT);
    const current = tokenPoints.geometry.getAttribute('position');
    for (let i = 0; i < current.array.length; i += 3) {
      let x = transition.starts[i] + (transition.targets[i] - transition.starts[i]) * t;
      let y = transition.starts[i + 1] + (transition.targets[i + 1] - transition.starts[i + 1]) * t;
      let z = transition.starts[i + 2] + (transition.targets[i + 2] - transition.starts[i + 2]) * t;
      const length = Math.hypot(x, y, z) || 1;
      const radialScale = SPHERE_RADIUS / length;
      x *= radialScale; y *= radialScale; z *= radialScale;
      current.array[i] = x;
      current.array[i + 1] = y;
      current.array[i + 2] = z;
    }
    current.needsUpdate = true;
    tokenPoints.geometry.computeBoundingSphere();
    updateEdgesFromPositions();
    updateMarkersFromPositions();
    updateLabelsFromPositions();
    if (rawT >= 1) transition = null;
  }

  if (state.autoRotate) {
    camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), delta * 0.18);
  }
  hoverRing.lookAt(camera.position);
  pinnedRing.lookAt(camera.position);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

// ---------------------------------------------------------------------------
// Model and token UI
// ---------------------------------------------------------------------------

function resetProjectionUi() {
  state.projection = null;
  state.pinnedIndex = null;
  state.hoveredIndex = null;
  $('projectionStatusPill').textContent = 'No projection';
  $('projectionStatusPill').className = 'status-pill';
  $('exportBtn').disabled = true;
  $('benchmarkWrap').classList.add('hidden');
  renderInspector();
  renderMetrics(null);
}

function markProjectionStale() {
  if (!state.projection) return;
  $('projectionStatusPill').textContent = 'Projection stale';
  $('projectionStatusPill').className = 'status-pill neutral';
}

async function refreshStatus() {
  const status = await api('/api/status');
  state.status = status;
  applyStatus(status);
}

function applyStatus(status) {
  const pill = $('modelStatusPill');
  pill.textContent = status.loaded ? `${status.model_name} · ${status.matrix_source}` : 'No model loaded';
  pill.className = `status-pill ${status.loaded ? 'good' : 'neutral'}`;
  $('modelVocabStat').textContent = status.loaded ? formatCompact(status.vocab_size) : '—';
  $('modelDimStat').textContent = status.loaded ? formatCompact(status.hidden_dim) : '—';
  $('modelSourceStat').textContent = status.loaded ? status.matrix_source : '—';
  $('modelMemoryStat').textContent = status.loaded ? formatBytes(status.memory_bytes) : '—';
  $('modelLoadNote').textContent = status.loaded
    ? `${status.load_strategy} · ${status.tensor_name} · ${status.dtype} · compute ${status.compute_device}`
    : 'The app prefers a safetensors-only matrix load so it does not need to instantiate the whole language model.';
  $('unloadModelBtn').disabled = !status.loaded;
  if (status.loaded) {
    $('modelNameInput').value = status.model_name;
    $('revisionInput').value = status.revision;
    $('deviceInput').value = status.compute_device;
  }
}

async function loadModel() {
  const body = {
    model_name: $('modelNameInput').value.trim(),
    revision: $('revisionInput').value.trim() || 'main',
    matrix_source: $('matrixSourceSelect').value,
    compute_device: $('deviceInput').value.trim() || 'auto',
    allow_download: $('allowDownloadInput').checked,
    force_reload: false,
  };
  if (!body.model_name) {
    showToast('Enter a model ID or path.', 'error');
    return;
  }
  setLoading(true, 'Loading vocabulary matrix', 'Resolving the tokenizer and the smallest usable embedding or LM-head tensor.');
  try {
    const status = await api('/api/model/load', { method: 'POST', body: JSON.stringify(body) });
    state.status = status;
    applyStatus(status);
    state.selected.clear();
    state.searchResults = [];
    resetProjectionUi();
    renderSearchResults();
    renderSelected();
    renderDemo();
    const first = await api('/api/tokens/0');
    state.selected.set(first.token_id, first);
    $('anchorIdInput').value = String(first.token_id);
    renderSelected();
    showToast(`Loaded ${status.model_name}`);
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    setLoading(false);
  }
}

async function unloadModelUi() {
  try {
    const status = await api('/api/model', { method: 'DELETE' });
    state.status = status;
    state.selected.clear();
    state.searchResults = [];
    resetProjectionUi();
    applyStatus(status);
    renderSearchResults();
    renderSelected();
    renderDemo();
    showToast('Model unloaded');
  } catch (error) {
    showToast(error.message, 'error');
  }
}

async function refreshLocalModels() {
  setLoading(true, 'Scanning local Hub cache', 'Looking for model repositories already present on this machine.');
  try {
    const payload = await api('/api/models/local');
    const select = $('localModelSelect');
    select.innerHTML = '<option value="">Choose a cached model…</option>';
    for (const model of payload.models) {
      const option = document.createElement('option');
      option.value = model.model_name;
      option.textContent = `${model.model_name}${model.size_bytes ? ` · ${formatBytes(model.size_bytes)}` : ''}`;
      select.appendChild(option);
    }
    $('localModelWrap').classList.remove('hidden');
    showToast(`${payload.models.length} cached model${payload.models.length === 1 ? '' : 's'} found`);
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    setLoading(false);
  }
}

async function performSearch() {
  if (!state.status?.loaded) {
    showToast('Load a model first.', 'error');
    return;
  }
  const params = new URLSearchParams({
    pattern: $('searchInput').value,
    mode: $('searchModeSelect').value,
    case_sensitive: String($('caseSensitiveInput').checked),
    limit: String(Math.max(1, Math.min(5000, Number($('searchLimitInput').value) || 200))),
  });
  try {
    const payload = await api(`/api/tokens/search?${params}`);
    state.searchResults = payload.results;
    renderSearchResults(payload);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

function renderSearchResults(payload = null) {
  const body = $('searchResultsBody');
  if (!state.searchResults.length) {
    body.innerHTML = '<tr><td colspan="4" class="empty-cell">No results.</td></tr>';
  } else {
    body.innerHTML = state.searchResults.map((row) => `
      <tr>
        <td class="mono">${row.token_id}</td>
        <td class="token-cell mono" title="${escapeHtml(row.raw)}">${escapeHtml(row.display)}</td>
        <td>${formatNumber(row.magnitude, 3)}</td>
        <td><button class="table-add" data-add-token="${row.token_id}" title="Add token">+</button></td>
      </tr>`).join('');
  }
  if (payload) {
    $('searchSummary').textContent = `${payload.total_matches.toLocaleString()} match${payload.total_matches === 1 ? '' : 'es'} · showing ${payload.results.length.toLocaleString()}`;
  } else {
    $('searchSummary').textContent = state.status?.loaded ? 'Search the loaded vocabulary.' : 'Load a model to search its vocabulary.';
  }
}

function addTokens(rows, replace = false) {
  if (replace) state.selected.clear();
  for (const row of rows) {
    if (state.selected.size >= 5000) break;
    state.selected.set(row.token_id, row);
  }
  renderSelected();
  markProjectionStale();
}

function selectedRows() {
  return [...state.selected.values()];
}

function renderSelected() {
  const rows = selectedRows();
  const anchorId = Number($('anchorIdInput').value);
  $('selectedCountBadge').textContent = `${rows.length.toLocaleString()} selected`;
  $('selectionStatusPill').textContent = `${rows.length.toLocaleString()} tokens`;
  const container = $('selectedTokens');
  if (!rows.length) {
    container.innerHTML = '<div class="empty-selection">No tokens selected.</div>';
    return;
  }
  const cap = 320;
  container.innerHTML = rows.slice(0, cap).map((row) => `
    <div class="selected-token-row ${row.token_id === anchorId ? 'anchor' : ''}">
      <span class="selected-token-id mono">${row.token_id}</span>
      <span class="selected-token-text mono" title="${escapeHtml(row.raw)}">${escapeHtml(row.display)}</span>
      <button class="remove-token" data-remove-token="${row.token_id}" title="Remove">×</button>
    </div>`).join('') + (rows.length > cap ? `<div class="empty-selection">…and ${(rows.length - cap).toLocaleString()} more tokens</div>` : '');
}

async function ensureAnchorSelected() {
  const anchorId = Number($('anchorIdInput').value);
  if (!Number.isInteger(anchorId) || anchorId < 0) throw new Error('Enter a valid anchor token ID.');
  if (!state.selected.has(anchorId)) {
    const token = await api(`/api/tokens/${anchorId}`);
    state.selected.set(token.token_id, token);
    renderSelected();
  }
  return anchorId;
}

async function loadSemanticVicinity() {
  if (!state.status?.loaded) return showToast('Load a model first.', 'error');
  const anchorId = Number($('anchorIdInput').value);
  const count = Math.max(2, Math.min(5000, Number($('vicinityCountInput').value) || 160));
  setLoading(true, 'Finding semantic vicinity', 'Scanning the vocabulary in blocks by cosine similarity.');
  try {
    const params = new URLSearchParams({ anchor_id: String(anchorId), limit: String(count), include_anchor: 'true' });
    const payload = await api(`/api/tokens/neighbors?${params}`);
    addTokens(payload.rows, true);
    showToast(`Selected ${payload.rows.length} semantic neighbors`);
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    setLoading(false);
  }
}

async function loadIdWindow() {
  if (!state.status?.loaded) return showToast('Load a model first.', 'error');
  const centerId = Number($('anchorIdInput').value);
  const count = Math.max(2, Math.min(5000, Number($('vicinityCountInput').value) || 160));
  try {
    const params = new URLSearchParams({ center_id: String(centerId), count: String(count) });
    const payload = await api(`/api/tokens/window?${params}`);
    addTokens(payload.rows, true);
    showToast(`Selected ${payload.rows.length} contiguous token IDs`);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

async function selectAnchorOnly() {
  if (!state.status?.loaded) return showToast('Load a model first.', 'error');
  try {
    const anchor = await ensureAnchorSelected();
    const row = state.selected.get(anchor);
    addTokens([row], true);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

// ---------------------------------------------------------------------------
// Projection UI
// ---------------------------------------------------------------------------

async function loadProjectionMethods() {
  try {
    const payload = await api('/api/projection/methods');
    state.methods = new Map(payload.methods.map((method) => [method.key, method]));
    const select = $('projectionMethodSelect');
    select.innerHTML = '';
    for (const method of payload.methods) {
      const option = document.createElement('option');
      option.value = method.key;
      option.disabled = !method.available;
      option.textContent = `${method.label}${method.available ? '' : ' · unavailable'}`;
      select.appendChild(option);
    }
    select.value = 'auto';
    updateMethodCard();
  } catch (error) {
    showToast(error.message, 'error');
  }
}

function updateMethodCard() {
  const method = state.methods.get($('projectionMethodSelect').value);
  if (!method) return;
  $('methodTitle').textContent = method.label;
  $('methodDescription').textContent = method.best_for;
  $('methodFamily').textContent = method.family;
  $('methodComplexity').textContent = method.complexity;
  $('methodCaveat').textContent = method.caveat;
}

function projectionPayload() {
  return {
    token_ids: selectedRows().map((row) => row.token_id),
    anchor_id: Number($('anchorIdInput').value),
    method: $('projectionMethodSelect').value,
    seed: Number($('projectionSeedInput').value) || 42,
    center_mode: $('centerModeSelect').value,
    manifold_neighbors: Number($('manifoldNeighborsInput').value) || 15,
    tsne_perplexity: Number($('tsnePerplexityInput').value) || 30,
    umap_min_dist: Number($('umapMinDistInput').value) || 0.1,
    align_anchor: $('alignAnchorInput').checked,
    edge_k: Number($('edgeKInput').value) || 0,
    max_edges: 4000,
  };
}

async function projectSelection() {
  if (!state.status?.loaded) return showToast('Load a model first.', 'error');
  try {
    await ensureAnchorSelected();
  } catch (error) {
    return showToast(error.message, 'error');
  }
  if (state.selected.size < 2) return showToast('Select at least two tokens.', 'error');
  const payload = projectionPayload();
  const label = state.methods.get(payload.method)?.label || payload.method;
  setLoading(true, `Projecting with ${label}`, 'Computing the embedding, radial sphere map, fidelity metrics, and original-space neighbor edges.');
  try {
    const result = await api('/api/projection', { method: 'POST', body: JSON.stringify(payload) });
    renderProjection(result);
    renderMetrics(result);
    $('projectionStatusPill').textContent = state.methods.get(result.actual_method)?.label || result.actual_method;
    $('projectionStatusPill').className = 'status-pill good';
    $('exportBtn').disabled = false;
    showToast(`Projected ${result.points.length.toLocaleString()} tokens with ${result.actual_method}`);
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    setLoading(false);
  }
}

function renderMetrics(result) {
  const metrics = result?.metrics;
  $('metricSpearman').textContent = metrics ? formatNumber(metrics.angular_spearman_rho, 3) : '—';
  $('metricStress').textContent = metrics ? formatNumber(metrics.stress_1, 3) : '—';
  $('metricMae').textContent = metrics ? `${formatNumber(metrics.mean_abs_angle_error_deg, 1)}°` : '—';
  $('metricKnn').textContent = metrics ? formatNumber(metrics.knn_recall_at_k, 3) : '—';
  $('metricKnnLabel').textContent = metrics ? `recall@${metrics.knn_k}` : 'local overlap';
  $('metricAnchor').textContent = metrics ? formatNumber(metrics.anchor_angle_spearman_rho, 3) : '—';
  $('metricRuntime').textContent = metrics ? `${formatNumber(metrics.runtime_ms, 0)} ms` : '—';
  const warnings = $('projectionWarnings');
  warnings.innerHTML = result?.warnings?.length
    ? result.warnings.map((warning) => `<div class="warning-item">${escapeHtml(warning)}</div>`).join('')
    : '';
}

async function benchmarkMethods() {
  if (!state.status?.loaded) return showToast('Load a model first.', 'error');
  try {
    await ensureAnchorSelected();
  } catch (error) {
    return showToast(error.message, 'error');
  }
  if (state.selected.size < 2) return showToast('Select at least two tokens.', 'error');
  const n = state.selected.size;
  const methods = ['spherical_pca', 'tangent_pca', 'random'];
  if (n <= 1200) methods.push('cosine_kernel');
  if (n <= 800) methods.push('angular_mds');
  if (n <= 500) methods.push('spherical_stress');
  if (n <= 1500) methods.push('isomap');
  if (state.methods.get('umap')?.available && n <= 5000) methods.push('umap');
  const body = {
    ...projectionPayload(),
    methods,
  };
  delete body.method;
  delete body.edge_k;
  delete body.max_edges;
  setLoading(true, 'Benchmarking projection methods', 'Running each eligible method on the same token matrix and comparing spherical distortion metrics.');
  try {
    const result = await api('/api/projection/compare', { method: 'POST', body: JSON.stringify(body) });
    renderBenchmark(result.rows);
    showToast(`Benchmarked ${result.rows.length} methods`);
  } catch (error) {
    showToast(error.message, 'error');
  } finally {
    setLoading(false);
  }
}

function renderBenchmark(rows) {
  $('benchmarkWrap').classList.remove('hidden');
  $('benchmarkBody').innerHTML = rows.map((row) => {
    if (!row.success) {
      return `<tr><td>${escapeHtml(row.requested_method)}</td><td colspan="5" class="comparison-error">${escapeHtml(row.error)}</td></tr>`;
    }
    const metrics = row.metrics;
    return `<tr class="comparison-row" data-benchmark-method="${escapeHtml(row.requested_method)}">
      <td>${escapeHtml(state.methods.get(row.actual_method)?.label || row.actual_method)}</td>
      <td>${formatNumber(metrics.angular_spearman_rho, 3)}</td>
      <td>${formatNumber(metrics.stress_1, 3)}</td>
      <td>${formatNumber(metrics.mean_abs_angle_error_deg, 1)}°</td>
      <td>${formatNumber(metrics.knn_recall_at_k, 3)}</td>
      <td>${formatNumber(metrics.runtime_ms, 0)}</td>
    </tr>`;
  }).join('');
}

function pinPoint(index) {
  if (!state.projection || index === null || index < 0 || index >= state.projection.points.length) return;
  state.pinnedIndex = index;
  renderInspector();
  rebuildLabels();
  updateMarkersFromPositions();
}

function renderInspector() {
  const point = state.pinnedIndex === null ? null : state.projection?.points?.[state.pinnedIndex];
  $('inspectorEmpty').classList.toggle('hidden', Boolean(point));
  $('inspectorContent').classList.toggle('hidden', !point);
  $('setPinnedAnchorBtn').disabled = !point;
  if (!point) return;
  $('inspectorToken').textContent = `${point.display} · ${point.token_id}`;
  const rows = [
    ['raw token', point.raw],
    ['token ID', point.token_id],
    ['vector norm', formatNumber(point.magnitude, 6)],
    ['angle to anchor', Number.isFinite(point.angle_to_anchor_deg) ? `${formatNumber(point.angle_to_anchor_deg, 4)}°` : '—'],
    ['cosine to anchor', formatNumber(point.cosine_to_anchor, 6)],
    ['sphere coordinate', `[${formatNumber(point.x, 4)}, ${formatNumber(point.y, 4)}, ${formatNumber(point.z, 4)}]`],
    ['special token', point.special ? 'yes' : 'no'],
  ];
  $('inspectorGrid').innerHTML = rows.map(([key, value]) => `<div class="vector-item"><span>${escapeHtml(key)}</span><span class="mono">${escapeHtml(value)}</span></div>`).join('');
}

function exportProjection() {
  if (!state.projection) return;
  const blob = new Blob([JSON.stringify(state.projection, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `vocabulary-sphere-${state.projection.actual_method}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function screenshot() {
  renderer.render(scene, camera);
  const link = document.createElement('a');
  link.href = renderer.domElement.toDataURL('image/png');
  link.download = 'vocabulary-sphere.png';
  link.click();
}

function syncAppearanceControls() {
  $('autoRotateInput').checked = state.autoRotate;
  $('showEdgesInput').checked = state.showEdges;
  $('showLabelsInput').checked = state.showLabels;
  $('toggleRotateBtn').classList.toggle('active', state.autoRotate);
  $('toggleEdgesBtn').classList.toggle('active', state.showEdges);
  $('toggleLabelsBtn').classList.toggle('active', state.showLabels);
  edgeLines.visible = state.showEdges && Boolean(state.projection);
  if (state.showEdges) updateEdgesFromPositions();
  rebuildLabels();
}

function setAutoRotate(value) {
  state.autoRotate = Boolean(value);
  syncAppearanceControls();
}
function setShowEdges(value) {
  state.showEdges = Boolean(value);
  syncAppearanceControls();
}
function setShowLabels(value) {
  state.showLabels = Boolean(value);
  syncAppearanceControls();
}

function isTypingTarget(element) {
  const tag = element?.tagName;
  return element?.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(tag);
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

$('loadModelBtn').addEventListener('click', loadModel);
$('unloadModelBtn').addEventListener('click', unloadModelUi);
$('refreshLocalBtn').addEventListener('click', refreshLocalModels);
$('localModelSelect').addEventListener('change', (event) => {
  if (event.target.value) $('modelNameInput').value = event.target.value;
});
$('searchBtn').addEventListener('click', performSearch);
$('searchInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') performSearch();
});
$('searchModeSelect').addEventListener('change', () => {
  const mode = $('searchModeSelect').value;
  $('searchInput').placeholder = mode === 'id' ? '42,90-110' : mode === 'literal' ? 'token text' : '^▁(cat|dog)';
});
$('searchResultsBody').addEventListener('click', (event) => {
  const button = event.target.closest('[data-add-token]');
  if (!button) return;
  const id = Number(button.dataset.addToken);
  const row = state.searchResults.find((item) => item.token_id === id);
  if (row) addTokens([row]);
});
$('addVisibleBtn').addEventListener('click', () => addTokens(state.searchResults));
$('replaceVisibleBtn').addEventListener('click', () => addTokens(state.searchResults, true));
$('semanticNeighborsBtn').addEventListener('click', loadSemanticVicinity);
$('idWindowBtn').addEventListener('click', loadIdWindow);
$('clearSelectionBtn').addEventListener('click', () => { state.selected.clear(); renderSelected(); markProjectionStale(); });
$('selectAnchorOnlyBtn').addEventListener('click', selectAnchorOnly);
$('selectedTokens').addEventListener('click', (event) => {
  const button = event.target.closest('[data-remove-token]');
  if (!button) return;
  state.selected.delete(Number(button.dataset.removeToken));
  renderSelected();
  markProjectionStale();
});
$('anchorIdInput').addEventListener('change', () => { renderSelected(); markProjectionStale(); });
$('projectionMethodSelect').addEventListener('change', updateMethodCard);
$('projectBtn').addEventListener('click', projectSelection);
$('benchmarkBtn').addEventListener('click', benchmarkMethods);
$('benchmarkBody').addEventListener('click', (event) => {
  const row = event.target.closest('[data-benchmark-method]');
  if (!row) return;
  $('projectionMethodSelect').value = row.dataset.benchmarkMethod;
  updateMethodCard();
  projectSelection();
});
$('exportBtn').addEventListener('click', exportProjection);
$('screenshotBtn').addEventListener('click', screenshot);
$('setPinnedAnchorBtn').addEventListener('click', async () => {
  const point = state.pinnedIndex === null ? null : state.projection?.points?.[state.pinnedIndex];
  if (!point) return;
  $('anchorIdInput').value = String(point.token_id);
  if (!state.selected.has(point.token_id)) state.selected.set(point.token_id, point);
  renderSelected();
  markProjectionStale();
  showToast(`Anchor set to token ${point.token_id}`);
});
$('pointSizeInput').addEventListener('input', (event) => {
  state.pointSize = Number(event.target.value);
  pointMaterial.size = state.pointSize;
  $('pointSizeLabel').textContent = state.pointSize.toFixed(3);
});
$('colorModeSelect').addEventListener('change', (event) => {
  state.colorMode = event.target.value;
  recolorPoints();
});
$('autoRotateInput').addEventListener('change', (event) => setAutoRotate(event.target.checked));
$('showEdgesInput').addEventListener('change', (event) => setShowEdges(event.target.checked));
$('showLabelsInput').addEventListener('change', (event) => setShowLabels(event.target.checked));
$('resetCameraBtn').addEventListener('click', resetCamera);
$('toggleRotateBtn').addEventListener('click', () => setAutoRotate(!state.autoRotate));
$('toggleEdgesBtn').addEventListener('click', () => setShowEdges(!state.showEdges));
$('toggleLabelsBtn').addEventListener('click', () => setShowLabels(!state.showLabels));

document.addEventListener('keydown', (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === '/' && !isTypingTarget(document.activeElement)) {
    event.preventDefault();
    $('searchInput').focus();
    return;
  }
  if (isTypingTarget(document.activeElement)) return;
  if (event.key === 'Enter') projectSelection();
  else if (event.key.toLowerCase() === 'r') resetCamera();
  else if (event.key.toLowerCase() === 'a') setAutoRotate(!state.autoRotate);
  else if (event.key.toLowerCase() === 'e') setShowEdges(!state.showEdges);
  else if (event.key.toLowerCase() === 'l') setShowLabels(!state.showLabels);
  else if (event.key === 'Escape') {
    state.pinnedIndex = null;
    renderInspector();
    rebuildLabels();
    updateMarkersFromPositions();
  }
});

async function initialize() {
  syncAppearanceControls();
  await Promise.all([loadProjectionMethods(), refreshStatus()]);
  renderSearchResults();
  renderSelected();
  renderMetrics(null);
}
initialize();
