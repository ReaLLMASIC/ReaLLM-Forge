import * as THREE from 'https://esm.sh/three@0.180.0';
import { OrbitControls } from 'https://esm.sh/three@0.180.0/examples/jsm/controls/OrbitControls.js';
import { LineSegments2 } from 'https://esm.sh/three@0.180.0/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'https://esm.sh/three@0.180.0/examples/jsm/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'https://esm.sh/three@0.180.0/examples/jsm/lines/LineMaterial.js';

const $ = (id) => document.getElementById(id);
const SPHERE_RADIUS = 2.18;
const BG = '#1b1b2f';
const APP_VERSION = '1.3.0';
const SETTINGS_SCHEMA = 'hf-vocab-sphere/settings';
const SETTINGS_FORMAT_VERSION = 1;
const MAX_SETTINGS_FILE_BYTES = 8 * 1024 * 1024;
const COLORS = {
  anchor: '#ff7b72',
  near: '#7dd3fc',
  middle: '#c77dff',
  orthogonal: '#ffd166',
  far: '#ff7b72',
  edge: '#58c4dd',
  sphere: '#ffffff',
  grid: '#8d99ae',
  resultant: '#52b788',
  resultantAlt: '#7dd3fc',
};

const state = {
  status: null,
  methods: new Map(),
  searchResults: [],
  tokenizedResults: [],
  selected: new Map(),
  projection: null,
  pinnedIndex: null,
  hoveredIndex: null,
  autoRotate: false,
  showEdges: true,
  showLabels: false,
  showLabelAliases: true,
  showLabelIds: true,
  showLabelMagnitudes: false,
  showLabelParticipation: false,
  showEdgeLabels: false,
  edgeColorMode: 'angle',
  edgeWidth: 1.8,
  nodeLabelSize: 13,
  edgeLabelSize: 11,
  edgeLabelLimit: 0,
  colorMode: 'anchor',
  pointSize: 0.075,
  sphereOpacity: 0.04,
  geometryMode: 'sphere',
  radialMode: 'surface',
  magnitudeShell: 0.18,
  arithmeticExpressions: [],
  arithmeticMode: 'expression',
  pendingProjectionMethod: null,
};

let pendingSettingsWorkspace = null;

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

function aliasForIndex(index) {
  let value = Number(index) + 1;
  let output = '';
  while (value > 0) {
    value -= 1;
    output = String.fromCharCode(65 + (value % 26)) + output;
    value = Math.floor(value / 26);
  }
  return output;
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
const labelCanvas = $('label-overlay');
const labelContext = labelCanvas.getContext('2d');
let labelPixelRatio = 1;
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

const sphereMaterial = new THREE.MeshBasicMaterial({
  color: COLORS.sphere,
  wireframe: true,
  transparent: true,
  opacity: state.sphereOpacity,
  depthWrite: false,
});
const sphere = new THREE.Mesh(new THREE.SphereGeometry(SPHERE_RADIUS, 48, 32), sphereMaterial);
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
const sphereGrid = createSphereGrid();
world.add(sphereGrid);

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

function createCartesianGrid() {
  const group = new THREE.Group();
  const size = SPHERE_RADIUS * 2.7;
  const divisions = 12;
  const make = () => {
    const grid = new THREE.GridHelper(size, divisions, COLORS.grid, COLORS.grid);
    const materials = Array.isArray(grid.material) ? grid.material : [grid.material];
    materials.forEach((material) => {
      material.transparent = true;
      material.opacity = 0.1;
      material.depthWrite = false;
    });
    return grid;
  };
  const xz = make();
  const xy = make();
  xy.rotation.x = Math.PI / 2;
  const yz = make();
  yz.rotation.z = Math.PI / 2;
  group.add(xz, xy, yz);
  group.visible = false;
  return group;
}

const cartesianGrid = createCartesianGrid();
world.add(cartesianGrid);

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

const edgeGeometry = new LineSegmentsGeometry();
const edgeMaterial = new LineMaterial({
  color: '#ffffff',
  linewidth: state.edgeWidth,
  vertexColors: true,
  transparent: true,
  opacity: 0.48,
  depthWrite: false,
  dashed: false,
  alphaToCoverage: true,
});
const edgeLines = new LineSegments2(edgeGeometry, edgeMaterial);
edgeLines.renderOrder = 1;
edgeLines.frustumCulled = false;
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

const resultantGroup = new THREE.Group();
world.add(resultantGroup);

let scenePoints = [];
let edgePairs = [];
let renderedEdgePairs = [];
let edgeBufferCount = 0;
let transition = null;
let labelEntries = [];
let edgeLabelEntries = [];
let resultantEntries = [];
let magnitudeProfile = null;

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

function clearLabels() {
  labelEntries = [];
}

function clearEdgeLabels() {
  edgeLabelEntries = [];
}

function labelMagnitudeText(point) {
  return state.showLabelMagnitudes && Number.isFinite(Number(point.magnitude))
    ? `‖v‖ ${formatNumber(point.magnitude, 3)}`
    : '';
}

function labelParticipationText(point) {
  return state.showLabelParticipation && Number.isFinite(Number(point.effective_dimension))
    ? `PR ${formatNumber(point.effective_dimension, 2)}`
    : '';
}

function labelTextForPoint(point) {
  const magnitudeText = labelMagnitudeText(point);
  const participationText = labelParticipationText(point);
  if (point.kind === 'resultant') {
    return [`${point.alias || 'R'}: ${point.label || point.display}`, magnitudeText, participationText]
      .filter((value) => value.length)
      .join(' · ');
  }
  const prefix = state.showLabelAliases && point.alias ? `${point.alias}: ` : '';
  const tokenText = point.display !== null && point.display !== undefined ? String(point.display) : '';
  const tokenId = state.showLabelIds && point.token_id !== null && point.token_id !== undefined
    ? String(point.token_id)
    : '';
  const body = [tokenText, tokenId, magnitudeText, participationText].filter((value) => value.length).join(' · ');
  return `${prefix}${body}`.trim();
}

function labelPriority(index, point) {
  if (state.pinnedIndex === index) return 4;
  if (point.kind === 'resultant') return 3;
  if (point.is_anchor) return 2;
  return 1;
}

function rebuildLabels() {
  clearLabels();
  if (!state.projection) return;
  const points = state.projection.points;
  const candidates = [];
  if (state.showLabels) {
    for (let index = 0; index < points.length; index += 1) candidates.push(index);
  }
  const anchorIndex = points.findIndex((point) => point.is_anchor);
  if (anchorIndex >= 0) candidates.push(anchorIndex);
  for (let index = 0; index < points.length; index += 1) {
    if (points[index].kind === 'resultant') candidates.push(index);
  }
  if (state.pinnedIndex !== null) candidates.push(state.pinnedIndex);
  labelEntries = [...new Set(candidates)]
    .filter((index) => index >= 0 && index < points.length)
    .map((index) => ({ index }))
    .sort((a, b) => labelPriority(a.index, points[a.index]) - labelPriority(b.index, points[b.index]));
}

function updateLabelsFromPositions() {
  // Labels are drawn in one screen-space canvas, so no per-label GPU objects
  // or texture limits are involved.
}

function rebuildEdgeLabels() {
  clearEdgeLabels();
  if (!state.projection || !state.showEdges || !state.showEdgeLabels || !edgeLines.visible) return;
  const available = renderedEdgePairs.length ? renderedEdgePairs : edgePairs;
  const requestedLimit = Math.max(0, Math.min(20000, Number(state.edgeLabelLimit) || 0));
  const rows = requestedLimit === 0 ? available : available.slice(0, requestedLimit);
  edgeLabelEntries = rows.map((edge) => ({ edge }));
}

function updateEdgeLabelsFromPositions() {
  // Edge labels share the screen-space label canvas with node labels.
}

const projectedLabelPosition = new THREE.Vector3();
function projectToLabelCanvas(position) {
  projectedLabelPosition.copy(position).project(camera);
  if (
    !Number.isFinite(projectedLabelPosition.x)
    || !Number.isFinite(projectedLabelPosition.y)
    || !Number.isFinite(projectedLabelPosition.z)
    || projectedLabelPosition.z < -1
    || projectedLabelPosition.z > 1
  ) return null;
  const width = labelCanvas.clientWidth;
  const height = labelCanvas.clientHeight;
  return {
    x: (projectedLabelPosition.x * 0.5 + 0.5) * width,
    y: (-projectedLabelPosition.y * 0.5 + 0.5) * height,
  };
}

function drawOutlinedLabel(text, screen, color, fontSize, options = {}) {
  if (!screen || !String(text ?? '').length) return;
  const width = labelCanvas.clientWidth;
  const height = labelCanvas.clientHeight;
  const margin = Math.max(30, fontSize * 4);
  if (screen.x < -margin || screen.x > width + margin || screen.y < -margin || screen.y > height + margin) return;
  labelContext.save();
  labelContext.globalAlpha = options.alpha ?? 1;
  labelContext.font = `700 ${fontSize}px Inter, Arial, sans-serif`;
  labelContext.textAlign = options.align || 'center';
  labelContext.textBaseline = 'middle';
  labelContext.lineJoin = 'round';
  labelContext.lineWidth = Math.max(2.5, fontSize * 0.25);
  labelContext.strokeStyle = 'rgba(8, 8, 20, 0.9)';
  labelContext.strokeText(text, screen.x, screen.y);
  labelContext.fillStyle = color;
  labelContext.fillText(text, screen.x, screen.y);
  labelContext.restore();
}

function drawLabelOverlay() {
  const width = labelCanvas.clientWidth;
  const height = labelCanvas.clientHeight;
  labelContext.setTransform(labelPixelRatio, 0, 0, labelPixelRatio, 0, 0);
  labelContext.clearRect(0, 0, width, height);
  if (!state.projection) return;
  const positions = tokenPoints.geometry.getAttribute('position');
  if (!positions) return;

  if (state.showEdges && state.showEdgeLabels && edgeLines.visible) {
    const edgeFontSize = Math.max(7, Number(state.edgeLabelSize) || 11);
    for (const entry of edgeLabelEntries) {
      const edge = entry.edge;
      if (edge.source_index >= positions.count || edge.target_index >= positions.count) continue;
      const a = new THREE.Vector3(
        positions.getX(edge.source_index),
        positions.getY(edge.source_index),
        positions.getZ(edge.source_index),
      );
      const b = new THREE.Vector3(
        positions.getX(edge.target_index),
        positions.getY(edge.target_index),
        positions.getZ(edge.target_index),
      );
      const midpoint = a.add(b).multiplyScalar(0.5);
      if (state.geometryMode === 'sphere' && midpoint.lengthSq() > 0.02) {
        midpoint.add(midpoint.clone().normalize().multiplyScalar(0.055));
      }
      const color = edgeColor(edge);
      drawOutlinedLabel(
        `${formatNumber(edge.angle_deg, 1)}°`,
        projectToLabelCanvas(midpoint),
        `#${color.getHexString()}`,
        edgeFontSize,
        { alpha: 0.94 },
      );
    }
  }

  const nodeFontSize = Math.max(8, Number(state.nodeLabelSize) || 13);
  for (const entry of labelEntries) {
    if (entry.index >= positions.count) continue;
    const point = state.projection.points[entry.index];
    if (!point) continue;
    const position = new THREE.Vector3(
      positions.getX(entry.index),
      positions.getY(entry.index),
      positions.getZ(entry.index),
    );
    const offset = position.lengthSq() > 0.02
      ? position.clone().normalize().multiplyScalar(0.13)
      : new THREE.Vector3(0, 0.14, 0);
    const screen = projectToLabelCanvas(position.add(offset));
    const color = point.kind === 'resultant' ? resultantColor(point) : point.is_anchor ? COLORS.anchor : '#ffffff';
    drawOutlinedLabel(labelTextForPoint(point), screen, color, nodeFontSize);
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

function resultantColor(pointOrIndex) {
  const palette = ['#52b788', '#7dd3fc', '#f4a261', '#c77dff', '#ffd166', '#58c4dd'];
  let index = Number(pointOrIndex) || 0;
  if (pointOrIndex && typeof pointOrIndex === 'object') {
    const match = String(pointOrIndex.alias || '').match(/^R(\d+)$/i);
    index = match ? Math.max(0, Number(match[1]) - 1) : Number(pointOrIndex.index) || 0;
  }
  return palette[((index % palette.length) + palette.length) % palette.length];
}

function colorForPoint(point, ranges) {
  if (point.kind === 'resultant') return new THREE.Color(resultantColor(point));
  if (point.is_anchor) return new THREE.Color(COLORS.anchor);
  if (state.colorMode === 'id') {
    const vocab = Math.max(1, state.status?.vocab_size || 1);
    return new THREE.Color().setHSL(((Number(point.token_id) || 0) / vocab + 0.54) % 1, 0.7, 0.64);
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
  updateLegend();
}

function updateLegend() {
  if (!state.projection) {
    $('legendCard').classList.add('hidden');
    return;
  }
  const nodeAngle = state.colorMode === 'anchor';
  const edgeAngle = state.showEdges && state.edgeColorMode === 'angle';
  $('legendCard').classList.toggle('hidden', !nodeAngle && !edgeAngle);
  $('legendTitle').textContent = nodeAngle && edgeAngle
    ? 'Node anchor angle · edge pair angle'
    : nodeAngle
      ? 'Node color: angle from anchor'
      : 'Edge color: original pair angle';
  $('legendNote').textContent = edgeAngle
    ? 'Edge colors and labels use exact original-space angles.'
    : 'Node colors use the original angle from the selected anchor.';
}

function computeMagnitudeProfile(points) {
  const values = points.map((point) => Number(point.magnitude)).filter((value) => Number.isFinite(value) && value >= 0);
  if (!values.length) return { low: 0, high: 1 };
  const low = Math.min(...values);
  const high = Math.max(...values);
  return { low, high };
}

function displayPositionForPoint(point) {
  const coordinate = new THREE.Vector3(Number(point.x) || 0, Number(point.y) || 0, Number(point.z) || 0);
  if (state.geometryMode === 'euclidean') return coordinate.multiplyScalar(SPHERE_RADIUS);
  const direction = coordinate.lengthSq() > 1e-16 ? coordinate.normalize() : new THREE.Vector3(0, 1, 0);
  let radius = SPHERE_RADIUS;
  if (state.radialMode === 'magnitude' && magnitudeProfile) {
    const magnitude = Number(point.magnitude);
    const span = magnitudeProfile.high - magnitudeProfile.low;
    const normalized = Number.isFinite(magnitude) && span > 1e-9
      ? Math.max(-1, Math.min(1, 2 * ((magnitude - magnitudeProfile.low) / span) - 1))
      : 0;
    radius *= 1 + state.magnitudeShell * normalized;
  }
  return direction.multiplyScalar(radius);
}

function buildProjectionTargets(data) {
  magnitudeProfile = computeMagnitudeProfile(data.points);
  const targets = new Float32Array(data.points.length * 3);
  data.points.forEach((point, index) => {
    const position = displayPositionForPoint(point);
    targets[index * 3] = position.x;
    targets[index * 3 + 1] = position.y;
    targets[index * 3 + 2] = position.z;
  });
  return targets;
}

function edgeColor(edge) {
  return state.edgeColorMode === 'angle'
    ? gradientColor((Number(edge.angle_deg) || 0) / 180)
    : new THREE.Color(COLORS.edge);
}

function edgeIsRenderable(edge, positions) {
  const source = Number(edge.source_index);
  const target = Number(edge.target_index);
  if (!Number.isInteger(source) || !Number.isInteger(target) || source < 0 || target < 0) return false;
  if (source >= positions.count || target >= positions.count || source === target) return false;
  return [
    positions.getX(source), positions.getY(source), positions.getZ(source),
    positions.getX(target), positions.getY(target), positions.getZ(target),
  ].every(Number.isFinite);
}

function ensureEdgeBuffers(count) {
  if (count === edgeBufferCount && edgeGeometry.getAttribute('instanceStart')) return;
  edgeGeometry.dispose();
  edgeGeometry.setPositions(new Float32Array(count * 6));
  edgeGeometry.setColors(new Float32Array(count * 6));
  edgeGeometry.instanceCount = count;
  edgeBufferCount = count;
  const positionBuffer = edgeGeometry.getAttribute('instanceStart')?.data;
  const colorBuffer = edgeGeometry.getAttribute('instanceColorStart')?.data;
  positionBuffer?.setUsage?.(THREE.DynamicDrawUsage);
  colorBuffer?.setUsage?.(THREE.DynamicDrawUsage);
}

function updateEdgesFromPositions() {
  if (!state.projection || !state.showEdges || !edgePairs.length) {
    renderedEdgePairs = [];
    edgeLines.visible = false;
    return;
  }
  const positions = tokenPoints.geometry.getAttribute('position');
  if (!positions) {
    renderedEdgePairs = [];
    edgeLines.visible = false;
    return;
  }

  renderedEdgePairs = edgePairs.filter((edge) => edgeIsRenderable(edge, positions));
  if (!renderedEdgePairs.length) {
    edgeLines.visible = false;
    return;
  }

  ensureEdgeBuffers(renderedEdgePairs.length);
  const positionBuffer = edgeGeometry.getAttribute('instanceStart').data;
  const colorBuffer = edgeGeometry.getAttribute('instanceColorStart').data;
  const positionArray = positionBuffer.array;
  const colorArray = colorBuffer.array;

  renderedEdgePairs.forEach((edge, index) => {
    const source = edge.source_index;
    const target = edge.target_index;
    const offset = index * 6;
    positionArray[offset] = positions.getX(source);
    positionArray[offset + 1] = positions.getY(source);
    positionArray[offset + 2] = positions.getZ(source);
    positionArray[offset + 3] = positions.getX(target);
    positionArray[offset + 4] = positions.getY(target);
    positionArray[offset + 5] = positions.getZ(target);
    const color = edgeColor(edge);
    colorArray[offset] = color.r;
    colorArray[offset + 1] = color.g;
    colorArray[offset + 2] = color.b;
    colorArray[offset + 3] = color.r;
    colorArray[offset + 4] = color.g;
    colorArray[offset + 5] = color.b;
  });

  positionBuffer.needsUpdate = true;
  colorBuffer.needsUpdate = true;
  edgeGeometry.instanceCount = renderedEdgePairs.length;
  edgeMaterial.linewidth = state.edgeWidth;
  edgeLines.visible = true;
}

function clearResultants() {
  for (const entry of resultantEntries) {
    resultantGroup.remove(entry.group);
    entry.group.traverse((child) => {
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    });
  }
  resultantEntries = [];
}

function makeResultantGlyph(index, point) {
  const color = resultantColor(point || index);
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.25,
    roughness: 0.5,
    transparent: true,
    opacity: 0.94,
    depthWrite: false,
  });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1, 12), material);
  const head = new THREE.Mesh(new THREE.ConeGeometry(0.075, 0.22, 16), material.clone());
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.115, 20, 16), material.clone());
  group.add(shaft, head, marker);
  group.renderOrder = 6;
  return { group, shaft, head, marker, index };
}

function rebuildResultants() {
  clearResultants();
  if (!state.projection) return;
  state.projection.points.forEach((point, index) => {
    if (point.kind !== 'resultant') return;
    const entry = makeResultantGlyph(index, point);
    entry.pointIndex = index;
    resultantEntries.push(entry);
    resultantGroup.add(entry.group);
  });
  updateResultantsFromPositions();
}

function updateResultantsFromPositions() {
  const positions = tokenPoints.geometry.getAttribute('position');
  if (!positions) return;
  for (const entry of resultantEntries) {
    if (entry.pointIndex >= positions.count) continue;
    const end = new THREE.Vector3(
      positions.getX(entry.pointIndex),
      positions.getY(entry.pointIndex),
      positions.getZ(entry.pointIndex),
    );
    const rawLength = end.length();
    const length = Math.max(0.001, rawLength);
    const direction = rawLength > 1e-9
      ? end.clone().multiplyScalar(1 / rawLength)
      : new THREE.Vector3(0, 1, 0);
    const headLength = Math.min(0.24, Math.max(0.025, length * 0.16), length * 0.65);
    const shaftLength = Math.max(0.0001, length - headLength);
    entry.group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    entry.shaft.scale.set(1, shaftLength, 1);
    entry.shaft.position.set(0, shaftLength / 2, 0);
    entry.head.scale.set(1, headLength / 0.22, 1);
    entry.head.position.set(0, shaftLength + headLength / 2, 0);
    entry.marker.position.set(0, length, 0);
  }
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

function applyScaffoldAppearance() {
  const spherical = state.geometryMode === 'sphere';
  sphere.visible = spherical && state.sphereOpacity > 0;
  sphereMaterial.opacity = state.sphereOpacity;
  sphereGrid.visible = spherical && state.sphereOpacity > 0;
  sphereGrid.children.forEach((line) => {
    line.material.opacity = Math.min(0.28, state.sphereOpacity * 2.4);
  });
  cartesianGrid.visible = !spherical;
  $('radialModeRow')?.classList.toggle('hidden', !spherical);
  $('magnitudeShellWrap')?.classList.toggle('hidden', !spherical || state.radialMode !== 'magnitude');
  $('sphereOpacityInput').disabled = !spherical;
  $('projectionPanelTitle').textContent = spherical ? 'Map geometry to S²' : 'Map geometry to free R³';
}

function animateToCurrentDisplay(duration = 520) {
  if (!state.projection) return;
  const targets = buildProjectionTargets(state.projection);
  const old = tokenPoints.geometry.getAttribute('position');
  const starts = old && old.count === state.projection.points.length
    ? new Float32Array(old.array)
    : new Float32Array(targets.length);
  transition = { starts, targets, startedAt: performance.now(), duration };
  applyScaffoldAppearance();
}

function renderProjection(data) {
  state.projection = data;
  state.geometryMode = data.geometry_mode || 'sphere';
  $('geometryModeSelect').value = state.geometryMode;
  scenePoints = data.points;
  edgePairs = data.edges || [];
  state.pinnedIndex = null;
  state.hoveredIndex = null;
  renderInspector();

  const targets = buildProjectionTargets(data);
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
  applyScaffoldAppearance();
  rebuildLabels();
  rebuildResultants();
  updateEdgesFromPositions();
  rebuildEdgeLabels();
  updateMarkersFromPositions();
}


function renderDemo() {
  state.projection = null;
  state.pinnedIndex = null;
  state.hoveredIndex = null;
  scenePoints = [];
  edgePairs = [];
  clearLabels();
  clearEdgeLabels();
  clearResultants();
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
  applyScaffoldAppearance();
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
    const identity = point.kind === 'resultant' ? `${point.alias || 'R'} · resultant` : `ID ${point.token_id}`;
    tooltip.innerHTML = `<strong class="mono">${escapeHtml(point.display)}</strong><span>${escapeHtml(identity)}${Number.isFinite(point.angle_to_anchor_deg) ? ` · ${formatNumber(point.angle_to_anchor_deg, 2)}° from anchor` : ''}</span>`;
    tooltip.classList.remove('hidden');
    $('sceneInspectorToken').textContent = point.kind === 'resultant' ? `${point.alias}: ${point.display}` : `${point.display} · ${point.token_id}`;
    $('sceneInspectorMeta').textContent = `norm ${formatNumber(point.magnitude, 4)}${point.kind === 'resultant' ? ` · ${point.expression}` : ''}${Number.isFinite(point.angle_to_anchor_deg) ? ` · angle ${formatNumber(point.angle_to_anchor_deg, 2)}°` : ''}`;
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
  labelPixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  labelCanvas.width = Math.max(1, Math.round(width * labelPixelRatio));
  labelCanvas.height = Math.max(1, Math.round(height * labelPixelRatio));
  edgeMaterial.resolution.set(Math.max(1, width), Math.max(1, height));
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
      current.array[i] = transition.starts[i] + (transition.targets[i] - transition.starts[i]) * t;
      current.array[i + 1] = transition.starts[i + 1] + (transition.targets[i + 1] - transition.starts[i + 1]) * t;
      current.array[i + 2] = transition.starts[i + 2] + (transition.targets[i + 2] - transition.starts[i + 2]) * t;
    }
    current.needsUpdate = true;
    tokenPoints.geometry.computeBoundingSphere();
    updateEdgesFromPositions();
    updateMarkersFromPositions();
    updateLabelsFromPositions();
    updateResultantsFromPositions();
    updateEdgeLabelsFromPositions();
    if (rawT >= 1) transition = null;
  }

  if (state.autoRotate) {
    camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), delta * 0.18);
  }
  hoverRing.lookAt(camera.position);
  pinnedRing.lookAt(camera.position);
  controls.update();
  renderer.render(scene, camera);
  drawLabelOverlay();
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
    state.tokenizedResults = [];
    state.arithmeticExpressions = [];
    resetProjectionUi();
    renderSearchResults();
    renderTokenizedResults();
    renderSelected();
    renderDemo();
    const restored = restorePendingSettingsWorkspace();
    if (!restored) {
      const first = await api('/api/tokens/0');
      state.selected.set(first.token_id, first);
      $('anchorIdInput').value = String(first.token_id);
      renderSelected();
    }
    showToast(restored ? `Loaded ${status.model_name} and restored saved workspace` : `Loaded ${status.model_name}`);
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
    state.tokenizedResults = [];
    state.arithmeticExpressions = [];
    resetProjectionUi();
    applyStatus(status);
    renderSearchResults();
    renderTokenizedResults();
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

function renderTokenizedResults(payload = null) {
  const body = $('tokenizedResultsBody');
  const projectable = state.tokenizedResults.filter((row) => row.projectable);
  if (!state.tokenizedResults.length) {
    body.innerHTML = '<tr><td colspan="4" class="empty-cell">No text tokenized yet.</td></tr>';
  } else {
    body.innerHTML = state.tokenizedResults.map((row) => `
      <tr class="${row.projectable ? '' : 'tokenized-unavailable'}">
        <td class="mono">${row.sequence_index}</td>
        <td class="mono">${row.token_id}</td>
        <td class="tokenized-piece" title="${escapeHtml(row.raw)}">
          <div class="tokenized-token mono">${escapeHtml(row.display)}</div>
          <div class="tokenized-decoded mono">${escapeHtml(row.decoded || '∅')}</div>
        </td>
        <td><button class="table-add" data-add-tokenized="${row.sequence_index}" title="${row.projectable ? 'Add token' : 'No vector row for this tokenizer token'}" ${row.projectable ? '' : 'disabled'}>+</button></td>
      </tr>`).join('');
  }
  $('addAllTokenizedBtn').disabled = projectable.length === 0;
  $('replaceTokenizedBtn').disabled = projectable.length === 0;
  if (payload) {
    const suffix = payload.truncated ? ` · first ${payload.returned_count.toLocaleString()} shown` : '';
    $('tokenizeSummary').textContent = `${payload.token_count.toLocaleString()} occurrence${payload.token_count === 1 ? '' : 's'} · ${payload.unique_token_count.toLocaleString()} unique${suffix}`;
  } else {
    $('tokenizeSummary').textContent = state.status?.loaded
      ? 'Tokenized occurrences will appear in sequence order.'
      : 'Load a model to tokenize text.';
  }
}

async function tokenizeSuppliedText() {
  if (!state.status?.loaded) return showToast('Load a model first.', 'error');
  const body = {
    text: $('tokenizeTextInput').value,
    add_special_tokens: $('tokenizeSpecialInput').checked,
    max_tokens: Math.max(1, Math.min(10000, Number($('tokenizeLimitInput').value) || 5000)),
  };
  try {
    const payload = await api('/api/tokens/tokenize', { method: 'POST', body: JSON.stringify(body) });
    state.tokenizedResults = payload.tokens;
    renderTokenizedResults(payload);
    showToast(`Tokenized ${payload.token_count.toLocaleString()} occurrence${payload.token_count === 1 ? '' : 's'}`);
  } catch (error) {
    showToast(error.message, 'error');
  }
}

function addAllTokenized(replace = false) {
  const rows = state.tokenizedResults.filter((row) => row.projectable);
  if (!rows.length) return showToast('No projectable token rows are available.', 'error');
  addTokens(rows, replace);
  showToast(`${replace ? 'Replaced with' : 'Added'} ${new Set(rows.map((row) => row.token_id)).size.toLocaleString()} unique token${rows.length === 1 ? '' : 's'}`);
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

function updateSelectionStatus() {
  const tokenCount = state.selected.size;
  const resultantCount = state.arithmeticExpressions.length;
  $('selectionStatusPill').textContent = resultantCount
    ? `${tokenCount.toLocaleString()} tokens + ${resultantCount.toLocaleString()} resultants`
    : `${tokenCount.toLocaleString()} tokens`;
}

function renderSelected() {
  const rows = selectedRows();
  const anchorId = Number($('anchorIdInput').value);
  $('selectedCountBadge').textContent = `${rows.length.toLocaleString()} selected`;
  updateSelectionStatus();
  const container = $('selectedTokens');
  renderArithmeticPanel();
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

function refreshSlerpAliasOptions(rows) {
  const from = $('slerpFromSelect');
  const to = $('slerpToSelect');
  const previousFrom = from.value;
  const previousTo = to.value;
  const options = rows.map((row, index) => {
    const alias = aliasForIndex(index);
    return `<option value="${alias}">${alias} · ${escapeHtml(row.display)} · ${row.token_id}</option>`;
  }).join('');
  from.innerHTML = options || '<option value="">Select vectors first</option>';
  to.innerHTML = options || '<option value="">Select vectors first</option>';
  const aliases = new Set(rows.map((_, index) => aliasForIndex(index)));
  from.value = aliases.has(previousFrom) ? previousFrom : rows.length ? 'A' : '';
  const defaultTo = rows.length > 1 ? 'B' : rows.length ? 'A' : '';
  to.value = aliases.has(previousTo) ? previousTo : defaultTo;
  from.disabled = rows.length === 0;
  to.disabled = rows.length === 0;
}

function syncArithmeticMode() {
  state.arithmeticMode = $('arithmeticModeSelect').value;
  const slerp = state.arithmeticMode === 'slerp';
  $('arithmeticExpressionWrap').classList.toggle('hidden', slerp);
  $('slerpControlsWrap').classList.toggle('hidden', !slerp);
  $('addArithmeticBtn').textContent = slerp ? 'Add SLERP resultant to map' : 'Add resultant to map';
}

function renderArithmeticPanel() {
  const rows = selectedRows();
  updateSelectionStatus();
  refreshSlerpAliasOptions(rows);
  syncArithmeticMode();
  const aliasContainer = $('arithmeticAliasGrid');
  if (!rows.length) {
    aliasContainer.innerHTML = '<div class="empty-selection">Select tokens to assign A, B, C…</div>';
  } else {
    const cap = 104;
    aliasContainer.innerHTML = rows.slice(0, cap).map((row, index) => `
      <div class="alias-item" title="${escapeHtml(row.raw)}">
        <span class="alias-symbol mono">${aliasForIndex(index)}</span>
        <span class="alias-token mono">${escapeHtml(row.display)} · ${row.token_id}</span>
      </div>`).join('') + (rows.length > cap ? `<div class="empty-selection">…and ${(rows.length - cap).toLocaleString()} more aliases</div>` : '');
  }

  const resultContainer = $('arithmeticResults');
  if (!state.arithmeticExpressions.length) {
    resultContainer.innerHTML = '<div class="empty-selection">No resultants defined.</div>';
  } else {
    resultContainer.innerHTML = state.arithmeticExpressions.map((item, index) => `
      <div class="arithmetic-result-row">
        <span class="arithmetic-result-alias mono">R${index + 1}</span>
        <div class="arithmetic-result-copy">
          <div class="arithmetic-result-label">${escapeHtml(item.label || `R${index + 1}`)}</div>
          <div class="arithmetic-result-expression mono" title="${escapeHtml(item.expression)}">${escapeHtml(item.expression)}</div>
        </div>
        <button class="remove-resultant" data-remove-resultant="${index}" title="Remove resultant">×</button>
      </div>`).join('');
  }
}

async function addArithmeticResult() {
  const rows = selectedRows();
  const slerpMode = $('arithmeticModeSelect').value === 'slerp';
  const rawFraction = Number($('slerpFractionInput').value);
  const fraction = Number.isFinite(rawFraction) ? rawFraction : 0;
  const fromAlias = $('slerpFromSelect').value;
  const toAlias = $('slerpToSelect').value;
  const expression = slerpMode
    ? `slerp(${fromAlias}, ${toAlias}, ${fraction.toFixed(4)})`
    : $('arithmeticExpressionInput').value.trim();
  const requestedLabel = $('arithmeticLabelInput').value.trim();
  const label = requestedLabel || (slerpMode
    ? `SLERP ${fromAlias}→${toAlias} · t=${fraction.toFixed(2)}`
    : `R${state.arithmeticExpressions.length + 1}`);
  if (!expression) return showToast('Enter a vector expression.', 'error');
  if (!rows.length) return showToast('Select vectors before defining arithmetic.', 'error');
  if (slerpMode && (rows.length < 2 || !fromAlias || !toAlias)) return showToast('Select at least two vectors for SLERP.', 'error');
  if (slerpMode && fromAlias === toAlias) return showToast('Choose two different aliases for SLERP.', 'error');
  if (state.arithmeticExpressions.length >= 12) return showToast('At most 12 resultants can be shown at once.', 'error');
  state.arithmeticExpressions.push({ expression, label });
  $('arithmeticLabelInput').value = '';
  renderArithmeticPanel();
  markProjectionStale();
  if (state.status?.loaded && state.selected.size >= 2) await projectSelection();
}

function clearArithmeticResults() {
  state.arithmeticExpressions = [];
  renderArithmeticPanel();
  markProjectionStale();
  if (state.projection) projectSelection();
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
    const preferredMethod = state.pendingProjectionMethod || 'auto';
    const preferredOption = [...select.options].find((option) => option.value === preferredMethod && !option.disabled);
    select.value = preferredOption ? preferredMethod : 'auto';
    state.pendingProjectionMethod = null;
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
    geometry_mode: $('geometryModeSelect').value,
    arithmetic_expressions: state.arithmeticExpressions.map((item) => ({ expression: item.expression, label: item.label || null })),
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
  setLoading(true, `Projecting with ${label}`, payload.geometry_mode === 'sphere' ? 'Computing the embedding, spherical map, fidelity metrics, resultants, and original-space edges.' : 'Computing the free 3-D embedding, fidelity metrics, resultants, and original-space edges.');
  try {
    const result = await api('/api/projection', { method: 'POST', body: JSON.stringify(payload) });
    renderProjection(result);
    renderMetrics(result);
    $('projectionStatusPill').textContent = `${state.methods.get(result.actual_method)?.label || result.actual_method} · ${result.geometry_mode === 'euclidean' ? 'R³' : 'S²'}`;
    $('projectionStatusPill').className = 'status-pill good';
    $('exportBtn').disabled = false;
    showToast(`Projected ${result.points.length.toLocaleString()} vectors with ${result.actual_method}`);
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
  $('metricGeometryNote').textContent = !metrics
    ? 'Choose a geometry and project a selection to calculate fidelity.'
    : metrics.metric_geometry === 'euclidean'
      ? `Free R³: Euclidean distances; stress and degree error use a fitted scale of ${formatNumber(metrics.low_distance_scale_to_radians, 4)} radians per display unit.`
      : 'S²: fidelity uses great-circle angles on the unit-sphere coordinates; magnitude-shell displacement is presentation-only.';
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
  const n = state.selected.size + state.arithmeticExpressions.length;
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
  setLoading(true, 'Benchmarking projection methods', `Running each eligible method on the same token/resultant matrix and comparing ${body.geometry_mode === 'euclidean' ? 'free-space' : 'spherical'} fidelity.`);
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
  $('setPinnedAnchorBtn').disabled = !point || point.kind === 'resultant' || point.token_id === null;
  if (!point) return;
  $('inspectorToken').textContent = point.kind === 'resultant'
    ? `${point.alias}: ${point.display}`
    : `${point.display} · ${point.token_id}`;
  const rows = point.kind === 'resultant'
    ? [
      ['type', 'derived resultant'],
      ['alias', point.alias || '—'],
      ['expression', point.expression || point.raw],
      ['uses', (point.referenced_aliases || []).join(', ') || '—'],
      ['vector norm', formatNumber(point.magnitude, 6)],
      ['angle to anchor', Number.isFinite(point.angle_to_anchor_deg) ? `${formatNumber(point.angle_to_anchor_deg, 4)}°` : '—'],
      ['cosine to anchor', formatNumber(point.cosine_to_anchor, 6)],
      [state.geometryMode === 'sphere' ? 'sphere coordinate' : 'free 3-D coordinate', `[${formatNumber(point.x, 4)}, ${formatNumber(point.y, 4)}, ${formatNumber(point.z, 4)}]`],
    ]
    : [
      ['alias', point.alias || '—'],
      ['raw token', point.raw],
      ['token ID', point.token_id],
      ['vector norm', formatNumber(point.magnitude, 6)],
      ['angle to anchor', Number.isFinite(point.angle_to_anchor_deg) ? `${formatNumber(point.angle_to_anchor_deg, 4)}°` : '—'],
      ['cosine to anchor', formatNumber(point.cosine_to_anchor, 6)],
      [state.geometryMode === 'sphere' ? 'sphere coordinate' : 'free 3-D coordinate', `[${formatNumber(point.x, 4)}, ${formatNumber(point.y, 4)}, ${formatNumber(point.z, 4)}]`],
      ['special token', point.special ? 'yes' : 'no'],
    ];
  $('inspectorGrid').innerHTML = rows.map(([key, value]) => `<div class="vector-item"><span>${escapeHtml(key)}</span><span class="mono">${escapeHtml(value)}</span></div>`).join('');
}



function finiteNumber(value, fallback, min = -Infinity, max = Infinity) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function finiteInteger(value, fallback, min = -Infinity, max = Infinity) {
  return Math.round(finiteNumber(value, fallback, min, max));
}

function booleanValue(value, fallback = false) {
  return typeof value === 'boolean' ? value : fallback;
}

function stringValue(value, fallback = '', maxLength = 100000) {
  return typeof value === 'string' ? value.slice(0, maxLength) : fallback;
}

function setSelectIfAvailable(id, value, fallback = null) {
  const select = $(id);
  const requested = String(value ?? '');
  const option = [...select.options].find((item) => item.value === requested && !item.disabled);
  if (option) {
    select.value = requested;
    return true;
  }
  if (fallback !== null) {
    const fallbackOption = [...select.options].find((item) => item.value === String(fallback) && !item.disabled);
    if (fallbackOption) select.value = String(fallback);
  }
  return false;
}

function compactSelectedToken(row) {
  return {
    token_id: Number(row.token_id),
    raw: String(row.raw ?? row.display ?? row.token_id),
    display: String(row.display ?? row.raw ?? row.token_id),
    magnitude: Number.isFinite(Number(row.magnitude)) ? Number(row.magnitude) : null,
    special: Boolean(row.special),
    present_in_tokenizer: row.present_in_tokenizer !== false,
  };
}

function buildSettingsSnapshot() {
  return {
    schema: SETTINGS_SCHEMA,
    format_version: SETTINGS_FORMAT_VERSION,
    app_version: APP_VERSION,
    saved_at: new Date().toISOString(),
    model: {
      model_name: $('modelNameInput').value.trim(),
      revision: $('revisionInput').value.trim() || 'main',
      matrix_source: $('matrixSourceSelect').value,
      compute_device: $('deviceInput').value.trim() || 'auto',
      allow_download: $('allowDownloadInput').checked,
      active: state.status?.loaded ? {
        model_name: state.status.model_name,
        revision: state.status.revision,
        matrix_source: state.status.matrix_source,
      } : null,
    },
    search: {
      mode: $('searchModeSelect').value,
      pattern: $('searchInput').value,
      case_sensitive: $('caseSensitiveInput').checked,
      limit: Number($('searchLimitInput').value) || 200,
      tokenization_text: $('tokenizeTextInput').value,
      tokenization_add_special_tokens: $('tokenizeSpecialInput').checked,
      tokenization_limit: Number($('tokenizeLimitInput').value) || 5000,
      vicinity_count: Number($('vicinityCountInput').value) || 160,
    },
    workspace: {
      anchor_id: Number($('anchorIdInput').value) || 0,
      selected_tokens: selectedRows().map(compactSelectedToken),
      arithmetic_expressions: state.arithmeticExpressions.map((item) => ({
        expression: String(item.expression ?? ''),
        label: item.label ? String(item.label) : null,
      })),
      arithmetic_editor: {
        mode: $('arithmeticModeSelect').value,
        expression: $('arithmeticExpressionInput').value,
        label: $('arithmeticLabelInput').value,
        slerp_from: $('slerpFromSelect').value,
        slerp_to: $('slerpToSelect').value,
        slerp_fraction: Number($('slerpFractionInput').value) || 0,
      },
    },
    projection: {
      method: $('projectionMethodSelect').value || 'auto',
      geometry_mode: $('geometryModeSelect').value,
      center_mode: $('centerModeSelect').value,
      seed: Number($('projectionSeedInput').value) || 42,
      edge_k: Number($('edgeKInput').value) || 0,
      manifold_neighbors: Number($('manifoldNeighborsInput').value) || 15,
      tsne_perplexity: Number($('tsnePerplexityInput').value) || 30,
      umap_min_dist: Number($('umapMinDistInput').value) || 0.1,
      align_anchor: $('alignAnchorInput').checked,
    },
    appearance: {
      auto_rotate: state.autoRotate,
      show_edges: state.showEdges,
      show_node_labels: state.showLabels,
      show_alias_letters: state.showLabelAliases,
      show_label_ids: state.showLabelIds,
      show_label_magnitudes: state.showLabelMagnitudes,
      show_label_participation: state.showLabelParticipation,
      show_edge_labels: state.showEdgeLabels,
      edge_color_mode: state.edgeColorMode,
      edge_width: state.edgeWidth,
      node_label_size: state.nodeLabelSize,
      edge_label_size: state.edgeLabelSize,
      edge_label_limit: state.edgeLabelLimit,
      color_mode: state.colorMode,
      point_size: state.pointSize,
      sphere_opacity: state.sphereOpacity,
      radial_mode: state.radialMode,
      magnitude_shell: state.magnitudeShell,
    },
    camera: {
      position: camera.position.toArray(),
      target: controls.target.toArray(),
      zoom: camera.zoom,
    },
  };
}

function downloadJson(payload, filename) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function saveSettingsFile() {
  const snapshot = buildSettingsSnapshot();
  const modelSlug = (snapshot.model.active?.model_name || snapshot.model.model_name || 'workspace')
    .replace(/[^a-z0-9._-]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'workspace';
  downloadJson(snapshot, `vocabulary-sphere-settings-${modelSlug}.json`);
  $('settingsStatus').textContent = `Saved ${snapshot.workspace.selected_tokens.length.toLocaleString()} tokens and ${snapshot.workspace.arithmetic_expressions.length.toLocaleString()} resultants.`;
  showToast('Settings saved to JSON');
}

function validateSettingsSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot)) {
    throw new Error('Settings file must contain a JSON object.');
  }
  if (snapshot.schema !== SETTINGS_SCHEMA) {
    throw new Error(`Unsupported settings schema. Expected ${SETTINGS_SCHEMA}.`);
  }
  const version = finiteInteger(snapshot.format_version, -1, -1, 100000);
  if (version < 1 || version > SETTINGS_FORMAT_VERSION) {
    throw new Error(`Unsupported settings format version ${snapshot.format_version}.`);
  }
  return snapshot;
}

function savedModelIdentity(snapshot) {
  const active = snapshot.model?.active;
  return {
    model_name: stringValue(active?.model_name ?? snapshot.model?.model_name, '', 500),
    revision: stringValue(active?.revision ?? snapshot.model?.revision, 'main', 200) || 'main',
    matrix_source: stringValue(active?.matrix_source ?? snapshot.model?.matrix_source, 'auto', 20),
  };
}

function settingsModelMatchesActive(snapshot) {
  if (!state.status?.loaded) return false;
  const saved = savedModelIdentity(snapshot);
  if (!saved.model_name || saved.model_name !== state.status.model_name || saved.revision !== state.status.revision) return false;
  return saved.matrix_source === 'auto' || saved.matrix_source === state.status.matrix_source;
}

function normalizedWorkspace(snapshot) {
  const workspace = snapshot.workspace && typeof snapshot.workspace === 'object' ? snapshot.workspace : {};
  const rawTokens = Array.isArray(workspace.selected_tokens) ? workspace.selected_tokens : [];
  const selectedTokens = [];
  const seen = new Set();
  for (const raw of rawTokens.slice(0, 5000)) {
    const tokenId = finiteInteger(raw?.token_id, -1, -1, Number.MAX_SAFE_INTEGER);
    if (tokenId < 0 || seen.has(tokenId)) continue;
    seen.add(tokenId);
    selectedTokens.push({
      token_id: tokenId,
      raw: stringValue(raw?.raw, String(tokenId), 10000),
      display: stringValue(raw?.display, stringValue(raw?.raw, String(tokenId), 10000), 10000),
      magnitude: Number.isFinite(Number(raw?.magnitude)) ? Number(raw.magnitude) : null,
      special: Boolean(raw?.special),
      present_in_tokenizer: raw?.present_in_tokenizer !== false,
    });
  }
  const expressions = Array.isArray(workspace.arithmetic_expressions)
    ? workspace.arithmetic_expressions.slice(0, 12).flatMap((item) => {
      const expression = stringValue(item?.expression, '', 2000).trim();
      if (!expression) return [];
      return [{ expression, label: stringValue(item?.label, '', 120) || null }];
    })
    : [];
  const editor = workspace.arithmetic_editor && typeof workspace.arithmetic_editor === 'object'
    ? workspace.arithmetic_editor
    : {};
  return {
    anchorId: finiteInteger(workspace.anchor_id, selectedTokens[0]?.token_id ?? 0, 0, Number.MAX_SAFE_INTEGER),
    selectedTokens,
    expressions,
    editor: {
      mode: editor.mode === 'slerp' ? 'slerp' : 'expression',
      expression: stringValue(editor.expression, '(A - B) + C', 2000),
      label: stringValue(editor.label, '', 120),
      slerpFrom: stringValue(editor.slerp_from, 'A', 20),
      slerpTo: stringValue(editor.slerp_to, 'B', 20),
      slerpFraction: finiteNumber(editor.slerp_fraction, 0.5, -1e9, 1e9),
    },
  };
}

function restoreWorkspace(workspace) {
  if (!state.status?.loaded) return false;
  state.selected.clear();
  const vocabSize = Number(state.status.vocab_size) || 0;
  for (const row of workspace.selectedTokens) {
    if (row.token_id >= vocabSize) continue;
    state.selected.set(row.token_id, row);
  }
  state.arithmeticExpressions = workspace.expressions;
  $('anchorIdInput').value = String(Math.min(Math.max(0, workspace.anchorId), Math.max(0, vocabSize - 1)));
  state.arithmeticMode = workspace.editor.mode;
  $('arithmeticModeSelect').value = state.arithmeticMode;
  $('arithmeticExpressionInput').value = workspace.editor.expression;
  $('arithmeticLabelInput').value = workspace.editor.label;
  $('slerpFractionInput').value = String(workspace.editor.slerpFraction);
  $('slerpFractionLabel').textContent = `t = ${workspace.editor.slerpFraction.toFixed(2)}`;
  resetProjectionUi();
  renderDemo();
  renderSelected();
  setSelectIfAvailable('slerpFromSelect', workspace.editor.slerpFrom, 'A');
  setSelectIfAvailable('slerpToSelect', workspace.editor.slerpTo, state.selected.size > 1 ? 'B' : 'A');
  return true;
}

function restorePendingSettingsWorkspace() {
  if (!pendingSettingsWorkspace || !settingsModelMatchesActive(pendingSettingsWorkspace.snapshot)) return false;
  const restored = restoreWorkspace(pendingSettingsWorkspace.workspace);
  if (restored) {
    $('settingsStatus').textContent = `Restored ${state.selected.size.toLocaleString()} tokens from ${pendingSettingsWorkspace.fileName}. Projection is ready to recompute.`;
    pendingSettingsWorkspace = null;
  }
  return restored;
}

function applyCameraSettings(cameraSettings) {
  if (!cameraSettings || typeof cameraSettings !== 'object') return;
  const position = Array.isArray(cameraSettings.position) ? cameraSettings.position.map(Number) : [];
  const target = Array.isArray(cameraSettings.target) ? cameraSettings.target.map(Number) : [];
  if (position.length === 3 && position.every(Number.isFinite) && new THREE.Vector3(...position).lengthSq() > 0.01) {
    camera.position.fromArray(position);
  }
  if (target.length === 3 && target.every(Number.isFinite)) controls.target.fromArray(target);
  camera.zoom = finiteNumber(cameraSettings.zoom, 1, 0.1, 10);
  camera.updateProjectionMatrix();
  controls.update();
}

function applySettingsControls(snapshot) {
  const model = snapshot.model && typeof snapshot.model === 'object' ? snapshot.model : {};
  const identity = savedModelIdentity(snapshot);
  $('modelNameInput').value = identity.model_name || stringValue(model.model_name, $('modelNameInput').value, 500);
  $('revisionInput').value = identity.revision || 'main';
  const requestedMatrixSource = ['input', 'output'].includes(identity.matrix_source)
    ? identity.matrix_source
    : model.matrix_source;
  setSelectIfAvailable('matrixSourceSelect', requestedMatrixSource, 'auto');
  $('deviceInput').value = stringValue(model.compute_device, 'auto', 50) || 'auto';
  $('allowDownloadInput').checked = booleanValue(model.allow_download, true);

  const search = snapshot.search && typeof snapshot.search === 'object' ? snapshot.search : {};
  setSelectIfAvailable('searchModeSelect', search.mode, 'regex');
  $('searchInput').value = stringValue(search.pattern, '', 256);
  $('caseSensitiveInput').checked = booleanValue(search.case_sensitive, false);
  $('searchLimitInput').value = String(finiteInteger(search.limit, 200, 1, 5000));
  $('tokenizeTextInput').value = stringValue(search.tokenization_text, '', 100000);
  $('tokenizeSpecialInput').checked = booleanValue(search.tokenization_add_special_tokens, false);
  $('tokenizeLimitInput').value = String(finiteInteger(search.tokenization_limit, 5000, 1, 10000));
  $('vicinityCountInput').value = String(finiteInteger(search.vicinity_count, 160, 2, 5000));
  updateSearchPlaceholder();

  const projection = snapshot.projection && typeof snapshot.projection === 'object' ? snapshot.projection : {};
  state.geometryMode = projection.geometry_mode === 'euclidean' ? 'euclidean' : 'sphere';
  setSelectIfAvailable('geometryModeSelect', state.geometryMode, 'sphere');
  setSelectIfAvailable('centerModeSelect', projection.center_mode, 'mean');
  $('projectionSeedInput').value = String(finiteInteger(projection.seed, 42, -2147483648, 2147483647));
  $('edgeKInput').value = String(finiteInteger(projection.edge_k, 2, 0, 20));
  $('manifoldNeighborsInput').value = String(finiteInteger(projection.manifold_neighbors, 15, 2, 250));
  $('tsnePerplexityInput').value = String(finiteNumber(projection.tsne_perplexity, 30, 2, 250));
  $('umapMinDistInput').value = String(finiteNumber(projection.umap_min_dist, 0.1, 0, 0.99));
  $('alignAnchorInput').checked = booleanValue(projection.align_anchor, true);
  const method = stringValue(projection.method, 'auto', 100) || 'auto';
  const methodApplied = setSelectIfAvailable('projectionMethodSelect', method, 'auto');
  state.pendingProjectionMethod = !methodApplied && state.methods.size === 0 ? method : null;
  updateMethodCard();

  const appearance = snapshot.appearance && typeof snapshot.appearance === 'object' ? snapshot.appearance : {};
  state.autoRotate = booleanValue(appearance.auto_rotate, false);
  state.showEdges = booleanValue(appearance.show_edges, true);
  state.showLabels = booleanValue(appearance.show_node_labels, false);
  state.showLabelAliases = booleanValue(appearance.show_alias_letters, true);
  state.showLabelIds = booleanValue(appearance.show_label_ids, true);
  state.showLabelMagnitudes = booleanValue(appearance.show_label_magnitudes, false);
  state.showLabelParticipation = booleanValue(appearance.show_label_participation, false);
  state.showEdgeLabels = booleanValue(appearance.show_edge_labels, false) && state.showEdges;
  state.edgeColorMode = appearance.edge_color_mode === 'uniform' ? 'uniform' : 'angle';
  state.edgeWidth = finiteNumber(appearance.edge_width, 1.8, 0.5, 8);
  state.nodeLabelSize = finiteNumber(appearance.node_label_size, 13, 8, 32);
  state.edgeLabelSize = finiteNumber(appearance.edge_label_size, 11, 7, 28);
  state.edgeLabelLimit = finiteInteger(appearance.edge_label_limit, 0, 0, 20000);
  state.colorMode = ['anchor', 'id', 'norm'].includes(appearance.color_mode) ? appearance.color_mode : 'anchor';
  state.pointSize = finiteNumber(appearance.point_size, 0.075, 0.025, 0.18);
  state.sphereOpacity = finiteNumber(appearance.sphere_opacity, 0.04, 0, 0.18);
  state.radialMode = appearance.radial_mode === 'magnitude' ? 'magnitude' : 'surface';
  state.magnitudeShell = finiteNumber(appearance.magnitude_shell, 0.18, 0.02, 0.45);
  pointMaterial.size = state.pointSize;
  syncAppearanceControls();
  recolorPoints();
  applyCameraSettings(snapshot.camera);
}

async function loadSettingsFile(file) {
  if (!file) return;
  if (file.size > MAX_SETTINGS_FILE_BYTES) {
    throw new Error(`Settings file is too large (${formatBytes(file.size)}). Maximum size is ${formatBytes(MAX_SETTINGS_FILE_BYTES)}.`);
  }
  const snapshot = validateSettingsSnapshot(JSON.parse(await file.text()));
  applySettingsControls(snapshot);
  resetProjectionUi();
  renderDemo();
  const workspace = normalizedWorkspace(snapshot);
  pendingSettingsWorkspace = { snapshot, workspace, fileName: file.name };
  const restored = restorePendingSettingsWorkspace();
  if (restored) {
    applyCameraSettings(snapshot.camera);
    showToast(`Loaded settings from ${file.name}`);
  } else {
    const identity = savedModelIdentity(snapshot);
    $('settingsStatus').textContent = `Loaded ${file.name}. Load ${identity.model_name || 'the referenced model'} to restore ${workspace.selectedTokens.length.toLocaleString()} saved tokens.`;
    showToast('Settings loaded; matching model is required for the saved token set');
  }
}

function exportProjection() {
  if (!state.projection) return;
  const blob = new Blob([JSON.stringify(state.projection, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `vocabulary-geometry-${state.projection.actual_method}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function screenshot() {
  renderer.render(scene, camera);
  drawLabelOverlay();
  const composite = document.createElement('canvas');
  composite.width = renderer.domElement.width;
  composite.height = renderer.domElement.height;
  const context = composite.getContext('2d');
  context.drawImage(renderer.domElement, 0, 0, composite.width, composite.height);
  context.drawImage(labelCanvas, 0, 0, composite.width, composite.height);
  const link = document.createElement('a');
  link.href = composite.toDataURL('image/png');
  link.download = 'vocabulary-geometry.png';
  link.click();
}

function syncAppearanceControls() {
  $('autoRotateInput').checked = state.autoRotate;
  $('showEdgesInput').checked = state.showEdges;
  $('showLabelsInput').checked = state.showLabels;
  $('showLabelAliasesInput').checked = state.showLabelAliases;
  $('showLabelIdsInput').checked = state.showLabelIds;
  $('showLabelMagnitudesInput').checked = state.showLabelMagnitudes;
  $('showLabelParticipationInput').checked = state.showLabelParticipation;
  $('showEdgeLabelsInput').checked = state.showEdgeLabels;
  $('pointSizeInput').value = String(state.pointSize);
  $('pointSizeLabel').textContent = state.pointSize.toFixed(3);
  $('colorModeSelect').value = state.colorMode;
  $('edgeColorModeSelect').value = state.edgeColorMode;
  $('edgeWidthInput').value = String(state.edgeWidth);
  $('edgeWidthLabel').textContent = `${state.edgeWidth.toFixed(1)} px`;
  $('nodeLabelSizeInput').value = String(state.nodeLabelSize);
  $('nodeLabelSizeLabel').textContent = `${state.nodeLabelSize.toFixed(0)} px`;
  $('edgeLabelSizeInput').value = String(state.edgeLabelSize);
  $('edgeLabelSizeLabel').textContent = `${state.edgeLabelSize.toFixed(0)} px`;
  $('sphereOpacityInput').value = String(state.sphereOpacity);
  $('sphereOpacityLabel').textContent = `${Math.round(state.sphereOpacity * 100)}%`;
  $('radialModeSelect').value = state.radialMode;
  $('magnitudeShellInput').value = String(state.magnitudeShell);
  $('magnitudeShellLabel').textContent = `${Math.round(state.magnitudeShell * 100)}%`;
  $('edgeLabelLimitInput').value = String(state.edgeLabelLimit);
  $('geometryModeSelect').value = state.geometryMode;
  $('toggleRotateBtn').classList.toggle('active', state.autoRotate);
  $('toggleEdgesBtn').classList.toggle('active', state.showEdges);
  $('toggleLabelsBtn').classList.toggle('active', state.showLabels);
  $('toggleEdgeLabelsBtn').classList.toggle('active', state.showEdgeLabels);
  edgeLines.visible = state.showEdges && Boolean(state.projection);
  applyScaffoldAppearance();
  if (state.showEdges) updateEdgesFromPositions();
  rebuildLabels();
  rebuildEdgeLabels();
  updateLegend();
}

function setAutoRotate(value) {
  state.autoRotate = Boolean(value);
  syncAppearanceControls();
}
function setShowEdges(value) {
  state.showEdges = Boolean(value);
  if (!state.showEdges) state.showEdgeLabels = false;
  syncAppearanceControls();
}
function setShowLabels(value) {
  state.showLabels = Boolean(value);
  syncAppearanceControls();
}
function setShowLabelAliases(value) {
  state.showLabelAliases = Boolean(value);
  syncAppearanceControls();
}
function setShowLabelIds(value) {
  state.showLabelIds = Boolean(value);
  syncAppearanceControls();
}
function setShowLabelMagnitudes(value) {
  state.showLabelMagnitudes = Boolean(value);
  syncAppearanceControls();
}
function setShowLabelParticipation(value) {
  state.showLabelParticipation = Boolean(value);
  syncAppearanceControls();
}
function setShowEdgeLabels(value) {
  state.showEdgeLabels = Boolean(value);
  if (state.showEdgeLabels) state.showEdges = true;
  syncAppearanceControls();
}


function isTypingTarget(element) {
  const tag = element?.tagName;
  return element?.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(tag);
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

$('saveSettingsBtn').addEventListener('click', saveSettingsFile);
$('loadSettingsBtn').addEventListener('click', () => {
  $('settingsFileInput').value = '';
  $('settingsFileInput').click();
});
$('settingsFileInput').addEventListener('change', async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await loadSettingsFile(file);
  } catch (error) {
    $('settingsStatus').textContent = `Could not load ${file.name}.`;
    showToast(error.message, 'error');
  } finally {
    event.target.value = '';
  }
});
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
function updateSearchPlaceholder() {
  const mode = $('searchModeSelect').value;
  $('searchInput').placeholder = mode === 'id' ? '42,90-110' : mode === 'literal' ? 'token text' : '^▁(cat|dog)';
}
$('searchModeSelect').addEventListener('change', updateSearchPlaceholder);
$('tokenizeTextBtn').addEventListener('click', tokenizeSuppliedText);
$('tokenizeTextInput').addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') tokenizeSuppliedText();
});
$('tokenizedResultsBody').addEventListener('click', (event) => {
  const button = event.target.closest('[data-add-tokenized]');
  if (!button) return;
  const sequenceIndex = Number(button.dataset.addTokenized);
  const row = state.tokenizedResults.find((item) => item.sequence_index === sequenceIndex);
  if (row?.projectable) addTokens([row]);
});
$('addAllTokenizedBtn').addEventListener('click', () => addAllTokenized(false));
$('replaceTokenizedBtn').addEventListener('click', () => addAllTokenized(true));
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
$('arithmeticModeSelect').addEventListener('change', syncArithmeticMode);
$('slerpFractionInput').addEventListener('input', (event) => {
  $('slerpFractionLabel').textContent = `t = ${Number(event.target.value).toFixed(2)}`;
});
$('addArithmeticBtn').addEventListener('click', addArithmeticResult);
$('arithmeticExpressionInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') addArithmeticResult();
});
$('clearArithmeticBtn').addEventListener('click', clearArithmeticResults);
$('arithmeticResults').addEventListener('click', (event) => {
  const button = event.target.closest('[data-remove-resultant]');
  if (!button) return;
  state.arithmeticExpressions.splice(Number(button.dataset.removeResultant), 1);
  renderArithmeticPanel();
  markProjectionStale();
  if (state.projection) projectSelection();
});
$('geometryModeSelect').addEventListener('change', (event) => {
  state.geometryMode = event.target.value;
  markProjectionStale();
  applyScaffoldAppearance();
});
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
  if (!point || point.kind === 'resultant' || point.token_id === null) return;
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
$('edgeColorModeSelect').addEventListener('change', (event) => {
  state.edgeColorMode = event.target.value;
  updateEdgesFromPositions();
  rebuildEdgeLabels();
  updateLegend();
});
$('edgeWidthInput').addEventListener('input', (event) => {
  state.edgeWidth = Number(event.target.value);
  edgeMaterial.linewidth = state.edgeWidth;
  $('edgeWidthLabel').textContent = `${state.edgeWidth.toFixed(1)} px`;
  updateEdgesFromPositions();
});
$('nodeLabelSizeInput').addEventListener('input', (event) => {
  state.nodeLabelSize = Number(event.target.value);
  $('nodeLabelSizeLabel').textContent = `${state.nodeLabelSize.toFixed(0)} px`;
});
$('edgeLabelSizeInput').addEventListener('input', (event) => {
  state.edgeLabelSize = Number(event.target.value);
  $('edgeLabelSizeLabel').textContent = `${state.edgeLabelSize.toFixed(0)} px`;
});
$('sphereOpacityInput').addEventListener('input', (event) => {
  state.sphereOpacity = Number(event.target.value);
  $('sphereOpacityLabel').textContent = `${Math.round(state.sphereOpacity * 100)}%`;
  applyScaffoldAppearance();
});
$('radialModeSelect').addEventListener('change', (event) => {
  state.radialMode = event.target.value;
  applyScaffoldAppearance();
  animateToCurrentDisplay();
});
$('magnitudeShellInput').addEventListener('input', (event) => {
  state.magnitudeShell = Number(event.target.value);
  $('magnitudeShellLabel').textContent = `${Math.round(state.magnitudeShell * 100)}%`;
  animateToCurrentDisplay(180);
});
$('edgeLabelLimitInput').addEventListener('change', (event) => {
  const parsed = Number(event.target.value);
  state.edgeLabelLimit = Math.max(0, Math.min(20000, Number.isFinite(parsed) ? Math.round(parsed) : 0));
  event.target.value = String(state.edgeLabelLimit);
  rebuildEdgeLabels();
});
$('autoRotateInput').addEventListener('change', (event) => setAutoRotate(event.target.checked));
$('showEdgesInput').addEventListener('change', (event) => setShowEdges(event.target.checked));
$('showEdgeLabelsInput').addEventListener('change', (event) => setShowEdgeLabels(event.target.checked));
$('showLabelsInput').addEventListener('change', (event) => setShowLabels(event.target.checked));
$('showLabelAliasesInput').addEventListener('change', (event) => setShowLabelAliases(event.target.checked));
$('showLabelIdsInput').addEventListener('change', (event) => setShowLabelIds(event.target.checked));
$('showLabelMagnitudesInput').addEventListener('change', (event) => setShowLabelMagnitudes(event.target.checked));
$('showLabelParticipationInput').addEventListener('change', (event) => setShowLabelParticipation(event.target.checked));
$('resetCameraBtn').addEventListener('click', resetCamera);
$('toggleRotateBtn').addEventListener('click', () => setAutoRotate(!state.autoRotate));
$('toggleEdgesBtn').addEventListener('click', () => setShowEdges(!state.showEdges));
$('toggleLabelsBtn').addEventListener('click', () => setShowLabels(!state.showLabels));
$('toggleEdgeLabelsBtn').addEventListener('click', () => setShowEdgeLabels(!state.showEdgeLabels));

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
  else if (event.key.toLowerCase() === 'g') setShowEdgeLabels(!state.showEdgeLabels);
  else if (event.key.toLowerCase() === 'l') setShowLabels(!state.showLabels);
  else if (event.key.toLowerCase() === 't') setShowLabelAliases(!state.showLabelAliases);
  else if (event.key.toLowerCase() === 'i') setShowLabelIds(!state.showLabelIds);
  else if (event.key.toLowerCase() === 'm') setShowLabelMagnitudes(!state.showLabelMagnitudes);
  else if (event.key.toLowerCase() === 'p') setShowLabelParticipation(!state.showLabelParticipation);
  else if (event.key === 'Escape') {
    state.pinnedIndex = null;
    renderInspector();
    rebuildLabels();
    updateMarkersFromPositions();
  }
});

async function initialize() {
  updateSearchPlaceholder();
  syncAppearanceControls();
  await Promise.all([loadProjectionMethods(), refreshStatus()]);
  renderSearchResults();
  renderTokenizedResults();
  renderSelected();
  renderMetrics(null);
}
initialize();
