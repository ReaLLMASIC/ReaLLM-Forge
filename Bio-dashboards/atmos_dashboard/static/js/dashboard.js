// Atmos dashboard: works with any Atmos device (Mini, Sphere S4 / S5, or anything that
// sends CSV / key=value / JSON). Rows arrive as {ts, v: {series_id: value}} where a series
// id is a metric, optionally tagged with its sensor: "co2", "temp@scd30", "rh@sen55".

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// ---------- metrics ----------
// Same priority as atmos_parser.PRIMARY_SOURCE_ORDER: SEN5x/SEN6x temperature & RH are
// compensated, SCD30 runs warm from its IR source, SFA3X last.
const SOURCE_ORDER = ['sen69c', 'sen68', 'sen66', 'sen55', 'sen54', 'sen5x', 'sen44', 'sht4x',
                      'scd30', 'scd41', 'scd40', 'scd4x', 'sfa3x', 'sfa30', 'sgp41', ''];
const CANONICAL = ['co2', 'pm1', 'pm25', 'pm4', 'pm10', 'temp', 'rh', 'voc', 'nox', 'hcho'];
const metricOf = (id) => id.split('@')[0];
const sourceOf = (id) => id.split('@')[1] || '';
const rank = (id) => { const i = SOURCE_ORDER.indexOf(sourceOf(id)); return i < 0 ? SOURCE_ORDER.length : i; };

// Best available value of `metric` in one row
function pick(v, metric) {
    let best = null, bestRank = Infinity;
    for (const id in v) {
        if (metricOf(id) !== metric) continue;
        const r = rank(id);
        if (r < bestRank) { best = v[id]; bestRank = r; }
    }
    return best;
}
const seen = new Set();          // every series id seen (history + live)

// ---------- status badge ----------
const BADGE = 'px-3 py-1 rounded-full text-xs font-semibold border ';
const TONES = {
    ok:   'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    wait: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    info: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
    bad:  'bg-red-500/20 text-red-400 border-red-500/30',
};
const setStatus = (text, tone) => { $('status').textContent = text; $('status').className = BADGE + TONES[tone]; };

// ---------- charts ----------
let windowMin = 15;
try { windowMin = parseInt(localStorage.getItem('atmos-window')) || 15; } catch (e) {}

const baseOptions = () => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
        x: { type: 'realtime', realtime: { duration: windowMin * 60000, refresh: 1000, delay: 2000 },
             ticks: { color: '#94a3b8', maxTicksLimit: 6 }, grid: { color: '#1e293b' } },
        y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
    },
    plugins: { legend: { display: false }, tooltip: { enabled: true } },
});
const line = (label, color, extra = {}) => ({ label, borderColor: color, backgroundColor: color,
    borderWidth: 2, pointRadius: 0, cubicInterpolationMode: 'monotone', spanGaps: false, data: [], ...extra });
const withLegend = (o) => { o.plugins.legend = { display: true, labels: { color: '#94a3b8', boxWidth: 12 } }; return o; };
const withRightAxis = (o, title) => {
    o.scales.y2 = { position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#94a3b8' },
                    title: { display: !!title, text: title, color: '#64748b' } };
    return o;
};

const charts = {
    co2: new Chart($('co2Chart'), { type: 'line',
        data: { datasets: [line('CO₂', '#fbbf24')] }, options: baseOptions() }),
    pm: new Chart($('pmChart'), { type: 'line',
        data: { datasets: [line('PM1.0', '#a3e635'), line('PM2.5', '#22d3ee', { borderWidth: 2.5 }), line('PM10', '#fb923c')] },
        options: withLegend(baseOptions()) }),
    climate: new Chart($('climateChart'), { type: 'line',
        data: { datasets: [line('Temp °C', '#fb7185'), line('RH %', '#38bdf8', { yAxisID: 'y2' })] },
        options: withRightAxis(withLegend(baseOptions()), '% RH') }),
    gas: new Chart($('gasChart'), { type: 'line',
        data: { datasets: [line('VOC idx', '#a78bfa'), line('NOx idx', '#818cf8'), line('HCHO ppb', '#f87171', { yAxisID: 'y2' })] },
        options: withRightAxis(withLegend(baseOptions()), 'ppb') }),
};
// which metric feeds which dataset
const SERIES = [
    [charts.co2, 0, 'co2'],
    [charts.pm, 0, 'pm1'], [charts.pm, 1, 'pm25'], [charts.pm, 2, 'pm10'],
    [charts.climate, 0, 'temp'], [charts.climate, 1, 'rh'],
    [charts.gas, 0, 'voc'], [charts.gas, 1, 'nox'], [charts.gas, 2, 'hcho'],
];

let lastTs = 0;
function pushRows(rows) {
    let grew = false;
    for (const r of rows) {
        for (const id in r.v) if (!seen.has(id)) { seen.add(id); grew = true; }
        if (r.ts <= lastTs) continue;                  // history + live can overlap
        lastTs = r.ts;
        for (const [chart, ds, metric] of SERIES) {
            const y = pick(r.v, metric);
            if (y != null) chart.data.datasets[ds].data.push({ x: r.ts, y });
        }
        recordTrends(r);
    }
    if (grew) applyLayout();
}

async function loadHistory() {
    for (const c of Object.values(charts)) c.data.datasets.forEach(d => { d.data.length = 0; });
    lastTs = 0;
    try {
        const { rows } = await (await fetch(`/api/history?minutes=${windowMin}`)).json();
        pushRows(rows);
        if (rows.length) renderTiles(rows[rows.length - 1]);
    } catch (e) {}
    for (const c of Object.values(charts)) c.update('none');
}

// window buttons (swap classes rather than stacking them -- see setPillActive in debug_panel.js)
function paintWindowButtons() {
    document.querySelectorAll('#window-btns button').forEach(b => setPillActive(b, parseInt(b.dataset.min) === windowMin));
}
$('window-btns').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-min]');
    if (!b) return;
    windowMin = parseInt(b.dataset.min);
    try { localStorage.setItem('atmos-window', windowMin); } catch (err) {}
    for (const c of Object.values(charts)) c.options.scales.x.realtime.duration = windowMin * 60000;
    paintWindowButtons();
    loadHistory();
});
paintWindowButtons();

// ---------- layout: show only what this device reports ----------
const has = (metric) => [...seen].some(id => metricOf(id) === metric);
const cardOf = (el) => el && el.closest('.rounded-2xl');
const TILE_METRICS = {                      // tile (by an element inside it) -> metrics it shows
    '#co2-val': ['co2'], '#pm25-val': ['pm25'], '#temp-val': ['temp', 'rh'],
    '#voc-val': ['voc'], '#nox-val': ['nox'], '#hcho-val': ['hcho'], '#pm1-val': ['pm1', 'pm4', 'pm10'],
};
function applyLayout() {
    if (!seen.size) return;
    for (const [sel, ms] of Object.entries(TILE_METRICS)) cardOf($(sel.slice(1)))?.classList.toggle('hidden', !ms.some(has));
    // Climate: hide the half that's missing; Particulates: hide missing sizes
    $('temp-val').parentElement.classList.toggle('hidden', !has('temp'));
    $('rh-val').parentElement.classList.toggle('hidden', !has('rh'));
    for (const m of ['pm1', 'pm4', 'pm10']) $(`${m}-val`).parentElement.classList.toggle('hidden', !has(m));
    // Chart lines for metrics this device doesn't have; whole chart card if none
    for (const [chart, ds, metric] of SERIES) chart.data.datasets[ds].hidden = !has(metric);
    for (const [key, chart] of Object.entries(charts)) {
        cardOf(chart.canvas)?.classList.toggle('hidden', !SERIES.some(([c, , m]) => c === chart && has(m)));
    }
    buildOtherTiles();
}

// Fields the dashboard doesn't know get a generic tile (any device works)
const otherTiles = {};
const OTHER_COLORS = ['#fda4af', '#fcd34d', '#86efac', '#93c5fd', '#c4b5fd', '#f9a8d4', '#5eead4', '#fdba74'];
function buildOtherTiles() {
    const unknown = [...seen].filter(id => !CANONICAL.includes(metricOf(id))).sort();
    const row = $('other-row');
    for (const id of unknown) {
        if (otherTiles[id]) continue;
        const color = OTHER_COLORS[Object.keys(otherTiles).length % OTHER_COLORS.length];
        const card = document.createElement('div');
        card.className = 'bg-slate-800 border border-slate-700 rounded-2xl p-5 shadow-xl text-center';
        card.innerHTML = `<h2 class="text-xs font-medium text-slate-400 tracking-wider mb-1 font-mono break-all">${esc(id)}</h2>
            <div class="text-4xl font-extrabold my-1" style="color:${color}" data-val>--</div>
            <div class="text-xs text-slate-500 h-4">unrecognised field</div>`;
        row.appendChild(card);
        otherTiles[id] = card.querySelector('[data-val]');
        trends.addMetric(id, { title: id, unit: '', tile: otherTiles[id], color, span: 0 });
    }
    $('other-section').classList.toggle('hidden', unknown.length === 0);
}

// ---------- tiles ----------
const fmt = (v, d = 0) => (v == null || Number.isNaN(v)) ? '--' : v.toFixed(d);

// Indoor CO₂ ventilation guidance
function co2Band(ppm) {
    if (ppm == null) return ['', 'text-slate-500'];
    if (ppm < 800)  return ['WELL VENTILATED', 'text-emerald-400'];
    if (ppm < 1000) return ['MODERATE', 'text-amber-300'];
    if (ppm < 1500) return ['POOR — VENTILATE', 'text-orange-400'];
    return ['VERY POOR — VENTILATE NOW', 'text-red-400'];
}
// US EPA PM2.5 AQI categories (2024 breakpoints); the official AQI uses a 24 h average
function pmBand(v) {
    if (v == null) return ['', 'text-slate-500'];
    if (v <= 9.0)   return ['GOOD', 'text-emerald-400'];
    if (v <= 35.4)  return ['MODERATE', 'text-amber-300'];
    if (v <= 55.4)  return ['UNHEALTHY FOR SENSITIVE GROUPS', 'text-orange-400'];
    if (v <= 125.4) return ['UNHEALTHY', 'text-red-400'];
    if (v <= 225.4) return ['VERY UNHEALTHY', 'text-fuchsia-400'];
    return ['HAZARDOUS', 'text-rose-300'];
}
const band = (el, [text, cls]) => { el.textContent = text; el.className = 'text-xs font-bold mt-1 h-4 ' + cls; };

function renderTiles(r) {
    const v = r.v, p = (m) => pick(v, m);
    $('co2-val').textContent = fmt(p('co2'));
    $('pm25-val').textContent = fmt(p('pm25'), 1);
    $('temp-val').textContent = fmt(p('temp'), 1);
    $('rh-val').textContent = fmt(p('rh'));
    $('voc-val').textContent = fmt(p('voc'));
    $('nox-val').textContent = fmt(p('nox'));
    $('hcho-val').textContent = fmt(p('hcho'), 1);
    $('pm1-val').textContent = fmt(p('pm1'), 1);
    $('pm4-val').textContent = fmt(p('pm4'), 1);
    $('pm10-val').textContent = fmt(p('pm10'), 1);
    band($('co2-band'), co2Band(p('co2')));
    band($('pm25-band'), pmBand(p('pm25')));
    // Sensirion indices: VOC 100 / NOx 1 = the sensor's learned baseline for this room
    const voc = p('voc'), nox = p('nox');
    $('voc-note').textContent = voc == null ? 'baseline 100' : voc > 150 ? 'above baseline' : voc < 80 ? 'below baseline' : 'near baseline';
    $('nox-note').textContent = nox == null ? 'baseline 1' : nox > 20 ? 'above baseline' : 'near baseline';
    for (const [id, el] of Object.entries(otherTiles)) if (v[id] != null) el.textContent = +v[id].toFixed(2);
}

// ---------- scanner panel follows the worker status ----------
const scannerPanel = $('scanner-panel');
const deviceList = $('device-list');
const note = (html, cls) => { deviceList.innerHTML = `<div class="${cls} col-span-3">${html}</div>`; };
const STALE_MS = 15000;          // sensor streams ~1 line/s
let uiIsConnected = false, lastPacketAt = 0, lastListKey = '';

function markLive() {
    lastPacketAt = Date.now();
    if (!uiIsConnected) {
        uiIsConnected = true;
        scannerPanel.style.display = 'none';
        setStatus('Telemetry Active 🟢', 'ok');
    }
}
function markIdle(text, tone) {
    uiIsConnected = false;
    scannerPanel.style.display = 'block';
    setStatus(text, tone);
    updateScannerList();
}
setInterval(() => {
    if (uiIsConnected && Date.now() - lastPacketAt > STALE_MS) markIdle('Signal Lost — Rescanning', 'wait');
}, 1000);

async function updateScannerList() {
    if (uiIsConnected) return;
    try {
        const st = await (await fetch('/api/status')).json();
        if (st.state === 'connecting') {
            lastListKey = '';
            const tries = st.attempt > 1 ? ` (attempt ${st.attempt}/${st.max_attempts})` : '';
            return note(`Handshaking with ${esc(st.address)}${tries}...`, 'text-emerald-400 font-bold py-4 animate-pulse');
        }
        if (st.state === 'streaming') {
            lastListKey = '';
            return note('Connected — waiting for the first reading...', 'text-emerald-400 font-bold py-4 animate-pulse');
        }
        const devices = await (await fetch('/api/scan-results')).json();
        const key = JSON.stringify(devices.map(d => d.address));
        if (key === lastListKey) {
            devices.forEach(d => {
                const el = deviceList.querySelector(`[data-rssi="${CSS.escape(d.address)}"]`);
                if (el) el.textContent = `${d.rssi} dBm`;
            });
            return;
        }
        lastListKey = key;
        if (devices.length === 0) return note('No Atmos devices detected... is it powered and advertising Bluetooth UART?', 'text-slate-500 text-sm animate-pulse');
        deviceList.innerHTML = devices.map(d => `
            <div class="bg-slate-700/50 p-4 rounded-xl border border-slate-600 flex flex-col gap-3 justify-between">
                <div>
                    <div class="font-bold text-slate-200">${esc(d.name)}</div>
                    <div class="text-xs text-slate-400 font-mono">${esc(d.address)} · <span data-rssi="${esc(d.address)}">${esc(d.rssi)} dBm</span></div>
                </div>
                <button data-mac="${esc(d.address)}" class="bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold py-2 px-4 rounded w-full transition-colors">
                    Connect Signal
                </button>
            </div>`).join('');
    } catch (err) {}
}

deviceList.addEventListener('click', async (e) => {
    const mac = e.target.closest('button[data-mac]')?.dataset.mac;
    if (!mac) return;
    lastListKey = '';
    note(`Handshaking with ${esc(mac)}...`, 'text-emerald-400 font-bold py-4 animate-pulse');
    await fetch('/api/connect', { method: 'POST', headers: {'Content-Type': 'application/json'},
                                  body: JSON.stringify({address: mac}) });
});
// Wi-Fi (TCP) connect: pre-fill with the last target
(async () => {
    try {
        const t = await (await fetch('/api/tcp-target')).json();
        if (t.host) { $('tcp-host').value = t.host; $('tcp-port').value = t.port || 8080; }
    } catch (e) {}
})();
$('tcp-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const host = $('tcp-host').value.trim(), port = parseInt($('tcp-port').value) || 8080;
    if (!host) { $('tcp-host').focus(); return; }
    lastListKey = '';
    note(`Connecting to ${esc(host)}:${port} over Wi-Fi...`, 'text-emerald-400 font-bold py-4 animate-pulse');
    try {
        await fetch('/api/connect-tcp', { method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify({ host, port }) });
    } catch (err) {}
});

setInterval(updateScannerList, 1000);
updateScannerList();

// ---------- live stream ----------
function connectWs() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.onopen = () => { if (!uiIsConnected) setStatus('System Online', 'info'); };
    ws.onclose = () => { markIdle('Server Offline 🔴 — retrying', 'bad'); setTimeout(connectWs, 2000); };
    ws.onmessage = (event) => {
        let msg; try { msg = JSON.parse(event.data); } catch (e) { return; }
        const rows = msg.rows || [];
        if (!rows.length) return;
        DebugPanel.onPacket(rows.length);
        markLive();
        pushRows(rows);
        renderTiles(rows[rows.length - 1]);
    };
}

// ---------- Tap-a-tile trends (see tile_trends.js) ----------
// Canonical tiles show the best source per metric; unknown fields are added as they appear.
const trends = createTileTrends({
    storageKey: 'atmos-trend',
    insertAfter: '#other-section',
    ring: ['ring-emerald-400/70'],
    metrics: {
        co2:  { title: 'Carbon Dioxide', unit: 'ppm', tile: '#co2-val', color: '#fbbf24', span: 150 },
        pm25: { title: 'Fine Particles (PM2.5)', unit: 'µg/m³', tile: '#pm25-val', color: '#22d3ee', span: 10, dp: 1 },
        climate: { title: 'Climate', tile: '#temp-val', series: [
            { id: 'temp', label: 'Temp', unit: '°C', color: '#fb7185', span: 2, dp: 1 },
            { id: 'rh',   label: 'RH',   unit: '%',  color: '#38bdf8', span: 10, axis: 'y2' } ] },
        voc:  { title: 'VOC Index', unit: '', tile: '#voc-val', color: '#a78bfa', span: 50 },
        nox:  { title: 'NOx Index', unit: '', tile: '#nox-val', color: '#818cf8', span: 5 },
        hcho: { title: 'Formaldehyde (HCHO)', unit: 'ppb', tile: '#hcho-val', color: '#f87171', span: 10, dp: 1 },
        particulates: { title: 'Particulates', tile: '#pm1-val', series: [
            { id: 'pm1',  label: 'PM1.0', unit: 'µg/m³', color: '#a3e635', span: 10, dp: 1 },
            { id: 'pm4',  label: 'PM4.0', unit: 'µg/m³', color: '#e879f9', span: 10, dp: 1 },
            { id: 'pm10', label: 'PM10',  unit: 'µg/m³', color: '#fb923c', span: 10, dp: 1 } ] },
    },
    prefill: async (metric, minutes) => {
        const { rows } = await (await fetch(`/api/history?minutes=${minutes}`)).json();
        const out = {};
        for (const m of CANONICAL) out[m] = [];
        for (const r of rows) {
            for (const m of CANONICAL) { const y = pick(r.v, m); if (y != null) out[m].push({ x: r.ts, y }); }
            for (const id in r.v) if (!CANONICAL.includes(metricOf(id))) (out[id] ??= []).push({ x: r.ts, y: r.v[id] });
        }
        return out;
    },
});
function recordTrends(r) {
    for (const m of CANONICAL) { const y = pick(r.v, m); if (y != null) trends.record(m, r.ts, y); }
    for (const id in r.v) if (!CANONICAL.includes(metricOf(id))) trends.record(id, r.ts, r.v[id]);
}

// Header subtitle: what's connected and how its data is being read
async function updateSubtitle() {
    try {
        const st = await (await fetch('/api/status')).json();
        const p = (st.debug || {}).parser || {}, profiles = (st.debug || {}).profiles || {};
        if (!p.format) return;
        const dev = p.effective_profile ? profiles[p.effective_profile]?.label
                  : p.column_source === 'device header' ? 'Device-described fields'
                  : p.column_source === 'names in each line' ? 'Named fields' : 'Custom layout';
        const n = Object.keys((st.debug || {}).fields || {}).length;
        $('subtitle').textContent = `${dev} · ${n} fields · ${p.format}` + (p.profile === 'auto' ? ' · auto-detected' : '');
    } catch (e) {}
}
setInterval(updateSubtitle, 5000);

loadHistory().then(connectWs);
