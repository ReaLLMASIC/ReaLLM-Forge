// --- Charts (same look as the Polar dashboard) ---
const chartOptions = (yMin, yMax) => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    scales: {
        // O2 readings arrive every ~2 s, so a 60 s window shows the trend
        x: { type: 'realtime', realtime: { duration: 60000, refresh: 250, delay: 2500 },
             ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
        y: { suggestedMin: yMin, suggestedMax: yMax, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
    },
    plugins: { legend: { display: false } }
});

const line = (color) => ({ borderColor: color, backgroundColor: color, borderWidth: 2.5,
                           pointRadius: 2, cubicInterpolationMode: 'monotone', data: [] });

const spo2Chart = new Chart(document.getElementById('spo2Chart').getContext('2d'), {
    type: 'line', data: { datasets: [line('#38bdf8')] }, options: chartOptions(88, 100)
});
const hrChart = new Chart(document.getElementById('hrChart').getContext('2d'), {
    type: 'line', data: { datasets: [line('#f43f5e')] }, options: chartOptions(50, 110)
});

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// --- Status badge ---
const BADGE = 'px-3 py-1 rounded-full text-xs font-semibold border ';
const TONES = {
    ok:   'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    wait: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    info: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
    bad:  'bg-red-500/20 text-red-400 border-red-500/30',
};
const setStatus = (text, tone) => { $('status').textContent = text; $('status').className = BADGE + TONES[tone]; };

// --- Scanner panel follows the worker's status ---
const scannerPanel = $('scanner-panel');
const deviceList = $('device-list');
const note = (html, cls) => { deviceList.innerHTML = `<div class="${cls} col-span-3">${html}</div>`; };

let workerStatus = 'Scanning';
let lastPacketAt = 0;
let lastListKey = '';
let pendingMac = null;

async function updateDeviceList() {
    if (workerStatus === 'Connected' || workerStatus === 'Calibrating') return;
    if (workerStatus === 'Connecting' || pendingMac) {
        lastListKey = '';
        return note(`Handshaking with ${esc(pendingMac || 'device')}...`, 'text-emerald-400 font-bold py-4 animate-pulse');
    }
    try {
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
        if (devices.length === 0) {
            return note('No O₂ monitors detected... put the probe on so it wakes up and advertises.',
                        'text-slate-500 text-sm animate-pulse');
        }
        deviceList.innerHTML = devices.map(d => `
            <div class="bg-slate-700/50 p-4 rounded-xl border border-slate-600 flex flex-col gap-3 justify-between">
                <div>
                    <div class="font-bold text-slate-200">${esc(d.name)}</div>
                    <div class="text-xs text-slate-400 font-mono">${esc(d.address)} · <span data-rssi="${esc(d.address)}">${esc(d.rssi)} dBm</span></div>
                </div>
                <button data-mac="${esc(d.address)}" class="bg-sky-600 hover:bg-sky-500 text-white text-sm font-bold py-2 px-4 rounded w-full transition-colors">
                    Connect Signal
                </button>
            </div>`).join('');
    } catch (err) {}
}

deviceList.addEventListener('click', async (e) => {
    const mac = e.target.closest('button[data-mac]')?.dataset.mac;
    if (!mac) return;
    pendingMac = mac;
    setTimeout(() => { pendingMac = null; }, 20000);  // give up showing "handshaking" if the worker never picks it up
    updateDeviceList();
    await fetch('/api/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({address: mac})
    });
});

setInterval(updateDeviceList, 1000);
updateDeviceList();

// --- Metrics ---
let hrMin = Infinity, hrMax = -Infinity;

function setBattery(pct) {
    if (!(pct > 0)) return;
    const [text, bar] = pct <= 20 ? ['text-red-400', 'bg-red-400']
                      : pct <= 40 ? ['text-amber-400', 'bg-amber-400']
                      : ['text-emerald-400', 'bg-emerald-400'];
    $('batt-val').textContent = pct;
    $('batt-num').className = `text-6xl font-extrabold my-2 tracking-tight ${text}`;
    $('batt-bar').className = `h-2 rounded-full transition-all duration-700 ${bar}`;
    $('batt-bar').style.width = `${Math.min(100, pct)}%`;
}

function clearReadings() {
    $('spo2-val').textContent = '--';
    $('hr-val').textContent = '--';
}

function applyStatus(status) {
    workerStatus = status;
    if (status === 'Connected' || status === 'Calibrating') {
        pendingMac = null;
        scannerPanel.style.display = 'none';
    } else {
        scannerPanel.style.display = 'block';
    }
    const [text, tone] = {
        Connected:   ['Telemetry Active 🟢', 'ok'],
        Calibrating: ['Sensor Settling...', 'info'],
        Connecting:  ['Connecting...', 'wait'],
        Scanning:    ['Scanning', 'wait'],
    }[status] || [status || 'Awaiting Data...', 'wait'];
    setStatus(text, tone);
    $('spo2-note').textContent = status === 'Calibrating' ? 'SENSOR SETTLING...' : '';
    $('spo2-note').className = 'text-xs font-bold mt-1 h-4 ' + (status === 'Calibrating' ? 'text-sky-400 animate-pulse' : 'text-slate-500');
}

// --- Stream (SSE reconnects on its own) ---
const eventSource = new EventSource('/api/vitals-stream');
eventSource.onopen = () => { if (!lastPacketAt) setStatus('System Online', 'info'); };
eventSource.onerror = () => setStatus('Server Offline 🔴 — retrying', 'bad');

eventSource.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }
    lastPacketAt = Date.now();
    DebugPanel.onPacket();
    applyStatus(data.status);
    setBattery(data.battery);
    if (data.status === 'Connected') {
        if (data.spo2 > 0) trends.sample('spo2', data.spo2);
        if (data.hr > 0) trends.sample('hr', data.hr);
    }
    if (data.battery > 0) trends.sample('battery', data.battery);

    if (data.status === 'Connected' && data.spo2 > 0) {
        const now = Date.now();
        $('spo2-val').textContent = data.spo2;
        spo2Chart.data.datasets[0].data.push({ x: now, y: data.spo2 });
        if (data.hr > 0) {
            $('hr-val').textContent = data.hr;
            hrChart.data.datasets[0].data.push({ x: now, y: data.hr });
            hrMin = Math.min(hrMin, data.hr);
            hrMax = Math.max(hrMax, data.hr);
            $('hr-range').textContent = `${hrMin} / ${hrMax}`;
        }
    } else if (data.status !== 'Calibrating') {
        clearReadings();
    }
};

// Worker rewrites data.json on every reading; if it goes quiet while "connected", say so
setInterval(() => {
    if ((workerStatus === 'Connected' || workerStatus === 'Calibrating') && Date.now() - lastPacketAt > 15000) {
        setStatus('Signal Lost — waiting for worker', 'wait');
        clearReadings();
    }
}, 1000);

// ---------- Tap-a-tile trends (see tile_trends.js) ----------
// History is pre-filled from the worker's daily CSV logs via /api/history.
const trends = createTileTrends({
    storageKey: 'viatom-trend',
    insertAfter: '#tile-row',
    ring: ['ring-sky-400/70'],
    metrics: {
        spo2:    { title: 'Blood Oxygen (SpO₂)', unit: '%',   tile: '#spo2-val', color: '#38bdf8', stepped: true, fixed: [90, 100] },
        hr:      { title: 'Pulse Rate',          unit: 'BPM', tile: '#hr-val',   color: '#f43f5e', stepped: true, span: 20 },
        battery: { title: 'Device Battery',      unit: '%',   tile: '#batt-val', color: '#34d399', stepped: true, fixed: [0, 100] },
    },
    prefill: async (metric, minutes) => (await fetch(`/api/history?minutes=${minutes}`)).json(),
});
