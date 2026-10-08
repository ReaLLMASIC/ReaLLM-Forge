const chartOptions = (delayTime) => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    scales: {
        // PMD frames arrive ~every 0.5 s, so the delay must exceed that for a smooth scroll
        x: { type: 'realtime', realtime: { duration: 5000, refresh: 40, delay: delayTime } },
        y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
    },
    plugins: { legend: { display: false } }
});

// --- Bedside-monitor ECG (see ecg_monitor.js) ---
let audioCtx = null, beepOn = false;
function beep() {
    if (!beepOn || !audioCtx) return;
    const t = audioCtx.currentTime;
    const osc = audioCtx.createOscillator(), g = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.15, t + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.08);
    osc.connect(g).connect(audioCtx.destination);
    osc.start(t);
    osc.stop(t + 0.09);
}

const monitor = new EcgMonitor({
    grid: document.getElementById('ecgGrid'),
    trace: document.getElementById('ecgTrace'),
    heart: document.getElementById('mon-heart'),
    noSignal: document.getElementById('ecg-nosignal'),
    onBeat: beep,
});

const ecgView = new EcgView({ monitor, storageKey: 'polar-ecg-view' });   // 🩺 Monitor / 📈 Classic

document.getElementById('gain-btn').addEventListener('click', (e) => {
    e.currentTarget.textContent = 'GAIN ' + monitor.cycleGain();
});
document.getElementById('beep-btn').addEventListener('click', (e) => {
    // Browsers only allow audio after a user gesture, so the context is created here
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    beepOn = !beepOn;
    e.currentTarget.textContent = beepOn ? '🔊 Beep on' : '🔇 Beep off';
    setPillActive(e.currentTarget, beepOn);
});
setInterval(() => {
    const btn = document.getElementById('gain-btn');
    if (btn.textContent.startsWith('GAIN AUTO')) btn.textContent = 'GAIN ' + monitor.gainLabel;
}, 1000);


const accChart = new Chart(document.getElementById('accChart').getContext('2d'), {
    type: 'line', data: { datasets: [
        { label: 'X', borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0, data: [] },
        { label: 'Y', borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0, data: [] },
        { label: 'Z', borderColor: '#fbbf24', borderWidth: 1.5, pointRadius: 0, data: [] }
    ]},
    options: { ...chartOptions(1000), plugins: { legend: { display: true, labels: { color: '#94a3b8' } } } }
});

const ppiChart = new Chart(document.getElementById('ppiChart').getContext('2d'), {
    type: 'line', data: { datasets: [{ borderColor: '#818cf8', backgroundColor: '#818cf8', borderWidth: 0, pointRadius: 4, data: [] }] },
    options: chartOptions(1500)
});

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const scannerPanel = $('scanner-panel');
const hrEl = $('hr-val'), ecgEl = $('ecg-val'), rmssdEl = $('rmssd-val'), labelEl = $('stress-label');

const BADGE = 'px-3 py-1 rounded-full text-xs font-semibold border ';
function setStatus(text, tone) {
    statusEl.textContent = text;
    statusEl.className = BADGE + {
        ok: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
        bad: 'bg-red-500/20 text-red-400 border-red-500/30',
        wait: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    }[tone];
}

// "Streaming" is driven by data freshness, not by the socket, so the scanner
// comes back on its own when the strap drops and the worker returns to scanning.
const STALE_MS = 5000;
let uiIsConnected = false;
let lastPacketAt = 0;

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
    document.getElementById('mon-hr').textContent = '--';
    scannerPanel.style.display = 'block';
    setStatus(text, tone);
    updateScannerList();
}
setInterval(() => {
    if (uiIsConnected && Date.now() - lastPacketAt > STALE_MS) markIdle('Signal Lost — Rescanning', 'wait');
}, 1000);

const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const deviceList = $('device-list');
const note = (html, cls) => { deviceList.innerHTML = `<div class="${cls} col-span-3">${html}</div>`; };
let lastListKey = '';

// The worker reports scanning / connecting / streaming; the panel follows it,
// so a poll mid-handshake can't redraw stale device cards over the progress message.
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
            return note('Connected — starting ECG / ACC streams...', 'text-emerald-400 font-bold py-4 animate-pulse');
        }

        const devices = await (await fetch('/api/scan-results')).json();
        const key = JSON.stringify(devices.map(d => d.address));
        if (key === lastListKey) {
            // same devices: just refresh RSSI in place, no DOM rebuild
            devices.forEach(d => {
                const el = deviceList.querySelector(`[data-rssi="${CSS.escape(d.address)}"]`);
                if (el) el.textContent = `${d.rssi} dBm`;
            });
            return;
        }
        lastListKey = key;
        if (devices.length === 0) {
            return note('No active straps detected... ensure pads are wet.', 'text-slate-500 text-sm animate-pulse');
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
    lastListKey = '';
    note(`Handshaking with ${esc(mac)}...`, 'text-emerald-400 font-bold py-4 animate-pulse');
    await fetch('/api/connect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({address: mac})
    });
});

setInterval(updateScannerList, 1000);
updateScannerList();

// --- HRV ---
const ppiHistory = [];
function pushPpi(ppi) {
    ppiHistory.push(ppi);
    if (ppiHistory.length > 20) ppiHistory.shift();
    if (ppiHistory.length <= 2) return null;

    let sum = 0;
    for (let i = 1; i < ppiHistory.length; i++) {
        const d = ppiHistory[i] - ppiHistory[i - 1];
        sum += d * d;
    }
    const rmssd = Math.sqrt(sum / (ppiHistory.length - 1));
    rmssdEl.textContent = rmssd.toFixed(1);

    const [text, color] = rmssd < 20 ? ['HIGH STRESS (SYMPATHETIC)', 'text-rose-500']
                        : rmssd < 50 ? ['MODERATE (BALANCED)', 'text-amber-400']
                        : ['RELAXED (PARASYMPATHETIC)', 'text-emerald-400'];
    labelEl.textContent = text;
    labelEl.className = 'text-xs font-bold mt-1 ' + color;
    return rmssd;
}

// --- Stream handling: each message is a batch {type, rows: [[...], ...]} ---
const monHrEl = document.getElementById('mon-hr');
const [accX, accY, accZ] = accChart.data.datasets.map(d => d.data);
const ppiData = ppiChart.data.datasets[0].data;

const handlers = {
    ecg(rows) {
        ecgView.push(rows);
        ecgPeakToPeak(rows);
        const [, mv, hr] = rows[rows.length - 1];
        ecgEl.textContent = mv.toFixed(2);
        if (hr > 0) hrEl.textContent = monHrEl.textContent = hr;
    },
    acc(rows) {
        for (const [ts, x, y, z] of rows) {
            accX.push({ x: ts, y: x });
            accY.push({ x: ts, y: y });
            accZ.push({ x: ts, y: z });
        }
    },
    ppi(rows) {
        for (const [ts, ppi, hr] of rows) {
            ppiData.push({ x: ts, y: ppi });
            const rmssd = pushPpi(ppi);
            if (rmssd != null) trends.record('rmssd', ts, rmssd);
            if (hr > 0) { hrEl.textContent = monHrEl.textContent = hr; trends.record('hr', ts, hr); }
        }
    },
};

function connectWs() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.onopen = () => { if (!uiIsConnected) setStatus('System Online', 'wait'); };
    ws.onclose = () => {
        markIdle('Disconnected 🔴 — retrying', 'bad');
        setTimeout(connectWs, 2000);
    };
    ws.onmessage = (event) => {
        const packet = JSON.parse(event.data);
        const h = handlers[packet.type];
        if (h && packet.rows && packet.rows.length) {
            DebugPanel.onPacket(packet.type, packet.rows.length);
            markLive();
            h(packet.rows);
        }
    };
}
connectWs();

// ---------- Tap-a-tile trends (see tile_trends.js) ----------
// History is pre-filled from this session's CSV logs via /api/history.
const trends = createTileTrends({
    storageKey: 'polar-trend',
    insertAfter: '#tile-row',
    ring: ['ring-rose-400/70'],
    metrics: {
        hr:     { title: 'Heart Rate', unit: 'BPM', tile: '#hr-val', color: '#f43f5e', stepped: true, span: 20 },
        // An instantaneous ECG voltage sampled once a second is noise; the per-second
        // peak-to-peak amplitude is meaningful: it drops when electrode contact degrades.
        ecg_pp: { title: 'ECG Amplitude (peak-to-peak / s)', unit: 'mV', tile: '#ecg-val', color: '#2dd4bf', span: 0.5, dp: 2 },
        rmssd:  { title: 'HRV (RMSSD)', unit: 'ms', tile: '#rmssd-val', color: '#818cf8', span: 20, dp: 1 },
    },
    prefill: async (metric, minutes) => (await fetch(`/api/history?metric=${metric}&minutes=${minutes}`)).json(),
});

// Per-second ECG peak-to-peak, from the samples as they arrive
let ppBucket = null, ppLo = Infinity, ppHi = -Infinity;
function ecgPeakToPeak(rows) {
    for (const [ts, mv] of rows) {
        const b = Math.floor(ts / 1000);
        if (ppBucket !== null && b !== ppBucket && ppHi > ppLo) {
            trends.record('ecg_pp', ppBucket * 1000 + 500, +(ppHi - ppLo).toFixed(3));
            ppLo = Infinity; ppHi = -Infinity;
        }
        ppBucket = b;
        if (mv < ppLo) ppLo = mv;
        if (mv > ppHi) ppHi = mv;
    }
}
