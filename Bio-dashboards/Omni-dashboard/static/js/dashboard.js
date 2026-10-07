const chartOptions = (delayTime) => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    scales: {
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

const ecgView = new EcgView({ monitor, storageKey: 'omni-ecg-view' });   // 🩺 Monitor / 📈 Classic

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

const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function fetchScanners() {
    try {
        const res = await fetch('/api/scanners');
        const state = await res.json();
        
        let anyScanning = false;

        ['polar', 'viatom'].forEach(type => {
            const statusEl = document.getElementById(`stat-${type}`);
            const wrapper = document.getElementById(`wrapper-${type}`);
            const list = document.getElementById(`list-${type}`);
            const deviceStatus = state.state[type].status;
            
            statusEl.innerText = `${type.toUpperCase()}: ${deviceStatus.toUpperCase()}`;
            
            if (deviceStatus === "Connected") {
                statusEl.className = "px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
                wrapper.style.display = "none";
            } else if (deviceStatus === "Connecting" || deviceStatus === "Calibrating") {
                statusEl.className = "px-3 py-1 rounded-full text-xs font-semibold bg-sky-500/20 text-sky-400 border border-sky-500/30";
                wrapper.style.display = "none";
            } else {
                anyScanning = true;
                wrapper.style.display = "block";
                statusEl.className = "px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30";
                
                list.innerHTML = '';
                if (state.devices[type].length === 0) {
                    list.innerHTML = `<div class="text-slate-500 text-sm italic py-2">Searching airspace...</div>`;
                } else {
                    state.devices[type].forEach(d => {
                        const div = document.createElement('div');
                        div.className = 'bg-slate-700/50 p-4 rounded-xl border border-slate-600 flex justify-between items-center';
                        div.innerHTML = `
                            <div>
                                <div class="font-bold text-slate-200 text-sm">${esc(d.name)}</div>
                                <div class="text-[10px] text-slate-400 font-mono mt-1">${esc(d.address)} · ${esc(d.rssi)} dBm</div>
                            </div>
                            <button class="bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold py-2 px-5 rounded transition-colors" 
                                    data-type="${type}" data-mac="${esc(d.address)}">
                                LOCK
                            </button>
                        `;
                        list.appendChild(div);
                    });
                }
            }
        });

        document.getElementById('scanner-panel').style.display = anyScanning ? "block" : "none";

    } catch(e) {}
}
setInterval(fetchScanners, 2000);

document.getElementById('scanner-panel').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-mac]');
    if (b) lockTarget(b.dataset.type, b.dataset.mac);
});

async function lockTarget(type, mac) {
    document.getElementById(`list-${type}`).innerHTML = `<div class="text-emerald-400 font-bold py-4 animate-pulse">Handshaking with ${esc(mac)}...</div>`;
    await fetch('/api/connect', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type: type, address: mac}) });
}

const $ = (id) => document.getElementById(id);
const [accX, accY, accZ] = accChart.data.datasets.map(d => d.data);
const ppiData = ppiChart.data.datasets[0].data;
const labelEl = $('stress-label');
const LABEL_BASE = 'text-[10px] sm:text-xs font-bold mt-2 text-center w-full h-4 ';

// EventSource reconnects on its own if the server restarts
const evtSource = new EventSource("/api/stream");
evtSource.onmessage = (e) => {
    const { polar, viatom } = JSON.parse(e.data);
    DebugPanel.onPacket();

    if (polar.status === "Connected") {
        if (polar.hr > 0) $('val-polar-hr').textContent = $('mon-hr').textContent = polar.hr;
        if (polar.rr > 0) $('val-rr').textContent = polar.rr;

        // One chart point per actual beat (not one per SSE tick)
        for (const [ts, rr] of polar.rr_buffer) ppiData.push({ x: ts, y: rr });

        const rmssd = polar.rmssd;
        $('val-rmssd').textContent = rmssd.toFixed(1);
        if (rmssd > 0) {
            const [text, color] = rmssd < 20 ? ['HIGH STRESS', 'text-rose-500']
                                : rmssd < 50 ? ['BALANCED', 'text-amber-400']
                                : ['RELAXED', 'text-emerald-400'];
            labelEl.textContent = text;
            labelEl.className = LABEL_BASE + color;
        }
        $('val-sdnn').textContent = polar.sdnn.toFixed(1);

        if (polar.ecg_buffer.length) ecgView.push(polar.ecg_buffer);   // rows are [ts_ms, mV]
        for (const [ts, x, y, z] of polar.acc_buffer) {
            accX.push({ x: ts, y: x });
            accY.push({ x: ts, y: y });
            accZ.push({ x: ts, y: z });
        }
    }

    if (polar.status !== "Connected") $('mon-hr').textContent = '--';

    if (viatom.status === "Connected") {
        if (viatom.spo2 > 0) $('val-spo2').textContent = viatom.spo2;
        if (viatom.hr > 0) $('val-viatom-hr').textContent = viatom.hr;
    }
};

// ---------- Tap-a-tile trends (see tile_trends.js) ----------
// History is pre-filled from this session's logs (logs_omni/) via /api/history.
const trends = createTileTrends({
    storageKey: 'omni-trend',
    insertAfter: '#tile-row',
    ring: ['ring-sky-400/70'],
    metrics: {
        spo2:      { title: 'O₂ Saturation', unit: '%',   tile: '#val-spo2',      color: '#818cf8', stepped: true, fixed: [90, 100] },
        viatom_hr: { title: 'Ring HR',       unit: 'BPM', tile: '#val-viatom-hr', color: '#22d3ee', stepped: true, span: 20 },
        polar_hr:  { title: 'ECG HR',        unit: 'BPM', tile: '#val-polar-hr',  color: '#f43f5e', stepped: true, span: 20 },
        rr:        { title: 'R-R Interval',  unit: 'ms',  tile: '#val-rr',        color: '#34d399', span: 200 },
        rmssd:     { title: 'HRV (RMSSD)',   unit: 'ms',  tile: '#val-rmssd',     color: '#c084fc', span: 20, dp: 1 },
        sdnn:      { title: 'SDNN',          unit: 'ms',  tile: '#val-sdnn',      color: '#a78bfa', span: 20, dp: 1 },
    },
    prefill: async (metric, minutes) => (await fetch(`/api/history?metric=${metric}&minutes=${minutes}`)).json(),
});
evtSource.addEventListener('message', (e) => {
    let d; try { d = JSON.parse(e.data); } catch (err) { return; }
    const { polar, viatom } = d;
    if (viatom.status === 'Connected') {
        if (viatom.spo2 > 0) trends.sample('spo2', viatom.spo2);
        if (viatom.hr > 0) trends.sample('viatom_hr', viatom.hr);
    }
    if (polar.status === 'Connected') {
        if (polar.hr > 0) trends.sample('polar_hr', polar.hr);
        if (polar.rmssd > 0) trends.sample('rmssd', polar.rmssd);
        if (polar.sdnn > 0) trends.sample('sdnn', polar.sdnn);
        for (const [ts, rr] of polar.rr_buffer) trends.record('rr', ts, rr);   // one point per beat
    }
});
