// Hydro-dash: HidrateSpark bottle. Polls the worker status (1 s) and the summary computed
// from the sip log (5 s, or immediately when a new sip arrives).

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ML_PER_OZ = 29.5735;
let units = 'ml';
let settings = {};
const vol = (ml, d = 0) => ml == null ? '--' : units === 'oz' ? (ml / ML_PER_OZ).toFixed(1) : Math.round(ml).toFixed(d);
const unitLabel = () => units === 'oz' ? 'fl oz' : 'mL';
const ago = (ts) => {
    const s = Math.max(0, Date.now() / 1000 - ts);
    return s < 60 ? 'just now' : s < 3600 ? `${Math.round(s / 60)} min ago` : s < 86400 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`;
};

// ---------- status badge ----------
const BADGE = 'px-3 py-1 rounded-full text-xs font-semibold border ';
const TONES = { ok: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', wait: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
                info: 'bg-sky-500/20 text-sky-400 border-sky-500/30', bad: 'bg-red-500/20 text-red-400 border-red-500/30' };
const setStatus = (text, tone) => { $('status').textContent = text; $('status').className = BADGE + TONES[tone]; };

// ---------- charts ----------
const axis = { ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } };
const hourChart = new Chart($('hourChart'), {
    type: 'bar',
    data: { labels: [...Array(24).keys()].map(h => `${h}`), datasets: [{ data: Array(24).fill(0), backgroundColor: '#22d3ee', borderRadius: 3 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } },
               scales: { x: axis, y: { ...axis, beginAtZero: true } } },
});
const weekChart = new Chart($('weekChart'), {
    type: 'bar',
    data: { labels: [], datasets: [
        { type: 'line', label: 'Goal', data: [], borderColor: '#f59e0b', borderDash: [6, 4], borderWidth: 2, pointRadius: 0 },
        { label: 'Intake', data: [], backgroundColor: '#38bdf8', borderRadius: 4 } ] },
    options: { responsive: true, maintainAspectRatio: false, animation: false,
               plugins: { legend: { display: true, labels: { color: '#94a3b8', boxWidth: 12 } } },
               scales: { x: axis, y: { ...axis, beginAtZero: true } } },
});

// ---------- summary (from the sip log) ----------
let lastSipSeen = null;
async function loadSummary() {
    let s;
    try { s = await (await fetch('/api/summary?days=7')).json(); } catch (e) { return; }
    const goal = s.goal_ml || 2500;
    $('today-val').textContent = vol(s.today_ml);
    $('today-goal').textContent = `${Math.round(100 * s.today_ml / goal)}% of ${vol(goal)} ${unitLabel()} goal`;
    $('goal-ring').style.strokeDashoffset = 314.16 * (1 - Math.min(1, s.today_ml / goal));
    $('sips-val').textContent = s.sips_today;
    $('refills-val').textContent = s.refills_today;
    if (s.last_sip) { $('last-val').textContent = vol(s.last_sip.ml); $('last-ago').textContent = ago(s.last_sip.ts); }
    const k = units === 'oz' ? 1 / ML_PER_OZ : 1;
    hourChart.data.datasets[0].data = s.hourly.map(v => +(v * k).toFixed(1));
    hourChart.update('none');
    weekChart.data.labels = s.daily.map(d => luxon.DateTime.fromISO(d.date).toFormat('ccc d'));
    weekChart.data.datasets[1].data = s.daily.map(d => +(d.ml * k).toFixed(1));
    weekChart.data.datasets[0].data = s.daily.map(() => +(goal * k).toFixed(1));
    weekChart.update('none');
    $('sip-list').innerHTML = s.recent.length ? s.recent.map(r =>
        `<div class="grid grid-cols-3 items-center py-1.5"><span class="text-slate-300">${new Date(r.ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
         <span class="font-mono text-cyan-300 text-right pr-6">${vol(r.ml)} ${unitLabel()}</span>
         <span class="text-xs text-right ${r.source === 'replayed' ? 'text-slate-500' : 'text-emerald-400'}">${r.source === 'replayed' ? 'synced from bottle' : 'live'}</span></div>`).join('')
        : '<div class="text-slate-500 py-2">No sips yet today.</div>';
    // trend: cumulative intake today
    for (const p of s.cumulative) trends.record('intake', p.x, p.y);
}

// ---------- live status + scanner ----------
let showAll = false;
$('show-all').addEventListener('change', (e) => { showAll = e.target.checked; lastListKey = ''; });
let lastListKey = '';

async function poll() {
    let st;
    try { st = await (await fetch('/api/status')).json(); } catch (e) { setStatus('Server offline', 'bad'); return; }
    settings = st.settings || settings;
    if (settings.units && settings.units !== units) { units = settings.units; applyUnits(); }
    const live = st.live || {};
    const streaming = st.state === 'streaming';
    $('scanner-panel').classList.toggle('hidden', streaming);
    if (streaming) setStatus('Bottle connected 🟢', 'ok');
    else if (st.state === 'connecting') setStatus(`Connecting${st.attempt > 1 ? ` (try ${st.attempt}/${st.max_attempts})` : ''}...`, 'wait');
    else setStatus('Looking for the bottle', 'info');

    // fill + battery
    if (live.fill_ml != null) {
        $('fill-val').textContent = vol(live.fill_ml);
        $('fill-bar').style.width = `${live.fill_pct ?? 0}%`;
        $('fill-note').textContent = `${live.fill_pct}% of ${vol(settings.capacity_ml)} ${unitLabel()}` + (settings.weight_full_raw == null ? ' (estimated)' : '');
        trends.sample('fill', live.fill_ml);
    } else {
        $('fill-val').textContent = '--';
        $('fill-note').textContent = streaming ? 'calibrate below to see the fill level' : '';
    }
    if (live.battery != null) { $('batt-val').textContent = live.battery; trends.sample('battery', live.battery); }
    if (live.last_sip && live.last_sip.ts !== lastSipSeen) { lastSipSeen = live.last_sip.ts; loadSummary(); }
    if (live.serial) $('subtitle').textContent = `HidrateSpark ${live.serial}${live.firmware ? ' · firmware ' + live.firmware : ''}`;
    if (!streaming) renderDevices(st);
}

async function renderDevices(st) {
    const list = $('device-list');
    if (st.state === 'connecting') {
        lastListKey = '';
        const phase = (st.debug || {}).phase || 'connecting';
        const err = (st.debug || {}).last_error;
        list.innerHTML = `<div class="col-span-3 py-2"><div class="text-cyan-300 font-bold animate-pulse">Connecting to ${esc(st.address)} (try ${esc(st.attempt)}/${esc(st.max_attempts)})...</div>
            <div class="text-xs text-slate-400 mt-1">${esc(phase)}</div>
            ${err && Date.now() / 1000 - err.at < 60 ? `<div class="text-xs text-amber-300 mt-1">last try: ${esc(err.error)}</div>` : ''}</div>`;
        return;
    }
    let devices = [];
    try { devices = await (await fetch('/api/scan-results')).json(); } catch (e) { return; }
    const shown = devices.filter(d => showAll || d.bottle);
    const key = JSON.stringify(shown.map(d => d.address)) + showAll;
    if (key === lastListKey) return;
    lastListKey = key;
    list.innerHTML = shown.length ? shown.map(d => `
        <div class="bg-slate-700/50 p-4 rounded-xl border ${d.bottle ? 'border-cyan-500/40' : 'border-slate-600'} flex flex-col gap-3">
            <div><div class="font-bold text-slate-200">${esc(d.name)} ${d.bottle ? '<span class="text-xs text-cyan-300">· HidrateSpark</span>' : ''}</div>
                 <div class="text-xs text-slate-400 font-mono">${esc(d.address)} · ${esc(d.rssi)} dBm</div>
                 ${(d.services || []).length || (d.mfr || []).length ? `<div class="text-[11px] text-slate-500 font-mono mt-1">${(d.services || []).length ? 'svc ' + esc(d.services.join(' ')) : ''}${(d.mfr || []).length ? ' · mfr ' + esc(d.mfr.join(' ')) : ''}</div>` : ''}</div>
            <button data-mac="${esc(d.address)}" class="bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-bold py-2 rounded transition-colors">Connect</button>
        </div>`).join('')
        : `<div class="text-slate-500 text-sm col-span-3">${devices.length && !showAll ? 'No HidrateSpark bottle seen yet (tick "Show all" if yours uses another name).' : 'Nothing in range yet...'}</div>`;
}
$('device-list').addEventListener('click', async (e) => {
    const mac = e.target.closest('button[data-mac]')?.dataset.mac;
    if (!mac) return;
    await fetch('/api/connect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ address: mac }) });
});

// ---------- settings + calibration ----------
function fillSettingsForm() {
    $('set-capacity').value = settings.capacity_ml ?? '';
    $('set-goal').value = settings.goal_ml ?? '';
    $('set-units').value = settings.units || 'ml';
}
function applyUnits() {
    for (const id of ['today-unit', 'fill-unit', 'last-unit']) $(id).textContent = unitLabel();
    loadSummary();
}
$('settings-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const body = { capacity_ml: +$('set-capacity').value || null, goal_ml: +$('set-goal').value || null, units: $('set-units').value };
    try { settings = await (await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json(); } catch (err) {}
    units = settings.units || 'ml'; applyUnits(); fillSettingsForm();
    $('settings-msg').textContent = 'Saved.';
    setTimeout(() => { $('settings-msg').textContent = ''; }, 2500);
});
document.querySelectorAll('button[data-cal]').forEach(b => b.addEventListener('click', async () => {
    const which = b.dataset.cal;
    if (which === 'reset') {
        await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reset_calibration: true }) });
        $('settings-msg').textContent = 'Calibration cleared.';
    } else {
        await fetch('/api/calibrate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ which }) });
        $('settings-msg').textContent = `Marked as ${which} (needs the bottle connected and upright).`;
    }
    setTimeout(() => { $('settings-msg').textContent = ''; }, 4000);
}));

// ---------- tap-a-tile trends (hydration is slow: hours and days) ----------
const trends = createTileTrends({
    storageKey: 'hydro-trend',
    insertAfter: '#tile-row-2',
    ring: ['ring-cyan-400/70'],
    historyMs: 8 * 24 * 3600 * 1000,
    sampleMs: 30000,
    windows: [[60, '1 hour'], [360, '6 hours'], [1440, '1 day'], [10080, '7 days']],
    metrics: {
        intake:  { title: 'Intake (cumulative, resets daily)', unit: 'mL', tile: '#today-val', color: '#22d3ee', stepped: true, span: 250 },
        fill:    { title: 'In the bottle', unit: 'mL', tile: '#fill-val', color: '#38bdf8', span: 200 },
        battery: { title: 'Bottle battery', unit: '%', tile: '#batt-val', color: '#34d399', stepped: true, fixed: [0, 100] },
    },
    prefill: async (metric, minutes) => (await fetch(`/api/history?metric=${metric}&minutes=${minutes}`)).json(),
});

(async () => {
    try { settings = await (await fetch('/api/settings')).json(); } catch (e) {}
    units = settings.units || 'ml';
    fillSettingsForm(); applyUnits();
    poll(); setInterval(poll, 1000);
    setInterval(loadSummary, 5000);
    setInterval(() => { if (lastSipSeen) $('last-ago').textContent = ago(lastSipSeen); }, 30000);
})();
