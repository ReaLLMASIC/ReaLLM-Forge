// Brain (Muse S Athena headband): live EEG, brain-wave bands, forehead blood oxygenation (fNIRS),
// pulse and head movement. Rows arrive on the brain stream as {type, rows} (eeg, optics, imu, bands,
// heart, fnirs, battery), in the columns of the session's CSV files (muse_dashboard/muse_worker.py).
import { icon, num, esc, dapi, toast, busy, $ } from '../ui.js';
import { Plot } from '../plot.js';
import { EegView } from '../eeg.js';
import { pageHead, bindHead, connectPanel } from '../device.js';

const KEY = 'brain';
const MAIN = ['TP9', 'AF7', 'AF8', 'TP10'];
const BANDS = [['Delta', 'δ', '1–4 Hz', '#818cf8'], ['Theta', 'θ', '4–8 Hz', '#38bdf8'], ['Alpha', 'α', '8–13 Hz', '#34d399'],
               ['Beta', 'β', '13–30 Hz', '#fbbf24'], ['Gamma', 'γ', '30–44 Hz', '#f472b6']];
const CONTACT = { 1: ['good', 'ok'], 2: ['fair', 'warn'], 4: ['bad', 'bad'], 0: ['off', 'off'] };
const FNIRS = [['lo', 'Left HbO', '#f43f5e', null], ['lr', 'Left HbR', '#3b82f6', null], ['ro', 'Right HbO', '#fb7185', [5, 4]], ['rr', 'Right HbR', '#60a5fa', [5, 4]]];
// the optics columns by how many channels the preset sends
const OPTICS = { 16: ['LO_NIR', 'RO_NIR', 'LO_IR', 'RO_IR', 'LI_NIR', 'RI_NIR', 'LI_IR', 'RI_IR', 'LO_RED', 'RO_RED', 'LO_AMB', 'RO_AMB', 'LI_RED', 'RI_RED', 'LI_AMB', 'RI_AMB'],
                 8: ['LO_NIR', 'RO_NIR', 'LO_IR', 'RO_IR', 'LI_NIR', 'RI_NIR', 'LI_IR', 'RI_IR'], 4: ['LI_NIR', 'RI_NIR', 'LI_IR', 'RI_IR'], 0: [] };
const legend = (items) => `<div class="legend" style="margin-bottom:8px">${items.map(([label, color, dashed]) =>
    `<span><i style="background:${color}${dashed ? ';opacity:.6' : ''}"></i>${label}</span>`).join('')}</div>`;

export async function mount(el, { store }) {
    el.innerHTML = `${pageHead(KEY)}
      <div data-connect></div>
      <div class="stack" data-live>
        <div class="card flush">
          <div class="eeg" data-eeg><canvas></canvas><div class="nosig" data-nosig>Waiting for the EEG…</div></div>
          <div class="row" style="padding:10px 14px;border-top:1px solid var(--line)">
            <span class="small muted">EEG · <span data-specs></span></span><span class="spacer"></span>
            <button class="btn sm" data-eeg-scale title="How tall the traces are drawn (per lane)"></button>
            <button class="btn sm" data-eeg-notch title="Mains notch: 60 Hz in the Americas, 50 Hz elsewhere"></button>
            <button class="btn sm" data-eeg-filter title="Show the filtered (1–40 Hz) or the raw signal"></button>
            <button class="btn sm" data-eeg-aux aria-pressed="false" title="Also show the 4 AUX channels">+ AUX</button>
          </div>
        </div>
        <div class="grid tiles" style="--c:var(--brain)">
          <div class="tile accent" style="--c:var(--heart)"><div class="label">Heart rate</div><div class="value"><span data-v="hr">--</span><small>bpm</small></div><div class="note" data-n="hr">from the forehead pulse</div></div>
          <div class="tile accent"><div class="label">Alpha share</div><div class="value"><span data-v="alpha">--</span><small>%</small></div><div class="note" data-n="alpha">8–13 Hz, of 1–44 Hz</div></div>
          <div class="tile"><div class="label">Sensor contact</div><div class="contact" data-contact></div><div class="note" data-n="contact">waiting for the EEG…</div></div>
          <div class="tile"><div class="label">Battery</div><div class="value"><span data-v="bat">--</span><small>%</small></div><div class="note" data-n="bat"></div></div>
        </div>
        <div class="grid cols-2">
          <div class="card"><div class="card-head"><h2>Brain waves</h2><span class="card-sub">share of the power now, sensors in contact</span></div>
            <div class="bands" data-bands>${BANDS.map(([name, sym, hz, c]) => `<div class="band" style="--c:${c}"><span class="name"><b>${sym} ${name}</b><small>${hz}</small></span>
              <span class="meter"><i></i></span><span class="pct num">--</span></div>`).join('')}</div></div>
          <div class="card"><div class="card-head"><h2>Over time</h2><span class="card-sub">last 2 minutes</span></div>
            ${legend(BANDS.map(b => [`${b[1]} ${b[0]}`, b[3]]))}<div class="chart short" data-band-trend></div></div>
        </div>
        <div class="grid cols-2">
          <div class="card"><div class="card-head"><h2>Blood oxygenation</h2><span class="card-sub" data-n="fnirs">fNIRS, change from the baseline</span><span class="spacer"></span>
              <button class="btn sm" data-zero title="Make the next 10 s the new zero (after settling in, before a task)">${icon('refresh')}Zero</button></div>
            ${legend(FNIRS.map(f => [f[1], f[2], !!f[3]]))}<div class="chart short" data-fnirs></div>
            <p class="tiny faint" style="margin-top:8px">HbO oxy-, HbR deoxy-haemoglobin, per side. The path length is assumed (3 cm × DPF 6): read the direction and shape, not the µM.</p></div>
          <div class="card"><div class="card-head"><h2>Pulse</h2><span class="card-sub" data-n="ppg">forehead PPG</span><span class="spacer"></span>
              <span class="small muted">HRV <b class="num" style="color:var(--text)" data-v="rmssd">--</b> ms</span></div>
            <div class="chart short" data-ppg></div></div>
        </div>
        <div class="grid cols-2">
          <div class="card"><div class="card-head"><h2>Motion</h2><span class="card-sub">accelerometer (g), last 10 s</span></div>
            ${legend([['X', '#38bdf8'], ['Y', '#a78bfa'], ['Z', '#fbbf24']])}<div class="chart short" data-acc></div></div>
          <div class="card"><div class="card-head"><h2>Head position</h2></div>
            <dl class="kv"><dt>Tilt forward / back</dt><dd data-v="pitch">--</dd><dt>Tilt left / right</dt><dd data-v="roll">--</dd>
              <dt>Turning</dt><dd data-v="turn">--</dd><dt>Movement</dt><dd data-v="still">--</dd></dl></div>
        </div>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-head"><h2>What the headband sends</h2></div>
        <label class="field"><select class="input" data-preset></select>
          <span class="hint">More optics channels give fNIRS on both sides; fewer save battery. Changing it restarts the session with new files.</span></label></div>
      <p class="small faint" style="margin-top:18px">Band powers are a share of the 1–44 Hz power from the last 2 s; HRV is RMSSD over the last 20 beats. A research tool, not a diagnosis.</p>`;

    const V = (k, text) => { const n = el.querySelector(`[data-v="${k}"]`); if (n) n.textContent = text; };
    const N = (k, text) => { const n = el.querySelector(`[data-n="${k}"]`); if (n) n.textContent = text; };

    // ---- EEG
    const eeg = new EegView({ canvas: $('[data-eeg] canvas', el), noSignal: $('[data-nosig]', el),
                              channels: [...MAIN, 'AUX1', 'AUX2', 'AUX3', 'AUX4'] });
    const paintEeg = () => {
        $('[data-specs]', el).textContent = eeg.specs;
        $('[data-eeg-scale]', el).textContent = `±${eeg.scale} µV`;
        $('[data-eeg-notch]', el).textContent = eeg.notch ? `${eeg.notch} Hz notch` : 'No notch';
        $('[data-eeg-filter]', el).textContent = eeg.filtered ? 'Filtered' : 'Raw';
        $('[data-eeg-aux]', el).setAttribute('aria-pressed', String(eeg.aux));
    };
    for (const [sel, fn] of [['scale', 'cycleScale'], ['notch', 'cycleNotch'], ['filter', 'toggleFilter'], ['aux', 'toggleAux']]) {
        $(`[data-eeg-${sel}]`, el).addEventListener('click', () => { eeg[fn](); paintEeg(); });
    }
    paintEeg();

    // ---- charts
    const live = (window, extra = {}) => ({ x: { window }, tick: 250, gap: 4000, ...extra });
    const bandTrend = new Plot($('[data-band-trend]', el), { series: BANDS.map((b, k) => ({ id: 'b' + k, label: b[0], color: b[3], unit: '%', dp: 1, width: 1.6 })),
        ...live(120e3, { tick: 1000 }), y: { zero: true, minSpan: 30 }, empty: 'Waiting for the bands' });
    const fnirs = new Plot($('[data-fnirs]', el), { series: FNIRS.map(([id, label, color, dash]) => ({ id, label, color, dash, unit: 'µM', dp: 2, width: 1.6 })),
        ...live(120e3, { tick: 1000 }), y: { minSpan: 1 }, empty: 'Waiting for the baseline' });
    const ppg = new Plot($('[data-ppg]', el), { series: [{ id: 'ppg', label: 'PPG', color: 'var(--heart)', width: 1.8 }],
        ...live(10e3), y: { format: () => '' }, empty: 'Waiting for a pulse' });
    const acc = new Plot($('[data-acc]', el), { series: [['x', '#38bdf8'], ['y', '#a78bfa'], ['z', '#fbbf24']].map(([id, color]) => ({ id, label: id.toUpperCase(), color, unit: 'g', dp: 2, width: 1.4 })),
        ...live(10e3), y: { minSpan: 0.4 }, empty: 'Waiting for motion data' });

    // ---- the streams
    function contact(codes) {
        $('[data-contact]', el).innerHTML = MAIN.map((ch, i) => {
            const [word, tone] = CONTACT[codes[i]] || CONTACT[0];
            return `<div title="${ch}: ${word}"><span class="dot ${tone}"></span><span>${ch}</span></div>`;
        }).join('');
        const good = codes.filter(c => c === 1 || c === 2).length;
        N('contact', good === 4 ? 'all four in contact' : good ? `${good} of 4 in contact: wet / adjust the others` : 'not on the head yet');
    }
    contact([0, 0, 0, 0]);

    let smooth = null;
    const bandRows = [...el.querySelectorAll('[data-bands] .band')];
    function onBands(rows) {
        for (const r of rows) {
            const codes = r.slice(21, 25);
            contact(codes);
            // shares of the power per channel in contact (the file holds log10 µV²), then their mean
            const use = MAIN.map((_, c) => c).filter(c => codes[c] === 1 || codes[c] === 2);
            if (!use.length) { V('alpha', '--'); smooth = null; continue; }
            const share = BANDS.map(() => 0);
            for (const c of use) {
                const p = r.slice(1 + 5 * c, 6 + 5 * c).map(v => 10 ** v), tot = p.reduce((a, b) => a + b, 0);
                p.forEach((v, k) => { share[k] += 100 * v / tot / use.length; });
            }
            // shown smoothed over ~3 s (the files keep every second as it was): one blink or jaw clench
            // would otherwise swing the shares from one second to the next
            smooth = smooth ? smooth.map((v, k) => v + (share[k] - v) * 0.35) : share;
            smooth.forEach((v, k) => bandTrend.push('b' + k, r[0], +v.toFixed(1)));
            V('alpha', smooth[2].toFixed(0));
            N('alpha', `8–13 Hz, ${use.length} of 4 channels`);
        }
        if (smooth) bandRows.forEach((row, k) => {
            row.querySelector('i').style.width = `${Math.min(100, smooth[k] * 1.6)}%`;      // the bar is full at 62 %
            row.querySelector('.pct').textContent = `${smooth[k].toFixed(0)}%`;
        });
    }

    let ppgChannel = null, opticsCols = null, ppgMean = null;
    function onOptics(rows) {
        const i = opticsCols && ppgChannel ? opticsCols.indexOf(ppgChannel) : -1;
        if (i < 0) return;
        for (const r of rows) {
            const x = r[i + 1];
            if (x == null) continue;
            ppgMean = ppgMean === null ? x : ppgMean + (x - ppgMean) / 64;      // ~1 s high-pass
            ppg.push('ppg', r[0], -(x - ppgMean));                             // more blood = less light: flip it
        }
    }

    const ibis = [];
    function onHeart(rows) {
        for (const [, ibi, hr] of rows) {
            V('hr', hr);
            ibis.push(ibi); if (ibis.length > 20) ibis.shift();
            if (ibis.length > 2) {
                let s = 0; for (let k = 1; k < ibis.length; k++) s += (ibis[k] - ibis[k - 1]) ** 2;
                V('rmssd', Math.sqrt(s / (ibis.length - 1)).toFixed(1));
            }
        }
    }

    let gyroAvg = 0;
    function onImu(rows) {
        for (const [ts, ax, ay, az, gx, gy, gz] of rows) {
            acc.push('x', ts, ax); acc.push('y', ts, ay); acc.push('z', ts, az);
            gyroAvg += (Math.hypot(gx, gy, gz) - gyroAvg) / 26;               // ~0.5 s
        }
        const [, ax, ay, az, , , gz] = rows[rows.length - 1];
        const deg = (r) => `${Math.round(r * 180 / Math.PI) || 0}°`;          // never "-0°"
        V('pitch', deg(Math.atan2(-ax, Math.hypot(ay, az))));
        V('roll', deg(Math.atan2(ay, az)));
        V('turn', `${Math.round(gz) || 0} °/s`);
        V('still', gyroAvg < 5 ? 'still' : gyroAvg < 30 ? 'some' : 'moving');
    }

    const handlers = {
        eeg: (rows) => eeg.push(rows), bands: onBands, optics: onOptics, heart: onHeart, imu: onImu,
        fnirs: (rows) => { for (const r of rows) FNIRS.forEach(([id], k) => fnirs.push(id, r[0], r[k + 1])); },
        battery: (rows) => V('bat', num(rows[rows.length - 1][1])),
    };
    const LIVE = new Set(['eeg', 'optics', 'imu']);           // these mean the headband is streaming now
    let lastData = 0;
    const offLive = store.on(`live:${KEY}`, (m) => {
        const h = handlers[m.type];
        if (!h || !m.rows || !m.rows.length) return;
        if (LIVE.has(m.type)) lastData = Date.now();
        h(m.rows);
    });

    $('[data-zero]', el).addEventListener('click', async (e) => {
        await busy(e.currentTarget, () => dapi(KEY, '/api/fnirs/zero', { method: 'POST' }));
        FNIRS.forEach(([id]) => fnirs.set(id, []));
        toast('Keep still for 10 seconds: that becomes the new zero', 'ok');
    });

    // ---- the preset (which streams the headband sends)
    const presetBox = $('[data-preset]', el);
    dapi(KEY, '/api/preset', { quiet: true }).then((p) => {
        presetBox.innerHTML = Object.entries(p.presets).map(([k, text]) => `<option value="${esc(k)}" ${k === p.preset ? 'selected' : ''}>${esc(text)}${k === p.default ? ' (default)' : ''}</option>`).join('');
    }).catch(() => { presetBox.closest('.card').hidden = true; });
    presetBox.addEventListener('change', async () => {
        try { await dapi(KEY, '/api/preset', { body: { preset: presetBox.value } }); toast('Saved — the session restarts with it', 'ok'); } catch (e) { /* the toast says why */ }
    });

    // ---- connected or not, and what the worker says about the signals
    let stopFinder = null;
    const connectBox = $('[data-connect]', el), liveBox = $('[data-live]', el);
    const offHead = bindHead(el, KEY, (st, s) => {
        const on = s.live || Date.now() - lastData < 5000;
        liveBox.hidden = !on;
        if (!on && !stopFinder) { stopFinder = connectPanel(connectBox, KEY); connectBox.hidden = false; }
        if (on && stopFinder) { stopFinder(); stopFinder = null; connectBox.innerHTML = ''; connectBox.hidden = true; }
        if (!st) return;
        const sig = st.signals || {}, layout = st.layout || {};
        opticsCols = OPTICS[layout.optics ?? 16] || null;
        if (sig.ppg && sig.ppg.channel !== ppgChannel) { ppgChannel = sig.ppg.channel; ppgMean = null; }
        N('ppg', sig.ppg ? (sig.ppg.channel ? `${sig.ppg.channel} · pulse clarity ${Math.round(100 * sig.ppg.quality)}%` : 'looking for a pulse…') : 'forehead PPG');
        N('hr', sig.ppg && !sig.ppg.hr && on ? 'no clear pulse yet: keep still' : 'from the forehead pulse');
        const f = sig.fnirs;
        N('fnirs', !f ? (st.layout && !layout.optics ? 'this preset has no optics' : 'fNIRS, change from the baseline')
            : f.baseline_s < f.of_s ? `baseline ${f.baseline_s}/${f.of_s} s — keep still` : `fNIRS, ${Object.values(f.optodes || {})[0] || ''} optodes`);
        N('bat', st.firmware ? `firmware ${st.firmware}` : '');
    });
    const latest = (store.latest || {})[KEY] || {};
    if (latest.battery != null) V('bat', num(latest.battery));

    return () => {
        offLive(); offHead(); stopFinder && stopFinder();
        eeg.destroy(); bandTrend.destroy(); fnirs.destroy(); ppg.destroy(); acc.destroy();
    };
}
