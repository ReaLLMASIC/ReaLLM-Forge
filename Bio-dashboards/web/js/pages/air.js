// Air (Atmos): CO₂, particles, climate and gases, with plain-language advice.
import { icon, esc, num, dapi, seg, prefs, toast, busy, $ } from '../ui.js';
import { Plot } from '../plot.js';
import { pageHead, bindHead, connectPanel } from '../device.js';
import { METRICS, CANONICAL, metricOf, pick, co2Band, pmBand, vocBand, overall } from '../air_meta.js';

const CHARTS = [
    { id: 'co2', title: 'Carbon dioxide', metrics: ['co2'], refs: [{ y: 1000, label: '1000 ppm', color: 'var(--warn)' }] },
    { id: 'pm', title: 'Particles', metrics: ['pm1', 'pm25', 'pm4', 'pm10'], refs: [{ y: 9, label: 'PM2.5 good ≤ 9', color: 'var(--ok)' }] },
    { id: 'climate', title: 'Temperature & humidity', metrics: ['temp', 'rh'], y2: ['rh'] },
    { id: 'gas', title: 'Gases', metrics: ['voc', 'nox', 'hcho'], y2: ['hcho'] },
];
const WIFI_FORM = `<form class="card tight" data-tcp style="margin-top:14px;background:var(--surface-2)">
    <div class="row" style="margin-bottom:10px">${icon('wifi')}<b>Connect over Wi-Fi</b><span class="small muted">— for an Atmos-Sphere streaming on your network</span></div>
    <div class="row" style="align-items:flex-end"><label class="field" style="flex:2;min-width:160px"><span>Address</span><input class="input" name="host" placeholder="192.168.1.50" autocomplete="off"></label>
    <label class="field" style="width:100px"><span>Port</span><input class="input" name="port" type="number" value="8080" min="1" max="65535"></label>
    <button class="btn primary" type="submit">Connect</button></div></form>`;

export async function mount(el, { store }) {
    el.innerHTML = `${pageHead('air')}
      <div data-connect></div>
      <div class="card" data-headline style="--c:var(--air);margin-bottom:16px;display:flex;gap:16px;align-items:center">
        <span class="glyph" style="width:46px;height:46px;border-radius:14px;display:grid;place-items:center;background:color-mix(in srgb,var(--air) 15%,transparent);color:var(--air)">${icon('air')}</span>
        <div style="flex:1"><h2 data-word style="font-size:1.25rem">Waiting for a reading…</h2><div class="muted" data-advice></div></div>
        <div class="small muted" data-profile style="text-align:right"></div>
      </div>
      <div class="grid tiles" data-tiles></div>
      <div class="row" style="margin:22px 0 12px"><h2 style="font-size:1.05rem">Over time</h2><span class="spacer"></span>
        <div class="seg" data-win><button data-v="15">15 min</button><button data-v="60">1 h</button><button data-v="360">6 h</button><button data-v="1440">24 h</button></div></div>
      <div class="grid cols-2" data-charts></div>
      <div class="card" data-others-card hidden style="margin-top:16px"><div class="card-head"><h2>Other readings</h2><span class="card-sub">fields Bio-dash doesn’t know yet, recorded as they are</span></div><div class="grid tiles" data-others></div></div>
      <div class="card" style="margin-top:16px"><details class="more" style="border:0;margin:0;padding:0"><summary>Device options</summary><div class="body stack" data-options>Loading…</div></details></div>`;

    let win = prefs.get('air.win', 60);
    const seen = new Set();
    const plots = {};
    const chartsBox = $('[data-charts]', el);
    for (const c of CHARTS) {
        const card = document.createElement('div');
        card.className = 'card';
        card.hidden = true;
        card.dataset.chart = c.id;
        card.innerHTML = `<div class="card-head"><h2>${c.title}</h2></div><div class="legend" style="margin-bottom:8px">${c.metrics.map(m => `<span data-leg="${m}" hidden><i style="background:${METRICS[m].color}"></i>${METRICS[m].label}${METRICS[m].unit ? ` (${METRICS[m].unit})` : ''}</span>`).join('')}</div><div class="chart"></div>`;
        chartsBox.appendChild(card);
        plots[c.id] = new Plot(card.querySelector('.chart'), {
            series: c.metrics.map(m => ({ id: m, label: METRICS[m].label, unit: METRICS[m].unit, color: METRICS[m].color, dp: METRICS[m].dp,
                                         axis: (c.y2 || []).includes(m) ? 'y2' : 'y', width: m === 'pm25' || c.metrics.length === 1 ? 2.2 : 1.6, fill: c.metrics.length === 1 })),
            x: { window: win * 60e3 }, refs: c.refs, y: { minSpan: c.id === 'co2' ? 100 : 2, showRefs: false }, y2: { minSpan: 5 }, gap: 60e3, tick: 2000,
        });
    }
    seg($('[data-win]', el), win, (v) => { win = +v; prefs.set('air.win', win); Object.values(plots).forEach(p => p.setWindow(win * 60e3)); load(); });

    let lastTs = 0, lastRow = null, lastData = 0;
    function addRows(rows, replace) {
        if (replace) { Object.values(plots).forEach(p => p.series.forEach(s => p.set(s.id, []))); lastTs = 0; }
        for (const r of rows) {
            for (const id in r.v) seen.add(id);
            if (r.ts <= lastTs) continue;
            lastTs = r.ts;
            for (const c of CHARTS) for (const m of c.metrics) { const y = pick(r.v, m); if (y != null) plots[c.id].push(m, r.ts, y); }
            lastRow = r;
        }
        layout();
    }
    function layout() {
        const has = (m) => [...seen].some(id => metricOf(id) === m);
        for (const c of CHARTS) {
            const card = chartsBox.querySelector(`[data-chart="${c.id}"]`);
            card.hidden = !c.metrics.some(has);
            for (const m of c.metrics) card.querySelector(`[data-leg="${m}"]`).hidden = !has(m);
        }
    }
    async function load() {
        const r = await dapi('air', `/api/history?minutes=${win}`, { quiet: true }).catch(() => null);
        if (!r) return;
        addRows(r.rows, true);
        if (lastRow) render(lastRow);
    }

    function band(m, v) { return m === 'co2' ? co2Band(v) : m === 'pm25' ? pmBand(v) : m === 'voc' ? vocBand(v) : null; }
    function render(r) {
        const v = r.v || {};
        const present = CANONICAL.filter(m => pick(v, m) != null);
        $('[data-tiles]', el).innerHTML = present.map(m => {
            const x = pick(v, m), meta = METRICS[m], b = band(m, x);
            return `<div class="tile"><div class="label"><i style="width:8px;height:8px;border-radius:2px;background:${meta.color}"></i>${meta.label}</div>
              <div class="value">${num(x, meta.dp)}${meta.unit ? `<small>${meta.unit}</small>` : ''}</div>
              <div class="note">${b ? `<span class="badge ${b.tone}">${b.word}</span>` : m === 'nox' ? (x > 20 ? 'above usual' : 'usual') : meta.long !== meta.label ? esc(meta.long) : ''}</div></div>`;
        }).join('');
        const o = overall(v);
        $('[data-word]', el).textContent = o ? `Air: ${o.word}` : 'Readings are coming in';
        $('[data-advice]', el).textContent = o ? o.advice : '';
        $('[data-headline]', el).style.setProperty('--air', o && o.tone !== 'ok' ? `var(--${o.tone})` : '');
        // fields the parser doesn't recognise
        const others = Object.keys(v).filter(id => !CANONICAL.includes(metricOf(id)));
        $('[data-others-card]', el).hidden = !others.length;
        $('[data-others]', el).innerHTML = others.map(id => `<div class="tile"><div class="label">${esc(id)}</div><div class="value" style="font-size:1.3rem">${num(v[id], 2)}</div></div>`).join('');
    }
    load();

    const offLive = store.on('live:air', (m) => {
        const rows = m.rows || [];
        if (!rows.length) return;
        lastData = Date.now();
        addRows(rows);
        render(rows[rows.length - 1]);
    });

    // options: device profile + Wi-Fi
    (async () => {
        const box = $('[data-options]', el);
        try {
            const [p, t] = await Promise.all([dapi('air', '/api/profile', { quiet: true }), dapi('air', '/api/tcp-target', { quiet: true })]);
            box.innerHTML = `<label class="field"><span>Which Atmos is this?</span>
                <select class="input" data-prof><option value="auto">Work it out from the data (recommended)</option>
                ${Object.entries(p.profiles).map(([k, x]) => `<option value="${k}" ${p.config.profile === k ? 'selected' : ''}>${esc(x.label)}</option>`).join('')}</select>
                <span class="hint">Only needed if readings land in the wrong boxes (a device that sends unlabeled numbers).</span></label>
              ${WIFI_FORM}`;
            const f = box.querySelector('[data-tcp]');
            if (t.host) { f.host.value = t.host; f.port.value = t.port || 8080; }
            box.querySelector('[data-prof]').addEventListener('change', async (e) => {
                await dapi('air', '/api/profile', { body: { profile: e.target.value } });
                toast('Saved — applies within a second', 'ok');
            });
        } catch (e) { box.textContent = 'Couldn’t load the options.'; }
    })();
    el.addEventListener('submit', async (e) => {
        const f = e.target.closest('form[data-tcp]');
        if (!f) return;
        e.preventDefault();
        const host = f.host.value.trim(), port = +f.port.value || 8080;
        if (!host) { f.host.focus(); return; }
        await busy(f.querySelector('button'), () => dapi('air', '/api/connect-tcp', { body: { host, port } }));
        toast(`Connecting to ${host}:${port}…`, 'ok');
    });

    let stopFinder = null;
    const connectBox = $('[data-connect]', el);
    const offHead = bindHead(el, 'air', (st, s) => {
        const live = s.live || Date.now() - lastData < 15000;
        $('[data-profile]', el).textContent = st && st.profile && st.profile !== 'auto' ? st.profile.toUpperCase() : '';
        if (!live && !stopFinder) { stopFinder = connectPanel(connectBox, 'air', WIFI_FORM); connectBox.hidden = false; }
        if (live && stopFinder) { stopFinder(); stopFinder = null; connectBox.innerHTML = ''; connectBox.hidden = true; }
        $('[data-headline]', el).hidden = !live;
    });
    const reload = setInterval(load, 10 * 60e3);
    return () => { offLive(); offHead(); stopFinder && stopFinder(); clearInterval(reload); Object.values(plots).forEach(p => p.destroy()); };
}
