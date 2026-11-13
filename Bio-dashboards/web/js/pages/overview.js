// Home: every device at a glance. A card per device (per person, for the worn ones) with its newest
// reading, a short trend and, when it isn't connected, what to do about it.
import { DEVICES, DEVICE_KEYS, dev, kindOf, personOf, icon, esc, num, pill, stateOf, dapi, ago } from '../ui.js';
import { sparkline } from '../plot.js';
import { pick, co2Band, pmBand } from '../air_meta.js';

const SPARK_MS = 30 * 60e3;

function greeting() {
    const h = new Date().getHours();
    return h < 5 ? 'Good night' : h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
}

export async function mount(el, { store }) {
    // the cards: every instance that is on, in the usual order (each person's strap, each person's ring, ...)
    const on = DEVICE_KEYS.flatMap(kind => Object.keys(store.app.instances).filter(k => kindOf(k) === kind && store.enabled(k)));
    const off = DEVICE_KEYS.filter(k => !store.kindOn(k));
    const whose = (k) => store.several() && store.app.devices[kindOf(k)].personal ? store.personName(personOf(k)) : '';
    el.innerHTML = `
      <div class="hello"><h1>${greeting()}</h1><p data-summary>${new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' })}</p></div>
      ${on.length ? `<div class="grid cols-2" data-cards>${on.map(k => card(k, whose(k))).join('')}</div>` : `
        <div class="card empty-state">${icon('bluetooth')}<h3>No devices yet</h3>
          <p>Tell Bio-dash which devices you have, and it will find them.</p>
          <p style="margin-top:16px"><a class="btn primary" href="#/setup">Set up devices</a></p></div>`}
      ${off.length && on.length ? `<div class="section-title">More devices</div><div class="grid cols-3">${off.map(k => `
        <a class="card tight add-card" href="#/settings" style="--c:${DEVICES[k].c}">${icon('plus')}Add ${esc(DEVICES[k].product)}</a>`).join('')}</div>` : ''}
      ${store.several() ? `<div class="section-title">Together</div>
      <a class="card" href="#/compare" style="display:flex;gap:14px;align-items:center">
        <span class="glyph" style="width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:var(--surface-2);color:var(--brand-2)">${icon('people')}</span>
        <span style="flex:1"><b>Compare</b><br><span class="muted small">${esc(store.people().map(p => store.personName(p.id)).join(', '))} side by side: heart rate, HRV, oxygen and ECG.</span></span>${icon('chevron')}</a>` : ''}
      <div class="section-title">Look back</div>
      <a class="card" href="#/analysis" style="display:flex;gap:14px;align-items:center">
        <span class="glyph" style="width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:var(--surface-2);color:var(--brand-2)">${icon('analysis')}</span>
        <span style="flex:1"><b>Analysis</b><br><span class="muted small">Days and nights side by side, night reports, and CSV export.</span></span>${icon('chevron')}</a>`;
    el.querySelectorAll('a.card svg:last-child').forEach(s => { s.style.width = '20px'; s.style.color = 'var(--faint)'; });

    // trends for the sparklines: history once, then the once-a-second "latest"
    const spark = Object.fromEntries(on.map(k => [k, []]));
    const load = async () => {
        const jobs = [];
        for (const k of on) {
            if (kindOf(k) === 'heart') jobs.push(dapi(k, '/api/history?metric=hr&minutes=30', { quiet: true }).then(r => { spark[k] = (r.hr || []).map(p => [p.x, p.y]); }));
            if (kindOf(k) === 'oxygen') jobs.push(dapi(k, '/api/history?minutes=30', { quiet: true }).then(r => { spark[k] = (r.spo2 || []).map(p => [p.x, p.y]); }));
            if (kindOf(k) === 'brain') jobs.push(dapi(k, '/api/history?metric=alpha&minutes=15', { quiet: true }).then(r => { spark[k] = (r.alpha || []).map(p => [p.x, p.y]); }));
        }
        if (on.includes('air')) jobs.push(dapi('air', '/api/history?minutes=30', { quiet: true }).then(r => { spark.air = (r.rows || []).map(x => [x.ts, pick(x.v, 'co2') ?? pick(x.v, 'pm25')]).filter(p => p[1] != null); }));
        if (on.includes('water')) {
            const since = Math.ceil((Date.now() - new Date().setHours(0, 0, 0, 0)) / 60000) + 1;
            jobs.push(dapi('water', `/api/history?metric=intake&minutes=${since}`, { quiet: true }).then(r => { spark.water = (r.intake || []).map(p => [p.x, p.y]); }));
        }
        await Promise.allSettled(jobs);
        paint();
    };

    function paint() {
        const L = store.latest || {};
        let live = 0;
        for (const k of on) {
            const c = el.querySelector(`[data-card="${k}"]`);
            if (!c) continue;
            const st = store.status[k], s = stateOf(st);
            if (s.live) live++;
            c.querySelector('[data-pill]').innerHTML = pill(st);
            const v = L[k] || {};
            const kind = kindOf(k);
            c.querySelector('[data-reading]').innerHTML = reading(kind, v, s);
            c.querySelector('[data-hint]').innerHTML = hint(k, st, s);
            const cv = c.querySelector('canvas');
            if (kind === 'heart' && v.hr && s.live) addPoint(spark[k], v.ts || Date.now(), v.hr);
            if (kind === 'oxygen' && v.spo2 && s.live) addPoint(spark[k], v.ts || Date.now(), v.spo2);
            if (kind === 'brain' && v.alpha != null && s.live) addPoint(spark[k], v.ts || Date.now(), v.alpha);
            if (k === 'air' && v.v && s.live) addPoint(spark.air, v.ts, pick(v.v, 'co2') ?? pick(v.v, 'pm25'));
            sparkline(cv, spark[k], dev(k).c, { window: k === 'water' ? null : SPARK_MS, min: k === 'water' ? 0 : undefined });
        }
        const sum = el.querySelector('[data-summary]');
        const date = new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
        sum.textContent = on.length ? `${date} · ${live ? `${live} of ${on.length} device${on.length > 1 ? 's' : ''} recording` : 'Nothing connected right now'}` : date;
    }

    load();
    const unsubs = [store.on('latest', paint), store.on('status', paint)];
    const reload = setInterval(load, 5 * 60e3);
    paint();
    return () => { unsubs.forEach(u => u()); clearInterval(reload); };
}

function addPoint(arr, x, y) {
    if (y == null || (arr.length && x <= arr[arr.length - 1][0])) return;
    arr.push([x, y]);
    while (arr.length && arr[0][0] < Date.now() - SPARK_MS * 1.2) arr.shift();
}

function card(k, whose) {
    const d = dev(k);
    return `<a class="card dev-card" href="#/${kindOf(k)}${whose ? '/' + personOf(k) : ''}" data-card="${k}" style="--c:${d.c}">
      <div class="top"><span class="glyph">${icon(d.icon)}</span><div><div class="name">${d.label}${whose ? ` <span class="whose">· ${esc(whose)}</span>` : ''}</div><div class="product">${esc(d.product)}</div></div>
        <span data-pill></span></div>
      <div class="reading" data-reading></div>
      <div class="hint" data-hint></div>
      <canvas class="spark" aria-hidden="true"></canvas></a>`;
}

const side = (label, value) => `<div>${label}<b>${value}</b></div>`;

function reading(k, v, s) {
    const fresh = s.live;
    if (k === 'heart') {
        return `<div class="main">${fresh && v.hr ? num(v.hr) : '--'}<small>bpm</small></div>
          <div class="side">${side('HRV', fresh && v.rmssd != null ? num(v.rmssd) + ' ms' : '--')}${side('R-R', fresh && v.rr ? num(v.rr) + ' ms' : '--')}</div>`;
    }
    if (k === 'oxygen') {
        const ok = fresh && v.status === 'Connected' && v.spo2 > 0;
        return `<div class="main">${ok ? num(v.spo2) : '--'}<small>% SpO₂</small></div>
          <div class="side">${side('Pulse', ok && v.hr ? num(v.hr) + ' bpm' : '--')}${side('Battery', v.battery ? num(v.battery) + '%' : '--')}</div>`;
    }
    if (k === 'brain') {
        const good = (v.contact || []).filter(c => c === 1 || c === 2).length;
        return `<div class="main">${fresh && v.alpha != null ? num(v.alpha) : '--'}<small>% alpha</small></div>
          <div class="side">${side('Pulse', fresh && v.hr ? num(v.hr) + ' bpm' : '--')}${side('Contact', fresh && v.contact ? `${good} of 4` : '--')}${side('Battery', v.battery != null ? num(v.battery) + '%' : '--')}</div>`;
    }
    if (k === 'air') {
        const vals = fresh ? v.v || {} : {};
        const co2 = pick(vals, 'co2'), pm = pick(vals, 'pm25');
        const band = co2 != null ? co2Band(co2) : pmBand(pm);
        const main = co2 != null ? `${num(co2)}<small>ppm CO₂</small>` : pm != null ? `${num(pm, 1)}<small>µg/m³ PM2.5</small>` : '--';
        return `<div class="main">${main}</div>
          <div class="side">${band ? side('Air', `<span class="badge ${band.tone}">${band.word}</span>`) : ''}${side('Temp', pick(vals, 'temp') != null ? num(pick(vals, 'temp'), 1) + '°' : '--')}${side('Humidity', pick(vals, 'rh') != null ? num(pick(vals, 'rh')) + '%' : '--')}</div>`;
    }
    if (k === 'water') {
        const today = v.today_ml, goal = v.goal_ml;
        const pct = goal ? Math.min(100, Math.round(100 * (today || 0) / goal)) : 0;
        return `<div class="main">${today != null ? num(today) : '--'}<small>${goal ? `of ${num(goal)} mL` : 'mL today'}</small></div>
          <div class="side">${side('Goal', goal ? pct + '%' : '--')}${side('In bottle', v.fill_ml != null && fresh ? num(v.fill_ml) + ' mL' : '--')}${side('Last sip', v.last_sip ? ago(v.last_sip.ts) : '--')}</div>`;
    }
    return '';
}

function hint(k, st, s) {
    if (!st || s.live) return '';
    if (st.state === 'connecting') return `Connecting to ${esc(st.name || 'your device')}…`;
    if ((st.worker || {}).state === 'restarting') return 'The Bluetooth helper stopped and is starting again. Open the device for details.';
    const ld = st.last_device || {};
    if (st.held) return 'Disconnected. Tap to pick a device.';
    if (ld.address && ld.auto !== false) return `Waiting for ${esc(ld.name || 'your device')} — ${esc(dev(k).find)}`;
    return `Tap to connect the ${esc(dev(k).product)}.`;
}
