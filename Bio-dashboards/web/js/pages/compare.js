// Compare: the people being followed, side by side. One row and one line per person for heart rate,
// HRV and blood oxygen, and their ECGs under each other. Shown when two or three people are set up
// (Settings → People). People are told apart by colour and by line style.
import { icon, esc, num, pill, dapi, seg, prefs, stateOf, PERSON_STYLE, $ } from '../ui.js';
import { Plot } from '../plot.js';

const ECG_WINDOW = 6000;
// A strap sends its ECG in bursts (about one every 0.5 s), so a strip whose right edge is "now" would have a
// growing gap at the end that jumps closed at each burst: the trace wiggles. Like the Heart page's monitor,
// each strip is drawn PLAY_DELAY_MS behind the strap's own clock instead, so there's always data to the edge.
const PLAY_DELAY_MS = 900;
const mark = (pid) => `<span class="pmark ${PERSON_STYLE[pid].mark}" style="--c:${PERSON_STYLE[pid].c}"></span>`;

export async function mount(el, { store }) {
    const people = store.people().map(p => ({
        id: p.id, name: store.personName(p.id), style: PERSON_STYLE[p.id],
        heart: p.devices.heart.enabled ? p.devices.heart.key : null,
        oxygen: p.devices.oxygen.enabled ? p.devices.oxygen.key : null,
    }));
    const hearts = people.filter(p => p.heart), rings = people.filter(p => p.oxygen);
    const cell = (p, what, unit) => `<td><span class="num big-num" data-v="${p.id}.${what}">--</span><small>${unit}</small></td>`;
    const chartCard = (title, sub, attr, who) => `<div class="card"><div class="card-head"><h2>${title}</h2><span class="card-sub">${sub}</span></div>
        <div class="legend" style="margin-bottom:8px">${who.map(p => `<span>${mark(p.id)}${esc(p.name)}</span>`).join('')}</div><div class="chart" ${attr}></div></div>`;

    el.innerHTML = `
      <div class="page-head" style="--c:var(--brand-2)"><div class="titles"><h1><span class="glyph">${icon('people')}</span>Compare</h1>
          <p class="sub">${people.map(p => esc(p.name)).join(', ')} — live, side by side</p></div>
        <div class="actions"><div class="seg" data-win><button data-v="5">5 min</button><button data-v="15">15 min</button><button data-v="60">1 hour</button></div>
          <a class="btn" href="#/settings">${icon('people')}<span>People</span></a></div></div>
      <div class="card flush"><div class="table-wrap"><table class="compare">
        <thead><tr><th></th><th>Heart rate</th><th>HRV</th><th>Blood oxygen</th><th>Pulse (ring)</th><th>Strap</th><th>Ring</th></tr></thead>
        <tbody>${people.map(p => `<tr>
          <th scope="row">${mark(p.id)}<span>${esc(p.name)}</span></th>
          ${cell(p, 'hr', 'bpm')}${cell(p, 'rmssd', 'ms')}${cell(p, 'spo2', '%')}${cell(p, 'pulse', 'bpm')}
          <td>${p.heart ? `<a href="#/heart/${p.id}" data-pill="${p.heart}"></a>` : '<span class="faint">off</span>'}</td>
          <td>${p.oxygen ? `<a href="#/oxygen/${p.id}" data-pill="${p.oxygen}"></a>` : '<span class="faint">off</span>'}</td></tr>`).join('')}</tbody></table></div></div>
      <div class="grid cols-2" style="margin-top:16px">
        ${hearts.length ? chartCard('Heart rate', 'bpm', 'data-hr', hearts) : ''}
        ${rings.length ? chartCard('Blood oxygen', '% SpO₂', 'data-spo2', rings) : ''}
        ${hearts.length ? chartCard('Heart-rate variability', 'RMSSD over 20 beats, ms', 'data-rmssd', hearts) : ''}
        ${rings.length ? chartCard('Pulse from the ring', 'bpm', 'data-pulse', rings) : ''}
      </div>
      ${hearts.length ? `<div class="card" style="margin-top:16px"><div class="card-head"><h2>ECG</h2><span class="card-sub">last ${ECG_WINDOW / 1000} seconds, mV</span></div>
        ${hearts.map(p => `<div class="ecg-row"><div class="who">${mark(p.id)}<span>${esc(p.name)}</span></div><div class="chart strip" data-ecg="${p.id}"></div></div>`).join('')}</div>` : ''}
      <p class="small faint" style="margin-top:18px">Each person’s strap and ring record to their own files, as always. Who wears what is set where you connect it: open Heart or Oxygen, pick the person, then <b>Device</b>.</p>`;

    let win = prefs.get('compare.win', 15);
    const plots = [];
    const trend = (attr, who, unit, extra = {}) => {
        const box = $(`[${attr}]`, el);
        if (!box) return null;
        const p = new Plot(box, { series: who.map(x => ({ id: x.id, label: x.name, color: x.style.c, dash: x.style.dash, unit, width: 2, ...extra.series })),
            x: { window: win * 60e3 }, y: extra.y || { minSpan: 10 }, refs: extra.refs, gap: extra.gap, empty: 'No readings in this window' });
        plots.push(p);
        return p;
    };
    const hr = trend('data-hr', hearts, 'bpm');
    const rmssd = trend('data-rmssd', hearts, 'ms', { series: { dp: 1 }, y: { minSpan: 10, zero: true } });
    const spo2 = trend('data-spo2', rings, '%', { y: { ceil: 101, minSpan: 6 }, refs: [{ y: 90, label: '90 %', color: 'var(--warn)' }], gap: 30e3 });
    const pulse = trend('data-pulse', rings, 'bpm', { gap: 30e3 });
    const clock = {};                          // per strap: how far the browser's clock is ahead of the newest sample
    const ecg = Object.fromEntries(hearts.map(p => {
        const follow = () => clock[p.id] == null ? Date.now() : Date.now() - clock[p.id] - PLAY_DELAY_MS;
        const plot = new Plot($(`[data-ecg="${p.id}"]`, el), { series: [{ id: 'ecg', label: p.name, color: p.style.c, unit: 'mV', dp: 2, width: 1.4, raw: true }],
            x: { window: ECG_WINDOW, hideAxis: true, follow }, y: { minSpan: 1 }, tick: 40, gap: 400, empty: 'Waiting for the ECG' });
        plots.push(plot);
        return [p.id, plot];
    }));
    seg($('[data-win]', el), win, (v) => { win = +v; prefs.set('compare.win', win); for (const p of [hr, rmssd, spo2, pulse]) p && p.setWindow(win * 60e3); load(); });

    async function load() {
        const m = Math.max(win, 15), jobs = [];
        const pts = (list) => (list || []).map(q => [q.x, q.y]);
        for (const p of hearts) {
            jobs.push(dapi(p.heart, `/api/history?metric=hr&minutes=${m}`, { quiet: true }).then(r => hr.set(p.id, pts(r.hr))));
            jobs.push(dapi(p.heart, `/api/history?metric=rmssd&minutes=${m}`, { quiet: true }).then(r => rmssd.set(p.id, pts(r.rmssd))));
        }
        for (const p of rings) {
            jobs.push(dapi(p.oxygen, `/api/history?minutes=${m}`, { quiet: true }).then(r => { spo2.set(p.id, pts(r.spo2)); pulse.set(p.id, pts(r.hr)); }));
        }
        await Promise.allSettled(jobs);
    }
    load();
    const reload = setInterval(load, 5 * 60e3);

    const V = (pid, what, text) => { const n = el.querySelector(`[data-v="${pid}.${what}"]`); if (n) n.textContent = text; };
    const offs = [];
    for (const p of hearts) {
        offs.push(store.on(`live:${p.heart}`, (msg) => {
            if (msg.type !== 'ecg' || !msg.rows.length) return;
            // the smallest lag seen is the true clock difference (later bursts only add network delay);
            // drift slowly toward the current one so a clock change is followed too
            const lag = Date.now() - msg.rows[msg.rows.length - 1][0];
            clock[p.id] = clock[p.id] == null || lag < clock[p.id] ? lag : clock[p.id] + (lag - clock[p.id]) * 0.02;
            for (const [ts, mv] of msg.rows) ecg[p.id].push('ecg', ts, mv);
        }));
    }
    for (const p of rings) {
        offs.push(store.on(`live:${p.oxygen}`, (d) => {
            if (d.status === 'Connected' && d.spo2 > 0) { spo2.push(p.id, d.ts, d.spo2); pulse.push(p.id, d.ts, d.hr); }
        }));
    }
    store.subscribe([...hearts.map(p => p.heart), ...rings.map(p => p.oxygen)]);

    // numbers and pills from the once-a-second summary (the same figures Home shows)
    function paint() {
        const L = store.latest || {};
        for (const p of people) {
            const h = (p.heart && L[p.heart]) || {}, o = (p.oxygen && L[p.oxygen]) || {};
            const hLive = p.heart && stateOf(store.status[p.heart]).live, oLive = p.oxygen && stateOf(store.status[p.oxygen]).live && o.status === 'Connected' && o.spo2 > 0;
            V(p.id, 'hr', hLive && h.hr ? num(h.hr) : '--');
            V(p.id, 'rmssd', hLive && h.rmssd != null ? num(h.rmssd) : '--');
            V(p.id, 'spo2', oLive ? num(o.spo2) : '--');
            V(p.id, 'pulse', oLive && o.hr ? num(o.hr) : '--');
            if (hLive && h.hr && h.ts) { hr.push(p.id, h.ts, h.hr); if (h.rmssd != null) rmssd.push(p.id, h.ts, h.rmssd); }
        }
        el.querySelectorAll('[data-pill]').forEach(a => { a.innerHTML = pill(store.status[a.dataset.pill]); });
    }
    offs.push(store.on('latest', paint), store.on('status', paint));
    paint();

    return () => { offs.forEach(off => off()); clearInterval(reload); plots.forEach(p => p.destroy()); };
}
