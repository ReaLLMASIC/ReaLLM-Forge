// Analysis: look back over any day or range — what was recorded, the numbers that matter,
// night reports, any signal on a chart, and CSV export. Reads the recordings; no Bluetooth.
import { icon, esc, num, api, seg, prefs, isoDay, dayLabel, duration, hm, toast, $ } from '../ui.js';
import { Plot } from '../plot.js';

const A = (p) => `/api/d/analysis${p}`;
const FAM = { heart: 'var(--heart)', oxygen: 'var(--oxygen)', air: 'var(--air)', water: 'var(--water)', other: 'var(--muted)' };
const FAM_LABEL = { heart: 'Heart', oxygen: 'Oxygen', air: 'Air', water: 'Water', other: 'Other' };
const PALETTE = ['#fb7185', '#a78bfa', '#34d399', '#38bdf8', '#f59e0b', '#22d3ee', '#e879f9', '#a3e635', '#fb923c', '#818cf8'];
const STEP_LABEL = (ms) => ms < 60e3 ? `${ms / 1000} s` : ms < 3600e3 ? `${ms / 60e3} min` : `${ms / 3600e3} h`;

const addDays = (iso, n) => { const [y, m, d] = iso.split('-').map(Number); return isoDay(new Date(y, m - 1, d + n)); };

export async function mount(el, { params, store }) {
    const today = isoDay(new Date());
    let start = params[0] || today, end = params[1] || params[0] || today;
    let days = {}, signals = [], picked = new Set(prefs.get('analysis.picked', ['heart.hr', 'oxygen.spo2'])), step = 'auto', showRaw = false;
    let plots = [];

    el.innerHTML = `
      <div class="page-head"><div class="titles"><h1><span class="glyph" style="--c:var(--brand-2)">${icon('analysis')}</span>Analysis</h1>
        <p class="sub">Look back over any day or range. Nothing here needs a device connected.</p></div></div>
      <div class="card" style="margin-bottom:16px">
        <div class="row" style="justify-content:space-between">
          <div class="seg" data-preset><button data-v="today">Today</button><button data-v="yesterday">Yesterday</button><button data-v="7">7 days</button><button data-v="30">30 days</button><button data-v="custom">Pick…</button></div>
          <div class="row tight"><button class="icon-btn" data-prev aria-label="Earlier">${icon('chevron').replace('<svg', '<svg style="transform:rotate(180deg)"')}</button>
            <b data-range-label style="min-width:150px;text-align:center"></b><button class="icon-btn" data-next aria-label="Later">${icon('chevron')}</button></div>
        </div>
        <form class="row" data-custom hidden style="margin-top:12px;align-items:flex-end">
          <label class="field"><span>From</span><input class="input" type="date" name="start"></label>
          <label class="field"><span>To</span><input class="input" type="date" name="end"></label>
          <button class="btn primary" type="submit">Show</button></form>
        <div class="row tight" data-strip style="margin-top:14px;overflow-x:auto;flex-wrap:nowrap;padding-bottom:4px"></div>
        <div class="row" data-units hidden style="margin-top:12px;align-items:flex-end"></div>
      </div>
      <div data-body></div>`;

    const body = $('[data-body]', el);

    function setRange(s, e, push = true) {
        start = s; end = e < s ? s : e;
        if (push) history.replaceState(null, '', `#/analysis/${start}${end !== start ? '/' + end : ''}`);
        paintRangeBar();
        load();
    }
    function preset() {
        if (start === end && start === today) return 'today';
        if (start === end && start === addDays(today, -1)) return 'yesterday';
        if (end === today && start === addDays(today, -6)) return '7';
        if (end === today && start === addDays(today, -29)) return '30';
        return 'custom';
    }
    const paintPreset = seg($('[data-preset]', el), preset(), (v) => {
        const f = $('[data-custom]', el);
        f.dataset.open = v === 'custom' ? '1' : '';
        f.hidden = v !== 'custom';
        if (v === 'today') setRange(today, today);
        else if (v === 'yesterday') setRange(addDays(today, -1), addDays(today, -1));
        else if (v === '7') setRange(addDays(today, -6), today);
        else if (v === '30') setRange(addDays(today, -29), today);
        else { f.start.value = start; f.end.value = end; f.start.focus(); }
    });
    $('[data-custom]', el).addEventListener('submit', (e) => { e.preventDefault(); const f = e.target; if (f.start.value) setRange(f.start.value, f.end.value || f.start.value); });
    const shift = (dir) => { const n = (Date.parse(end) - Date.parse(start)) / 86400e3 + 1; setRange(addDays(start, dir * n), addDays(end, dir * n)); };
    $('[data-prev]', el).addEventListener('click', () => shift(-1));
    $('[data-next]', el).addEventListener('click', () => shift(1));

    function paintRangeBar() {
        paintPreset(preset());
        $('[data-custom]', el).hidden = !$('[data-custom]', el).dataset.open;
        $('[data-range-label]', el).textContent = start === end ? dayLabel(start, { weekday: 'short', month: 'short', day: 'numeric' })
            : `${dayLabel(start, { month: 'short', day: 'numeric' })} – ${dayLabel(end, { month: 'short', day: 'numeric' })}`;
        $('[data-next]', el).disabled = end >= today;
        // the last 21 days, each with a dot per kind of recording
        const strip = [];
        for (let i = 20; i >= 0; i--) {
            const d = addDays(today, -i), fams = days[d] || [], inRange = d >= start && d <= end;
            strip.push(`<button class="chip" data-day="${d}" aria-pressed="${inRange}" style="flex:none;flex-direction:column;gap:3px;padding:6px 8px;border-radius:10px;min-width:44px;${fams.length ? '' : 'opacity:.55'}">
              <span class="tiny" style="font-weight:700">${dayLabel(d, { weekday: 'narrow' })}</span><span>${+d.slice(8)}</span>
              <span style="display:flex;gap:2px;height:6px">${fams.filter(f => FAM[f]).map(f => `<i style="width:6px;height:6px;border-radius:50%;background:${FAM[f]};opacity:1"></i>`).join('')}</span></button>`);
        }
        $('[data-strip]', el).innerHTML = strip.join('');
        const cur = $('[data-strip] [aria-pressed="true"]', el);
        cur && cur.scrollIntoView({ block: 'nearest', inline: 'center' });
    }
    $('[data-strip]', el).addEventListener('click', (e) => { const b = e.target.closest('[data-day]'); if (b) setRange(b.dataset.day, b.dataset.day); });

    // ------------------------------------------------------------------ whose strap and ring
    // Recordings are kept per device. When the range holds more than one strap or ring (several people),
    // one of each is shown rather than everyone's merged into one line; the first person's by default.
    let unitsQ = '', unitsFor = '';
    const chosenUnit = {};
    async function pickUnits() {
        const box = $('[data-units]', el);
        const found = await api(A(`/api/units?start=${start}&end=${end}`), { quiet: true });
        const owner = (kind, unit) => { const p = store.people().find(x => (x.devices[kind] || {}).unit === unit); return p && store.several() ? store.personName(p.id) : ''; };
        const kinds = ['heart', 'oxygen'].filter(k => (found[k] || []).length > 1);
        for (const k of kinds) {
            const mine = (store.people()[0].devices[k] || {}).unit;
            if (!found[k].includes(chosenUnit[k])) chosenUnit[k] = found[k].includes(mine) ? mine : found[k][0];
        }
        box.hidden = !kinds.length;
        const sig = JSON.stringify([found, kinds]);
        if (sig !== unitsFor) {
            unitsFor = sig;
            box.innerHTML = kinds.map(k => `<label class="field"><span>${k === 'heart' ? 'Heart strap' : 'Oxygen ring'}</span><select class="input" data-unit="${k}">
              ${found[k].map(u => `<option value="${esc(u)}" ${u === chosenUnit[k] ? 'selected' : ''}>${esc(u)}${owner(k, u) ? ` — ${esc(owner(k, u))}` : ''}</option>`).join('')}</select></label>`).join('')
              + (kinds.length ? '<span class="small muted" style="flex:1;min-width:180px">More than one was recorded in this range. Analysis shows one of each, so two people aren’t mixed into one line.</span>' : '');
            box.querySelectorAll('[data-unit]').forEach(sel => sel.addEventListener('change', () => { chosenUnit[sel.dataset.unit] = sel.value; load(); }));
        }
        // the other family keeps its only unit (if any), so naming one family's unit doesn't drop the other's
        const all = ['heart', 'oxygen'].flatMap(k => kinds.includes(k) ? [chosenUnit[k]] : found[k] || []);
        unitsQ = kinds.length ? `&units=${encodeURIComponent(all.join('|'))}` : '';
    }

    // ------------------------------------------------------------------ load + render
    let loadSeq = 0;
    async function load() {
        const seq = ++loadSeq;
        plots.forEach(p => p.destroy()); plots = [];
        body.innerHTML = `<div class="grid cols-2">${'<div class="card skeleton" style="height:150px"></div>'.repeat(4)}</div>`;
        try { await pickUnits(); } catch (e) { unitsQ = ''; }
        if (seq !== loadSeq) return;
        const q = `start=${start}&end=${end}${unitsQ}`;
        let sum, sig;
        try {
            [sum, sig] = await Promise.all([api(A(`/api/summary?${q}`)), api(A(`/api/signals?${q}`))]);
        } catch (e) {
            body.innerHTML = `<div class="card empty-state">${icon('warn')}<h3>Couldn’t read the recordings</h3><p>${esc(e.message)}</p></div>`;
            return;
        }
        if (seq !== loadSeq) return;
        signals = sig.signals;
        const anything = Object.keys(sum.timeline).length || Object.keys(sum.cards).length;
        if (!anything) {
            body.innerHTML = `<div class="card empty-state">${icon('calendar')}<h3>Nothing recorded ${start === end ? 'that day' : 'in that range'}</h3>
              <p>Pick a day with dots in the strip above.</p></div>`;
            return;
        }
        body.innerHTML = `
          <div class="grid cols-2" data-cards>${cards(sum)}</div>
          ${sum.nights.length ? `<div class="section-title">Nights</div><div class="card"><div class="list">${sum.nights.map(n => `
            <div class="item" style="--c:var(--oxygen)">${icon('moon')}<div class="grow"><div class="title">Night of ${dayLabel(n.night)}</div><div class="desc">${duration(n.minutes)} of oxygen readings</div></div>
              <a class="btn sm" href="${A('/night/' + n.night)}" target="_blank" rel="noopener">${icon('external')}Night report</a></div>`).join('')}</div></div>` : ''}
          <div class="section-title">When things were recording</div>
          <div class="card">${timeline(sum)}</div>
          <div class="section-title">Explore</div>
          <div class="card">
            <div class="card-head"><h2>Signals</h2><span class="card-sub">tap to add to the charts</span><span class="spacer"></span>
              <label class="switch small"><input type="checkbox" data-raw ${showRaw ? 'checked' : ''}><span class="text"><b style="font-weight:600">Every column</b></span></label></div>
            <div data-chips></div>
            <div class="row" style="margin-top:14px">
              <label class="field" style="flex-direction:row;align-items:center;gap:8px"><span>Detail</span><select class="input" data-step style="width:auto"><option value="auto">Automatic</option>${sig.steps.map(s => `<option value="${s}">${STEP_LABEL(s)}</option>`).join('')}</select></label>
              <span class="spacer"></span>
              <button class="btn" data-export>${icon('download')}Download CSV</button>
              <button class="btn" data-export-mcint title="Whole numbers with a fixed range per column, for ReaLLM-Forge data/csv_mc_int">${icon('download')}For ReaLLM-Forge</button></div>
            <div class="stack" data-charts style="margin-top:16px"></div>
          </div>
          ${sum.per_day.length ? `<div class="section-title">Day by day</div><div class="card"><div class="table-wrap">${perDay(sum)}</div></div>` : ''}`;
        $('[data-step]', body).value = step;
        $('[data-step]', body).addEventListener('change', (e) => { step = e.target.value; drawCharts(); });
        $('[data-raw]', body).addEventListener('change', (e) => { showRaw = e.target.checked; paintChips(); });
        $('[data-export]', body).addEventListener('click', () => exportCsv(false));
        $('[data-export-mcint]', body).addEventListener('click', () => exportCsv(true));
        $('[data-chips]', body).addEventListener('click', (e) => {
            const c = e.target.closest('[data-sig]');
            if (!c) return;
            const id = c.dataset.sig;
            picked.has(id) ? picked.delete(id) : picked.add(id);
            prefs.set('analysis.picked', [...picked]);
            paintChips(); drawCharts();
        });
        paintChips();
        drawCharts();
    }

    function paintChips() {
        const box = $('[data-chips]', body);
        const groups = {};
        for (const s of signals) {
            if (s.id.startsWith('raw.') && !showRaw && !picked.has(s.id)) continue;
            (groups[s.family] = groups[s.family] || []).push(s);
        }
        box.innerHTML = Object.entries(groups).map(([f, list]) => `<div style="margin-bottom:10px"><div class="tiny muted" style="font-weight:700;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em">${FAM_LABEL[f] || f}</div>
          <div class="chips">${list.map(s => `<button class="chip" data-sig="${esc(s.id)}" aria-pressed="${picked.has(s.id)}" style="--c:${FAM[s.family] || 'var(--muted)'}"><i></i>${esc(s.label)}</button>`).join('')}</div></div>`).join('')
            || '<p class="muted small">No signals in this range.</p>';
    }

    let chartSeq = 0;
    async function drawCharts() {
        const box = $('[data-charts]', body), seq = ++chartSeq;
        plots.forEach(p => p.destroy()); plots = [];
        const ids = [...picked].filter(id => signals.some(s => s.id === id));
        if (!ids.length) { box.innerHTML = '<p class="muted small">Pick a signal above to chart it.</p>'; return; }
        box.innerHTML = '<div class="skeleton" style="height:220px"></div>';
        let r;
        try { r = await api(A(`/api/series?start=${start}&end=${end}&ids=${encodeURIComponent(ids.join(','))}&step=${step}${unitsQ}`)); }
        catch (e) { box.innerHTML = `<p class="muted small">${esc(e.message)}</p>`; return; }
        if (seq !== chartSeq) return;
        // one chart per unit, so things measured alike share an axis
        const byUnit = {};
        for (const id of ids) {
            const s = signals.find(x => x.id === id), m = r.meta[id];
            if (!s || !m) continue;
            (byUnit[s.unit || s.label] = byUnit[s.unit || s.label] || []).push(s);
        }
        box.innerHTML = Object.entries(byUnit).map(([u, list], i) => `<div><div class="legend" style="margin-bottom:6px">${list.map((s, j) => `<span><i style="background:${colorFor(s, i, j)}"></i>${esc(s.label)}${s.unit ? ` (${esc(s.unit)})` : ''}</span>`).join('')}</div><div class="chart" data-unit="${i}"></div></div>`).join('')
            + `<p class="tiny faint">${STEP_LABEL(r.step)} per point${r.missing.length ? ` · not in this range: ${esc(r.missing.join(', '))}` : ''}. The shaded band is the lowest–highest within each point.</p>`;
        const xs = Array.from({ length: r.n }, (_, k) => r.first + k * r.step);
        Object.entries(byUnit).forEach(([u, list], i) => {
            const p = new Plot(box.querySelector(`[data-unit="${i}"]`), {
                series: list.map((s, j) => ({ id: s.id, label: s.label, unit: s.unit, color: colorFor(s, i, j), dp: s.unit === 'mV' ? 2 : 1,
                                              type: s.kind === 'events' ? 'bar' : 'line', width: 1.8 })),
                x: { min: Date.parse(start + 'T00:00:00'), max: Date.parse(end + 'T00:00:00') + 86400e3, step: list.some(s => s.kind === 'events') ? r.step : undefined },
                gap: r.step * 1.5, empty: 'Nothing recorded',
            });
            for (const s of list) {
                const d = r.series[s.id];
                if (!d) continue;
                const pts = [], lo = [], hi = [];
                d.mean.forEach((v, k) => { if (v != null) { pts.push([xs[k], v]); lo.push(d.min ? d.min[k] : null); hi.push(d.max ? d.max[k] : null); } });
                p.set(s.id, pts, d.min ? { lo, hi } : null);
            }
            plots.push(p);
        });
    }
    function colorFor(s, i, j) {
        // the family colour for the first signal of a family, then distinct colours
        return j === 0 ? (FAM[s.family] || PALETTE[i % PALETTE.length]) : PALETTE[(i * 3 + j) % PALETTE.length];
    }

    function exportCsv(mc) {
        const ids = [...picked].filter(id => signals.some(s => s.id === id));
        if (!ids.length) { toast('Pick at least one signal first', 'bad'); return; }
        const st = step === 'auto' ? (mc ? '10s' : '1min') : (+step >= 60e3 ? `${+step / 60e3}min` : `${+step / 1000}s`);
        const q = `start=${start}&end=${end}&ids=${encodeURIComponent(ids.join(','))}&step=${st}${unitsQ}`;
        const a = document.createElement('a');
        a.href = A(mc ? `/api/export_mcint.zip?${q}` : `/api/export.csv?${q}&minmax=1`);
        a.download = '';
        document.body.appendChild(a); a.click(); a.remove();
    }

    try { days = (await api(A('/api/days'), { quiet: true })).days || {}; } catch (e) { days = {}; }
    paintRangeBar();
    load();
    return {
        cleanup: () => { plots.forEach(p => p.destroy()); },
        update: (p) => { if (p[0] && (p[0] !== start || (p[1] || p[0]) !== end)) setRange(p[0], p[1] || p[0], false); },
    };
}

// ---------------------------------------------------------------------- pieces
const stat = (label, value, note = '') => `<div><div class="tiny muted" style="font-weight:650">${label}</div><div style="font-size:1.35rem;font-weight:700" class="num">${value}</div>${note ? `<div class="tiny muted">${note}</div>` : ''}</div>`;
const mm = (s, dp = 0) => s ? `${num(s.min, dp)}–${num(s.max, dp)}` : '--';

function cards(sum) {
    const c = sum.cards, out = [];
    const card = (fam, title, html, hours) => `<div class="card" style="--c:${FAM[fam]}"><div class="card-head"><span class="glyph" style="width:30px;height:30px;border-radius:9px;display:grid;place-items:center;background:color-mix(in srgb,${FAM[fam]} 15%,transparent);color:${FAM[fam]}">${icon(fam === 'heart' ? 'heart' : fam === 'oxygen' ? 'oxygen' : fam === 'air' ? 'air' : 'water').replace('<svg', '<svg style="width:17px;height:17px"')}</span><h2>${title}</h2><span class="spacer"></span><span class="card-sub">${hours != null ? duration(hours * 60) + ' recorded' : ''}</span></div>
        <div class="row" style="gap:22px;align-items:flex-start">${html}</div></div>`;
    if (c.heart) {
        const h = c.heart;
        out.push(card('heart', 'Heart', stat('Average', `${num(h.hr && h.hr.avg)}<small class="muted"> bpm</small>`, `range ${mm(h.hr)}`)
            + stat('Resting', h.resting_hr ? `${num(h.resting_hr)}<small class="muted"> bpm</small>` : '--', 'lowest 5-min average')
            + stat('HRV', h.rmssd_median ? `${num(h.rmssd_median)}<small class="muted"> ms</small>` : '--', 'median RMSSD')
            + stat('Beats', num(h.beats)), h.hours));
    }
    if (c.oxygen) {
        const o = c.oxygen;
        out.push(card('oxygen', 'Oxygen', stat('Average SpO₂', o.spo2 ? `${num(o.spo2.avg, 1)}<small class="muted"> %</small>` : '--', o.spo2 ? `lowest ${num(o.spo2.min)} %` : '')
            + stat('Below 90 %', duration(o.below_90_min), `below 95 %: ${duration(o.below_95_min)}`)
            + stat('Pulse', o.pulse ? `${num(o.pulse.avg)}<small class="muted"> bpm</small>` : '--', `range ${mm(o.pulse)}`), o.hours));
    }
    if (c.air) {
        const a = c.air, co2 = a.metrics.find(m => m.id.startsWith('air.co2')), pm = a.metrics.find(m => m.id.startsWith('air.pm25')), t = a.metrics.find(m => m.id.startsWith('air.temp'));
        out.push(card('air', 'Air', (co2 ? stat('CO₂ average', `${num(co2.avg)}<small class="muted"> ppm</small>`, co2.over_1000_min ? `over 1000 ppm for ${duration(co2.over_1000_min)}` : `peak ${num(co2.max)} ppm`) : '')
            + (pm ? stat('PM2.5 average', `${num(pm.avg, 1)}<small class="muted"> µg/m³</small>`, `peak ${num(pm.max, 1)}`) : '')
            + (t ? stat('Temperature', `${num(t.avg, 1)}<small class="muted"> °C</small>`, `${num(t.min, 1)}–${num(t.max, 1)} °C`) : ''), a.hours));
    }
    if (c.water) {
        const w = c.water;
        out.push(card('water', 'Water', stat(sum.per_day.length ? 'Per day' : 'Drunk', `${num(sum.per_day.length ? w.per_day_ml : w.intake_ml)}<small class="muted"> mL</small>`, w.goal_pct != null ? `${w.goal_pct}% of goal` : '')
            + stat('Sips', num(w.sips), w.largest_sip_ml ? `largest ${num(w.largest_sip_ml)} mL` : '')
            + stat('Refills', num(w.refills)), null));
    }
    return out.join('');
}

function timeline(sum) {
    const a = sum.start, b = sum.end + 1, span = b - a;
    const fams = Object.keys(sum.timeline).filter(f => sum.timeline[f].length);
    if (!fams.length) return '<p class="muted small">No continuous recordings in this range.</p>';
    const lanes = fams.map(f => `<div class="lane" style="--c:${FAM[f] || 'var(--muted)'}"><span>${FAM_LABEL[f] || f}</span><div class="track">${sum.timeline[f].map(([x, y]) =>
        `<i style="left:${((x - a) / span * 100).toFixed(3)}%;width:${((y - x) / span * 100).toFixed(3)}%" title="${hm(x)}–${hm(y)}"></i>`).join('')}</div></div>`).join('');
    const ticks = [];
    const n = span <= 86400e3 + 1 ? 4 : Math.min(7, Math.round(span / 86400e3));
    for (let i = 0; i <= n; i++) {
        const t = a + span * i / n;
        ticks.push(`<span>${span <= 86400e3 + 1 ? hm(t) : new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span>`);
    }
    return `<div class="timeline">${lanes}<div class="axis">${ticks.join('')}</div></div>`;
}

function perDay(sum) {
    const rows = sum.per_day.map(d => `<tr><td>${dayLabel(d.day)}</td>
        <td class="r">${d.hours.heart ? duration(d.hours.heart * 60) : '—'}</td><td class="r">${d.hr_avg != null ? num(d.hr_avg) : '—'}</td>
        <td class="r">${d.spo2_avg != null ? num(d.spo2_avg, 1) : '—'}</td><td class="r">${d.spo2_min != null ? num(d.spo2_min) : '—'}</td>
        <td class="r">${d.co2_avg != null ? num(d.co2_avg) : '—'}</td><td class="r">${d.intake_ml != null ? num(d.intake_ml) : '—'}</td></tr>`).join('');
    return `<table class="t"><thead><tr><th>Day</th><th class="r">Heart recorded</th><th class="r">Avg HR</th><th class="r">Avg SpO₂</th><th class="r">Lowest SpO₂</th><th class="r">Avg CO₂</th><th class="r">Water mL</th></tr></thead><tbody>${rows}</tbody></table>`;
}
