// Bio-dash Analysis page. Data comes from app.py; nothing here talks to a device.
(() => {
    const $ = (id) => document.getElementById(id);
    const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
    const store = {                                  // per-viewer conveniences only; the page works without them
        get(k, d) { try { const v = localStorage.getItem('analysis.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
        set(k, v) { try { localStorage.setItem('analysis.' + k, JSON.stringify(v)); } catch (e) {} },
    };
    const DEFAULT_SIGNALS = ['heart.hr', 'oxygen.spo2', 'air.co2', 'water.fill', 'water.total'];
    const S = { from: null, to: null, families: {}, signals: [], selected: new Set(store.get('selected', [])),
                step: store.get('step', 'auto'), band: store.get('band', true), data: null, charts: [], hoverT: null };

    // ---------------------------------------------------------------- dates
    const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    const parse = (s) => { const [y, m, d] = s.split('-').map(Number); return new Date(y, m - 1, d); };
    const addDays = (s, n) => { const d = parse(s); d.setDate(d.getDate() + n); return iso(d); };
    const span = () => Math.round((parse(S.to) - parse(S.from)) / 864e5) + 1;
    const nice = (s, opts) => parse(s).toLocaleDateString(undefined, opts || { weekday: 'short', month: 'short', day: 'numeric' });
    const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const when = (ms) => (span() > 1 ? new Date(ms).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' : '') + clock(ms);
    const dur = (min) => min >= 90 ? `${(min / 60).toFixed(1)} h` : `${Math.round(min)} min`;
    const num = (v, d = 0) => v == null ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });

    function setRange(from, to, preset) {
        if (to < from) [from, to] = [to, from];
        S.from = from; S.to = to;
        $('from').value = from; $('to').value = to;
        document.querySelectorAll('#presets button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.preset === preset)));
        store.set('range', { from, to, preset: preset || null });
        loadAll();
    }
    function preset(p) {
        const today = iso(new Date());
        if (p === 'today') setRange(today, today, p);
        else if (p === 'yesterday') { const y = addDays(today, -1); setRange(y, y, p); }
        else setRange(addDays(today, -(Number(p) - 1)), today, p);
    }
    document.querySelectorAll('#presets button').forEach(b => b.addEventListener('click', () => preset(b.dataset.preset)));
    $('from').addEventListener('change', () => $('from').value && setRange($('from').value, $('to').value || $('from').value));
    $('to').addEventListener('change', () => $('to').value && setRange($('from').value || $('to').value, $('to').value));
    $('prev').addEventListener('click', () => { const n = span(); setRange(addDays(S.from, -n), addDays(S.to, -n)); });
    $('next').addEventListener('click', () => { const n = span(); setRange(addDays(S.from, n), addDays(S.to, n)); });

    async function getJSON(url) {
        const r = await fetch(url);
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.detail || `request failed (${r.status})`);
        return body;
    }
    const q = () => `start=${S.from}&end=${S.to}`;

    // ---------------------------------------------------------------- load
    async function loadAll() {
        ['summary', 'cards', 'perday', 'signals'].forEach(id => $(id).classList.add('loading'));
        try {
            const [sum, sig] = await Promise.all([getJSON(`/api/summary?${q()}`), getJSON(`/api/signals?${q()}`)]);
            S.summary = sum; S.signals = sig.signals;
            renderTimeline(sum); renderCards(sum); renderPerDay(sum); renderChips();
            await loadSeries();
        } catch (e) {
            showNotice(`Couldn't load this range: ${e.message}`);
        } finally {
            ['summary', 'cards', 'perday', 'signals'].forEach(id => $(id).classList.remove('loading'));
        }
    }
    function showNotice(text, html) {
        const n = $('notice');
        n.hidden = !text && !html;
        if (html) n.innerHTML = html; else n.textContent = text || '';
    }

    // ---------------------------------------------------------------- timeline
    const tip = $('tip');
    function showTip(ev, lines) {
        tip.replaceChildren(...lines.map((l, i) => { const d = el('div', '', l); if (i === 0) d.style.fontWeight = '700'; return d; }));
        tip.style.display = 'block';
        const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
        tip.style.left = x + 'px'; tip.style.top = (ev.clientY + 14) + 'px';
    }
    const hideTip = () => { tip.style.display = 'none'; };

    function renderTimeline(sum) {
        const box = $('timeline'); box.replaceChildren();
        const start = sum.start, end = sum.end, total = end - start + 1;
        $('range-label').textContent = S.from === S.to ? nice(S.from, { weekday: 'long', month: 'long', day: 'numeric' })
            : `${nice(S.from)} – ${nice(S.to)} · ${span()} days`;
        const fams = Object.keys(sum.families).filter(f => sum.timeline[f]);
        if (!fams.length) {
            box.append(el('p', 'empty-card', 'Nothing was recorded in this range. Pick another day, or use ‹ › to step through.'));
            return;
        }
        const grid = el('div', 'lanes');
        for (const f of fams) {
            const meta = sum.families[f], segs = sum.timeline[f];
            const lab = el('div', 'lane-label'); const dot = el('span', 'dot'); dot.style.background = meta.color;
            lab.append(dot, document.createTextNode(meta.label));
            const lane = el('div', 'lane');
            let ms = 0;
            for (const [a, b] of segs) {
                ms += b - a;
                const seg = el('i'); seg.style.background = meta.color;
                seg.style.left = (100 * (a - start) / total) + '%';
                seg.style.width = (100 * (b - a) / total) + '%';
                seg.tabIndex = 0;
                const lines = [meta.label, `${when(a)} – ${when(b)}`, dur((b - a) / 60000)];
                seg.addEventListener('pointermove', ev => showTip(ev, lines));
                seg.addEventListener('pointerleave', hideTip);
                seg.addEventListener('focus', () => { const r = seg.getBoundingClientRect(); showTip({ clientX: r.left, clientY: r.bottom }, lines); });
                seg.addEventListener('blur', hideTip);
                lane.append(seg);
            }
            grid.append(lab, lane, el('div', 'lane-hours num', dur(ms / 60000)));
        }
        // time axis under the lanes
        const axis = el('div', 'axis');
        const days = span();
        const wide = box.clientWidth;                     // fewer labels on a narrow screen
        if (days <= 2) {
            const every = (days === 1 ? 3 : 6) * (wide < 520 ? 4 : wide < 820 ? 2 : 1);
            for (let h = 0; h <= 24 * days; h += every) {
                const t = start + h * 3600e3;
                const s = el('span', '', h % 24 === 0 && h ? nice(iso(new Date(t)), { month: 'short', day: 'numeric' }) : clock(t));
                s.style.left = (100 * h / (24 * days)) + '%'; axis.append(s);
            }
        } else {
            const every = Math.ceil(days / (wide < 520 ? 3 : wide < 820 ? 6 : 10));
            for (let d = 0; d < days; d += every) {
                const s = el('span', '', nice(addDays(S.from, d), { month: 'short', day: 'numeric' }));
                s.style.left = (100 * d / days) + '%'; axis.append(s);
            }
        }
        grid.append(el('div'), axis, el('div'));
        box.append(grid);
    }

    // ---------------------------------------------------------------- cards
    function card(fam, sum, build) {
        const meta = sum.families[fam];
        const c = el('div', 'card dev');
        const h = el('h3'); const dot = el('span', 'dot'); dot.style.background = meta.color;
        h.append(dot, document.createTextNode(meta.label));
        c.append(h);
        const data = sum.cards[fam];
        if (!data) { c.append(el('div', 'hrs', 'nothing recorded'), el('p', 'empty-card', 'No recordings from this device in the range.')); return c; }
        c.append(el('div', 'hrs', `${dur(data.hours * 60)} recorded`));
        build(c, data);
        return c;
    }
    function hero(c, value, unit, sub) {
        const h = el('div', 'hero'); h.append(el('b', 'num', value), el('small', '', unit)); c.append(h);
        if (sub) c.append(el('div', 'hero-sub', sub));
    }
    function kv(c, rows) {
        const dl = el('dl', 'kv');
        for (const [k, v] of rows) if (v != null) dl.append(el('dt', '', k), el('dd', 'num', v));
        c.append(dl);
    }
    function renderCards(sum) {
        const box = $('cards'); box.replaceChildren();
        box.append(card('heart', sum, (c, d) => {
            hero(c, num(d.hr && d.hr.avg), 'bpm', 'average heart rate');
            kv(c, [['Lowest – highest', d.hr ? `${num(d.hr.min)} – ${num(d.hr.max)} bpm` : null],
                   ['Resting (lowest 5-min average)', d.resting_hr != null ? `${num(d.resting_hr)} bpm` : null],
                   ['HRV (median RMSSD)', d.rmssd_median != null ? `${num(d.rmssd_median)} ms` : null],
                   ['Beats recorded', num(d.beats)]]);
        }));
        box.append(card('oxygen', sum, (c, d) => {
            hero(c, num(d.spo2 && d.spo2.avg, 1), '% SpO₂', `average · lowest ${num(d.spo2 && d.spo2.min)} %`);
            kv(c, [['Time below 95 %', dur(d.below_95_min)], ['Time below 90 %', dur(d.below_90_min)],
                   ['Time below 88 %', dur(d.below_88_min)],
                   ['Pulse', d.pulse ? `${num(d.pulse.min)} – ${num(d.pulse.max)} bpm (avg ${num(d.pulse.avg)})` : null]]);
            if (sum.nights && sum.nights.length) {
                const n = el('div', 'nights');
                for (const x of sum.nights) {
                    const a = el('a', 'pill', `🌙 Night of ${nice(x.night, { month: 'short', day: 'numeric' })}`);
                    a.href = `/night/${x.night}`; a.target = '_blank'; a.rel = 'noopener';
                    a.title = `Overnight report: ${dur(x.minutes)} of SpO₂ readings`;
                    n.append(a);
                }
                c.append(el('div', 'hrs', ''), n);
            }
        }));
        box.append(card('air', sum, (c, d) => {
            const co2 = d.metrics.find(m => m.id.startsWith('air.co2'));
            if (co2) hero(c, num(co2.avg), 'ppm CO₂', `average · peak ${num(co2.max)} ppm` + (co2.over_1000_min ? ` · ${dur(co2.over_1000_min)} above 1000` : ''));
            kv(c, d.metrics.filter(m => m !== co2).map(m => [m.label, `${num(m.avg, 1)} ${m.unit} (${num(m.min, 1)} – ${num(m.max, 1)})`]));
        }));
        box.append(card('water', sum, (c, d) => {
            hero(c, num(d.intake_ml), 'mL drunk', d.goal_pct != null ? `${d.goal_pct} % of ${num(d.goal_ml)} mL` + (span() > 1 ? ' a day' : '') : '');
            if (d.goal_pct != null) {
                const bar = el('div', 'bar'); const fill = el('i'); fill.style.width = Math.min(100, d.goal_pct) + '%';
                fill.style.background = sum.families.water.color; bar.append(fill); c.append(bar);
            }
            kv(c, [['Sips', num(d.sips)], ['Per day', span() > 1 ? `${num(d.per_day_ml)} mL` : null],
                   ['Largest sip', d.largest_sip_ml != null ? `${num(d.largest_sip_ml)} mL` : null], ['Refills', num(d.refills)],
                   ['Left in the bottle', d.fill_last_ml != null ? `${num(d.fill_last_ml)} mL` : null]]);
        }));
    }

    function renderPerDay(sum) {
        const sec = $('perday'), t = $('perday-table');
        sec.hidden = !(sum.per_day && sum.per_day.length > 1);
        if (sec.hidden) return;
        t.replaceChildren();
        const head = el('tr');
        ['Day', 'Heart', 'Avg HR', 'O2 ring', 'SpO₂ avg', 'SpO₂ low', 'Avg CO₂', 'Drunk'].forEach(h => head.append(el('th', '', h)));
        const thead = el('thead'); thead.append(head);
        const tb = el('tbody');
        for (const r of sum.per_day) {
            const tr = el('tr');
            const cell = (v, suffix = '') => { const td = el('td', v == null ? 'na num' : 'num', v == null ? '—' : v + suffix); return td; };
            const dayCell = el('td'); const a = el('a', '', nice(r.day)); a.href = '#'; a.style.color = 'inherit';
            a.addEventListener('click', ev => { ev.preventDefault(); setRange(r.day, r.day); });
            dayCell.append(a);
            tr.append(dayCell,
                cell(r.hours.heart ? dur(r.hours.heart * 60) : null), cell(r.hr_avg != null ? num(r.hr_avg) : null, ' bpm'),
                cell(r.hours.oxygen ? dur(r.hours.oxygen * 60) : null), cell(r.spo2_avg != null ? num(r.spo2_avg, 1) : null, ' %'),
                cell(r.spo2_min != null ? num(r.spo2_min) : null, ' %'), cell(r.co2_avg != null ? num(r.co2_avg) : null, ' ppm'),
                cell(r.intake_ml != null ? num(r.intake_ml) : null, ' mL'));
            tb.append(tr);
        }
        t.append(thead, tb);
    }

    // ---------------------------------------------------------------- signal picker
    function renderChips() {
        const box = $('chips'); box.replaceChildren();
        const ids = new Set(S.signals.map(s => s.id));
        // first visit: a sensible set; afterwards, whatever was picked (and still exists)
        if (![...S.selected].some(id => ids.has(id)))
            S.selected = new Set(S.signals.filter(s => DEFAULT_SIGNALS.some(d => s.id === d || s.id.startsWith(d + '@'))).map(s => s.id).slice(0, 6));
        if (!S.signals.length) { box.append(el('p', 'empty-card', 'No signals in this range.')); return; }
        const fams = {}; const extra = [];
        for (const s of S.signals) (s.id.startsWith('raw.') ? extra : (fams[s.family] = fams[s.family] || [])).push(s);
        const chip = (s) => {
            const b = el('button', 'chip'); b.setAttribute('aria-pressed', String(S.selected.has(s.id)));
            const dot = el('span', 'dot'); dot.style.background = s.color;
            b.append(dot, document.createTextNode(s.label + (s.unit ? ` (${s.unit})` : '')));
            b.title = `${s.id} · ${s.readings.toLocaleString()} readings`;
            b.addEventListener('click', () => {
                S.selected.has(s.id) ? S.selected.delete(s.id) : S.selected.add(s.id);
                b.setAttribute('aria-pressed', String(S.selected.has(s.id)));
                store.set('selected', [...S.selected]); loadSeries();
            });
            return b;
        };
        for (const f of Object.keys(S.summary.families)) {
            if (!fams[f]) continue;
            const row = el('div', 'chip-row'); row.append(el('span', 'fam', f));
            fams[f].forEach(s => row.append(chip(s))); box.append(row);
        }
        if (extra.length) {
            const d = el('details', 'more'); d.append(el('summary', '', `More signals (${extra.length}): every other column the dashboards record`));
            const row = el('div', 'chip-row'); row.style.marginTop = '6px'; extra.forEach(s => row.append(chip(s)));
            d.append(row); if (extra.some(s => S.selected.has(s.id))) d.open = true; box.append(d);
        }
    }
    $('step').value = S.step;
    $('step').addEventListener('change', () => { S.step = $('step').value; store.set('step', S.step); loadSeries(); });
    $('band').checked = S.band;
    $('band').addEventListener('change', () => { S.band = $('band').checked; store.set('band', S.band); drawCharts(); updateExport(); });

    function updateExport() {
        const ids = S.signals.filter(s => S.selected.has(s.id)).map(s => s.id);
        const a = $('export');
        const step = S.data ? `${S.data.step / 1000}s` : (S.step === 'auto' ? '1min' : S.step);
        a.href = `/api/export.csv?${q()}&ids=${encodeURIComponent(ids.join(','))}&step=${encodeURIComponent(step)}&minmax=${S.band ? 1 : 0}`;
        a.style.opacity = ids.length ? '' : '.4';
        a.style.pointerEvents = ids.length ? '' : 'none';
        a.title = ids.length ? 'The selected signals, at the step shown, as one CSV' : 'Pick at least one signal';
    }

    let seriesSeq = 0;
    async function loadSeries() {
        updateExport();
        const ids = S.signals.filter(s => S.selected.has(s.id)).map(s => s.id);
        if (!ids.length) { S.data = null; drawCharts(); return; }
        const seq = ++seriesSeq;
        $('plots').classList.add('loading');
        try {
            const data = await getJSON(`/api/series?${q()}&ids=${encodeURIComponent(ids.join(','))}&step=${encodeURIComponent(S.step)}`);
            if (seq !== seriesSeq) return;
            S.data = data; showNotice('');
            drawCharts(); updateExport();
        } catch (e) {
            if (seq === seriesSeq) showNotice(e.message);
        } finally {
            if (seq === seriesSeq) $('plots').classList.remove('loading');
        }
    }

    // ---------------------------------------------------------------- charts
    const alpha = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };
    const crosshair = {
        id: 'crosshair',
        afterDatasetsDraw(chart) {
            if (S.hoverT == null) return;
            const x = chart.scales.x.getPixelForValue(S.hoverT);
            if (x < chart.chartArea.left || x > chart.chartArea.right) return;
            const ctx = chart.ctx; ctx.save();
            ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(x, chart.chartArea.top); ctx.lineTo(x, chart.chartArea.bottom); ctx.stroke(); ctx.restore();
        },
    };
    let raf = null;
    function setHover(t) {
        S.hoverT = t;
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = null; S.charts.forEach(c => c.draw()); renderReadout(); });
    }

    function renderReadout() {
        const r = $('readout'); r.replaceChildren();
        const d = S.data;
        if (!d || S.hoverT == null) { r.append(el('span', '', d ? 'Hover a chart to read every signal at that moment.' : '')); return; }
        const k = Math.max(0, Math.min(d.n - 1, Math.floor((S.hoverT - d.first) / d.step)));
        const t = d.first + k * d.step;
        r.append(el('b', 'num', when(t) + (d.step >= 60000 ? '' : ':' + String(new Date(t).getSeconds()).padStart(2, '0'))));
        for (const [id, s] of Object.entries(d.series)) {
            const m = d.meta[id], v = s.mean[k];
            const item = el('span');
            const key = el('i', 'key'); key.style.background = m.color;
            const val = el('b', 'num', v == null ? '—' : num(v, Math.abs(v) < 100 ? 1 : 0));
            item.append(key, val, document.createTextNode(v == null ? ` ${m.label}` : ` ${m.unit} ${m.label}`));
            if (v != null && s.n[k] === 0) item.append(el('span', 'est', ' (estimate)'));
            r.append(item);
        }
    }

    function drawCharts() {
        S.charts.forEach(c => c.destroy()); S.charts = [];
        const box = $('plots'); box.replaceChildren();
        const d = S.data;
        renderReadout();
        if (!d) { box.append(el('p', 'empty-card', 'Pick one or more signals above.')); return; }
        if (d.missing && d.missing.length) box.append(el('p', 'note', `Not in this range: ${d.missing.join(', ')}`));
        const xs = Array.from({ length: d.n }, (_, k) => d.first + k * d.step + d.step / 2);
        const days = span();
        for (const [id, s] of Object.entries(d.series)) {
            const m = d.meta[id];
            const wrap = el('div', 'plot');
            const head = el('div', 'plot-head'); head.append(el('span', '', m.label), el('small', '', m.unit + (m.kind === 'events' ? ' per step' : '')));
            const cbox = el('div', 'plot-box'); const canvas = el('canvas'); cbox.append(canvas);
            wrap.append(head, cbox); box.append(wrap);
            const pts = (arr) => arr.map((v, k) => ({ x: xs[k], y: v }));
            const datasets = [];
            const est = (m.kind === 'readings' || m.kind === 'state')        // a running total between sips is exact, not an estimate
                ? (ctx) => (s.n[ctx.p0DataIndex] === 0 || s.n[ctx.p1DataIndex] === 0) ? [4, 4] : undefined : undefined;
            if (m.kind === 'events') {
                datasets.push({ type: 'bar', data: pts(s.mean), backgroundColor: m.color, borderRadius: 2, barPercentage: 1, categoryPercentage: 1, minBarLength: 0 });
            } else {
                if (S.band && s.min && s.max) {
                    datasets.push({ data: pts(s.max), borderWidth: 0, pointRadius: 0, fill: '+1', backgroundColor: alpha(m.color, .22), spanGaps: false });
                    datasets.push({ data: pts(s.min), borderWidth: 0, pointRadius: 0, fill: false, spanGaps: false });
                }
                datasets.push({ data: pts(s.mean), borderColor: m.color, borderWidth: 2, pointRadius: 0, pointHitRadius: 0, fill: false,
                                stepped: (m.kind === 'state' || m.kind === 'daily_total') ? 'before' : false, spanGaps: false,
                                segment: { borderDash: est } });
            }
            const chart = new Chart(canvas, {
                type: 'line',
                data: { datasets },
                options: {
                    animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
                    events: ['mousemove', 'mouseout', 'touchstart', 'touchmove'],
                    plugins: { legend: { display: false }, tooltip: { enabled: false } },
                    scales: {
                        x: { type: 'time', min: S.summary.start, max: S.summary.end,
                             time: { unit: days <= 2 ? 'hour' : 'day', stepSize: days === 1 ? 3 : (days === 2 ? 6 : Math.ceil(days / 10)),
                                     displayFormats: { hour: 'HH:mm', day: 'MMM d' } },
                             grid: { color: '#243247', drawBorder: false }, ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true, font: { size: 11 } } },
                        y: { grid: { color: '#243247', drawBorder: false }, ticks: { color: '#94a3b8', maxTicksLimit: 4, font: { size: 11 } },
                             beginAtZero: m.kind === 'events' || m.kind === 'daily_total' },
                    },
                    onHover: (ev, _els, ch) => {
                        if (ev.type === 'mouseout') return setHover(null);
                        const x = ch.scales.x.getValueForPixel(ev.x);
                        setHover(x >= S.summary.start && x <= S.summary.end ? x : null);
                    },
                },
                plugins: [crosshair],
            });
            canvas.addEventListener('mouseleave', () => setHover(null));
            S.charts.push(chart);
        }
    }

    // ---------------------------------------------------------------- start
    (async () => {
        let info;
        try { info = await getJSON('/api/days'); } catch (e) { showNotice(`The analysis server isn't answering: ${e.message}`); return; }
        S.families = info.families;
        $('where').textContent = `Recordings: ${info.root}`;
        const days = Object.keys(info.days);
        if (!days.length) {
            showNotice('', `No recordings found in <code></code> yet. Record with any dashboard first, or try it with made-up data: ` +
                `<code>python3 analysis_dashboard/sample_data.py /tmp/biodash-demo</code> then restart with ` +
                `<code>BIODASH_DATA_DIR=/tmp/biodash-demo</code>.`);
            $('notice').querySelector('code').textContent = info.root;
        }
        const saved = store.get('range', null);
        if (saved && saved.preset) return preset(saved.preset);
        if (saved && saved.from) return setRange(saved.from, saved.to);
        const last = days.length ? days[days.length - 1] : info.today;
        setRange(last, last, last === info.today ? 'today' : null);
    })();
})();
