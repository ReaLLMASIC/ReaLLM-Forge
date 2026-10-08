// Tap-a-tile trends (shared by all dashboards).
//
// Tap a tile to open a live line graph of its reading under the tile row; tap it
// again, ✕, or Esc to close. Windows: 1 / 5 / 15 / 60 min, with min / avg / max.
//
//   const trends = createTileTrends({
//       storageKey: 'polar-trend',                 // remembers the chosen window
//       insertAfter: '#tile-row',                  // the panel is inserted after this element
//       ring: ['ring-rose-400/70'],                // accent for the selected tile (literal classes)
//       metrics: {
//           hr:   { title: 'Heart Rate', unit: 'BPM', tile: '#hr-val', color: '#f43f5e', stepped: true, span: 20 },
//           clim: { title: 'Climate', tile: '#temp-val', series: [
//                       { id: 'temp', label: 'Temp', unit: '°C', color: '#fb7185', span: 2 },
//                       { id: 'rh',   label: 'RH',   unit: '%',  color: '#38bdf8', span: 10, axis: 'y2' } ] },
//       },
//       prefill: async (metricKey, minutes) => ({ seriesId: [{x, y}, ...] }),   // optional, from the server's CSV
//       windows: [[1, '1 min'], [5, '5 min'], ...],   // optional: [minutes, label] choices (default 1/5/15/60 min)
//   });
//   trends.record('hr', Date.now(), 72);      // add a point (series id)
//   trends.sample('hr', 72);                  // add a point at most once per second
//   trends.addMetric('battery_mv', {...})      // a tile created at runtime (tile: element or selector)
//
// Drawing is kept honest: whole-number readings can be drawn as steps, curves use
// monotone interpolation (never overshoots real values), and each series has a
// minimum y-range (`span`) or a fixed one (`fixed: [lo, hi]`) so a 1-unit wobble
// doesn't fill the chart.

function createTileTrends({ metrics, storageKey, insertAfter, ring, prefill = null,
                            historyMs = 60 * 60 * 1000, sampleMs = 1000,
                            windows = [[1, '1 min'], [5, '5 min'], [15, '15 min'], [60, '1 hour']] }) {
    const PILL = 'bg-slate-700/50 text-slate-300 border-slate-600';
    const SELECTED = ['ring-2', 'ring-offset-2', 'ring-offset-slate-900', ...(ring || ['ring-sky-400/70'])];
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

    const history = {};
    const lastSample = {};
    // Normalise: every metric gets a series list; single-series metrics use their own key
    function normalise(key, m) {
        if (!m.series) m.series = [{ id: key, label: m.title, unit: m.unit, color: m.color,
                                     span: m.span, fixed: m.fixed, stepped: m.stepped, dp: m.dp }];
        for (const s of m.series) {
            s.dp ??= m.dp ?? 0; s.unit ??= m.unit; s.stepped ??= m.stepped;
            history[s.id] ??= [];
        }
    }
    for (const [key, m] of Object.entries(metrics)) normalise(key, m);

    // ---------- panel (built here so each dashboard doesn't repeat the markup) ----------
    const panel = document.createElement('div');
    panel.className = 'hidden bg-slate-800 border border-slate-700 rounded-2xl p-4 shadow-xl w-full';
    panel.innerHTML = `
        <div class="flex flex-wrap items-center gap-x-4 gap-y-2 mb-3">
            <h2 data-t="title" class="text-sm font-medium text-slate-400 uppercase tracking-wider">Trend</h2>
            <span data-t="stats" class="text-xs font-mono text-slate-400"></span>
            <div class="ml-auto flex flex-wrap items-center gap-2">
                <div data-t="windows" class="flex gap-2">
                    ${windows.map(([m, l]) =>
                        `<button data-min="${m}" class="px-3 py-1 rounded-full text-xs font-semibold border ${PILL} hover:text-white transition-colors">${l}</button>`).join('')}
                </div>
                <button data-t="close" class="w-7 h-7 rounded-full border ${PILL} hover:text-white transition-colors text-sm leading-none" title="Close (Esc)">✕</button>
            </div>
        </div>
        <div class="relative h-56 w-full">
            <canvas data-t="canvas"></canvas>
            <div data-t="empty" class="hidden absolute inset-0 flex items-center justify-center pointer-events-none">
                <span data-t="empty-text" class="text-xs text-slate-500"></span>
            </div>
        </div>`;
    document.querySelector(insertAfter).insertAdjacentElement('afterend', panel);
    const q = (t) => panel.querySelector(`[data-t="${t}"]`);

    let current = null;
    let windowMin = windows[Math.min(1, windows.length - 1)][0];
    try { windowMin = parseInt(localStorage.getItem(storageKey + '-window')) || windowMin; } catch (e) {}
    if (!windows.some(([m]) => m === windowMin)) windowMin = windows[0][0];

    const chart = new Chart(q('canvas').getContext('2d'), {
        type: 'line',
        data: { datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            interaction: { mode: 'nearest', axis: 'x', intersect: false },
            scales: {
                x: { type: 'realtime', realtime: { duration: windowMin * 60000, refresh: 1000, delay: 1000 },
                     ticks: { color: '#94a3b8', maxTicksLimit: 7 }, grid: { color: '#1e293b' } },
                y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                y2: { display: false, position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#94a3b8' } },
            },
            plugins: { legend: { display: false, labels: { color: '#94a3b8', boxWidth: 12 } } },
        },
    });

    // ---------- recording ----------
    function prune(arr, now) {
        const cutoff = now - historyMs;
        let i = 0;
        while (i < arr.length && arr[i].x < cutoff) i++;
        if (i) arr.splice(0, i);
    }
    function record(id, x, y) {
        if (!(id in history) || y == null || !Number.isFinite(y)) return;
        const h = history[id];
        if (h.length && x <= h[h.length - 1].x) return;          // keep time order, drop duplicates
        h.push({ x, y });
        prune(h, x);
        if (current) {
            const ds = chart.data.datasets.find(d => d._sid === id);
            if (ds) ds.data.push({ x, y });
        }
    }
    function sample(id, y, now = Date.now()) {
        if (now - (lastSample[id] || 0) < sampleMs) return;
        lastSample[id] = now;
        record(id, now, y);
    }

    // ---------- y ranges ----------
    function range(s, pts) {
        let lo = Infinity, hi = -Infinity;
        for (const p of pts) { if (p.y < lo) lo = p.y; if (p.y > hi) hi = p.y; }
        if (s.fixed) return { min: Math.min(s.fixed[0], Math.floor(lo)), max: Math.max(s.fixed[1], Math.ceil(hi)) };
        if (!Number.isFinite(lo)) return {};
        const pad = Math.max(0, ((s.span || 0) - (hi - lo)) / 2);
        return { suggestedMin: lo - pad >= 0 || lo < 0 ? lo - pad : 0, suggestedMax: hi + pad };
    }

    function refreshStats() {
        if (!current) return;
        const m = metrics[current];
        const cutoff = Date.now() - windowMin * 60000;
        const parts = [];
        let any = false;
        for (const axis of ['y', 'y2']) {
            const sc = chart.options.scales[axis];
            sc.min = sc.max = sc.suggestedMin = sc.suggestedMax = undefined;
        }
        for (const s of m.series) {
            const pts = history[s.id].filter(p => p.x >= cutoff);
            if (!pts.length) continue;
            any = true;
            Object.assign(chart.options.scales[s.axis === 'y2' ? 'y2' : 'y'], range(s, pts));
            let lo = Infinity, hi = -Infinity, sum = 0;
            for (const p of pts) { lo = Math.min(lo, p.y); hi = Math.max(hi, p.y); sum += p.y; }
            const f = (v) => v.toFixed(s.dp);
            const label = m.series.length > 1 ? `${s.label} ` : '';
            parts.push(`${label}min ${f(lo)} · avg ${f(sum / pts.length)} · max ${f(hi)} ${s.unit || ''}`);
        }
        q('stats').textContent = parts.join('   |   ');
        q('empty').classList.toggle('hidden', any);
    }

    // ---------- open / close ----------
    const tileOf = {};
    const paintWindows = () => q('windows').querySelectorAll('button').forEach(b => setPillActive(b, parseInt(b.dataset.min) === windowMin));

    async function open(key) {
        current = key;
        const m = metrics[key];
        const unit = m.series.length === 1 && m.series[0].unit ? ` <span class="normal-case">(${esc(m.series[0].unit)})</span>` : '';
        q('title').innerHTML = esc(m.title) + unit;
        q('empty-text').textContent = prefill ? 'Loading history…' : 'No readings yet. History starts when this page is opened.';
        const multi = m.series.length > 1;
        chart.options.plugins.legend.display = multi;
        chart.options.scales.y2.display = m.series.some(s => s.axis === 'y2');
        chart.options.scales.x.realtime.duration = windowMin * 60000;
        const build = () => {
            chart.data.datasets = m.series.map(s => ({
                _sid: s.id, label: `${s.label}${s.unit ? ' (' + s.unit + ')' : ''}`,
                borderColor: s.color, backgroundColor: s.color, borderWidth: multi ? 2 : 2.5,
                pointRadius: 0, pointHoverRadius: 4, yAxisID: s.axis === 'y2' ? 'y2' : 'y',
                stepped: s.stepped ? 'before' : false, cubicInterpolationMode: 'monotone',
                data: history[s.id].map(p => ({ ...p })),              // the chart prunes its own copy
            }));
            refreshStats();
            chart.update('none');
        };
        panel.classList.remove('hidden');
        for (const [k, t] of Object.entries(tileOf)) SELECTED.forEach(c => t.classList.toggle(c, k === key));
        paintWindows();
        build();
        panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        if (prefill) {
            try {
                const got = await prefill(key, windowMin) || {};
                if (current !== key) return;
                for (const [id, pts] of Object.entries(got)) {
                    if (!(id in history) || !pts.length) continue;
                    // server history fills in everything older than what this page recorded itself
                    const firstLocal = history[id].length ? history[id][0].x : Infinity;
                    const older = pts.filter(p => p.x < firstLocal && Number.isFinite(p.y));
                    history[id] = older.concat(history[id]);
                }
            } catch (e) {}
            q('empty-text').textContent = 'No readings in this window yet.';
            if (current === key) build();
        }
    }

    function close() {
        current = null;
        panel.classList.add('hidden');
        chart.data.datasets = [];
        Object.values(tileOf).forEach(t => SELECTED.forEach(c => t.classList.remove(c)));
    }

    // ---------- wire up tiles ----------
    function wire(key, m) {
        const inner = typeof m.tile === 'string' ? document.querySelector(m.tile) : m.tile;
        const tile = inner && inner.closest('.rounded-2xl');
        if (!tile || tileOf[key]) return;
        tileOf[key] = tile;
        tile.classList.add('cursor-pointer', 'select-none', 'hover:bg-slate-700/40', 'transition-colors');
        tile.setAttribute('role', 'button');
        tile.setAttribute('tabindex', '0');
        tile.title = 'Tap for a trend graph';
        const go = () => (current === key ? close() : open(key));
        tile.addEventListener('click', go);
        tile.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
    }
    for (const [key, m] of Object.entries(metrics)) wire(key, m);

    // Metrics discovered at runtime (e.g. unknown fields from a new device)
    function addMetric(key, m) {
        if (metrics[key]) return;
        metrics[key] = m;
        normalise(key, m);
        wire(key, m);
    }
    q('close').addEventListener('click', close);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && current) close(); });
    q('windows').addEventListener('click', (e) => {
        const b = e.target.closest('button[data-min]');
        if (!b) return;
        windowMin = parseInt(b.dataset.min);
        try { localStorage.setItem(storageKey + '-window', windowMin); } catch (err) {}
        if (current) open(current);
    });
    setInterval(refreshStats, 1000);

    return { record, sample, open, close, addMetric, history };
}
