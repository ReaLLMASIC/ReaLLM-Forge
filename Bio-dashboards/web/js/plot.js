// A small, fast canvas chart for time series (and simple bars), made for this app.
// Replaces Chart.js + luxon + the streaming plugin (≈290 kB) with ≈7 kB gzipped, and only
// redraws when something changed and the chart is on screen.
//
//   const p = new Plot(el, { series: [{ id: 'hr', label: 'Heart rate', color: 'var(--heart)', unit: 'bpm' }],
//                            x: { window: 15 * 60e3 } });
//   p.set('hr', [[ts, y], ...]);  p.push('hr', ts, y);
//
// series options: type 'line' | 'bar' | 'dots' | 'step', axis 'y' | 'y2', dp, width, fill, band (set lo/hi),
//                 hidden, dash, raw (every sample, no averaging: waveforms).   x: { window (ms, follows now), ahead (ms shown past now: forecasts), min, max,
//                 format(x), step (bar width), kind 'time' | 'value', follow() -> the instant at the right edge
//                 (instead of now: a live trace drawn a little behind, so data arriving in bursts scrolls smoothly) }
//                 dots with lo/hi get range bars
//                 y / y2: { min, max, zero, label, format }   refs: [{ y, label, color, axis, dash }]
import { cssVar, hm, hms, esc } from './ui.js';

const TIME_STEPS = [1e3, 2e3, 5e3, 10e3, 15e3, 30e3, 60e3, 120e3, 300e3, 600e3, 900e3, 1800e3, 3600e3, 7200e3,
                    10800e3, 21600e3, 43200e3, 86400e3, 172800e3, 604800e3];

function niceStep(range, target) {
    const raw = range / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}
function lowerBound(xs, x) {
    let lo = 0, hi = xs.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (xs[m] < x) lo = m + 1; else hi = m; }
    return lo;
}

export class Plot {
    constructor(el, opts = {}) {
        this.el = el;
        this.o = opts;
        this.o.x = this.o.x || {};
        this.o.y = this.o.y || {};
        this.series = (opts.series || []).map(s => ({ type: 'line', axis: 'y', width: 2, dp: 0, ...s, xs: [], ys: [], lo: null, hi: null }));
        this.canvas = document.createElement('canvas');
        this.tip = document.createElement('div');
        this.tip.className = 'tip';
        this.tip.hidden = true;
        this.emptyEl = document.createElement('div');
        this.emptyEl.className = 'empty';
        this.emptyEl.textContent = opts.empty || 'No readings yet';
        el.append(this.canvas, this.emptyEl, this.tip);
        this.ctx = this.canvas.getContext('2d');
        this.dirty = true;
        this.visible = true;
        this.hover = null;
        this._colors = {};
        this._resize = () => { this.dirty = true; this.schedule(); };
        this.ro = new ResizeObserver(this._resize);
        this.ro.observe(el);
        this.io = new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; if (this.visible) this.schedule(); });
        this.io.observe(el);
        this._theme = () => { this._colors = {}; this.dirty = true; this.schedule(); };
        window.addEventListener('biodash-theme', this._theme);
        const move = (e) => { const r = this.canvas.getBoundingClientRect(); this.hover = { x: e.clientX - r.left, y: e.clientY - r.top }; this.schedule(); };
        this.canvas.addEventListener('pointermove', move);
        this.canvas.addEventListener('pointerdown', move);
        this.canvas.addEventListener('pointerleave', () => { this.hover = null; this.tip.hidden = true; this.schedule(); });
        if (this.o.x.window) {
            // a live window slides with the clock: redraw once a second while visible
            this.timer = setInterval(() => { if (this.visible && !document.hidden) { this.dirty = true; this.schedule(); } }, this.o.tick || 1000);
        }
        this.schedule();
    }

    destroy() {
        this.ro.disconnect(); this.io.disconnect();
        clearInterval(this.timer);
        window.removeEventListener('biodash-theme', this._theme);
        this.el.innerHTML = '';
    }

    get(id) { return this.series.find(s => s.id === id); }

    set(id, pts, band) {
        const s = this.get(id);
        if (!s) return;
        s.xs = []; s.ys = [];
        for (const p of pts || []) {
            const x = Array.isArray(p) ? p[0] : p.x, y = Array.isArray(p) ? p[1] : p.y;
            s.xs.push(x); s.ys.push(y);
        }
        if (band) { s.lo = band.lo; s.hi = band.hi; }
        this.dirty = true; this.schedule();
    }

    push(id, x, y) {
        const s = this.get(id);
        if (!s || y === null || y === undefined || Number.isNaN(y)) return;
        const n = s.xs.length;
        if (n && x < s.xs[n - 1]) return;               // history and live can overlap
        s.xs.push(x); s.ys.push(y);
        const w = this.o.x.window;
        if (w && s.xs.length > 50 && s.xs[0] < this.now() - w * 1.5) {
            const cut = lowerBound(s.xs, this.now() - w * 1.1);
            s.xs.splice(0, cut); s.ys.splice(0, cut);
        }
        this.dirty = true; this.schedule();
    }

    now() { return this.o.x.follow ? this.o.x.follow() : Date.now(); }
    setWindow(ms) { this.o.x.window = ms; this.dirty = true; this.schedule(); }
    setAhead(ms) { this.o.x.ahead = ms; this.dirty = true; this.schedule(); }
    setHidden(id, hidden) { const s = this.get(id); if (s) { s.hidden = hidden; this.dirty = true; this.schedule(); } }

    schedule() {
        if (this._raf) return;
        this._raf = requestAnimationFrame(() => { this._raf = null; if (this.visible) this.draw(); });
    }

    color(c) {
        if (!c) return '#888';
        if (!c.startsWith('var(')) return c;
        return this._colors[c] || (this._colors[c] = cssVar(c.slice(4, -1), this.el) || '#888');
    }

    // ------------------------------------------------------------ drawing
    draw() {
        const { el, canvas, ctx } = this;
        const W = el.clientWidth, H = el.clientHeight;
        if (!W || !H) return;
        const dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
            canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);

        const o = this.o, shown = this.series.filter(s => !s.hidden);
        // x range
        let x0, x1;
        if (o.x.window) { x1 = (o.x.max ?? this.now()) + (o.x.ahead || 0); x0 = x1 - o.x.window; }
        else {
            x0 = o.x.min ?? Math.min(...shown.map(s => s.xs[0] ?? Infinity));
            x1 = o.x.max ?? Math.max(...shown.map(s => s.xs[s.xs.length - 1] ?? -Infinity));
            if (o.x.step) x1 += o.x.step;
        }
        const has = Number.isFinite(x0) && Number.isFinite(x1) && shown.some(s => s.xs.length);
        this.emptyEl.hidden = has;
        if (!has) return;
        if (x1 <= x0) { x0 -= 30e3; x1 += 30e3; }

        // y ranges (per axis, from what's visible)
        const range = (axis) => {
            let lo = Infinity, hi = -Infinity;
            for (const s of shown) {
                if (s.axis !== axis) continue;
                const a = lowerBound(s.xs, x0 - (o.x.step || 0)), b = lowerBound(s.xs, x1 + 1);
                for (let i = a; i < b; i++) {
                    const v = s.ys[i];
                    if (v === null || v === undefined) continue;
                    const l = s.lo ? s.lo[i] ?? v : v, h = s.hi ? s.hi[i] ?? v : v;
                    if (l < lo) lo = l;
                    if (h > hi) hi = h;
                }
            }
            const cfg = o[axis] || {};
            for (const r of o.refs || []) if ((r.axis || 'y') === axis && cfg.showRefs !== false && Number.isFinite(lo)) { lo = Math.min(lo, r.y); hi = Math.max(hi, r.y); }
            if (!Number.isFinite(lo)) return null;
            if (cfg.zero || shown.some(s => s.axis === axis && s.type === 'bar')) lo = Math.min(0, lo);
            if (cfg.min !== undefined) lo = Math.min(cfg.min, lo);
            if (cfg.max !== undefined) hi = Math.max(cfg.max, hi);
            if (cfg.floor !== undefined) lo = Math.max(cfg.floor, lo);
            if (cfg.ceil !== undefined) hi = Math.min(cfg.ceil, hi);
            if (hi - lo < (cfg.minSpan || 1e-9)) { const m = (hi + lo) / 2, h = (cfg.minSpan || Math.max(1, Math.abs(m) * 0.1)) / 2; lo = m - h; hi = m + h; }
            const pad = (hi - lo) * 0.08;
            if (!(cfg.zero && lo === 0)) lo -= pad;
            hi += pad;
            const step = niceStep(hi - lo, H < 170 ? 3 : 4);
            return { lo, hi, step, cfg };
        };
        const ry = range('y'), ry2 = range('y2');

        // layout
        ctx.font = '11px ' + (cssVar('--font') || 'system-ui');
        const fmtY = (r, v) => (r.cfg.format ? r.cfg.format(v) : (+v.toFixed(r.step < 1 ? Math.min(3, Math.ceil(-Math.log10(r.step))) : 0)).toLocaleString());
        const ticks = (r) => { const t = []; if (!r) return t; for (let v = Math.ceil(r.lo / r.step) * r.step; v <= r.hi + 1e-9; v += r.step) t.push(+v.toFixed(10)); return t; };
        const ty = ticks(ry), ty2 = ticks(ry2);
        const wLabel = (r, t) => Math.max(0, ...t.map(v => ctx.measureText(fmtY(r, v)).width));
        const left = ry ? Math.ceil(wLabel(ry, ty)) + 10 : 6;
        const right = ry2 ? Math.ceil(wLabel(ry2, ty2)) + 10 : 8;
        const top = 8, bottom = o.x.hideAxis ? 6 : 22;
        const pw = W - left - right, ph = H - top - bottom;
        if (pw < 20 || ph < 20) return;
        const X = (x) => left + ((x - x0) / (x1 - x0)) * pw;
        const Y = (r, y) => top + ph - ((y - r.lo) / (r.hi - r.lo)) * ph;
        this._geo = { X, x0, x1, left, pw, top, ph };

        const grid = this.color('var(--grid)'), axis = this.color('var(--axis)');
        ctx.lineWidth = 1;
        ctx.strokeStyle = grid;
        ctx.fillStyle = axis;
        ctx.textBaseline = 'middle';
        // y grid + labels
        ctx.beginPath();
        ctx.textAlign = 'right';
        for (const v of ty) { const y = Math.round(Y(ry, v)) + .5; ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.fillText(fmtY(ry, v), left - 6, y); }
        ctx.stroke();
        ctx.textAlign = 'left';
        for (const v of ty2) ctx.fillText(fmtY(ry2, v), left + pw + 6, Y(ry2, v));

        // x labels
        if (!o.x.hideAxis) {
            ctx.textAlign = 'center';
            ctx.textBaseline = 'alphabetic';
            const maxLabels = Math.max(2, Math.floor(pw / 70));
            if (o.x.kind === 'value') {
                const st = o.x.labelStep || niceStep(x1 - x0, maxLabels);
                for (let v = Math.ceil(x0 / st) * st; v < x1; v += st) {
                    const x = X(v + (o.x.step ? o.x.step / 2 : 0));
                    if (x >= left && x <= left + pw) ctx.fillText(o.x.format ? o.x.format(v) : String(v), x, H - 6);
                }
            } else {
                const span = x1 - x0;
                const st = TIME_STEPS.find(s => span / s <= maxLabels) || 604800e3;
                const tz = new Date().getTimezoneOffset() * 60e3;
                let v = Math.ceil((x0 - tz) / st) * st + tz;
                if (st >= 86400e3) { const d = new Date(x0); d.setHours(0, 0, 0, 0); v = d.getTime(); while (v < x0) v += 86400e3; }
                for (; v <= x1; v += st) {
                    const x = X(v + (o.x.step ? o.x.step / 2 : 0));
                    if (x < left + 12 || x > left + pw - 12) continue;
                    const label = o.x.format ? o.x.format(v) : st >= 86400e3 ? new Date(v).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
                                : st < 60e3 ? hms(v) : hm(v);
                    ctx.fillText(label, x, H - 6);
                }
            }
        }

        // reference lines (goal, thresholds)
        for (const r of o.refs || []) {
            const ax = (r.axis || 'y') === 'y' ? ry : ry2;
            if (!ax) continue;
            const y = Math.round(Y(ax, r.y)) + .5;
            ctx.save();
            ctx.strokeStyle = this.color(r.color || 'var(--muted)');
            ctx.setLineDash(r.dash || [4, 4]);
            ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
            if (r.label) { ctx.fillStyle = ctx.strokeStyle; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(r.label, left + pw - 4, y - 3); }
            ctx.restore();
        }

        // series
        ctx.save();
        ctx.beginPath(); ctx.rect(left, top - 2, pw, ph + 4); ctx.clip();
        const gap = o.gap ?? (o.x.window ? Math.max(15e3, o.x.window / 40) : (x1 - x0) / 30);
        for (const s of shown) {
            const r = s.axis === 'y2' ? ry2 : ry;
            if (!r || !s.xs.length) continue;
            const col = this.color(s.color);
            const a = Math.max(0, lowerBound(s.xs, x0 - (o.x.step || 0)) - 1), b = Math.min(s.xs.length, lowerBound(s.xs, x1) + 1);
            if (s.type === 'bar') {
                const bw = Math.max(2, (o.x.step ? (o.x.step / (x1 - x0)) * pw : 6) * 0.72);
                ctx.fillStyle = col;
                for (let i = a; i < b; i++) {
                    const v = s.ys[i];
                    if (v == null) continue;
                    const cx = X(s.xs[i] + (o.x.step || 0) / 2), y = Y(r, Math.max(v, r.lo)), y0 = Y(r, Math.max(0, r.lo));
                    ctx.globalAlpha = s.alpha ? s.alpha(i, v) : 1;
                    roundRect(ctx, cx - bw / 2, Math.min(y, y0), bw, Math.max(1, Math.abs(y0 - y)), Math.min(4, bw / 3));
                }
                ctx.globalAlpha = 1;
                continue;
            }
            if (s.lo && s.hi && s.type !== 'dots') {  // min-max band behind the line
                ctx.fillStyle = col; ctx.globalAlpha = .16;
                const flush = (from, to) => {
                    ctx.beginPath();
                    for (let i = from; i <= to; i++) ctx.lineTo(X(s.xs[i]), Y(r, s.hi[i]));
                    for (let i = to; i >= from; i--) ctx.lineTo(X(s.xs[i]), Y(r, s.lo[i]));
                    ctx.closePath(); ctx.fill();
                };
                let start = -1;
                for (let i = a; i <= b; i++) {
                    const ok = i < b && s.lo[i] != null && s.hi[i] != null && (start < 0 || s.xs[i] - s.xs[i - 1] <= gap);
                    if (ok && start < 0) start = i;
                    else if (!ok && start >= 0) { flush(start, i - 1); start = (i < b && s.lo[i] != null) ? i : -1; }
                }
                ctx.globalAlpha = 1;
            }
            if (s.type === 'dots') {
                ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = s.width;
                for (let i = a; i < b; i++) {
                    if (s.ys[i] == null) continue;
                    const x = X(s.xs[i]);
                    if (s.lo && s.hi && s.lo[i] != null) { ctx.globalAlpha = .55; ctx.beginPath(); ctx.moveTo(x, Y(r, s.lo[i])); ctx.lineTo(x, Y(r, s.hi[i])); ctx.stroke(); ctx.globalAlpha = 1; }
                    ctx.beginPath(); ctx.arc(x, Y(r, s.ys[i]), s.radius || 2.5, 0, 6.3);
                    if (s.hollow) ctx.stroke(); else ctx.fill();
                }
                continue;
            }
            // more points than pixels: draw each 4 px column's average, with its spread as a soft band
            let xs = s.xs, ys = s.ys, from = a, to = b;
            if ((s.type === 'line' || s.type === 'step') && !s.lo && !s.raw && b - a > pw / 2) {
                const d = decimate(s.xs, s.ys, a, b, x0, x1, pw, 4);
                ctx.fillStyle = col; ctx.globalAlpha = .1;
                for (let i = 0; i < d.xs.length; i++) {
                    const x = X(d.xs[i]), yh = Y(r, d.hi[i]), yl = Y(r, d.lo[i]);
                    ctx.fillRect(x - 2, yh, 4, Math.max(1, yl - yh));
                }
                ctx.globalAlpha = 1;
                xs = d.xs; ys = d.ys; from = 0; to = d.xs.length;
            }
            ctx.strokeStyle = col;
            ctx.lineWidth = s.width;
            ctx.lineJoin = 'round'; ctx.lineCap = 'round';
            if (s.dash) ctx.setLineDash(s.dash);
            // the line, as runs broken at gaps; each run also gets its own soft fill
            const runs = [];
            let run = null;
            for (let i = from; i < to; i++) {
                const v = ys[i];
                if (v == null) { run = null; continue; }
                const x = X(xs[i]), y = Y(r, v);
                if (run && xs[i] - xs[i - 1] > gap) run = null;
                if (!run) { run = []; runs.push(run); }
                else if (s.type === 'step') run.push([x, run[run.length - 1][1]]);
                run.push([x, y]);
            }
            ctx.beginPath();
            for (const rr of runs) { ctx.moveTo(rr[0][0], rr[0][1]); for (let k = 1; k < rr.length; k++) ctx.lineTo(rr[k][0], rr[k][1]); }
            ctx.stroke();
            ctx.setLineDash([]);
            if (s.fill && runs.length) {
                const g = ctx.createLinearGradient(0, top, 0, top + ph);
                g.addColorStop(0, withAlpha(col, .28)); g.addColorStop(1, withAlpha(col, 0));
                ctx.fillStyle = g;
                for (const rr of runs) {
                    if (rr.length < 2) continue;
                    ctx.beginPath(); ctx.moveTo(rr[0][0], top + ph);
                    for (const [x, y] of rr) ctx.lineTo(x, y);
                    ctx.lineTo(rr[rr.length - 1][0], top + ph); ctx.closePath(); ctx.fill();
                }
            }
        }
        ctx.restore();

        // hover: crosshair + tooltip with every series' value at that time
        if (this.hover && this.hover.x >= left && this.hover.x <= left + pw) {
            const xv = x0 + ((this.hover.x - left) / pw) * (x1 - x0) - (o.x.step ? o.x.step / 2 : 0);
            const rows = [];
            let snapX = null;
            for (const s of shown) {
                if (!s.xs.length) continue;
                let i = lowerBound(s.xs, xv);
                if (i >= s.xs.length || (i > 0 && xv - s.xs[i - 1] < s.xs[i] - xv)) i--;
                if (i < 0 || s.ys[i] == null) continue;
                const tol = o.x.step ? o.x.step : Math.max(gap, (x1 - x0) / 50);
                if (Math.abs(s.xs[i] - xv) > tol) continue;
                snapX = snapX ?? s.xs[i];
                const v = s.ys[i];
                const val = s.format ? s.format(v) : (+v).toLocaleString(undefined, { maximumFractionDigits: s.dp });
                const extra = s.lo && s.hi && s.lo[i] != null ? ` <span class="faint">(${(+s.lo[i]).toFixed(s.dp)}–${(+s.hi[i]).toFixed(s.dp)})</span>` : '';
                rows.push(`<div><i style="background:${this.color(s.color)}"></i>${esc(s.label || s.id)}: <b style="display:inline;color:inherit;font-size:inherit">${val}</b> ${esc(s.unit || '')}${extra}</div>`);
            }
            if (rows.length && snapX !== null) {
                const cx = X(snapX + (o.x.step ? o.x.step / 2 : 0));
                ctx.strokeStyle = axis; ctx.globalAlpha = .5; ctx.setLineDash([3, 3]);
                ctx.beginPath(); ctx.moveTo(Math.round(cx) + .5, top); ctx.lineTo(Math.round(cx) + .5, top + ph); ctx.stroke();
                ctx.setLineDash([]); ctx.globalAlpha = 1;
                const head = o.x.tipFormat ? o.x.tipFormat(snapX) : o.x.kind === 'value' ? (o.x.format ? o.x.format(snapX) : snapX)
                    : (x1 - x0 > 2 * 86400e3 ? new Date(snapX).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : hms(snapX));
                this.tip.innerHTML = `<b>${esc(head)}</b>${rows.join('')}`;
                this.tip.hidden = false;
                const tw = this.tip.offsetWidth;
                this.tip.style.left = Math.min(W - tw / 2 - 2, Math.max(tw / 2 + 2, cx)) + 'px';
                this.tip.style.top = Math.max(0, Math.min(this.hover.y - 70, H - this.tip.offsetHeight - 24)) + 'px';
            } else this.tip.hidden = true;
        } else this.tip.hidden = true;
        this.dirty = false;
    }
}

// mean / min / max of the points in each `px`-wide column (x = the column's mean time)
function decimate(xs, ys, a, b, x0, x1, pw, px) {
    const cols = Math.max(1, Math.floor(pw / px)), span = (x1 - x0) / cols;
    const out = { xs: [], ys: [], lo: [], hi: [] };
    let col = -1, n = 0, sx = 0, sy = 0, lo = 0, hi = 0;
    const flush = () => { if (n) { out.xs.push(sx / n); out.ys.push(sy / n); out.lo.push(lo); out.hi.push(hi); } };
    for (let i = a; i < b; i++) {
        const y = ys[i];
        if (y == null) continue;
        const c = Math.floor(xs[i] / span);           // columns fixed in time, not to the edge: a scrolling chart
                                                      // keeps the same samples in each column, so it doesn't shimmer
        if (c !== col) { flush(); col = c; n = 0; sx = 0; sy = 0; lo = y; hi = y; }
        n++; sx += xs[i]; sy += y;
        if (y < lo) lo = y;
        if (y > hi) hi = y;
    }
    flush();
    return out;
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.fill();
}
export function withAlpha(col, a) {
    if (col.startsWith('#')) {
        let h = col.slice(1);
        if (h.length === 3) h = h.split('').map(c => c + c).join('');
        const n = parseInt(h.slice(0, 6), 16);
        return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
    }
    const m = col.match(/rgba?\(([^)]+)\)/);
    if (m) { const p = m[1].split(',').slice(0, 3).join(','); return `rgba(${p},${a})`; }
    return col;
}

// A tiny sparkline: no axes, no hover; draws a line (and soft fill) of the last values.
export function sparkline(canvas, pts, color, { min, max, window } = {}) {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) return;
    const dpr = window_dpr();
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!pts || pts.length < 2) return;
    const x1 = Date.now(), x0 = window ? x1 - window : pts[0][0];
    let lo = min ?? Infinity, hi = max ?? -Infinity;
    for (const [x, y] of pts) if (x >= x0) { if (y < lo) lo = y; if (y > hi) hi = y; }
    if (!Number.isFinite(lo)) return;
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const X = (x) => ((x - x0) / (x1 - x0)) * (W - 2) + 1, Y = (y) => H - 3 - ((y - lo) / (hi - lo)) * (H - 6);
    const col = color.startsWith('var(') ? cssVar(color.slice(4, -1), canvas) : color;
    if (pts.length > W / 4) {             // smooth: one averaged point per ~4 px
        const d = decimate(pts.map(p => p[0]), pts.map(p => p[1]), 0, pts.length, x0, x1, W, 4);
        pts = d.xs.map((x, i) => [x, d.ys[i]]);
        lo = Math.min(...d.ys); hi = Math.max(...d.ys);
        if (min !== undefined) lo = Math.min(lo, min);
        if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    }
    ctx.beginPath();
    let started = false, fx = 0, lx = 0;
    for (const [x, y] of pts) { if (x < x0) continue; if (!started) { ctx.moveTo(X(x), Y(y)); fx = X(x); started = true; } else ctx.lineTo(X(x), Y(y)); lx = X(x); }
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
    ctx.lineTo(lx, H); ctx.lineTo(fx, H); ctx.closePath();
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, withAlpha(col, .22)); g.addColorStop(1, withAlpha(col, 0));
    ctx.fillStyle = g; ctx.fill();
}
const window_dpr = () => window.devicePixelRatio || 1;
