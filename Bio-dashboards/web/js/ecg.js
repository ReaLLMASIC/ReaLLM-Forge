// Bedside-monitor ECG: a sweeping pen with an erase gap over ECG paper (25 mm/s), auto or
// fixed gain, an R-wave flash and an optional beep. Colours come from the theme (--ecg-*).
//
//   const mon = new EcgMonitor(container, { onBeat });
//   mon.push(rows)       rows = [[ts_ms, mV, hr], ...] straight from the strap's stream
//   mon.cycleGain()      AUTO -> 5 -> 10 -> 20 mm/mV -> AUTO
//   mon.setForecast(f)   {t0, dt, rep, lo, hi}: mV on a dt grid from t0 (ms, strap clock), or null
//   mon.destroy()
//
// Samples carry the strap's own clock (evenly spaced) and are replayed at real speed a little
// behind "now", so the pen moves smoothly although Bluetooth delivers them in ~0.5 s bursts.
//
// A forecast is drawn ahead of the pen (dashed representative sample + band), through the same 0.5 Hz
// high-pass as the trace, so it shows where the pen is about to go; the pen draws the real ECG over it.
import { cssVar } from './ui.js';

export class EcgMonitor {
    constructor(box, { onBeat, beatEl, noSignalEl } = {}) {
        this.box = box;
        this.grid = document.createElement('canvas');
        this.trace = document.createElement('canvas');
        this.fcCanvas = document.createElement('canvas');
        box.prepend(this.grid, this.trace, this.fcCanvas);
        this.recent = []; this.fc = null;     // the last seconds of [ts, mV] (to seed the forecast's filter); the forecast
        this.onBeat = onBeat || (() => {});
        this.beatEl = beatEl;
        this.noSignalEl = noSignalEl;
        this.MM_PER_SEC = 25;
        this.PLAY_DELAY_MS = 900;
        this.GAP_FRAC = 0.025;
        this.gainModes = ['AUTO', 5, 10, 20];
        this.gainIdx = 0;
        this.autoMmPerMv = 10;
        this.queue = []; this.qHead = 0;
        this.clockOffset = null; this.lastPlotted = null; this.lastSampleAt = 0;
        this.hpPrevIn = 0; this.hpPrevOut = 0; this.hpLastTs = null;
        this.envelope = 0.5; this.lastBeatTs = 0; this.aboveThr = false;
        this.alive = true;
        this.readTheme();
        this._onTheme = () => { this.readTheme(); this._drawGrid(); };
        window.addEventListener('biodash-theme', this._onTheme);
        this.ro = new ResizeObserver(() => this._resize());
        this.ro.observe(box);
        this._resize();
        this._frame = this._frame.bind(this);
        requestAnimationFrame(this._frame);
    }

    destroy() {
        this.alive = false;
        this.ro.disconnect();
        window.removeEventListener('biodash-theme', this._onTheme);
    }

    readTheme() {
        const v = (n) => cssVar(n, this.box);
        this.theme = { background: v('--ecg-bg'), gridMinor: v('--ecg-grid-minor'), gridMajor: v('--ecg-grid-major'),
                       trace: v('--ecg-trace'), glow: v('--ecg-glow'),
                       forecast: v('--ecg-forecast') || '#a78bfa', forecastBand: v('--ecg-forecast-band') || 'rgba(167,139,250,.2)' };
    }

    get gainLabel() {
        const m = this.gainModes[this.gainIdx];
        return m === 'AUTO' ? `Auto (${this.autoMmPerMv.toFixed(0)} mm/mV)` : `${m} mm/mV`;
    }
    cycleGain() { this.gainIdx = (this.gainIdx + 1) % this.gainModes.length; return this.gainLabel; }

    push(rows) {
        if (!rows.length) return;
        const cand = Date.now() - rows[rows.length - 1][0];
        if (this.clockOffset === null || cand < this.clockOffset) this.clockOffset = cand;
        else this.clockOffset += (cand - this.clockOffset) * 0.02;
        for (const r of rows) this.queue.push(r);
        if (this.queue.length > 30000) { this.queue = this.queue.slice(-3000); this.qHead = 0; }   // tab was hidden a while
        this.lastSampleAt = Date.now();
        for (const r of rows) this.recent.push([r[0], r[1]]);
        const keepFrom = rows[rows.length - 1][0] - 4000;
        let k = 0;
        while (k < this.recent.length && this.recent[k][0] < keepFrom) k++;
        if (k > 256) this.recent = this.recent.slice(k);
    }

    // The forecast through the same high-pass as the trace: the filter starts 2.5 s before the forecast on the
    // real samples (it settles in ~1.5 s), then runs on through each forecast series.
    setForecast(f) {
        if (!f || !f.rep || f.rep.length < 2) { this.fc = null; return; }
        const hp = (s, ts, mv) => {
            if (s.last === null || ts - s.last > 500) { s.prevIn = mv; s.prevOut = 0; }
            else {
                const rc = 1 / (2 * Math.PI * 0.5), a = rc / (rc + (ts - s.last) / 1000);
                s.prevOut = a * (s.prevOut + mv - s.prevIn);
                s.prevIn = mv;
            }
            s.last = ts;
            return s.prevOut;
        };
        const seed = { prevIn: 0, prevOut: 0, last: null };
        for (const [ts, mv] of this.recent) if (ts >= f.t0 - 2500 && ts < f.t0) hp(seed, ts, mv);
        const run = (vals) => { const s = { ...seed }; return vals.map((v, k) => hp(s, f.t0 + k * f.dt, v)); };
        this.fc = { t0: f.t0, dt: f.dt, n: f.rep.length, rep: run(f.rep), lo: run(f.lo), hi: run(f.hi) };
    }

    _resize() {
        const W = this.box.clientWidth, H = this.box.clientHeight;
        if (!W || !H) return;
        const dpr = window.devicePixelRatio || 1;
        this.W = W; this.H = H;
        for (const c of [this.grid, this.trace, this.fcCanvas]) {
            c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
            c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        this.sweepSec = Math.min(10, Math.max(4, Math.round(W / (this.MM_PER_SEC * 4.5))));
        this.sweepMs = this.sweepSec * 1000;
        this.pxPerMm = W / (this.MM_PER_SEC * this.sweepSec);
        this.baseline = H * 0.6;
        this._drawGrid();
        this.trace.getContext('2d').clearRect(0, 0, W, H);
        this.lastPlotted = null;
    }

    _drawGrid() {
        if (!this.W) return;
        const g = this.grid.getContext('2d'), { W, H, pxPerMm } = this;
        g.fillStyle = this.theme.background; g.fillRect(0, 0, W, H);
        const lines = (step, color) => {
            g.strokeStyle = color; g.lineWidth = 1; g.beginPath();
            for (let x = 0; x <= W; x += step) { g.moveTo(Math.round(x) + .5, 0); g.lineTo(Math.round(x) + .5, H); }
            for (let y = this.baseline % step; y <= H; y += step) { g.moveTo(0, Math.round(y) + .5); g.lineTo(W, Math.round(y) + .5); }
            g.stroke();
        };
        lines(pxPerMm, this.theme.gridMinor);
        lines(pxPerMm * 5, this.theme.gridMajor);
    }

    _filter(ts, mv) {          // 0.5 Hz high-pass: removes baseline wander; reset across gaps
        if (this.hpLastTs === null || ts - this.hpLastTs > 500) { this.hpPrevIn = mv; this.hpPrevOut = 0; }
        else {
            const dt = (ts - this.hpLastTs) / 1000, rc = 1 / (2 * Math.PI * 0.5), a = rc / (rc + dt);
            this.hpPrevOut = a * (this.hpPrevOut + mv - this.hpPrevIn);
            this.hpPrevIn = mv;
        }
        this.hpLastTs = ts;
        return this.hpPrevOut;
    }

    _detectBeat(ts, y) {
        const a = Math.abs(y);
        this.envelope = Math.max(a, this.envelope * 0.9985);
        const thr = Math.max(0.15, this.envelope * 0.55);
        if (a > thr && !this.aboveThr && ts - this.lastBeatTs > 280) {
            this.lastBeatTs = ts;
            if (this.beatEl) {
                this.beatEl.style.opacity = '1'; this.beatEl.style.transform = 'scale(1.25)';
                clearTimeout(this._flashT);
                this._flashT = setTimeout(() => { this.beatEl.style.opacity = ''; this.beatEl.style.transform = ''; }, 140);
            }
            this.onBeat();
        }
        this.aboveThr = a > thr * 0.8;
    }

    _mmPerMv() {
        const m = this.gainModes[this.gainIdx];
        if (m !== 'AUTO') return m;
        const target = (this.H * 0.35) / this.pxPerMm / Math.max(0.3, this.envelope);
        this.autoMmPerMv += (Math.min(40, Math.max(2.5, target)) - this.autoMmPerMv) * 0.01;
        return this.autoMmPerMv;
    }

    _frame() {
        if (!this.alive) return;
        requestAnimationFrame(this._frame);
        if (this.noSignalEl) this.noSignalEl.hidden = Date.now() - this.lastSampleAt < 3000;
        if (this.clockOffset === null || !this.W || document.hidden) return;
        const playTs = Date.now() - this.clockOffset - this.PLAY_DELAY_MS;
        while (this.qHead < this.queue.length && this.queue[this.qHead][0] < playTs - 2000) this.qHead++;
        const ctx = this.trace.getContext('2d');
        ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.strokeStyle = this.theme.trace; ctx.shadowColor = this.theme.glow; ctx.shadowBlur = 4;
        const gap = this.W * this.GAP_FRAC, mmPerMv = this._mmPerMv();
        let drew = false;
        ctx.beginPath();
        while (this.qHead < this.queue.length && this.queue[this.qHead][0] <= playTs) {
            const [ts, mv] = this.queue[this.qHead++];
            const y = this._filter(ts, mv);
            this._detectBeat(ts, y);
            const x = ((ts % this.sweepMs) / this.sweepMs) * this.W;
            const py = this.baseline - y * mmPerMv * this.pxPerMm;
            ctx.clearRect(x, 0, gap, this.H);
            if (x + gap > this.W) ctx.clearRect(0, 0, x + gap - this.W, this.H);
            const prev = this.lastPlotted;
            if (prev && x >= prev.x && ts - prev.ts < 200) { ctx.moveTo(prev.x, prev.y); ctx.lineTo(x, py); drew = true; }
            this.lastPlotted = { ts, x, y: py };
        }
        if (drew) ctx.stroke();
        this._drawForecast(playTs, mmPerMv);
        if (this.qHead > 4000) { this.queue = this.queue.slice(this.qHead); this.qHead = 0; }
    }

    // The forecast's part ahead of the pen, on its own canvas (redrawn every frame)
    _drawForecast(playTs, mmPerMv) {
        const ctx = this.fcCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.W, this.H);
        const f = this.fc;
        if (!f || f.t0 + (f.n - 1) * f.dt <= playTs) return;
        const limit = playTs + this.sweepMs * 0.85;          // never wrap round onto the pen
        const X = (ts) => ((ts % this.sweepMs) / this.sweepMs) * this.W;
        const Y = (v) => this.baseline - v * mmPerMv * this.pxPerMm;
        // a wider erase bar while forecasting: the old sweep from the pen to the forecast's end is cleared
        // (the pen redraws it within ~2 s), so the forecast isn't drawn over an old trace
        const tc = this.trace.getContext('2d');
        const xa = X(playTs) + 4, xb = X(Math.min(f.t0 + (f.n - 1) * f.dt, limit));   // +4: the pen tip's glow
        if (xb >= xa) tc.clearRect(xa, 0, xb - xa, this.H);
        else { tc.clearRect(xa, 0, this.W - xa, this.H); tc.clearRect(0, 0, xb, this.H); }
        const runs = [];
        let cur = [];
        for (let k = 0; k < f.n; k++) {
            const ts = f.t0 + k * f.dt;
            if (ts <= playTs || ts >= limit) continue;
            const x = X(ts);
            if (cur.length && x < cur[cur.length - 1].x) { runs.push(cur); cur = []; }   // right-edge wrap
            cur.push({ x, k });
        }
        if (cur.length) runs.push(cur);
        ctx.save();
        for (const run of runs) {
            if (run.length < 2) continue;
            ctx.beginPath();
            run.forEach((p, i) => (i ? ctx.lineTo(p.x, Y(f.hi[p.k])) : ctx.moveTo(p.x, Y(f.hi[p.k]))));
            for (let i = run.length - 1; i >= 0; i--) ctx.lineTo(run[i].x, Y(f.lo[run[i].k]));
            ctx.closePath();
            ctx.fillStyle = this.theme.forecastBand;
            ctx.fill();
            ctx.beginPath();
            run.forEach((p, i) => (i ? ctx.lineTo(p.x, Y(f.rep[p.k])) : ctx.moveTo(p.x, Y(f.rep[p.k]))));
            ctx.setLineDash([6, 5]);
            ctx.lineWidth = 1.6;
            ctx.strokeStyle = this.theme.forecast;
            ctx.stroke();
        }
        ctx.restore();
    }
}
