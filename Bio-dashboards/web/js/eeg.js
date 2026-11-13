// Live EEG: one lane per channel, the newest sample at the right edge, the last 8 s across.
// Display filters (the recording is untouched): 1 Hz high-pass, 40 Hz low-pass and a mains notch, each a
// 2nd-order biquad (RBJ cookbook) run sample by sample, so switching them on shows the next samples filtered.
// Drawn per pixel column (min / max of the samples under it), so it stays light at 256 Hz x 8 channels.
//   const view = new EegView({ canvas, noSignal, channels: ['TP9', 'AF7', 'AF8', 'TP10', 'AUX1', ...] });
//   view.push(rows)          // [[ts_ms, uV, uV, ...], ...] as the stream sends them
//   view.destroy()
import { prefs } from './ui.js';

class Biquad {
    constructor(kind, f0, fs, q = Math.SQRT1_2) {
        const w = 2 * Math.PI * f0 / fs, c = Math.cos(w), a = Math.sin(w) / (2 * q);
        let b0, b1, b2;
        if (kind === 'hp') { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; }
        else if (kind === 'lp') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = (1 - c) / 2; }
        else { b0 = 1; b1 = -2 * c; b2 = 1; }                                         // notch
        const a0 = 1 + a;
        Object.assign(this, { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: -2 * c / a0, a2: (1 - a) / a0, x1: 0, x2: 0, y1: 0, y2: 0 });
    }
    step(x) {
        const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
        this.x2 = this.x1; this.x1 = x; this.y2 = this.y1; this.y1 = y;
        return y;
    }
    /** Start in the steady state for a constant input x (no ring at the start); -> the steady output. */
    prime(x) {
        const y = x * (this.b0 + this.b1 + this.b2) / (1 + this.a1 + this.a2);
        this.x1 = this.x2 = x; this.y1 = this.y2 = y;
        return y;
    }
}

export class EegView {
    constructor({ canvas, noSignal = null, channels, rate = 256, seconds = 8 }) {
        Object.assign(this, { canvas, noSignal, channels, rate, seconds });
        this.ctx = canvas.getContext('2d');
        this.cap = rate * seconds;
        this.t = new Float64Array(this.cap);
        this.v = channels.map(() => new Float32Array(this.cap));
        this.head = 0; this.count = 0; this.lastAt = 0; this.lastTs = 0;
        const saved = prefs.get('brain.eeg', {});
        this.scale = saved.scale || 100;               // uV from a lane's centre to its edge
        this.notch = saved.notch ?? 60;                // 60, 50 or 0 (off)
        this.filtered = saved.filtered ?? true;
        this.aux = saved.aux ?? false;
        this.theme();
        this._makeFilters();
        this._theme = () => this.theme();
        window.addEventListener('biodash-theme', this._theme);
        const draw = () => {                                                                       // ~30 fps, only while on screen
            if (this.gone) return;
            if (!document.hidden) this.draw();
            this.timer = setTimeout(() => requestAnimationFrame(draw), 33);
        };
        requestAnimationFrame(draw);
    }
    destroy() { this.gone = true; clearTimeout(this.timer); window.removeEventListener('biodash-theme', this._theme); }
    /** Colours from the theme's tokens (app.css); re-read when the theme changes. */
    theme() {
        const css = getComputedStyle(this.canvas), t = (n, d) => css.getPropertyValue(n).trim() || d;
        const aux = t('--faint', '#5d6a85');
        this.colors = [t('--brain', '#e879f9'), '#38bdf8', '#34d399', '#fbbf24', aux, aux, aux, aux];
        this.lane = t('--grid', 'rgba(148,163,184,0.08)');
        this.guide = t('--line', 'rgba(148,163,184,0.22)');
        this.text = t('--axis', '#8592ab');
    }
    _save() { prefs.set('brain.eeg', { scale: this.scale, notch: this.notch, filtered: this.filtered, aux: this.aux }); }
    _makeFilters() {
        this.filters = this.channels.map(() => [new Biquad('hp', 1, this.rate), new Biquad('lp', 40, this.rate),
                                                ...(this.notch ? [new Biquad('notch', this.notch, this.rate, 8)] : [])]);
        this.primed = false;
        this.mean = this.channels.map(() => null);
    }
    get specs() { return this.filtered ? `1-40 Hz${this.notch ? ` · ${this.notch} Hz notch` : ''}` : 'raw (offset removed)'; }
    cycleScale() { const s = [25, 50, 100, 200, 500]; this.scale = s[(s.indexOf(this.scale) + 1) % s.length]; this._save(); return `±${this.scale} µV`; }
    cycleNotch() { this.notch = { 60: 50, 50: 0, 0: 60 }[this.notch]; this._makeFilters(); this._save(); return this.notch ? `${this.notch} Hz` : 'No notch'; }
    toggleFilter() { this.filtered = !this.filtered; this._makeFilters(); this._save(); return this.filtered ? 'Filtered' : 'Raw'; }
    toggleAux() { this.aux = !this.aux; this._save(); return this.aux; }

    push(rows) {
        const nch = Math.min(this.channels.length, rows[0].length - 1);
        if (!this.primed) {
            for (let c = 0; c < nch; c++) { let y = rows[0][c + 1]; for (const f of this.filters[c]) y = f.prime(y); }
            this.primed = true;
        }
        for (const r of rows) {
            if (r[0] < this.lastTs - 2000) this.count = 0;          // a new session's clock: start over
            this.lastTs = r[0];
            const i = this.head;
            this.t[i] = r[0];
            for (let c = 0; c < this.channels.length; c++) {
                const x = c < nch ? r[c + 1] : NaN;
                let y;
                if (!Number.isFinite(x)) y = NaN;
                else if (this.filtered) { y = x; for (const f of this.filters[c]) y = f.step(y); }
                else { this.mean[c] = this.mean[c] === null ? x : this.mean[c] + (x - this.mean[c]) / 256; y = x - this.mean[c]; }
                this.v[c][i] = y;
            }
            this.head = (i + 1) % this.cap;
            this.count = Math.min(this.count + 1, this.cap);
        }
        this.lastAt = Date.now();
    }

    draw() {
        const cv = this.canvas, dpr = window.devicePixelRatio || 1;
        const W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
        if (!W || !H) return;
        if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
        const g = this.ctx;
        g.clearRect(0, 0, W, H);
        if (this.noSignal) this.noSignal.style.display = Date.now() - this.lastAt > 3000 ? 'flex' : 'none';
        const lanes = this.channels.map((n, c) => c).filter(c => c < 4 || this.aux);
        const lh = H / lanes.length, left = 46 * dpr;
        g.font = `600 ${11 * dpr}px ui-monospace, monospace`;
        g.textBaseline = 'middle';
        // lane guides: centre line and +-scale
        lanes.forEach((c, k) => {
            const mid = lh * (k + 0.5);
            g.fillStyle = this.lane; g.fillRect(left, lh * k + 2 * dpr, W - left, lh - 4 * dpr);
            g.strokeStyle = this.guide; g.lineWidth = dpr;
            g.beginPath(); g.moveTo(left, mid); g.lineTo(W, mid); g.stroke();
            g.fillStyle = this.colors[c]; g.fillText(this.channels[c], 6 * dpr, mid);
        });
        if (!this.count) return;
        const end = this.lastTs, span = this.seconds * 1000, x0 = end - span, pxPerMs = (W - left) / span;
        // one-second ticks
        g.strokeStyle = this.lane;
        for (let s = Math.ceil(x0 / 1000) * 1000; s <= end; s += 1000) {
            const x = left + (s - x0) * pxPerMs;
            g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
        }
        const n = this.count, start = (this.head - n + this.cap) % this.cap;
        lanes.forEach((c, k) => {
            const mid = lh * (k + 0.5), ky = (lh / 2 - 3 * dpr) / this.scale, v = this.v[c];
            g.strokeStyle = this.colors[c]; g.lineWidth = 1.2 * dpr;
            g.save(); g.beginPath(); g.rect(left, lh * k, W - left, lh); g.clip();
            g.beginPath();
            let col = -1, lo = 0, hi = 0, started = false;
            const flush = () => {
                if (col < 0) return;
                const ylo = mid - hi * ky, yhi = mid - lo * ky, x = left + col;
                if (!started) { g.moveTo(x, ylo); started = true; } else g.lineTo(x, ylo);
                if (yhi - ylo > 1) g.lineTo(x, yhi);
            };
            for (let j = 0; j < n; j++) {
                const i = (start + j) % this.cap, ts = this.t[i];
                if (ts < x0) continue;
                const y = v[i];
                if (!Number.isFinite(y)) continue;
                const px = Math.floor((ts - x0) * pxPerMs);
                if (px !== col) { flush(); col = px; lo = hi = y; }
                else { if (y < lo) lo = y; if (y > hi) hi = y; }
            }
            flush();
            g.stroke(); g.restore();
        });
        // scale bar
        g.fillStyle = this.text;
        g.fillText(`±${this.scale} µV`, W - 70 * dpr, 12 * dpr);
    }
}
