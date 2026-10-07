// Bedside-monitor style ECG: sweeping pen with an erase gap, ECG-paper grid at
// 25 mm/s, auto or fixed gain, R-wave flash + optional beep.
// Colours are options, so it can follow the dashboard theme.
//
//   const mon = new EcgMonitor({ grid, trace, heart, noSignal, onBeat, theme });
//   mon.push(rows)      // rows = [[ts_ms, mV, hr], ...] straight from the websocket
//   mon.cycleGain()     // AUTO -> 5 -> 10 -> 20 mm/mV -> AUTO
//
// Sample timestamps come from the Polar's own clock (evenly spaced). They are
// replayed at real speed a little behind "now", so the pen moves smoothly even
// though BLE delivers ECG in ~0.5 s bursts, and so the viewer's clock doesn't
// have to match the Orin's.

class EcgMonitor {
    constructor({ grid, trace, heart, noSignal, onBeat, theme = {} }) {
        this.grid = grid;
        this.trace = trace;
        this.heart = heart;
        this.noSignal = noSignal;
        this.onBeat = onBeat || (() => {});

        this.MM_PER_SEC = 25;       // standard paper speed
        this.PLAY_DELAY_MS = 900;   // must exceed the ~560 ms PMD frame spacing
        this.GAP_FRAC = 0.025;      // width of the erase bar ahead of the pen
        // Defaults match the dashboard: slate-800 screen, rose-500 trace
        this.theme = {
            trace: '#f43f5e',
            glow: 'rgba(244, 63, 94, 0.55)',
            background: '#1e293b',
            gridMinor: 'rgba(148, 163, 184, 0.07)',
            gridMajor: 'rgba(148, 163, 184, 0.16)',
            ...theme,
        };

        this.gainModes = ['AUTO', 5, 10, 20];   // mm per mV
        this.gainIdx = 0;
        this.autoMmPerMv = 10;

        this.queue = [];
        this.qHead = 0;
        this.clockOffset = null;    // local ms - sensor ms (lag + clock skew)
        this.lastPlotted = null;    // {ts, x, y}
        this.lastSampleAt = 0;

        // Display filter state: 0.5 Hz high-pass removes baseline wander
        this.hpPrevIn = 0;
        this.hpPrevOut = 0;
        this.hpLastTs = null;
        this.envelope = 0.5;        // running |mV| peak, for auto gain + beat detection

        this.lastBeatTs = 0;
        this.aboveThr = false;

        this._resize = this._resize.bind(this);
        new ResizeObserver(this._resize).observe(this.trace.parentElement);
        this._resize();
        requestAnimationFrame(this._frame.bind(this));
    }

    get gainLabel() {
        const m = this.gainModes[this.gainIdx];
        return m === 'AUTO' ? `AUTO (${this.autoMmPerMv.toFixed(0)} mm/mV)` : `${m} mm/mV`;
    }

    cycleGain() {
        this.gainIdx = (this.gainIdx + 1) % this.gainModes.length;
        return this.gainLabel;
    }

    // ---------- input ----------
    push(rows) {
        if (!rows.length) return;
        const cand = Date.now() - rows[rows.length - 1][0];
        // Track the smallest arrival lag seen (best case); drift upward only slowly
        if (this.clockOffset === null || cand < this.clockOffset) this.clockOffset = cand;
        else this.clockOffset += (cand - this.clockOffset) * 0.02;

        for (const r of rows) this.queue.push(r);
        this.lastSampleAt = Date.now();
    }

    // ---------- layout ----------
    _resize() {
        const box = this.trace.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.W = Math.max(50, box.width);
        this.H = Math.max(50, box.height);
        for (const c of [this.grid, this.trace]) {
            c.width = Math.round(this.W * dpr);
            c.height = Math.round(this.H * dpr);
            c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        // Pick a sweep length that keeps ~4-5 px per mm, like a real screen
        this.sweepSec = Math.min(10, Math.max(4, Math.round(this.W / (this.MM_PER_SEC * 4.5))));
        this.sweepMs = this.sweepSec * 1000;
        this.pxPerMm = this.W / (this.MM_PER_SEC * this.sweepSec);
        this.baseline = this.H * 0.62;   // R-waves are mostly upward: leave more room above
        this._drawGrid();
        this.trace.getContext('2d').clearRect(0, 0, this.W, this.H);
        this.lastPlotted = null;
    }

    _drawGrid() {
        const g = this.grid.getContext('2d');
        const { W, H, pxPerMm } = this;
        g.fillStyle = this.theme.background;
        g.fillRect(0, 0, W, H);
        const lines = (step, color, width) => {
            g.strokeStyle = color;
            g.lineWidth = width;
            g.beginPath();
            for (let x = 0; x <= W; x += step) { g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, H); }
            for (let y = this.baseline % step; y <= H; y += step) { g.moveTo(0, Math.round(y) + 0.5); g.lineTo(W, Math.round(y) + 0.5); }
            g.stroke();
        };
        lines(pxPerMm, this.theme.gridMinor, 1);        // 1 mm
        lines(pxPerMm * 5, this.theme.gridMajor, 1);    // 5 mm (0.2 s)
    }

    // ---------- signal processing ----------
    _filter(ts, mv) {
        // First-order high-pass at 0.5 Hz; reset across gaps so a reconnect doesn't ring
        if (this.hpLastTs === null || ts - this.hpLastTs > 500) {
            this.hpPrevIn = mv;
            this.hpPrevOut = 0;
        } else {
            const dt = (ts - this.hpLastTs) / 1000;
            const rc = 1 / (2 * Math.PI * 0.5);
            const a = rc / (rc + dt);
            this.hpPrevOut = a * (this.hpPrevOut + mv - this.hpPrevIn);
            this.hpPrevIn = mv;
        }
        this.hpLastTs = ts;
        return this.hpPrevOut;
    }

    _detectBeat(ts, y) {
        const a = Math.abs(y);
        this.envelope = Math.max(a, this.envelope * 0.9985);   // ~5 s decay at 130 Hz
        const thr = Math.max(0.15, this.envelope * 0.55);
        if (a > thr && !this.aboveThr && ts - this.lastBeatTs > 280) {
            this.lastBeatTs = ts;
            this._flash();
            this.onBeat();
        }
        this.aboveThr = a > thr * 0.8;
    }

    _mmPerMv() {
        const m = this.gainModes[this.gainIdx];
        if (m !== 'AUTO') return m;
        // Aim for the biggest deflection to reach ~35% of the height; move smoothly
        const target = (this.H * 0.35) / this.pxPerMm / Math.max(0.3, this.envelope);
        this.autoMmPerMv += (Math.min(40, Math.max(2.5, target)) - this.autoMmPerMv) * 0.01;
        return this.autoMmPerMv;
    }

    _flash() {
        if (!this.heart) return;
        this.heart.style.opacity = '1';
        this.heart.style.transform = 'scale(1.25)';
        clearTimeout(this._flashT);
        this._flashT = setTimeout(() => {
            this.heart.style.opacity = '0.25';
            this.heart.style.transform = 'scale(1)';
        }, 140);
    }

    // ---------- drawing ----------
    _frame() {
        requestAnimationFrame(this._frame.bind(this));
        if (this.clockOffset === null) return;

        const playTs = Date.now() - this.clockOffset - this.PLAY_DELAY_MS;
        // If we fell far behind (tab was hidden), skip ahead instead of racing
        while (this.qHead < this.queue.length && this.queue[this.qHead][0] < playTs - 2000) this.qHead++;

        const ctx = this.trace.getContext('2d');
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.strokeStyle = this.theme.trace;
        ctx.shadowColor = this.theme.glow;
        ctx.shadowBlur = 4;
        const gap = this.W * this.GAP_FRAC;
        const mmPerMv = this._mmPerMv();

        let drew = false;
        ctx.beginPath();
        while (this.qHead < this.queue.length && this.queue[this.qHead][0] <= playTs) {
            const [ts, mv] = this.queue[this.qHead++];
            const y = this._filter(ts, mv);
            this._detectBeat(ts, y);

            const x = ((ts % this.sweepMs) / this.sweepMs) * this.W;
            const py = this.baseline - y * mmPerMv * this.pxPerMm;

            // Erase bar just ahead of the pen (wraps at the right edge)
            ctx.clearRect(x, 0, gap, this.H);
            if (x + gap > this.W) ctx.clearRect(0, 0, x + gap - this.W, this.H);

            const prev = this.lastPlotted;
            if (prev && x >= prev.x && ts - prev.ts < 200) {
                ctx.moveTo(prev.x, prev.y);
                ctx.lineTo(x, py);
                drew = true;
            }
            this.lastPlotted = { ts, x, y: py };
        }
        if (drew) ctx.stroke();

        // Compact the queue now and then
        if (this.qHead > 4000) {
            this.queue = this.queue.slice(this.qHead);
            this.qHead = 0;
        }

        if (this.noSignal) {
            this.noSignal.style.display = Date.now() - this.lastSampleAt > 3000 ? 'flex' : 'none';
        }
    }
}
