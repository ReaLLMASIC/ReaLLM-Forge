// Live forecasts on the Heart page: the three Forecast buttons, what they draw, and a live error readout.
//
// polar_dashboard/forecast_service.py (started by server.py when the models are installed) follows the first
// person's strap, forecasts the next second every 0.5 s and posts it to the Heart API, which sends it on the
// heart stream as ecg_forecast / acc_forecast / rr_forecast. A forecast carries, per lane, the median, the
// band (lo / hi: the range of the samples) and one representative sample (`rep`) on a dt_ms grid starting at
// t0 (ms), plus the last real row. Three models, one overlay each:
//   ecg   ECG only            the next second ahead of the ECG pen
//   acc   ECG + accelerometer the next second of X / Y / Z on the Motion chart
//   rr    ECG + R-R           the next beats on the Beat-to-beat chart: a ring at the median R-R, a bar over the samples' range
// The dashed line is the representative sample: an average of the samples would smear a beat whose timing is
// uncertain; the band shows the spread.
//
//   const fc = new Forecasts({ monitor, acc, rr, buttons: { ecg, acc, rr }, status: { ecg, acc, rr } });
//   fc.onMessage(m)            *_forecast | forecast_settings messages from the heart stream
//   fc.onEcgRows(rows); fc.onAccRows(rows)     the live rows, for the error readout
//   fc.destroy()
import { dapi } from './ui.js';

const KINDS = ['ecg', 'acc', 'rr'];
const ACC_LANES = { x: 'acc_x_mg', y: 'acc_y_mg', z: 'acc_z_mg' };
const AHEAD_MS = 1300;                    // room right of "now" on a chart while its forecast is on

// Running forecast error: every forecast against the samples that then arrived (|actual - median| over its
// horizon, and the same for "hold the last value"), averaged over the last 30 s of forecasts.
export class ForecastErrors {
    constructor(lanes) { this.lanes = lanes; this.open = []; this.done = []; }

    add(fc) {
        const idx = this.lanes.map(n => fc.lanes.indexOf(n));
        if (idx.some(i => i < 0)) return;
        this.open.push({ t0: fc.t0, dt: fc.dt_ms, n: fc.median[idx[0]].length,
                         med: idx.map(i => fc.median[i]), last: idx.map(i => fc.last[i]), sum: 0, hold: 0, cnt: 0 });
        if (this.open.length > 8) this.open.shift();
    }

    observe(ts, values) {
        for (const o of this.open) {
            const pos = (ts - o.t0) / o.dt;
            if (pos < 0 || pos > o.n - 1) continue;
            const k = Math.floor(pos), w = pos - k;
            for (let j = 0; j < values.length; j++) {
                const m = o.med[j];
                const pred = k + 1 < o.n ? m[k] * (1 - w) + m[k + 1] * w : m[k];
                o.sum += Math.abs(values[j] - pred);
                o.hold += Math.abs(values[j] - o.last[j]);
                o.cnt++;
            }
        }
        const now = Date.now();
        this.open = this.open.filter(o => {
            if (ts <= o.t0 + (o.n - 1) * o.dt) return true;
            if (o.cnt) this.done.push({ mae: o.sum / o.cnt, hold: o.hold / o.cnt, at: now });
            return false;
        });
        this.done = this.done.filter(d => d.at >= now - 30000);
    }

    summary() {
        if (!this.done.length) return null;
        const avg = (key) => this.done.reduce((s, d) => s + d[key], 0) / this.done.length;
        return { mae: avg('mae'), hold: avg('hold'), n: this.done.length };
    }

    reset() { this.open = []; this.done = []; }
}

export class Forecasts {
    constructor({ monitor, acc, rr, buttons, status }) {
        Object.assign(this, { monitor, acc, rr, buttons, status });
        this.on = { ecg: false, acc: false, rr: false };
        this.latest = {}; this.receivedAt = {}; this.every = {}; this.service = null; this.available = false;
        this.errors = { ecg: new ForecastErrors(['ecg_uv']), acc: new ForecastErrors(Object.values(ACC_LANES)) };
        this._clicks = KINDS.map(k => { const fn = () => this.toggle(k); buttons[k].addEventListener('click', fn); return [buttons[k], fn]; });
        this.refresh();
        this.timer = setInterval(() => this.refresh(), 2000);      // the service's state (it heartbeats the API)
        this.ticker = setInterval(() => this._tick(), 500);
    }

    destroy() {
        clearInterval(this.timer); clearInterval(this.ticker);
        for (const [b, fn] of this._clicks) b.removeEventListener('click', fn);
        this.monitor.setForecast(null);
    }

    async refresh() {
        let view;
        try { view = await dapi('heart', '/api/forecast/settings', { quiet: true }); }
        catch (e) { this.available = false; this._render(); return; }        // the Heart API isn't answering: no buttons
        this._apply(view);
    }

    async toggle(kind) {
        try { this._apply(await dapi('heart', '/api/forecast/settings', { body: { [kind]: !this.on[kind] } })); }
        catch (e) { /* the toast says why */ }
    }

    _apply(view) {
        if (!view || !view.settings) return;
        this.available = true;
        this.service = view.service || null;
        for (const k of KINDS) {
            const was = this.on[k];
            this.on[k] = !!view.settings[k];
            if (was !== this.on[k]) {
                if (!this.on[k]) this._clear(k);
                if (k === 'acc') this.acc.setAhead(this.on.acc ? AHEAD_MS : 0);
                if (k === 'rr') this.rr.setAhead(this.on.rr ? AHEAD_MS : 0);
            }
        }
        this._render();
    }

    onMessage(m) {
        if (m.type === 'forecast_settings') return this._apply(m);
        const kind = (m.type || '').replace('_forecast', '');
        if (!KINDS.includes(kind) || !this.on[kind] || !m.lanes) return;
        const now = Date.now();
        if (this.receivedAt[kind]) {                // how often this kind arrives (slow on a CPU, or with all three on)
            const gap = now - this.receivedAt[kind];
            this.every[kind] = this.every[kind] ? this.every[kind] * 0.7 + gap * 0.3 : gap;
        }
        this.latest[kind] = m; this.receivedAt[kind] = now;
        const times = [m.t0 - m.dt_ms, ...m.median[0].map((_, i) => m.t0 + i * m.dt_ms)];     // from the last real row, so it joins the trace
        const lane = (name, what) => { const i = m.lanes.indexOf(name); return i < 0 ? null : [m.last[i], ...m[what][i]]; };
        if (kind === 'ecg') {
            if (!lane('ecg_uv', 'rep')) return;
            this.errors.ecg.add(m);
            this.monitor.setForecast({ t0: times[0], dt: m.dt_ms, rep: lane('ecg_uv', 'rep'), lo: lane('ecg_uv', 'lo'), hi: lane('ecg_uv', 'hi') });
        } else if (kind === 'acc') {
            this.errors.acc.add(m);
            for (const [axis, name] of Object.entries(ACC_LANES)) {
                const rep = lane(name, 'rep');
                if (rep) this.acc.set('f' + axis, rep.map((y, i) => [times[i], y]), { lo: lane(name, 'lo'), hi: lane(name, 'hi') });
            }
        } else {
            const beats = m.beats || [];
            this.rr.set('next', beats.map(b => [b.t, b.rr]), { lo: beats.map(b => b.lo), hi: beats.map(b => b.hi) });
        }
        this._render();
    }

    onEcgRows(rows) { if (this.on.ecg) for (const r of rows) this.errors.ecg.observe(r[0], [r[1]]); }
    onAccRows(rows) { if (this.on.acc) for (const r of rows) this.errors.acc.observe(r[0], [r[1], r[2], r[3]]); }

    _clear(kind, forget = true) {             // forget: also how often it came (when switched off)
        delete this.latest[kind];
        if (forget) { delete this.receivedAt[kind]; delete this.every[kind]; }
        if (kind === 'ecg') { this.errors.ecg.reset(); this.monitor.setForecast(null); }
        else if (kind === 'acc') { this.errors.acc.reset(); for (const a of 'xyz') this.acc.set('f' + a, [], { lo: [], hi: [] }); }
        else this.rr.set('next', [], { lo: [], hi: [] });
    }

    _tick() {
        // the service stopped: nothing for 3 s, or for three of its usual gaps when it's slower than that
        for (const k of KINDS) if (this.latest[k] && Date.now() - this.receivedAt[k] > Math.max(3000, 3 * (this.every[k] || 0))) this._clear(k, false);
        this._render();
    }

    _text(kind) {
        const svc = this.service || {}, ov = (svc.overlays || {})[kind] || {}, p = this.latest[kind];
        if (!svc.online) return ['Forecast service isn’t running', true, 'It starts with Bio-dash when the models are in polar_dashboard/forecast/models'];
        if (svc.state === 'unavailable') return ['Forecasts unavailable', true, svc.detail || ''];
        if (ov.error) return [ov.error, true, ov.error];
        if (!p) return [svc.state === 'loading model' ? 'Loading the model…' : svc.state === 'waiting for data' ? 'Waiting for data' : 'Starting…', false, ''];
        const about = `${p.model}: ${p.samples} samples from ${p.context_rows} rows of context. A research demo, not a medical device.`;
        if (kind === 'rr') {
            const sc = p.rr_score || ov.rr_score;                // forecast_service.py: {n, model_ms, hold_ms}
            return [`${(p.beats || []).length ? 'next beat ' + Math.round(p.beats[0].rr) + ' ms' : 'no beat within the horizon'}`
                + (sc && sc.n && sc.model_ms != null ? ` · error ${Math.round(sc.model_ms)} ms (hold ${Math.round(sc.hold_ms)})` : ''), false,
                `${about} Ring = the samples’ median R-R for the next beat, bar = their range; error = each beat against the last forecast made before it (hold = repeating the previous R-R).`];
        }
        let text = `${(p.median[0].length * p.dt_ms / 1000).toFixed(1)} s ahead · ${Math.round(p.compute_ms)} ms`;
        const e = this.errors[kind].summary();
        if (e) text += kind === 'ecg' ? ` · error ${e.mae.toFixed(3)} mV (hold ${e.hold.toFixed(3)})` : ` · error ${e.mae.toFixed(0)} mg (hold ${e.hold.toFixed(0)})`;
        const m = this.monitor, end = p.t0 + (p.median[0].length - 1) * p.dt_ms;
        if (kind === 'ecg' && m.clockOffset !== null && end <= Date.now() - m.clockOffset - m.PLAY_DELAY_MS) {
            return [`Forecasts arrive too late to draw (${Math.round(p.compute_ms)} ms each; this needs a GPU)`, true, about];
        }
        return [text, false, `${about} Band = range of the samples, dashed = one representative sample; error = mean |actual − median| over the last 30 s (hold = repeating the last value).`];
    }

    _render() {
        for (const k of KINDS) {
            const b = this.buttons[k], st = this.status[k];
            b.hidden = !this.available;
            b.setAttribute('aria-pressed', String(this.on[k]));
            let text = '', warn = false, title = '';
            // a forecast this page can't read must never take the buttons away (they'd be the only way to switch it off)
            if (this.on[k]) {
                try { [text, warn, title] = this._text(k); }
                catch (e) { console.error(`forecast status (${k})`, e); text = 'Forecast on'; }
            }
            st.textContent = text.length > 90 ? text.slice(0, 88) + '…' : text;
            st.title = title || text;
            st.classList.toggle('warn', warn);
        }
    }
}
