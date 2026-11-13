// Heart (Polar H10): live ECG, heart rate, HRV, beat-to-beat intervals and motion, for one person's strap
// (`device`: "heart", or "heart-p2" ... for another person's). The forecasts follow the first person's.
import { icon, num, dapi, seg, prefs, $ } from '../ui.js';
import { Plot } from '../plot.js';
import { EcgMonitor } from '../ecg.js';
import { Forecasts } from '../forecast.js';
import { pageHead, bindHead, connectPanel } from '../device.js';

function hrvWord(rmssd) {
    if (rmssd == null) return ['', ''];
    if (rmssd < 20) return ['Low — stressed or working hard', 'warn'];
    if (rmssd < 50) return ['Balanced', 'ok'];
    return ['High — relaxed', 'ok'];
}
function contactWord(pp) {
    if (pp == null) return ['', ''];
    if (pp < 0.25) return ['Weak — wet the pads', 'warn'];
    if (pp < 0.6) return ['OK', 'ok'];
    return ['Good contact', 'ok'];
}

export async function mount(el, { store, device: key }) {
    const canForecast = key === 'heart' && store.app.forecast;
    const fcBtn = (kind, title) => `<button class="btn sm" data-fc="${kind}" aria-pressed="false" hidden title="${title}">${icon('sparkle')}<span>Forecast</span></button>`;
    el.innerHTML = `${pageHead(key)}
      <div data-connect></div>
      <div class="callout warn" data-warn hidden style="margin-bottom:16px">${icon('warn')}<span data-warn-text></span></div>
      <div class="stack" data-live>
        <div class="card flush">
          <div class="ecg" data-ecg>
            <div class="overlay"><div class="hr"><span data-hr>--</span><small>bpm</small></div>
              <svg class="beat" viewBox="0 0 24 24" fill="currentColor" data-beat><path d="M12 21s-7.5-4.6-9.2-9.4C1.6 8.2 3.8 5 7.2 5c2 0 3.6 1.1 4.8 2.8C13.2 6.1 14.8 5 16.8 5c3.4 0 5.6 3.2 4.4 6.6C19.5 16.4 12 21 12 21z"/></svg>
              <span class="spacer"></span></div>
            <div class="nosig" data-nosig hidden>Waiting for the ECG…</div>
          </div>
          <div class="row" style="padding:10px 14px;border-top:1px solid var(--line)">
            <span class="small muted">ECG · 25 mm/s</span><span class="small muted fc-status" data-fc-status="ecg"></span><span class="spacer"></span>
            ${fcBtn('ecg', 'Draw the next second, as the ECG model predicts it, ahead of the trace')}
            <button class="btn sm" data-gain title="How tall the trace is drawn">Size: Auto</button>
            <button class="btn sm" data-beep aria-pressed="false">${icon('mute')}<span>Beep off</span></button>
          </div>
        </div>
        <div class="grid tiles" style="--c:var(--heart)">
          <div class="tile accent"><div class="label">Heart rate</div><div class="value"><span data-v="hr">--</span><small>bpm</small></div><div class="note" data-n="hr"></div></div>
          <div class="tile"><div class="label">Heart-rate variability</div><div class="value"><span data-v="rmssd">--</span><small>ms</small></div><div class="note" data-n="rmssd"></div></div>
          <div class="tile"><div class="label">Last beat-to-beat</div><div class="value"><span data-v="rr">--</span><small>ms</small></div><div class="note">R-R interval</div></div>
          <div class="tile"><div class="label">ECG signal</div><div class="value"><span data-v="pp">--</span><small>mV</small></div><div class="note" data-n="pp"></div></div>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="card-head"><h2>Trend</h2><span class="spacer"></span>
          <div class="seg" data-win><button data-v="5">5 min</button><button data-v="15">15 min</button><button data-v="60">1 hour</button></div></div>
        <div class="legend" style="margin-bottom:8px"><span><i style="background:var(--heart)"></i>Heart rate (bpm)</span><span><i style="background:var(--oxygen)"></i>HRV, RMSSD (ms)</span></div>
        <div class="chart" data-trend></div>
      </div>
      <div class="grid cols-2" style="margin-top:16px" data-live2>
        <div class="card"><div class="card-head"><h2>Beat-to-beat</h2><span class="card-sub">last 2 minutes</span><span class="spacer"></span>
            ${fcBtn('rr', 'Mark the next beats, as the ECG + R-R model predicts them')}</div>
          <div class="chart short" data-rr></div><div class="small muted fc-status" data-fc-status="rr"></div></div>
        <div class="card"><div class="card-head"><h2>Motion</h2><span class="card-sub">accelerometer, last 20 s</span><span class="spacer"></span>
            ${fcBtn('acc', 'Draw the next second of movement, as the ECG + accelerometer model predicts it')}</div>
          <div class="legend" style="margin-bottom:6px"><span><i style="background:#38bdf8"></i>X</span><span><i style="background:#a78bfa"></i>Y</span><span><i style="background:#fbbf24"></i>Z</span></div>
          <div class="chart short" data-acc></div><div class="small muted fc-status" data-fc-status="acc"></div></div>
      </div>
      <p class="small faint" style="margin-top:18px">HRV here is RMSSD over the last 20 beats. It’s a guide, not a diagnosis.${canForecast
        ? ' Forecasts come from small models trained on H10 recordings: a research demo. <a href="/lab" target="_blank" rel="noopener">Open the model lab</a>.' : ''}</p>`;

    let win = prefs.get('heart.win', 15);
    const trend = new Plot($('[data-trend]', el), {
        series: [{ id: 'hr', label: 'Heart rate', color: 'var(--heart)', unit: 'bpm', type: 'line', fill: true },
                 { id: 'rmssd', label: 'HRV', color: 'var(--oxygen)', unit: 'ms', axis: 'y2', dp: 1, width: 1.6 }],
        x: { window: win * 60e3 }, y: { minSpan: 10 }, y2: { minSpan: 10, zero: true }, empty: 'No heart data in this window',
    });
    const rr = new Plot($('[data-rr]', el), { series: [{ id: 'rr', label: 'R-R', color: 'var(--heart)', unit: 'ms', type: 'dots' },
                                                       { id: 'next', label: 'Forecast', color: 'var(--forecast)', unit: 'ms', type: 'dots', hollow: true, radius: 4, width: 2 }],
        x: { window: 120e3 }, y: { minSpan: 80 }, tick: 500, empty: 'Waiting for beats' });
    const acc = new Plot($('[data-acc]', el), {
        series: [{ id: 'x', label: 'X', color: '#38bdf8', unit: 'mg', width: 1.4 }, { id: 'y', label: 'Y', color: '#a78bfa', unit: 'mg', width: 1.4 },
                 { id: 'z', label: 'Z', color: '#fbbf24', unit: 'mg', width: 1.4 },
                 ...[['fx', '#38bdf8'], ['fy', '#a78bfa'], ['fz', '#fbbf24']].map(([id, color]) =>
                     ({ id, label: `${id[1].toUpperCase()} forecast`, color, unit: 'mg', width: 1.4, dash: [5, 4] }))],
        x: { window: 20e3 }, y: { minSpan: 200 }, gap: 3000, tick: 250, empty: 'Waiting for motion data' });
    seg($('[data-win]', el), win, (v) => { win = +v; prefs.set('heart.win', win); trend.setWindow(win * 60e3); loadHistory(); });

    async function loadHistory() {
        const m = Math.max(win, 15);
        const [h, r] = await Promise.all([dapi(key, `/api/history?metric=hr&minutes=${m}`, { quiet: true }).catch(() => ({})),
                                          dapi(key, `/api/history?metric=rmssd&minutes=${m}`, { quiet: true }).catch(() => ({}))]);
        trend.set('hr', h.hr || []);
        trend.set('rmssd', r.rmssd || []);
    }
    loadHistory();

    // ECG
    let audio = null, beepOn = false;
    const mon = new EcgMonitor($('[data-ecg]', el), {
        beatEl: $('[data-beat]', el), noSignalEl: $('[data-nosig]', el),
        onBeat: () => {
            if (!beepOn || !audio) return;
            const t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
            o.frequency.value = 880;
            g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.15, t + 0.005); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.08);
            o.connect(g).connect(audio.destination); o.start(t); o.stop(t + 0.09);
        },
    });
    const part = (attr) => Object.fromEntries(['ecg', 'acc', 'rr'].map(k => [k, $(`[${attr}="${k}"]`, el)]));
    const fc = canForecast ? new Forecasts({ monitor: mon, acc, rr, buttons: part('data-fc'), status: part('data-fc-status') }) : null;
    $('[data-gain]', el).addEventListener('click', (e) => { e.currentTarget.textContent = 'Size: ' + mon.cycleGain(); });
    const gainTimer = setInterval(() => { const b = $('[data-gain]', el); if (b && b.textContent.startsWith('Size: Auto')) b.textContent = 'Size: ' + mon.gainLabel.replace(/ \(.*/, ''); }, 2000);
    $('[data-beep]', el).addEventListener('click', (e) => {
        audio = audio || new (window.AudioContext || window.webkitAudioContext)();     // needs a tap first (browser rule)
        beepOn = !beepOn;
        e.currentTarget.setAttribute('aria-pressed', String(beepOn));
        e.currentTarget.innerHTML = `${icon(beepOn ? 'sound' : 'mute')}<span>${beepOn ? 'Beep on' : 'Beep off'}</span>`;
    });

    // values
    const V = (k, text) => { const n = el.querySelector(`[data-v="${k}"]`); if (n) n.textContent = text; };
    const N = (k, [text, tone]) => { const n = el.querySelector(`[data-n="${k}"]`); if (n) { n.textContent = text; n.style.color = tone === 'warn' ? 'var(--warn)' : ''; } };
    const ppis = [];
    let ppBucket = null, ppLo = Infinity, ppHi = -Infinity, lastData = 0;
    const offLive = store.on(`live:${key}`, (m) => {
        if (m.type.startsWith('forecast') || m.type.endsWith('_forecast')) return fc && fc.onMessage(m);
        const rows = m.rows || [];
        if (!rows.length) return;
        lastData = Date.now();
        if (m.type === 'ecg') {
            mon.push(rows);
            fc && fc.onEcgRows(rows);
            for (const [ts, mv] of rows) {
                const b = Math.floor(ts / 1000);
                if (ppBucket !== null && b !== ppBucket && ppHi > ppLo) {
                    const pp = ppHi - ppLo;
                    V('pp', num(pp, 2)); N('pp', contactWord(pp));
                    ppLo = Infinity; ppHi = -Infinity;
                }
                ppBucket = b;
                if (mv < ppLo) ppLo = mv;
                if (mv > ppHi) ppHi = mv;
            }
            const hr = rows[rows.length - 1][2];
            if (hr > 0) { V('hr', hr); $('[data-hr]', el).textContent = hr; }
        } else if (m.type === 'acc') {
            for (const [ts, x, y, z] of rows) { acc.push('x', ts, x); acc.push('y', ts, y); acc.push('z', ts, z); }
            fc && fc.onAccRows(rows);
        } else if (m.type === 'ppi') {
            for (const [ts, ppi, hr] of rows) {
                ppis.push(ppi);
                if (ppis.length > 20) ppis.shift();
                rr.push('rr', ts, ppi);
                V('rr', ppi);
                if (hr > 0) { V('hr', hr); $('[data-hr]', el).textContent = hr; trend.push('hr', ts, hr); }
                if (ppis.length > 2) {
                    let s = 0;
                    for (let i = 1; i < ppis.length; i++) s += (ppis[i] - ppis[i - 1]) ** 2;
                    const rm = Math.sqrt(s / (ppis.length - 1));
                    V('rmssd', num(rm, 1)); N('rmssd', hrvWord(rm));
                    trend.push('rmssd', ts, +rm.toFixed(1));
                }
            }
        }
    });

    // connected or not
    let stopFinder = null;
    const connectBox = $('[data-connect]', el), liveBox = $('[data-live]', el), live2 = $('[data-live2]', el);
    const offHead = bindHead(el, key, (st, s) => {
        const live = s.live || Date.now() - lastData < 5000;
        liveBox.hidden = !live; live2.hidden = !live;
        $('[data-warn]', el).hidden = !(live && st && st.warning);
        $('[data-warn-text]', el).textContent = (st && st.warning) || '';
        if (!live && !stopFinder) { stopFinder = connectPanel(connectBox, key); connectBox.hidden = false; }
        if (live && stopFinder) { stopFinder(); stopFinder = null; connectBox.innerHTML = ''; connectBox.hidden = true; }
        if (!live) { V('hr', '--'); $('[data-hr]', el).textContent = '--'; }
    });

    return () => {
        offLive(); offHead(); stopFinder && stopFinder(); clearInterval(gainTimer);
        fc && fc.destroy(); mon.destroy(); trend.destroy(); rr.destroy(); acc.destroy();
        audio && audio.close();
    };
}
