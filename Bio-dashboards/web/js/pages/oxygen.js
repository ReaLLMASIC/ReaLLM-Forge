// Oxygen (Checkme O2 ring): SpO₂, pulse and battery, live and over the last hours, for one person's ring
// (`device`: "oxygen", or "oxygen-p2" ... for another person's).
import { num, dapi, seg, prefs, duration, $ } from '../ui.js';
import { Plot } from '../plot.js';
import { pageHead, bindHead, connectPanel } from '../device.js';

function spo2Word(v) {
    if (!v) return ['', ''];
    if (v >= 95) return ['Normal', 'ok'];
    if (v >= 90) return ['A little low', 'warn'];
    return ['Low', 'bad'];
}

export async function mount(el, { store, device: key }) {
    el.innerHTML = `${pageHead(key)}
      <div data-connect></div>
      <div class="card" data-live style="--c:var(--oxygen)">
        <div class="row" style="gap:28px;align-items:flex-end">
          <div><div class="small muted" style="font-weight:650">Blood oxygen</div><div class="big" style="color:var(--oxygen)"><span data-v="spo2">--</span><small>% SpO₂</small></div>
            <div style="margin-top:6px" data-band></div></div>
          <div><div class="small muted" style="font-weight:650">Pulse</div><div class="big" style="font-size:2.4rem"><span data-v="hr">--</span><small>bpm</small></div></div>
          <div><div class="small muted" style="font-weight:650">Ring battery</div><div class="big" style="font-size:2.4rem"><span data-v="bat">--</span><small>%</small></div></div>
        </div>
        <div class="callout" data-settle hidden style="margin-top:14px"><span>Settling — keep your hand still for a few seconds while the ring finds your pulse.</span></div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="card-head"><h2>Over time</h2><span class="spacer"></span>
          <div class="seg" data-win><button data-v="15">15 min</button><button data-v="60">1 h</button><button data-v="360">6 h</button><button data-v="1440">24 h</button></div></div>
        <div class="legend" style="margin-bottom:8px"><span><i style="background:var(--oxygen)"></i>SpO₂ (%)</span><span><i style="background:var(--heart)"></i>Pulse (bpm)</span></div>
        <div class="chart tall" data-chart></div>
        <div class="grid tiles" style="margin-top:14px" data-stats></div>
      </div>
      <p class="small faint" style="margin-top:18px">Pulse oximeters read a few percent off on cold hands, dark nail polish or with movement. A guide, not a diagnosis.</p>`;

    let win = prefs.get('oxygen.win', 60);
    const plot = new Plot($('[data-chart]', el), {
        series: [{ id: 'spo2', label: 'SpO₂', color: 'var(--oxygen)', unit: '%', type: 'line', fill: true },
                 { id: 'hr', label: 'Pulse', color: 'var(--heart)', unit: 'bpm', axis: 'y2', width: 1.5 }],
        x: { window: win * 60e3 }, y: { ceil: 101, minSpan: 6 }, y2: { minSpan: 20 },
        refs: [{ y: 90, label: '90 %', color: 'var(--warn)' }], gap: 30e3, tick: 2000, empty: 'No readings in this window',
    });
    seg($('[data-win]', el), win, (v) => { win = +v; prefs.set('oxygen.win', win); plot.setWindow(win * 60e3); load(); });

    let hist = { spo2: [], hr: [] };
    async function load() {
        const r = await dapi(key, `/api/history?minutes=${win}`, { quiet: true }).catch(() => null);
        if (!r) return;
        hist = { spo2: r.spo2.map(p => [p.x, p.y]), hr: r.hr.map(p => [p.x, p.y]) };
        plot.set('spo2', hist.spo2);
        plot.set('hr', hist.hr);
        stats();
    }
    function stats() {
        const cut = Date.now() - win * 60e3, pts = hist.spo2.filter(p => p[0] >= cut);
        const box = $('[data-stats]', el);
        if (!pts.length) { box.innerHTML = ''; return; }
        const vals = pts.map(p => p[1]);
        const below = (lim) => { let ms = 0; for (let i = 1; i < pts.length; i++) if (pts[i - 1][1] < lim) ms += Math.min(30e3, pts[i][0] - pts[i - 1][0]); return ms / 60e3; };
        const t = (label, value, note = '') => `<div class="tile"><div class="label">${label}</div><div class="value" style="font-size:1.4rem">${value}</div><div class="note">${note}</div></div>`;
        box.innerHTML = t('Lowest', `${Math.min(...vals)}<small>%</small>`) + t('Average', `${num(vals.reduce((a, b) => a + b, 0) / vals.length, 1)}<small>%</small>`)
            + t('Below 95 %', duration(below(95))) + t('Below 90 %', duration(below(90)));
    }
    load();
    const reload = setInterval(load, 5 * 60e3);

    let lastData = 0, lastRow = 0;
    const offLive = store.on(`live:${key}`, (d) => {
        const ok = d.status === 'Connected' && d.spo2 > 0;
        $('[data-v="spo2"]', el).textContent = ok ? d.spo2 : '--';
        $('[data-v="hr"]', el).textContent = ok && d.hr ? d.hr : '--';
        $('[data-v="bat"]', el).textContent = d.battery || '--';
        $('[data-settle]', el).hidden = d.status !== 'Calibrating';
        const [w, tone] = spo2Word(ok ? d.spo2 : 0);
        $('[data-band]', el).innerHTML = w ? `<span class="badge ${tone}">${w}</span>` : '';
        if (ok) {
            lastData = Date.now();
            plot.push('spo2', d.ts, d.spo2);
            plot.push('hr', d.ts, d.hr);
            if (d.ts - lastRow > 5000) { lastRow = d.ts; hist.spo2.push([d.ts, d.spo2]); stats(); }
        }
    });
    const latest = (store.latest || {})[key];
    if (latest && latest.status) store.emit(`live:${key}`, { ...latest, ts: latest.ts || Date.now() });

    let stopFinder = null;
    const connectBox = $('[data-connect]', el), liveBox = $('[data-live]', el);
    const offHead = bindHead(el, key, (st, s) => {
        const live = s.live || Date.now() - lastData < 8000;
        liveBox.hidden = !live;
        if (!live && !stopFinder) { stopFinder = connectPanel(connectBox, key); connectBox.hidden = false; }
        if (live && stopFinder) { stopFinder(); stopFinder = null; connectBox.innerHTML = ''; connectBox.hidden = true; }
    });
    return () => { offLive(); offHead(); stopFinder && stopFinder(); clearInterval(reload); plot.destroy(); };
}
