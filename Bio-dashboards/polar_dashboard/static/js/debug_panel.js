// Debug panel: Bluetooth radio, link, measured stream rates, browser-side stats.
// Toggle with the 🐞 button or the "D" key. Polls /api/status once a second while open.

// Swap (don't stack) colour classes: in the compiled CSS the slate base classes come
// after the rose ones, so adding rose on top of slate would silently lose.
function setPillActive(el, on) {
    const off = ['bg-slate-700/50', 'text-slate-300', 'border-slate-600'];
    const act = ['bg-rose-500/10', 'text-rose-400', 'border-rose-500/40'];
    el.classList.remove(...(on ? off : act));
    el.classList.add(...(on ? act : off));
}

const DebugPanel = (() => {
    const panel = document.getElementById('debug-panel');
    const btn = document.getElementById('debug-btn');
    const esc = (s) => String(s ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

    // Browser-side counters, fed from dashboard.js
    const browser = { msgs: 0, rows: { ecg: 0, acc: 0, ppi: 0 }, lastT: performance.now(), last: null };
    function onPacket(type, n) {
        browser.msgs++;
        if (type in browser.rows) browser.rows[type] += n;
    }

    // --- helpers ---
    const row = (k, v, cls = 'text-slate-200') =>
        `<div class="flex justify-between gap-3 py-1 border-b border-slate-700/50 last:border-0">
            <span class="text-slate-400">${k}</span><span class="font-mono ${cls} text-right">${v}</span></div>`;
    const section = (title, body) =>
        `<div class="bg-slate-900/40 border border-slate-700 rounded-xl p-4">
            <h3 class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">${title}</h3>${body}</div>`;
    const health = (measured, nominal) => {
        if (!nominal || !measured) return 'text-slate-500';
        const err = Math.abs(measured - nominal) / nominal;
        return err < 0.02 ? 'text-emerald-400' : err < 0.10 ? 'text-amber-400' : 'text-red-400';
    };
    const pill = (text, cls) => `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold border whitespace-nowrap ${cls}">${text}</span>`;
    const placementPill = (a) => a.placement === 'internal'
        ? `<span title="${esc(a.placement_basis)}">${pill('INTERNAL', 'bg-slate-600/40 text-slate-300 border-slate-500/40')}</span>`
        : a.placement === 'external'
        ? `<span title="${esc(a.placement_basis)}">${pill('EXTERNAL', 'bg-violet-500/20 text-violet-300 border-violet-500/30')}</span>` : '';
    const fmtDur = (s) => s == null ? '—' : s < 60 ? `${Math.floor(s)} s` : `${Math.floor(s / 60)} m ${Math.floor(s % 60)} s`;

    function renderRadio(dbg) {
        const inUse = dbg.adapter;
        if (!dbg.adapters || !dbg.adapters.length) {
            return section('Bluetooth Radio', row('Adapters', 'none reported yet', 'text-slate-500'));
        }
        const cards = dbg.adapters.map(a => {
            const used = a.name === inUse;
            const chip = [a.vendor, a.product].filter(Boolean).join(' · ') || (a.usb_id ? `ID ${a.usb_id}` : 'unknown chip');
            return `<div class="rounded-lg border ${used ? 'border-rose-500/40 bg-rose-500/5' : 'border-slate-700'} p-3 mb-2 last:mb-0">
                <div class="mb-1">
                    <span class="font-mono font-bold text-slate-100">${esc(a.address || a.name)}</span>
                    <span class="text-[11px] font-mono text-slate-500 ml-1 whitespace-nowrap" title="hci numbers follow enumeration order and can change">currently ${esc(a.name)}</span>
                </div>
                <div class="flex flex-wrap gap-1 mb-1.5">
                        ${placementPill(a)}
                        ${used ? pill('IN USE', 'bg-rose-500/20 text-rose-400 border-rose-500/30') : ''}
                        ${a.powered === false ? pill('OFF', 'bg-red-500/20 text-red-400 border-red-500/30')
                          : a.powered ? pill('POWERED', 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30') : ''}
                </div>
                <div class="text-xs text-slate-300">${esc(chip)}</div>
                <div class="text-xs font-mono text-slate-500 mt-0.5" title="${esc(a.placement_basis || '')}">${esc(a.bus || 'bus ?')}${a.placement && a.placement !== 'unknown' ? ' · ' + esc(a.placement) : ''}${a.usb_id ? ' · ' + esc(a.usb_id) : ''}</div>
                ${a.alias ? `<div class="text-xs text-slate-500">alias: ${esc(a.alias)}</div>` : ''}
                ${a.error ? `<div class="text-xs text-amber-400">${esc(a.error)}</div>` : ''}
            </div>`;
        }).join('');
        const note = (inUse ? '' : `<div class="text-xs text-slate-500 mt-1">Radio in use is identified once a device is selected.</div>`) +
            `<div class="text-[11px] text-slate-500 mt-1">hci numbers follow plug-in order and can swap between boots; the address is the stable identity.</div>`;
        const r = dbg.radio || {}, cur = (r.config || 'auto').toUpperCase();
        const picker = `<div class="mt-3 pt-3 border-t border-slate-700">
            <label class="block text-xs text-slate-400 mb-1">Radio for this dashboard</label>
            <select data-radio class="w-full bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-xs font-mono text-slate-200 focus:border-rose-500 focus:outline-none">
                <option value="auto" ${cur === 'AUTO' ? 'selected' : ''}>Auto</option>
                ${(dbg.adapters || []).map(a => `<option value="${esc(a.address)}" ${cur === (a.address || '').toUpperCase() ? 'selected' : ''}>${esc(a.address)} (${esc(a.name)}${a.placement && a.placement !== 'unknown' ? ', ' + esc(a.placement) : ''})</option>`).join('')}
            </select>
            <div class="text-[11px] text-slate-500 mt-0.5">→ ${esc(r.hci || 'system default')} · ${esc(r.reason || '—')}</div>
            <div class="text-[11px] text-slate-500 mt-1">Saved by address. Applies on the next scan; a live connection isn't dropped.</div>
        </div>`;
        return section('Bluetooth Radio', cards + note + picker);
    }

    function renderLink(st, dbg) {
        const d = dbg.device || {};
        const up = dbg.connected_at ? (Date.now() / 1000 - dbg.connected_at) : null;
        const stateCls = { streaming: 'text-emerald-400', connecting: 'text-amber-400', scanning: 'text-sky-400' }[st.state] || 'text-slate-300';
        return section('Link',
            row('State', esc(st.state) + (st.state === 'connecting' && st.attempt ? ` (${st.attempt}/${st.max_attempts})` : ''), stateCls) +
            row('Device', esc(d.name)) +
            row('Address', esc(d.address)) +
            row('RSSI at scan', d.rssi != null ? `${d.rssi} dBm` : '—',
                d.rssi == null ? 'text-slate-500' : d.rssi > -70 ? 'text-emerald-400' : d.rssi > -85 ? 'text-amber-400' : 'text-red-400') +
            row('Radio', esc(dbg.adapter)) +
            row('Connected for', fmtDur(up)) +
            row('Saving to', dbg.recording ? `<span title="${esc(dbg.recording)}">${esc(dbg.recording.split('/').slice(-3).join('/'))}</span>` : '—', 'text-slate-300 text-xs'));
    }

    function renderRates(dbg) {
        const r = dbg.rates || {}, n = dbg.nominal || {};
        return section('Stream Rates (worker, 5 s window)',
            row('ECG', r.ecg_hz != null ? `${r.ecg_hz} / ${n.ecg_hz} Hz` : '—', health(r.ecg_hz, n.ecg_hz)) +
            row('ACC', r.acc_hz != null ? `${r.acc_hz} / ${n.acc_hz} Hz` : '—', health(r.acc_hz, n.acc_hz)) +
            row('ECG frames', r.ecg_frames_per_s != null ? `${r.ecg_frames_per_s}/s × ${r.ecg_samples_per_frame}` : '—') +
            row('ACC frames', r.acc_frames_per_s != null ? `${r.acc_frames_per_s}/s × ${r.acc_samples_per_frame}` : '—') +
            row('HR notifications', r.hr_notifications_per_s != null ? `${r.hr_notifications_per_s}/s` : '—') +
            row('R-R intervals', r.rr_per_min != null ? `${r.rr_per_min}/min` : '—') +
            row('Dropped ECG / ACC', r.ecg_dropped != null ? `${r.ecg_dropped} / ${r.acc_dropped}` : '—',
                (r.ecg_dropped || r.acc_dropped) ? 'text-amber-400' : 'text-emerald-400'));
    }

    function renderBrowser(dbg) {
        const now = performance.now(), dt = (now - browser.lastT) / 1000;
        const rate = (k) => dt > 0 ? (browser.rows[k] / dt).toFixed(1) : '—';
        const out = {
            msgs: dt > 0 ? (browser.msgs / dt).toFixed(1) : '—',
            ecg: rate('ecg'), acc: rate('acc'),
        };
        browser.msgs = 0; browser.rows = { ecg: 0, acc: 0, ppi: 0 }; browser.lastT = now;
        browser.last = out;

        const mon = (typeof monitor !== 'undefined') ? monitor : null;
        const buffered = mon ? mon.queue.length - mon.qHead : null;
        const v = dbg.versions || {};
        return section('Browser & Versions',
            row('WebSocket msgs', `${out.msgs}/s`) +
            row('ECG / ACC rows', `${out.ecg} / ${out.acc} /s`) +
            row('ECG display delay', mon ? `${mon.PLAY_DELAY_MS} ms` : '—') +
            row('ECG buffered', buffered != null ? `${buffered} samples` : '—') +
            row('Arrival offset', mon && mon.clockOffset != null ? `${Math.round(mon.clockOffset)} ms` : '—', 'text-slate-400') +
            row('bleak', esc(v['bleak'])) +
            row('polar-python', esc(v['polar-python'])) +
            row('Python', esc(v['python'])));
    }

    async function refresh() {
        if (panel.classList.contains('hidden')) return;
        if (panel.contains(document.activeElement) && document.activeElement.tagName === 'SELECT') return;   // don't redraw an open dropdown
        let st = {};
        try { st = await (await fetch('/api/status')).json(); } catch (e) {}
        const dbg = st.debug || {};
        document.getElementById('debug-body').innerHTML =
            renderRadio(dbg) + renderLink(st, dbg) + renderRates(dbg) + renderBrowser(dbg);
        document.getElementById('debug-updated').textContent = new Date().toLocaleTimeString();
    }

    document.getElementById('debug-body').addEventListener('change', async (e) => {
        const sel = e.target.closest('select[data-radio]');
        if (!sel) return;
        sel.disabled = true;
        try {
            await fetch('/api/radio', { method: 'POST', headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify({ address: sel.value }) });
        } catch (err) {}
        sel.blur();
        refresh();
    });

    function setOpen(open) {
        panel.classList.toggle('hidden', !open);
        setPillActive(btn, open);
        try { localStorage.setItem('polar-debug-open', open ? '1' : '0'); } catch (e) {}
        if (open) { browser.msgs = 0; browser.lastT = performance.now(); refresh(); }
    }
    const toggle = () => setOpen(panel.classList.contains('hidden'));

    btn.addEventListener('click', toggle);
    document.addEventListener('keydown', (e) => {
        if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey && !e.altKey &&
            !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) toggle();
    });
    setInterval(refresh, 1000);
    try { if (localStorage.getItem('polar-debug-open') === '1') setOpen(true); } catch (e) {}

    return { onPacket };
})();
