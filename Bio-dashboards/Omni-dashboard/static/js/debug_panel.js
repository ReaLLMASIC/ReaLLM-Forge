// Omni debug panel: radio routing (which radio carries the Polar vs the O2 + scanning),
// both device links, scanner state. Toggle with the 🐞 button or "D"; polls /api/debug.

// Swap (don't stack) colour classes: in the compiled CSS the slate base classes can
// come after the accent ones, so adding an accent on top of slate would silently lose.
function setPillActive(el, on) {
    const off = ['bg-slate-700/50', 'text-slate-300', 'border-slate-600'];
    const act = ['bg-sky-500/10', 'text-sky-400', 'border-sky-500/40'];
    el.classList.remove(...(on ? off : act));
    el.classList.add(...(on ? act : off));
}

const DebugPanel = (() => {
    const panel = document.getElementById('debug-panel');
    const body = document.getElementById('debug-body');
    const btn = document.getElementById('debug-btn');
    const esc = (s) => String(s ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

    const arrivals = [];
    const onPacket = () => { arrivals.push(performance.now()); };

    // --- helpers ---
    const row = (k, v, cls = 'text-slate-200') =>
        `<div class="flex justify-between gap-3 py-1 border-b border-slate-700/50 last:border-0">
            <span class="text-slate-400">${k}</span><span class="font-mono ${cls} text-right break-all">${v}</span></div>`;
    const section = (title, content) =>
        `<div class="bg-slate-900/40 border border-slate-700 rounded-xl p-4">
            <h3 class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">${title}</h3>${content}</div>`;
    const pill = (text, cls) => `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold border whitespace-nowrap ${cls}">${text}</span>`;
    const fmtDur = (s) => s == null ? '—' : s < 60 ? `${Math.floor(s)} s` : `${Math.floor(s / 60)} m ${Math.floor(s % 60)} s`;
    const ago = (t) => t ? Date.now() / 1000 - t : null;
    const countCls = (n) => n ? 'text-amber-400' : 'text-emerald-400';
    const rssiCls = (r) => r == null ? 'text-slate-500' : r > -70 ? 'text-emerald-400' : r > -85 ? 'text-amber-400' : 'text-red-400';
    const health = (m, n, tol = [0.02, 0.10]) => (!n || !m) ? 'text-slate-500'
        : Math.abs(m - n) / n < tol[0] ? 'text-emerald-400' : Math.abs(m - n) / n < tol[1] ? 'text-amber-400' : 'text-red-400';
    const stateCls = (st) => ({ Connected: 'text-emerald-400', Calibrating: 'text-sky-400', Connecting: 'text-amber-400' }[st] || 'text-slate-400');

    // ---------- radios + routing ----------
    function renderRadios(d) {
        const r = d.routing || {}, cfg = d.config || {}, list = d.adapters || [];
        const cards = list.length ? list.map(a => {
            const roles = [];
            if (a.name === r.polar) roles.push(pill('POLAR', 'bg-rose-500/20 text-rose-400 border-rose-500/30'));
            if (a.name === r.viatom) roles.push(pill('O2 + SCAN', 'bg-sky-500/20 text-sky-400 border-sky-500/30'));
            const place = a.placement === 'internal' ? pill('INTERNAL', 'bg-slate-600/40 text-slate-300 border-slate-500/40')
                        : a.placement === 'external' ? pill('EXTERNAL', 'bg-violet-500/20 text-violet-300 border-violet-500/30') : '';
            const chip = [a.vendor, a.product].filter(Boolean).join(' · ') || (a.usb_id ? `ID ${a.usb_id}` : 'unknown chip');
            return `<div class="rounded-lg border ${roles.length ? 'border-sky-500/30 bg-sky-500/5' : 'border-slate-700'} p-3 mb-2">
                <div class="mb-1">
                    <span class="font-mono font-bold text-slate-100">${esc(a.address || a.name)}</span>
                    <span class="text-[11px] font-mono text-slate-500 ml-1 whitespace-nowrap">currently ${esc(a.name)}</span>
                </div>
                <div class="flex flex-wrap gap-1 mb-1.5">${roles.join('')}${place}
                    ${a.powered === false ? pill('OFF', 'bg-red-500/20 text-red-400 border-red-500/30') : ''}</div>
                <div class="text-xs text-slate-300">${esc(chip)}</div>
                <div class="text-xs font-mono text-slate-500 mt-0.5" title="${esc(a.placement_basis || '')}">${esc(a.bus || 'bus ?')}${a.placement && a.placement !== 'unknown' ? ' · ' + esc(a.placement) : ''}${a.usb_id ? ' · ' + esc(a.usb_id) : ''}</div>
                ${a.error ? `<div class="text-xs text-amber-400">${esc(a.error)}</div>` : ''}
            </div>`;
        }).join('') : row('Adapters', 'none reported yet', 'text-slate-500');

        const options = (role) => {
            const cur = (cfg[role] || 'auto').toUpperCase();
            return `<option value="auto" ${cur === 'AUTO' ? 'selected' : ''}>Auto</option>` + list.map(a =>
                `<option value="${esc(a.address)}" ${cur === (a.address || '').toUpperCase() ? 'selected' : ''}>` +
                `${esc(a.address)} (${esc(a.name)}${a.placement && a.placement !== 'unknown' ? ', ' + esc(a.placement) : ''})</option>`).join('');
        };
        const picker = (role, label) => `
            <label class="block text-xs text-slate-400 mt-2 mb-1">${label}</label>
            <select data-role="${role}" class="w-full bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-xs font-mono text-slate-200 focus:border-sky-500 focus:outline-none">${options(role)}</select>
            <div class="text-[11px] text-slate-500 mt-0.5">→ ${esc(r[role])} · ${esc((r.why || {})[role])}</div>`;

        const shared = r.polar && r.polar === r.scan;
        return section('Bluetooth Radios & Routing',
            cards +
            `<div class="mt-3 pt-3 border-t border-slate-700">
                ${picker('polar', 'Polar H10 radio')}
                ${picker('viatom', 'O2 + scanning radio')}
                ${shared ? `<div class="text-[11px] text-amber-400 mt-2">One radio is carrying the Polar and scanning, so scanning pauses while the H10 streams. A second radio (e.g. a USB dongle) fixes this.</div>` : ''}
                <div class="text-[11px] text-slate-500 mt-2">Saved by address, so hci renumbering doesn't matter. Applies on the next connection.</div>
            </div>`);
    }

    function renderPolar(d) {
        const p = d.polar || {}, r = p.rates || {}, n = p.nominal || {}, dev = p.device || {};
        const live = p.state === 'Connected';
        return section('Polar H10 Link',
            row('State', esc(p.state), stateCls(p.state)) +
            row('Device', esc(dev.name)) +
            row('RSSI at scan', dev.rssi != null ? `${dev.rssi} dBm` : '—', rssiCls(dev.rssi)) +
            row('Radio', esc(p.adapter)) +
            row('Connected for', fmtDur(ago(p.connected_at))) +
            row('Saving to', p.recording ? `<span title="${esc(p.recording)}">${esc(p.recording.split('/').slice(-3).join('/'))}</span>` : '—', 'text-slate-300 text-xs') +
            row('ECG', live ? `${r.ecg_hz} / ${n.ecg_hz} Hz` : '—', live ? health(r.ecg_hz, n.ecg_hz) : 'text-slate-500') +
            row('ACC', live ? `${r.acc_hz} / ${n.acc_hz} Hz` : '—', live ? health(r.acc_hz, n.acc_hz) : 'text-slate-500') +
            row('ECG frames', live ? `${r.ecg_frames_per_s}/s × ${r.ecg_samples_per_frame}` : '—') +
            row('HR notifications', live ? `${r.hr_notifications_per_s}/s` : '—') +
            row('Dropped ECG / ACC', r.ecg_dropped != null ? `${r.ecg_dropped} / ${r.acc_dropped}` : '—',
                (r.ecg_dropped || r.acc_dropped) ? 'text-amber-400' : 'text-emerald-400') +
            row('Link failures', p.link_failures ?? '—', countCls(p.link_failures)));
    }

    function renderViatom(d) {
        const v = d.viatom || {}, r = v.rates || {}, c = v.counters || {}, dev = v.device || {}, g = v.gatt || {};
        const live = v.state === 'Connected' || v.state === 'Calibrating';
        const age = ago(v.last_reading_at);
        return section('Checkme O2 Link',
            row('State', esc(v.state), stateCls(v.state)) +
            row('Device', esc(dev.name)) +
            row('RSSI at scan', dev.rssi != null ? `${dev.rssi} dBm` : '—', rssiCls(dev.rssi)) +
            row('Radio', esc(v.adapter)) +
            row('Connected for', fmtDur(ago(v.connected_at))) +
            row('Saving to', v.recording ? `<span title="${esc(v.recording)}">${esc(v.recording.split('/').slice(-3).join('/'))}</span>` : '—', 'text-slate-300 text-xs') +
            row('Valid readings', live ? `${r.readings_per_s} / ${(v.nominal || {}).readings_per_s} /s` : '—',
                live ? health(r.readings_per_s, (v.nominal || {}).readings_per_s, [0.15, 0.4]) : 'text-slate-500') +
            row('Last reading', age == null ? '—' : `${age.toFixed(1)} s ago`,
                age == null ? 'text-slate-500' : age < 5 ? 'text-emerald-400' : age < 15 ? 'text-amber-400' : 'text-red-400') +
            row('Fragments / calibrating', `${c.fragments_skipped ?? '—'} / ${c.calibrating_packets ?? '—'}`, 'text-slate-300') +
            row('Out-of-range SpO₂', c.out_of_range ?? '—', countCls(c.out_of_range)) +
            row('Write failures', c.write_failures ?? '—', countCls(c.write_failures)) +
            row('Write mode · MTU', `${esc(g.write_mode)} · ${esc(g.mtu)}`) +
            row('Link failures', v.link_failures ?? '—', countCls(v.link_failures)));
    }

    function renderScanner(d) {
        const s = d.scanner || {}, v = d.versions || {}, r = d.routing || {};
        const now = performance.now();
        while (arrivals.length && now - arrivals[0] > 10000) arrivals.shift();
        const mon = (typeof monitor !== 'undefined') ? monitor : null;
        const sweep = ago(s.last_sweep_at);
        return section('Scanner, Browser & Versions',
            row('Scan radio', esc(r.scan)) +
            row('Scanner', s.scanning ? 'scanning' : s.paused_reason ? 'paused' : 'idle',
                s.scanning ? 'text-sky-400' : 'text-slate-400') +
            (s.paused_reason ? row('Paused because', esc(s.paused_reason), 'text-slate-400') : '') +
            row('Last sweep', sweep == null ? '—' : `${sweep.toFixed(0)} s ago`) +
            row('SSE updates (10 s)', `${(arrivals.length / 10).toFixed(1)}/s`) +
            row('ECG buffered', mon ? `${mon.queue.length - mon.qHead} samples` : '—') +
            row('bleak', esc(v['bleak'])) +
            row('polar-python', esc(v['polar-python'])) +
            row('Python', esc(v['python'])));
    }

    async function refresh() {
        if (panel.classList.contains('hidden')) return;
        // Don't redraw under an open dropdown
        if (panel.contains(document.activeElement) && document.activeElement.tagName === 'SELECT') return;
        let d = {};
        try { d = await (await fetch('/api/debug')).json(); } catch (e) {}
        body.innerHTML = renderRadios(d) + renderPolar(d) + renderViatom(d) + renderScanner(d);
        document.getElementById('debug-updated').textContent = new Date().toLocaleTimeString();
    }

    // Radio pickers
    body.addEventListener('change', async (e) => {
        const sel = e.target.closest('select[data-role]');
        if (!sel) return;
        sel.disabled = true;
        try {
            await fetch('/api/radios', { method: 'POST', headers: {'Content-Type': 'application/json'},
                                         body: JSON.stringify({ role: sel.dataset.role, address: sel.value }) });
        } catch (err) {}
        sel.blur();
        refresh();
    });

    function setOpen(open) {
        panel.classList.toggle('hidden', !open);
        setPillActive(btn, open);
        try { localStorage.setItem('omni-debug-open', open ? '1' : '0'); } catch (e) {}
        if (open) refresh();
    }
    const toggle = () => setOpen(panel.classList.contains('hidden'));

    btn.addEventListener('click', toggle);
    document.addEventListener('keydown', (e) => {
        if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey && !e.altKey &&
            !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) toggle();
    });
    setInterval(refresh, 1000);
    try { if (localStorage.getItem('omni-debug-open') === '1') setOpen(true); } catch (e) {}

    return { onPacket };
})();
