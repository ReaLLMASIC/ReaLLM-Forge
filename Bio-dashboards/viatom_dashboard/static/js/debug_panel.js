// Debug panel: Bluetooth radio, link, GATT and measured data rates for the O2 Ultra.
// Toggle with the 🐞 button or the "D" key. Polls /api/debug once a second while open.

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
    const btn = document.getElementById('debug-btn');
    const esc = (s) => String(s ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

    // Browser-side SSE arrivals over a 10 s window (updates are slow, ~1 per 2 s)
    const arrivals = [];
    const onPacket = () => { arrivals.push(performance.now()); };

    // --- helpers ---
    const row = (k, v, cls = 'text-slate-200') =>
        `<div class="flex justify-between gap-3 py-1 border-b border-slate-700/50 last:border-0">
            <span class="text-slate-400">${k}</span><span class="font-mono ${cls} text-right break-all">${v}</span></div>`;
    const section = (title, body) =>
        `<div class="bg-slate-900/40 border border-slate-700 rounded-xl p-4">
            <h3 class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">${title}</h3>${body}</div>`;
    const pill = (text, cls) => `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold border whitespace-nowrap ${cls}">${text}</span>`;
    const placementPill = (a) => a.placement === 'internal'
        ? `<span title="${esc(a.placement_basis)}">${pill('INTERNAL', 'bg-slate-600/40 text-slate-300 border-slate-500/40')}</span>`
        : a.placement === 'external'
        ? `<span title="${esc(a.placement_basis)}">${pill('EXTERNAL', 'bg-violet-500/20 text-violet-300 border-violet-500/30')}</span>` : '';
    const fmtDur = (s) => s == null ? '—' : s < 60 ? `${Math.floor(s)} s` : `${Math.floor(s / 60)} m ${Math.floor(s % 60)} s`;
    const shortUuid = (u) => u ? `${u.slice(0, 8)}…${u.slice(-4)}` : '—';
    const countCls = (n, bad = 'text-amber-400') => n ? bad : 'text-emerald-400';

    function renderRadio(dbg) {
        const inUse = dbg.adapter;
        if (!dbg.adapters || !dbg.adapters.length) {
            return section('Bluetooth Radio', row('Adapters', 'none reported yet', 'text-slate-500'));
        }
        const cards = dbg.adapters.map(a => {
            const used = a.name === inUse;
            const chip = [a.vendor, a.product].filter(Boolean).join(' · ') || (a.usb_id ? `ID ${a.usb_id}` : 'unknown chip');
            return `<div class="rounded-lg border ${used ? 'border-sky-500/40 bg-sky-500/5' : 'border-slate-700'} p-3 mb-2 last:mb-0">
                <div class="mb-1">
                    <span class="font-mono font-bold text-slate-100">${esc(a.address || a.name)}</span>
                    <span class="text-[11px] font-mono text-slate-500 ml-1 whitespace-nowrap" title="hci numbers follow enumeration order and can change">currently ${esc(a.name)}</span>
                </div>
                <div class="flex flex-wrap gap-1 mb-1.5">
                        ${placementPill(a)}
                        ${used ? pill('IN USE', 'bg-sky-500/20 text-sky-400 border-sky-500/30') : ''}
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
            <select data-radio class="w-full bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-xs font-mono text-slate-200 focus:border-sky-500 focus:outline-none">
                <option value="auto" ${cur === 'AUTO' ? 'selected' : ''}>Auto</option>
                ${(dbg.adapters || []).map(a => `<option value="${esc(a.address)}" ${cur === (a.address || '').toUpperCase() ? 'selected' : ''}>${esc(a.address)} (${esc(a.name)}${a.placement && a.placement !== 'unknown' ? ', ' + esc(a.placement) : ''})</option>`).join('')}
            </select>
            <div class="text-[11px] text-slate-500 mt-0.5">→ ${esc(r.hci || 'system default')} · ${esc(r.reason || '—')}</div>
            <div class="text-[11px] text-slate-500 mt-1">Saved by address. Applies on the next scan; a live connection isn't dropped.</div>
        </div>`;
        return section('Bluetooth Radio', cards + note + picker);
    }

    function renderLink(dbg) {
        const d = dbg.device || {}, c = dbg.counters || {};
        const up = dbg.connected_at ? (Date.now() / 1000 - dbg.connected_at) : null;
        const stateCls = { Connected: 'text-emerald-400', Calibrating: 'text-sky-400', Connecting: 'text-amber-400',
                           Scanning: 'text-sky-400' }[dbg.state] || 'text-slate-300';
        return section('Link',
            row('State', esc(dbg.state), stateCls) +
            row('Device', esc(d.name)) +
            row('Address', esc(d.address)) +
            row('RSSI at scan', d.rssi != null ? `${d.rssi} dBm` : '—',
                d.rssi == null ? 'text-slate-500' : d.rssi > -70 ? 'text-emerald-400' : d.rssi > -85 ? 'text-amber-400' : 'text-red-400') +
            row('Radio', esc(dbg.adapter)) +
            row('Connected for', fmtDur(up)) +
            row('Saving to', dbg.recording ? `<span title="${esc(dbg.recording)}">${esc(dbg.recording.split('/').slice(-3).join('/'))}</span>` : '—', 'text-slate-300 text-xs') +
            row('Link failures', c.link_failures ?? '—', countCls(c.link_failures, 'text-amber-400')));
    }

    function renderRates(dbg) {
        const r = dbg.rates || {}, n = dbg.nominal || {}, c = dbg.counters || {};
        const age = dbg.last_reading_at ? Date.now() / 1000 - dbg.last_reading_at : null;
        const nominal = n.readings_per_s;
        const readCls = !r.readings_per_s ? 'text-slate-500'
            : Math.abs(r.readings_per_s - nominal) / nominal < 0.15 ? 'text-emerald-400'
            : Math.abs(r.readings_per_s - nominal) / nominal < 0.4 ? 'text-amber-400' : 'text-red-400';
        return section('Data Rates (worker, 20 s window)',
            row('Valid readings', r.readings_per_s != null ? `${r.readings_per_s} / ${nominal} /s` : '—', readCls) +
            row('Notifications', r.notifications_per_s != null ? `${r.notifications_per_s}/s` : '—') +
            row('Polls sent', r.polls_per_s != null ? `${r.polls_per_s}/s (every ${n.poll_interval_s} s)` : '—') +
            row('Last reading', age == null ? '—' : `${age.toFixed(1)} s ago`,
                age == null ? 'text-slate-500' : age < 5 ? 'text-emerald-400' : age < 15 ? 'text-amber-400' : 'text-red-400') +
            row('Fragments skipped', c.fragments_skipped ?? '—', 'text-slate-300') +
            row('Calibrating packets', c.calibrating_packets ?? '—', 'text-slate-300') +
            row('Out-of-range SpO₂', c.out_of_range ?? '—', countCls(c.out_of_range)) +
            row('Write failures', c.write_failures ?? '—', countCls(c.write_failures)));
    }

    function renderGatt(dbg) {
        const g = dbg.gatt || {}, v = dbg.versions || {};
        const now = performance.now();
        while (arrivals.length && now - arrivals[0] > 10000) arrivals.shift();
        const sse = (arrivals.length / 10).toFixed(2);
        return section('GATT, Browser & Versions',
            row('Notify char', `<span title="${esc(g.notify_uuid)}">${esc(shortUuid(g.notify_uuid))}</span>`) +
            row('Write char', `<span title="${esc(g.write_uuid)}">${esc(shortUuid(g.write_uuid))}</span>`) +
            row('Write mode', esc(g.write_mode)) +
            row('MTU', esc(g.mtu)) +
            row('SSE updates (10 s)', `${sse}/s`) +
            row('bleak', esc(v['bleak'])) +
            row('Flask', esc(v['flask'])) +
            row('Python', esc(v['python'])));
    }

    async function refresh() {
        if (panel.classList.contains('hidden')) return;
        if (panel.contains(document.activeElement) && document.activeElement.tagName === 'SELECT') return;   // don't redraw an open dropdown
        let dbg = {};
        try { dbg = await (await fetch('/api/debug')).json(); } catch (e) {}
        document.getElementById('debug-body').innerHTML =
            renderRadio(dbg) + renderLink(dbg) + renderRates(dbg) + renderGatt(dbg);
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
        try { localStorage.setItem('viatom-debug-open', open ? '1' : '0'); } catch (e) {}
        if (open) refresh();
    }
    const toggle = () => setOpen(panel.classList.contains('hidden'));

    btn.addEventListener('click', toggle);
    document.addEventListener('keydown', (e) => {
        if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey && !e.altKey &&
            !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) toggle();
    });
    setInterval(refresh, 1000);
    try { if (localStorage.getItem('viatom-debug-open') === '1') setOpen(true); } catch (e) {}

    return { onPacket };
})();
