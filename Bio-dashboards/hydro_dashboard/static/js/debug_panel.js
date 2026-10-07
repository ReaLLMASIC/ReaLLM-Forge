// Hydro-dash debug panel: radio, link, bottle protocol, sensors, and every characteristic
// with its last raw frame (the capture view for checking / decoding new firmware).
// Toggle with the 🐞 button or the "D" key. Polls /api/status once a second while open.

// Swap (don't stack) colour classes: in the compiled CSS the slate base classes come
// after the accent ones, so adding the accent on top of slate would silently lose.
function setPillActive(el, on) {
    const off = ['bg-slate-700/50', 'text-slate-300', 'border-slate-600'];
    const act = ['bg-cyan-500/10', 'text-cyan-400', 'border-cyan-500/40'];
    el.classList.remove(...(on ? off : act));
    el.classList.add(...(on ? act : off));
}

const DebugPanel = (() => {
    const panel = document.getElementById('debug-panel');
    const btn = document.getElementById('debug-btn');
    const esc = (s) => String(s ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

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
            return `<div class="rounded-lg border ${used ? 'border-cyan-500/40 bg-cyan-500/5' : 'border-slate-700'} p-3 mb-2 last:mb-0">
                <div class="mb-1">
                    <span class="font-mono font-bold text-slate-100">${esc(a.address || a.name)}</span>
                    <span class="text-[11px] font-mono text-slate-500 ml-1 whitespace-nowrap" title="hci numbers follow enumeration order and can change">currently ${esc(a.name)}</span>
                </div>
                <div class="flex flex-wrap gap-1 mb-1.5">
                        ${placementPill(a)}
                        ${used ? pill('IN USE', 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30') : ''}
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
            <select data-radio class="w-full bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-xs font-mono text-slate-200 focus:border-cyan-500 focus:outline-none">
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

    function renderProtocol(st, dbg) {
        const c = dbg.counters || {};
        const hs = dbg.handshake || '—';
        const hsCls = hs === 'complete' ? 'text-emerald-400' : hs.startsWith('failed') ? 'text-red-400' : 'text-slate-400';
        const known = { '016e11b1-6c8a-4074-9e5a-076053f93784': '016e11b1… (PRO / PRO 2)', 'bf2d1ba1-c473-49f2-9571-0ce69036c642': 'modern (bf2d1ba1…)' };
        return section('Bottle Protocol',
            row('Handshake', esc(hs), hsCls) +
            row('Sip records via', dbg.sip_char ? esc(known[dbg.sip_char] || dbg.sip_char) : 'not found', dbg.sip_char ? 'text-slate-200' : 'text-amber-400') +
            row('Sip frames', esc(c.sip_frames ?? '—')) +
            row('Sips live / replayed', `${c.sips_live ?? '—'} / ${c.sips_replayed ?? '—'}`) +
            row('Duplicates dropped', esc(c.duplicates ?? '—'), 'text-slate-300') +
            row('Undecoded sip frames', esc(c.unknown_frames ?? '—'), c.unknown_frames ? 'text-amber-400' : 'text-emerald-400') +
            row('Drain writes', esc(c.drain_writes ?? '—'), 'text-slate-300') +
            row('Skipped records', esc(c.skipped_records ?? '—'), c.skipped_records ? 'text-amber-400' : 'text-emerald-400') +
            row('Drain paused (stuck record)', esc(c.drain_paused ?? '—'), c.drain_paused ? 'text-amber-400' : 'text-emerald-400') +
            row('Link failures', esc(c.link_failures ?? '—'), c.link_failures ? 'text-amber-400' : 'text-emerald-400') +
            (dbg.last_error ? `<div class="mt-2 text-xs"><div class="text-slate-500">Last connection error (attempt ${esc(dbg.last_error.attempt)}, ${Math.round(Date.now() / 1000 - dbg.last_error.at)} s ago)</div>
                <div class="font-mono text-amber-300 break-all mt-0.5">${esc(dbg.last_error.error)}</div></div>` : ''));
    }

    function renderSensors(st) {
        const l = st.live || {}, s = st.settings || {}, r = (st.debug || {}).rates || {};
        const age = l.last_notify_at ? Date.now() / 1000 - l.last_notify_at : null;
        const cal = s.weight_full_raw != null ? `full ${s.weight_full_raw}` + (s.weight_empty_raw != null ? ` · empty ${s.weight_empty_raw}` : '') : 'not calibrated';
        return section('Sensors',
            row('Weight raw / settled', `${esc(l.weight_raw)} / ${esc(l.weight_stable)}`) +
            row('Weight readings', r.weight != null ? `${r.weight}/s` : '—', 'text-slate-300') +
            row('Calibration anchors', esc(cal), s.weight_full_raw != null ? 'text-slate-200' : 'text-amber-400') +
            row('Cap', l.cap_open == null ? '—' : l.cap_open ? 'open' : 'closed') +
            row('Battery', l.battery != null ? `${l.battery} %` : '—') +
            row('Serial / firmware', `${esc(l.serial)} / ${esc(l.firmware)}`, 'text-slate-300 text-xs') +
            row('Notifications', r.notifications != null ? `${r.notifications}/s` : '—', 'text-slate-300') +
            row('Last notification', age == null ? '—' : `${age.toFixed(1)} s ago`, age == null ? 'text-slate-500' : age < 5 ? 'text-emerald-400' : 'text-amber-400'));
    }

    function renderGatt(dbg) {
        const g = dbg.gatt || [], frames = dbg.last_frames || {};
        if (!g.length) return `<div class="md:col-span-2 xl:col-span-4">${section('Bottle Services &amp; Raw Frames', row('Characteristics', 'connect to the bottle to list them', 'text-slate-500'))}</div>`;
        const rows = g.map(c => {
            const f = frames[c.char.toLowerCase()];
            const ago = f ? `${Math.max(0, Date.now() / 1000 - f.at).toFixed(0)} s ago` : '';
            return `<tr class="border-b border-slate-700/50 align-top">
                <td class="py-1 pr-3 font-mono text-[11px] text-slate-300 whitespace-nowrap" title="service ${esc(c.service)}">${esc(c.char)}</td>
                <td class="py-1 pr-3 text-[11px] ${c.known ? 'text-cyan-300' : 'text-slate-500'}">${esc(c.known || 'unknown')}</td>
                <td class="py-1 pr-3 text-[11px] text-slate-400">${esc((c.props || []).join(', '))}</td>
                <td class="py-1 font-mono text-[11px] text-slate-200 break-all">${f ? esc(f.hex) : '<span class="text-slate-600">—</span>'} <span class="text-slate-500">${ago}</span></td></tr>`;
        }).join('');
        return `<div class="md:col-span-2 xl:col-span-4">${section(`Bottle Services &amp; Raw Frames (${g.length} characteristics; every notification is also saved to the session's _raw.csv)`,
            `<div class="overflow-x-auto"><table class="w-full text-left"><thead><tr class="text-[10px] uppercase text-slate-500">
             <th class="pr-3 font-medium">Characteristic</th><th class="pr-3 font-medium">Known as</th><th class="pr-3 font-medium">Properties</th><th class="font-medium">Last frame</th></tr></thead>
             <tbody>${rows}</tbody></table></div>`)}</div>`;
    }

    async function refresh() {
        if (panel.classList.contains('hidden')) return;
        if (panel.contains(document.activeElement) && document.activeElement.tagName === 'SELECT') return;   // don't redraw an open dropdown
        let st = {};
        try { st = await (await fetch('/api/status')).json(); } catch (e) {}
        const dbg = st.debug || {};
        document.getElementById('debug-body').innerHTML =
            renderRadio(dbg) + renderLink(st, dbg) + renderProtocol(st, dbg) + renderSensors(st) + renderGatt(dbg);
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
        try { localStorage.setItem('hydro-debug-open', open ? '1' : '0'); } catch (e) {}
        if (open) refresh();
    }
    const toggle = () => setOpen(panel.classList.contains('hidden'));

    btn.addEventListener('click', toggle);
    document.addEventListener('keydown', (e) => {
        if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey && !e.altKey &&
            !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) toggle();
    });
    setInterval(refresh, 1000);
    try { if (localStorage.getItem('hydro-debug-open') === '1') setOpen(true); } catch (e) {}

    return {};
})();
