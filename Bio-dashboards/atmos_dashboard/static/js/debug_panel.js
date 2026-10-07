// Debug panel: Bluetooth radio, link, UART stream health for the Atmos-Mini.
// Toggle with the 🐞 button or the "D" key. Polls /api/status once a second while open.

// Swap (don't stack) colour classes: in the compiled CSS the slate base classes can
// come after the accent ones, so adding an accent on top of slate would silently lose.
function setPillActive(el, on) {
    const off = ['bg-slate-700/50', 'text-slate-300', 'border-slate-600'];
    const act = ['bg-emerald-500/10', 'text-emerald-400', 'border-emerald-500/40'];
    el.classList.remove(...(on ? off : act));
    el.classList.add(...(on ? act : off));
}

const DebugPanel = (() => {
    const panel = document.getElementById('debug-panel');
    const btn = document.getElementById('debug-btn');
    const esc = (s) => String(s ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

    // Browser-side arrivals over a 10 s window (data is slow, ~1 line/s)
    const arrivals = [];
    const onPacket = (rows) => { const t = performance.now(); for (let i = 0; i < rows; i++) arrivals.push(t); };

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
    const countCls = (n) => n ? 'text-amber-400' : 'text-emerald-400';

    function renderRadio(dbg) {
        const inUse = dbg.adapter;
        if (!dbg.adapters || !dbg.adapters.length) {
            return section('Bluetooth Radio', row('Adapters', 'none reported yet', 'text-slate-500'));
        }
        const cards = dbg.adapters.map(a => {
            const used = a.name === inUse;
            const chip = [a.vendor, a.product].filter(Boolean).join(' · ') || (a.usb_id ? `ID ${a.usb_id}` : 'unknown chip');
            return `<div class="rounded-lg border ${used ? 'border-emerald-500/40 bg-emerald-500/5' : 'border-slate-700'} p-3 mb-2 last:mb-0">
                <div class="mb-1">
                    <span class="font-mono font-bold text-slate-100">${esc(a.address || a.name)}</span>
                    <span class="text-[11px] font-mono text-slate-500 ml-1 whitespace-nowrap" title="hci numbers follow enumeration order and can change">currently ${esc(a.name)}</span>
                </div>
                <div class="flex flex-wrap gap-1 mb-1.5">
                        ${placementPill(a)}
                        ${used ? pill('IN USE', 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30') : ''}
                        ${a.powered === false ? pill('OFF', 'bg-red-500/20 text-red-400 border-red-500/30')
                          : a.powered ? pill('POWERED', 'bg-sky-500/20 text-sky-400 border-sky-500/30') : ''}
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
            <select data-radio class="w-full bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-xs font-mono text-slate-200 focus:border-emerald-500 focus:outline-none">
                <option value="auto" ${cur === 'AUTO' ? 'selected' : ''}>Auto</option>
                ${(dbg.adapters || []).map(a => `<option value="${esc(a.address)}" ${cur === (a.address || '').toUpperCase() ? 'selected' : ''}>${esc(a.address)} (${esc(a.name)}${a.placement && a.placement !== 'unknown' ? ', ' + esc(a.placement) : ''})</option>`).join('')}
            </select>
            <div class="text-[11px] text-slate-500 mt-0.5">→ ${esc(r.hci || 'system default')} · ${esc(r.reason || '—')}</div>
            <div class="text-[11px] text-slate-500 mt-1">Saved by address. Applies on the next scan; a live connection isn't dropped.</div>
        </div>`;
        return section('Bluetooth Radio', cards + note + picker);
    }

    function renderLink(st, dbg) {
        const d = dbg.device || {}, c = dbg.counters || {};
        const up = dbg.connected_at ? (Date.now() / 1000 - dbg.connected_at) : null;
        const stateCls = { streaming: 'text-emerald-400', connecting: 'text-amber-400', scanning: 'text-sky-400' }[st.state] || 'text-slate-300';
        return section('Link',
            row('State', esc(st.state) + (st.state === 'connecting' && st.attempt ? ` (${st.attempt}/${st.max_attempts})` : ''), stateCls) +
            row('Transport', dbg.transport === 'tcp' ? 'Wi-Fi (TCP)' : dbg.transport === 'ble' ? 'Bluetooth' : '—') +
            row('Device', esc(d.name)) +
            row('Address', esc(d.address)) +
            row('RSSI at scan', d.rssi != null ? `${d.rssi} dBm` : '—',
                d.rssi == null ? 'text-slate-500' : d.rssi > -70 ? 'text-emerald-400' : d.rssi > -85 ? 'text-amber-400' : 'text-red-400') +
            row('Radio', esc(dbg.adapter)) +
            row('Connected for', fmtDur(up)) +
            row('Saving to', dbg.recording ? `<span title="${esc(dbg.recording)}">${esc(dbg.recording.split('/').slice(-3).join('/'))}</span>` : '—', 'text-slate-300 text-xs') +
            row('Link failures', c.link_failures ?? '—', countCls(c.link_failures)));
    }

    function renderStream(dbg) {
        const r = dbg.rates || {}, c = dbg.counters || {};
        const age = dbg.last_line_at ? Date.now() / 1000 - dbg.last_line_at : null;
        const interval = r.line_interval_s != null
            ? `${r.line_interval_s} s ± ${r.line_jitter_s != null ? Math.round(r.line_jitter_s * 1000) : '?'} ms` : '—';
        return section('UART Stream (worker, 15 s window)',
            row('Readings', r.lines_per_s != null ? `${r.lines_per_s}/s` : '—', r.lines_per_s ? 'text-emerald-400' : 'text-slate-500') +
            row('Line interval', interval) +
            row('Last reading', age == null ? '—' : `${age.toFixed(1)} s ago`,
                age == null ? 'text-slate-500' : age < 3 ? 'text-emerald-400' : age < 8 ? 'text-amber-400' : 'text-red-400') +
            row('Notifications', r.notifications_per_s != null ? `${r.notifications_per_s}/s` : '—') +
            row('Throughput', r.bytes_per_s != null ? `${r.bytes_per_s} B/s` : '—') +
            row('Bytes / notification', esc(r.bytes_per_notification)) +
            row('Header lines', c.header_lines ?? '—', 'text-slate-300') +
            row('Rejected lines', c.rejected_lines ?? '—', countCls(c.rejected_lines)) +
            row('Buffer overflows', c.buffer_overflows ?? '—', c.buffer_overflows ? 'text-red-400' : 'text-emerald-400') +
            row('Rows logged', dbg.samples_written ?? '—'));
    }

    function renderGatt(dbg) {
        const g = dbg.gatt || {}, v = dbg.versions || {};
        const now = performance.now();
        while (arrivals.length && now - arrivals[0] > 10000) arrivals.shift();
        return section('UART, Browser & Versions',
            row('NUS service', `<span title="${esc(g.service)}">${esc(shortUuid(g.service))}</span>`) +
            row('TX characteristic', `<span title="${esc(g.tx_char)}">${esc(shortUuid(g.tx_char))}</span>`) +
            row('MTU', esc(g.mtu)) +
            row('Rows to browser (10 s)', `${(arrivals.length / 10).toFixed(2)}/s`) +
            row('bleak', esc(v['bleak'])) +
            row('FastAPI', esc(v['fastapi'])) +
            row('Python', esc(v['python'])));
    }

    // ---------- device profile (full width) ----------
    const SOURCE_ORDER = ['sen69c', 'sen68', 'sen66', 'sen55', 'sen54', 'sen5x', 'sen44', 'sht4x',
                          'scd30', 'scd41', 'scd40', 'scd4x', 'sfa3x', 'sfa30', 'sgp41', ''];
    const CANON = ['co2', 'pm1', 'pm25', 'pm4', 'pm10', 'temp', 'rh', 'voc', 'nox', 'hcho'];
    function renderProfile(dbg, cfg) {
        const p = dbg.parser || {}, profiles = dbg.profiles || {}, fields = dbg.fields || {};
        const saved = (cfg.config || {});
        const eff = p.effective_profile && profiles[p.effective_profile];
        const opt = (v, l) => `<option value="${v}" ${(saved.profile || 'auto') === v ? 'selected' : ''}>${esc(l)}</option>`;
        const assumed = eff && !eff.order_confirmed;

        // which source each canonical metric's tile is using
        const shown = {};
        for (const id of Object.keys(fields)) {
            const [m, src = ''] = id.split('@');
            if (!CANON.includes(m)) continue;
            const r = SOURCE_ORDER.indexOf(src);
            if (!(m in shown) || r < SOURCE_ORDER.indexOf(shown[m].split('@')[1] || '')) shown[m] = id;
        }
        const tagOf = (id) => {
            const m = id.split('@')[0];
            if (!CANON.includes(m)) return pill('OTHER TILE', 'bg-violet-500/20 text-violet-300 border-violet-500/30');
            return shown[m] === id ? pill('ON TILE', 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30')
                                   : pill('ALT SOURCE', 'bg-slate-600/40 text-slate-300 border-slate-500/40');
        };
        const fieldRows = Object.entries(fields).map(([id, f]) =>
            `<div class="flex items-center justify-between gap-2 py-1 border-b border-slate-700/50">
                <span class="font-mono text-slate-200 text-xs">${esc(id)}</span>
                <span class="flex items-center gap-2"><span class="font-mono text-slate-300 text-xs">${esc(+(+f.value).toFixed(2))}</span>${tagOf(id)}</span>
            </div>`).join('') || `<div class="text-xs text-slate-500">No readings yet.</div>`;

        const expected = (eff || profiles[saved.profile] || {}).columns;
        return `<div class="md:col-span-2 xl:col-span-4 bg-slate-900/40 border border-slate-700 rounded-xl p-4">
            <h3 class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3">Device Profile</h3>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                    <label class="block text-xs text-slate-400 mb-1">Device type</label>
                    <select data-profile class="w-full bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-sm text-slate-200 focus:border-emerald-500 focus:outline-none">
                        ${opt('auto', 'Auto (detect from the data)')}
                        ${Object.entries(profiles).map(([k, v]) => opt(k, v.label)).join('')}
                    </select>
                    <div class="mt-3">
                        ${row('Line format', esc(p.format || '—'))}
                        ${row('Fields per line', esc(p.field_count ?? '—'))}
                        ${row('Columns from', esc(p.column_source || '—'), p.column_source?.startsWith('generic') ? 'text-amber-400' : 'text-slate-200')}
                        ${row('Detected device', esc(eff ? eff.label : '—'))}
                        ${row('Log file', esc(p.log_file || '—'), 'text-slate-400')}
                    </div>
                    ${assumed ? `<div class="text-[11px] text-amber-400 mt-2">The ${esc(eff.label)} column order is assumed
                        (each Sensirion library's read order, sensors as SCD30 → SEN5x → SFA3X). If values land on the wrong
                        tiles, have the firmware print a header line or set the column map below.</div>` : ''}
                    ${p.device_header ? `<div class="text-[11px] text-slate-400 mt-2">Device header: <span class="font-mono">${esc(p.device_header.join(', '))}</span></div>` : ''}
                    <label class="block text-xs text-slate-400 mt-4 mb-1">Custom column map <span class="text-slate-500">(headerless CSV only; comma-separated, e.g. co2@scd30, temp, rh, pm2.5)</span></label>
                    <div class="flex gap-2">
                        <input data-columns type="text" value="${esc((saved.columns || []).join(', '))}"
                               placeholder="${esc((expected || []).join(', '))}"
                               class="flex-grow bg-slate-800 border border-slate-600 rounded-lg px-2 py-1.5 text-xs font-mono text-slate-200 placeholder-slate-600 focus:border-emerald-500 focus:outline-none">
                        <button data-apply class="px-3 py-1 rounded-full text-xs font-semibold border bg-slate-700/50 text-slate-300 border-slate-600 hover:text-white">Apply</button>
                        <button data-clear class="px-3 py-1 rounded-full text-xs font-semibold border bg-slate-700/50 text-slate-300 border-slate-600 hover:text-white">Clear</button>
                    </div>
                    <div class="text-[11px] text-slate-500 mt-1">Applies within a second, no reconnect. A header line from the device always takes priority.</div>
                </div>
                <div>
                    <div class="text-xs text-slate-400 mb-1">Detected fields <span class="text-slate-500">(${Object.keys(fields).length})</span></div>
                    <div class="max-h-72 overflow-y-auto pr-1">${fieldRows}</div>
                </div>
            </div>
        </div>`;
    }

    async function saveProfile(profile, columns) {
        try {
            await fetch('/api/profile', { method: 'POST', headers: {'Content-Type': 'application/json'},
                                          body: JSON.stringify({ profile, columns }) });
        } catch (e) {}
        setTimeout(refresh, 1200);   // the worker picks it up within a second
    }
    const body = document.getElementById('debug-body');
    body.addEventListener('change', (e) => {
        const sel = e.target.closest('select[data-profile]');
        if (!sel) return;
        const cols = body.querySelector('input[data-columns]')?.value.split(',').map(c => c.trim()).filter(Boolean);
        sel.blur();
        saveProfile(sel.value, cols && cols.length ? cols : null);
    });
    body.addEventListener('click', (e) => {
        const apply = e.target.closest('button[data-apply]'), clear = e.target.closest('button[data-clear]');
        if (!apply && !clear) return;
        const input = body.querySelector('input[data-columns]');
        if (clear) input.value = '';
        const cols = input.value.split(',').map(c => c.trim()).filter(Boolean);
        input.blur();
        saveProfile(body.querySelector('select[data-profile]').value, cols.length ? cols : null);
    });
    body.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.matches('input[data-columns]')) body.querySelector('button[data-apply]').click();
    });

    async function refresh() {
        if (panel.classList.contains('hidden')) return;
        if (panel.contains(document.activeElement) && document.activeElement.tagName === 'SELECT') return;   // don't redraw an open dropdown
        // don't redraw under the user's cursor
        const ae = document.activeElement;
        if (panel.contains(ae) && ['INPUT', 'SELECT'].includes(ae.tagName)) return;
        let st = {}, cfg = {};
        try {
            [st, cfg] = await Promise.all([(await fetch('/api/status')).json(), (await fetch('/api/profile')).json()]);
        } catch (e) {}
        const dbg = st.debug || {};
        document.getElementById('debug-body').innerHTML =
            renderRadio(dbg) + renderLink(st, dbg) + renderStream(dbg) + renderGatt(dbg) + renderProfile(dbg, cfg);
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
        try { localStorage.setItem('atmos-debug-open', open ? '1' : '0'); } catch (e) {}
        if (open) refresh();
    }
    const toggle = () => setOpen(panel.classList.contains('hidden'));

    btn.addEventListener('click', toggle);
    document.addEventListener('keydown', (e) => {
        if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey && !e.altKey &&
            !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) toggle();
    });
    setInterval(refresh, 1000);
    try { if (localStorage.getItem('atmos-debug-open') === '1') setOpen(true); } catch (e) {}

    return { onPacket };
})();
