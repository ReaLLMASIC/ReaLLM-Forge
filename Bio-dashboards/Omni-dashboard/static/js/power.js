// Shutdown / reboot card at the bottom of the debug panel (shared by all dashboards).
// Two taps to act: the first arms the button (red, 4 s), the second sends the request.
// The server only allows it from this computer unless BIODASH_ALLOW_POWER=1 -- see bt_debug.py.
(function () {
    const body = document.getElementById('debug-body');
    if (!body) return;
    const card = document.createElement('div');
    card.id = 'power-card';
    card.className = 'mt-4 bg-slate-900/60 border border-slate-700 rounded-xl p-4 text-sm';
    body.insertAdjacentElement('afterend', card);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const BTN = 'px-4 py-1.5 rounded-full text-xs font-semibold border transition-colors ';
    const IDLE = 'bg-slate-700/50 text-slate-200 border-slate-600 hover:bg-slate-700';
    const ARMED = 'bg-red-600 text-white border-red-500 hover:bg-red-500';
    let info = null, armed = null, armTimer = null, done = false;

    function render(msg = '', msgCls = 'text-slate-500') {
        if (done) return;
        const ok = info && info.allowed;
        const label = (a) => armed === a ? (a === 'shutdown' ? 'Confirm shut down?' : 'Confirm reboot?') : (a === 'shutdown' ? '⏻ Shut down' : '⟳ Reboot');
        const how = !info ? 'checking…' : info.dry_run ? 'DRY RUN — nothing will actually happen'
                  : ok ? `${esc(info.reason)} · via ${info.method === 'sudo' ? 'sudo' : 'systemctl'}` : esc(info.reason);
        card.innerHTML = `
            <div class="flex flex-wrap items-center justify-between gap-3">
                <div>
                    <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider">System${info ? ' · ' + esc(info.host) : ''}</div>
                    <div class="text-xs mt-1 ${ok ? (info.dry_run ? 'text-amber-400' : 'text-slate-500') : 'text-amber-400'}">${how}</div>
                </div>
                <div class="flex gap-2">
                    <button data-power="reboot" class="${BTN}${armed === 'reboot' ? ARMED : IDLE}" ${ok ? '' : 'disabled style="opacity:.4;cursor:not-allowed"'}>${label('reboot')}</button>
                    <button data-power="shutdown" class="${BTN}${armed === 'shutdown' ? ARMED : IDLE}" ${ok ? '' : 'disabled style="opacity:.4;cursor:not-allowed"'}>${label('shutdown')}</button>
                </div>
            </div>
            ${info && info.fix ? `<div class="mt-3 text-xs text-slate-400">To allow it without a password prompt, run once on this computer:
                <pre class="mt-1 p-2 rounded bg-slate-950 text-slate-300 font-mono text-[11px] whitespace-pre-wrap break-all">${esc(info.fix)}</pre></div>` : ''}
            ${msg ? `<div class="mt-2 text-xs ${msgCls}">${msg}</div>` : ''}`;
    }

    async function load() {
        try { info = await (await fetch('/api/system/power')).json(); } catch (e) { info = null; }
        render();
    }

    card.addEventListener('click', async (e) => {
        const action = e.target.closest('button[data-power]')?.dataset.power;
        if (!action || !info || !info.allowed || done) return;
        if (armed !== action) {                       // first tap: arm
            armed = action; render();
            clearTimeout(armTimer);
            armTimer = setTimeout(() => { armed = null; render(); }, 4000);
            return;
        }
        clearTimeout(armTimer); armed = null;         // second tap: go
        let res = null, out = {};
        try {
            res = await fetch('/api/system/power', { method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Biodash-Power': '1' },
                body: JSON.stringify({ action }) });
            out = await res.json();
        } catch (err) { out = { error: 'no reply from the dashboard' }; }
        if (res && res.ok) {
            if (out.dry_run) { render(`Dry run: ${action} accepted (nothing happened).`, 'text-amber-400'); return; }
            done = true;
            card.innerHTML = `<div class="text-sm text-slate-300">${action === 'shutdown' ? '⏻ Shutting down' : '⟳ Rebooting'} ${esc(info.host)} in 2 s…
                <span class="text-slate-500">${action === 'reboot' ? 'The dashboards come back on their own if they run as services (biodash.py).' : 'The dashboards will go offline.'}</span></div>`;
        } else {
            render(esc(out.error || 'refused') + (out.fix ? ' — see the command above.' : ''), 'text-red-400');
            if (out.fix && info) { info.fix = out.fix; render(esc(out.error), 'text-red-400'); }
        }
    });

    load();
    setInterval(() => { if (!document.getElementById('debug-panel')?.classList.contains('hidden') && !armed) load(); }, 15000);
})();
