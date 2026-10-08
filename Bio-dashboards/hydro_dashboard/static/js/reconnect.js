// Auto-reconnect bar (shared by all dashboards): shows the remembered device(s) at the
// top of the scanner panel, with Pause / Resume and Forget. The worker does the actual
// reconnecting -- it picks the remembered device as soon as it's back in range
// (or, for a Wi-Fi device, retries its address every 15 s).
(() => {
    const panel = document.getElementById('scanner-panel');
    if (!panel) return;
    const bar = document.createElement('div');
    bar.className = 'hidden mb-4 flex flex-col gap-2';
    const heading = panel.querySelector('h2');
    (heading || panel.firstChild) ? panel.insertBefore(bar, heading ? heading.nextSibling : panel.firstChild) : panel.appendChild(bar);

    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const ROLE_LABEL = { device: '', polar: 'Polar H10 · ', viatom: 'O2 ring · ' };
    const BTN = 'px-3 py-1 rounded-full text-xs font-semibold border bg-slate-700/50 text-slate-300 border-slate-600 hover:bg-slate-700 hover:text-white transition-colors';

    function line(role, d) {
        const name = esc(d.name || d.address);
        const addr = d.transport === 'tcp' ? '' : ` <span class="font-mono text-slate-500 text-xs">${esc(d.address)}</span>`;
        const text = !d.auto
            ? `<span class="text-slate-400">Last device: <b class="text-slate-200">${name}</b>${addr} · auto-reconnect off</span>`
            : d.transport === 'tcp'
            ? `<span class="text-emerald-300">↻ Auto-reconnect: retrying <b>${name}</b> every 15 s</span>`
            : `<span class="text-emerald-300">↻ Auto-reconnect: waiting for <b>${name}</b>${addr} to come into range</span>`;
        return `<div class="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm rounded-xl border ${d.auto ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-slate-700 bg-slate-900/30'} px-3 py-2">
            ${role in ROLE_LABEL ? (ROLE_LABEL[role] ? `<span class="text-xs text-slate-500">${ROLE_LABEL[role]}</span>` : '') : `<span class="text-xs text-slate-500">${esc(role)} · </span>`}${text}
            <span class="ml-auto flex gap-2">
                <button data-role="${esc(role)}" data-act="${d.auto ? 'pause' : 'resume'}" class="${BTN}">${d.auto ? 'Pause' : 'Resume'}</button>
                <button data-role="${esc(role)}" data-act="forget" class="${BTN}">Forget</button>
            </span>
        </div>`;
    }

    async function refresh() {
        let devices = {};
        try { devices = (await (await fetch('/api/last-device')).json()).devices || {}; } catch (e) { return; }
        const rows = Object.entries(devices).filter(([, d]) => d);
        bar.innerHTML = rows.map(([role, d]) => line(role, d)).join('');
        bar.classList.toggle('hidden', rows.length === 0);
    }

    bar.addEventListener('click', async (e) => {
        const b = e.target.closest('button[data-act]');
        if (!b) return;
        const body = { role: b.dataset.role };
        if (b.dataset.act === 'forget') body.forget = true; else body.auto = b.dataset.act === 'resume';
        b.disabled = true;
        try {
            await fetch('/api/last-device', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        } catch (err) {}
        refresh();
    });

    refresh();
    setInterval(refresh, 3000);
})();
