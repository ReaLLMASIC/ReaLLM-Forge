// Links between the Bio-dash dashboards on this machine, with a live dot for each
// (green = its server answers). Same file in every dashboard; it fills <nav id="dash-nav">.
(() => {
    const DASHBOARDS = [
        { name: 'Polar H10', port: 5001 },
        { name: 'Viatom O2', port: 5003 },
        { name: 'Atmos',     port: 5002 },
        { name: 'Omni',      port: 5000 },
        { name: 'Hydro',     port: 5004 },
    ];
    const nav = document.getElementById('dash-nav');
    if (!nav) return;
    const here = parseInt(location.port || (location.protocol === 'https:' ? '443' : '80'));
    const urlFor = (d) => `${location.protocol}//${location.hostname}:${d.port}/`;
    const DOT = 'inline-block w-1.5 h-1.5 rounded-full ';

    nav.innerHTML = DASHBOARDS.map(d => d.port === here
        ? `<span class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold border bg-slate-700 text-white border-slate-500 flex items-center gap-1.5" aria-current="page">
               <span class="${DOT}bg-emerald-400"></span>${d.name}</span>`
        : `<a href="${urlFor(d)}" class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold border bg-slate-800 text-slate-400 border-slate-700 hover:text-white hover:border-slate-500 transition-colors flex items-center gap-1.5">
               <span data-port="${d.port}" class="${DOT}bg-slate-600" title="checking…"></span>${d.name}</a>`).join('');

    // A no-cors fetch resolves if the server answers at all (the response is opaque)
    // and rejects if nothing is listening -- enough for an up/down dot.
    async function ping(d) {
        const dot = nav.querySelector(`[data-port="${d.port}"]`);
        if (!dot) return;
        const ctl = new AbortController();
        const t = setTimeout(() => ctl.abort(), 2000);
        let up = false;
        try { await fetch(urlFor(d), { mode: 'no-cors', cache: 'no-store', signal: ctl.signal }); up = true; } catch (e) {}
        clearTimeout(t);
        dot.className = DOT + (up ? 'bg-emerald-400' : 'bg-slate-600');
        dot.title = up ? 'running' : 'not running';
    }
    const pingAll = () => DASHBOARDS.filter(d => d.port !== here).forEach(ping);
    pingAll();
    setInterval(pingAll, 10000);

    // ---------- ☀ Keep screen on (Screen Wake Lock API) ----------
    // Stops the display that shows this dashboard from dimming / locking. Browsers only
    // allow it on a secure page (https:// or http://localhost); elsewhere the toggle
    // explains why it's off. The lock drops when the tab is hidden, so it's re-acquired
    // when the tab becomes visible again. (The computer itself is kept awake by the
    // service / launch.sh via systemd-inhibit.)
    const KEY = 'biodash-keep-awake';
    const btn = document.createElement('button');
    btn.type = 'button';
    const BASE = 'px-2.5 py-0.5 rounded-full text-[11px] font-semibold border transition-colors flex items-center gap-1 ';
    const OFF = 'bg-slate-800 text-slate-400 border-slate-700 hover:text-white hover:border-slate-500';
    const ON = 'bg-amber-500/15 text-amber-300 border-amber-500/40';
    const NA = 'bg-slate-800 text-slate-600 border-slate-700 cursor-not-allowed';
    nav.appendChild(btn);

    const supported = 'wakeLock' in navigator && window.isSecureContext;
    let lock = null;
    let wanted = false;
    try { wanted = localStorage.getItem(KEY) === '1'; } catch (e) {}

    function paint() {
        if (!supported) {
            btn.className = BASE + NA;
            btn.textContent = '☀ Keep screen on';
            btn.title = window.isSecureContext
                ? 'This browser does not support keeping the screen on.'
                : 'Browsers only allow this on https:// or http://localhost -- open the dashboard via localhost on this machine to use it.';
            return;
        }
        btn.className = BASE + (lock ? ON : OFF);
        btn.textContent = lock ? '☀ Screen stays on' : '☀ Keep screen on';
        btn.title = lock ? 'Click to let the screen sleep again' : 'Keep this screen from dimming or locking';
    }
    async function acquire() {
        if (!supported || !wanted || lock || document.visibilityState !== 'visible') return paint();
        try {
            lock = await navigator.wakeLock.request('screen');
            lock.addEventListener('release', () => { lock = null; paint(); });
        } catch (e) { lock = null; }
        paint();
    }
    btn.addEventListener('click', async () => {
        if (!supported) return;
        wanted = !lock;                                   // held -> turn off, not held -> turn on
        try { localStorage.setItem(KEY, wanted ? '1' : '0'); } catch (e) {}
        if (lock) { await lock.release(); lock = null; paint(); }
        else acquire();
    });
    document.addEventListener('visibilitychange', acquire);
    paint();
    acquire();
})();
