// Bio-dash sidebar: everything that used to sit in the top bar, in one place.
//   Dashboards  links to the other dashboards, with a live dot for each
//   View        debug panel, keep the screen on
//   Updates     check GitHub, apply a patch (same version) or upgrade (new version)
//   Power       reboot / shut down this computer
// Same file in every dashboard. It builds its own markup and styles, so a template only
// needs to load it. Closed when a page opens; ☰ in the header opens it (beside the page on
// wide screens, as a slide-over drawer on narrow ones).
(() => {
    const DASHBOARDS = [                                  // listed in port order
        { name: 'Omni',      port: 5000, color: '#818cf8' },
        { name: 'Polar H10', port: 5001, color: '#f43f5e' },
        { name: 'Atmos',     port: 5002, color: '#34d399' },
        { name: 'Viatom O2', port: 5003, color: '#38bdf8' },
        { name: 'Hydro',     port: 5004, color: '#22d3ee' },
    ];
    const here = parseInt(location.port || (location.protocol === 'https:' ? '443' : '80'));
    const urlFor = (d) => `${location.protocol}//${location.hostname}:${d.port}/`;
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const store = { get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } },
                    set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };
    const HEADER = { 'Content-Type': 'application/json', 'X-Biodash-Power': '1' };

    // ---------------------------------------------------------------- styles
    const css = document.createElement('style');
    css.textContent = `
    :root{--sb-w:236px}
    #sb{position:fixed;inset:0 auto 0 0;width:var(--sb-w);z-index:40;background:#0f172a;border-right:1px solid #1e293b;
        display:flex;flex-direction:column;transform:translateX(-100%);transition:transform .18s ease;
        font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#cbd5e1}
    body.sb-open #sb{transform:none}
    #sb-scroll{flex:1;overflow-y:auto;padding:4px 10px 14px;scrollbar-width:thin}
    #sb-head{display:flex;align-items:center;justify-content:space-between;padding:14px 14px 10px 16px}
    #sb-brand{font-weight:700;font-size:15px;color:#f1f5f9;letter-spacing:.01em}
    #sb-ver{font:600 11px ui-monospace,monospace;color:#64748b;margin-left:6px}
    .sb-x{background:none;border:0;color:#64748b;font-size:18px;line-height:1;padding:4px 6px;border-radius:6px;cursor:pointer}
    .sb-x:hover{color:#f1f5f9;background:#1e293b}
    .sb-h{font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#64748b;margin:16px 8px 6px}
    .sb-item{display:flex;align-items:center;gap:9px;width:100%;padding:7px 9px;border-radius:8px;border:0;background:none;
        color:#cbd5e1;font:inherit;text-align:left;text-decoration:none;cursor:pointer}
    .sb-item:hover{background:#1e293b;color:#fff}
    .sb-item[aria-current]{background:#1e293b;color:#fff;font-weight:600;box-shadow:inset 3px 0 0 var(--c,#94a3b8)}
    .sb-item.on{color:#fcd34d}
    .sb-item:disabled{color:#475569;cursor:not-allowed}.sb-item:disabled:hover{background:none}
    .sb-dot{width:7px;height:7px;border-radius:50%;background:#475569;flex:none}
    .sb-dot.up{background:#34d399;box-shadow:0 0 0 3px rgba(52,211,153,.15)}
    .sb-ico{width:16px;text-align:center;flex:none;opacity:.9}
    .sb-tag{margin-left:auto;font-size:10.5px;color:#64748b}
    .sb-box{margin:0 2px;padding:9px 10px;border:1px solid #1e293b;border-radius:10px;background:#111c31}
    .sb-note{font-size:11.5px;color:#94a3b8;margin:2px 0 8px;overflow-wrap:anywhere}
    .sb-note.warn{color:#fbbf24}.sb-note.err{color:#f87171}.sb-note.ok{color:#34d399}
    .sb-btn{display:block;width:100%;padding:6px 10px;margin-top:6px;border-radius:8px;border:1px solid #334155;
        background:#1e293b;color:#e2e8f0;font:600 12px/1.3 inherit;cursor:pointer;text-align:center}
    .sb-btn:hover{background:#334155}.sb-btn.go{background:#0e7490;border-color:#0891b2;color:#fff}
    .sb-btn.go:hover{background:#0891b2}.sb-btn.armed{background:#dc2626;border-color:#ef4444;color:#fff}
    .sb-btn:disabled{opacity:.45;cursor:not-allowed}
    .sb-row{display:flex;gap:6px}.sb-row .sb-btn{margin-top:0;white-space:nowrap;padding:6px 4px}
    .sb-list{margin:6px 0 2px;padding:0;list-style:none;font-size:11.5px;color:#94a3b8;max-height:150px;overflow-y:auto}
    .sb-list li{padding:3px 0;overflow-wrap:anywhere;border-top:1px solid #1e293b}.sb-list li:first-child{border-top:0}
    .sb-pre{margin:6px 0 0;padding:6px;border-radius:6px;background:#020617;color:#cbd5e1;
        font:10.5px/1.4 ui-monospace,monospace;white-space:pre-wrap;word-break:break-all;max-height:140px;overflow:auto}
    #sb-foot{padding:8px 16px 12px;font-size:10.5px;color:#475569;border-top:1px solid #1e293b}
    #sb-shade{display:none;position:fixed;inset:0;z-index:39;background:rgba(2,6,23,.6)}
    #sb-open{background:#1e293b;border:1px solid #334155;color:#cbd5e1;border-radius:8px;padding:5px 9px;font-size:15px;
        line-height:1;cursor:pointer;margin-right:12px;flex:none}
    #sb-open:hover{color:#fff;border-color:#64748b}
    body.sb-open #sb-open{display:none}
    @media (min-width:1024px){body{transition:padding-left .18s ease}body.sb-open{padding-left:var(--sb-w)}}
    @media (max-width:1023px){body.sb-open #sb-shade{display:block}#sb{box-shadow:8px 0 30px rgba(0,0,0,.5)}}
    @media (prefers-reduced-motion:reduce){#sb,body{transition:none}}`;
    document.head.appendChild(css);

    // ---------------------------------------------------------------- frame
    const sb = document.createElement('aside');
    sb.id = 'sb';
    sb.setAttribute('aria-label', 'Bio-dash menu');
    sb.innerHTML = `
        <div id="sb-head"><div><span id="sb-brand">Bio-dash</span><span id="sb-ver"></span></div>
            <button class="sb-x" id="sb-close" title="Hide the menu" aria-label="Hide the menu">«</button></div>
        <div id="sb-scroll">
            <div class="sb-h">Dashboards</div><nav id="sb-nav" aria-label="Dashboards"></nav>
            <div class="sb-h">View</div><div id="sb-view"></div>
            <div class="sb-h">Updates</div><div class="sb-box" id="sb-update"></div>
            <div class="sb-h">Power</div><div class="sb-box" id="sb-power"></div>
        </div>
        <div id="sb-foot"></div>`;
    const shade = document.createElement('div');
    shade.id = 'sb-shade';
    document.body.append(sb, shade);

    const wide = () => window.matchMedia('(min-width:1024px)').matches;
    const setOpen = (open) => document.body.classList.toggle('sb-open', open);
    // the button that brings the menu back sits at the left of the page header
    const opener = document.createElement('button');
    opener.id = 'sb-open';
    opener.type = 'button';
    opener.title = 'Menu';
    opener.setAttribute('aria-label', 'Open the menu');
    opener.textContent = '☰';
    const bar = document.querySelector('header > div') || document.querySelector('header');
    if (bar) { bar.style.justifyContent = 'flex-start'; bar.prepend(opener); bar.lastElementChild.style.marginLeft = 'auto'; }
    else { opener.style.cssText = 'position:fixed;top:10px;left:10px;z-index:38'; document.body.append(opener); }
    opener.addEventListener('click', () => setOpen(true));
    sb.querySelector('#sb-close').addEventListener('click', () => setOpen(false));
    shade.addEventListener('click', () => setOpen(false));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !wide()) setOpen(false); });
    setOpen(false);                                       // every page opens with the menu closed

    // ---------------------------------------------------------------- dashboards
    const nav = sb.querySelector('#sb-nav');
    nav.innerHTML = DASHBOARDS.map(d => d.port === here
        ? `<a class="sb-item" aria-current="page" style="--c:${d.color}"><span class="sb-dot up"></span>${d.name}<span class="sb-tag">here</span></a>`
        : `<a class="sb-item" href="${urlFor(d)}"><span class="sb-dot" data-port="${d.port}" title="checking…"></span>${d.name}<span class="sb-tag">:${d.port}</span></a>`).join('');
    // A no-cors fetch resolves if the server answers at all and rejects if nothing listens.
    async function ping(d) {
        const dot = nav.querySelector(`[data-port="${d.port}"]`);
        if (!dot) return;
        const ctl = new AbortController();
        const t = setTimeout(() => ctl.abort(), 2000);
        let up = false;
        try { await fetch(urlFor(d), { mode: 'no-cors', cache: 'no-store', signal: ctl.signal }); up = true; } catch (e) {}
        clearTimeout(t);
        dot.classList.toggle('up', up);
        dot.title = up ? 'running' : 'not running';
    }
    const pingAll = () => DASHBOARDS.filter(d => d.port !== here).forEach(ping);
    pingAll();
    setInterval(pingAll, 10000);

    // ---------------------------------------------------------------- view: debug + keep awake
    const view = sb.querySelector('#sb-view');
    const dbgBtn = document.getElementById('debug-btn');           // the original toggle keeps its behaviour (and the D key)
    const dbgPanel = document.getElementById('debug-panel');
    if (dbgBtn) {
        dbgBtn.style.display = 'none';
        const b = document.createElement('button');
        b.className = 'sb-item'; b.type = 'button';
        const paint = () => {
            const on = dbgPanel && !dbgPanel.classList.contains('hidden');
            b.classList.toggle('on', !!on);
            b.innerHTML = `<span class="sb-ico">🐞</span>Debug panel<span class="sb-tag">${on ? 'on' : 'D'}</span>`;
        };
        b.addEventListener('click', () => { dbgBtn.click(); if (!wide()) setOpen(false); });
        if (dbgPanel) new MutationObserver(paint).observe(dbgPanel, { attributes: true, attributeFilter: ['class'] });
        paint();
        view.append(b);
    }
    // Screen Wake Lock: browsers only allow it on https:// or http://localhost. The lock drops
    // when the tab is hidden, so it's taken again when the tab comes back.
    const wake = document.createElement('button');
    wake.className = 'sb-item'; wake.type = 'button';
    view.append(wake);
    const canWake = 'wakeLock' in navigator && window.isSecureContext;
    let lock = null, wanted = store.get('biodash-keep-awake') === '1';
    function paintWake() {
        wake.disabled = !canWake;
        wake.classList.toggle('on', !!lock);
        wake.innerHTML = `<span class="sb-ico">☀</span>${lock ? 'Screen stays on' : 'Keep screen on'}<span class="sb-tag">${canWake ? (lock ? 'on' : 'off') : 'n/a'}</span>`;
        wake.title = canWake ? (lock ? 'Click to let the screen sleep again' : 'Keep this screen from dimming or locking')
            : (window.isSecureContext ? 'This browser does not support keeping the screen on.'
                : 'Browsers only allow this on https:// or http://localhost — open the dashboard via localhost on this machine.');
    }
    async function acquire() {
        if (!canWake || !wanted || lock || document.visibilityState !== 'visible') return paintWake();
        try { lock = await navigator.wakeLock.request('screen'); lock.addEventListener('release', () => { lock = null; paintWake(); }); }
        catch (e) { lock = null; }
        paintWake();
    }
    wake.addEventListener('click', async () => {
        if (!canWake) return;
        wanted = !lock;
        store.set('biodash-keep-awake', wanted ? '1' : '0');
        if (lock) { await lock.release(); lock = null; paintWake(); } else acquire();
    });
    document.addEventListener('visibilitychange', acquire);
    paintWake(); acquire();

    // two taps to act: the first arms a button (red, 4 s), the second confirms
    let armed = null, armTimer = null;
    function arm(id, repaint) {
        if (armed === id) { clearTimeout(armTimer); armed = null; return true; }
        armed = id; repaint();
        clearTimeout(armTimer);
        armTimer = setTimeout(() => { armed = null; repaint(); }, 4000);
        return false;
    }

    // ---------------------------------------------------------------- updates
    const upd = sb.querySelector('#sb-update');
    let info = null, checking = false, running = null, note = null;
    const ago = (t) => { const s = Date.now() / 1000 - t; return s < 90 ? 'just now' : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`; };

    function paintUpdate() {
        const v = info && info.version ? `V${info.version}` : '';
        sb.querySelector('#sb-ver').textContent = v;
        if (running) {
            const st = running.status || {};
            upd.innerHTML = `<div class="sb-note warn">${running.action === 'upgrade' ? 'Upgrading' : 'Applying the patch'}…</div>
                <div class="sb-note">${esc(st.stage || 'Starting')}</div>
                ${st.log && st.log.length ? `<pre class="sb-pre">${esc(st.log.slice(-8).join('\n'))}</pre>` : ''}
                <div class="sb-note" style="margin:8px 0 0">The dashboards restart when it finishes; this page reloads by itself.</div>`;
            return;
        }
        let h = '';
        if (!info) h += `<div class="sb-note">Not checked yet.</div>`;
        else if (info.error) h += `<div class="sb-note err">${esc(info.error)}</div>`;
        else {
            const p = info.patch, u = info.upgrade;
            if (!p && !u) h += `<div class="sb-note ok">${v} is up to date.</div>`;
            if (p) h += `<div class="sb-note warn">Patch for ${v}: ${p.count} change${p.count === 1 ? '' : 's'}</div>
                <ul class="sb-list">${p.commits.map(c => `<li>${esc(c.subject)}</li>`).join('')}</ul>`;
            if (u) h += `<div class="sb-note warn" style="margin-top:8px">New version: V${esc(u.version)}</div>
                ${u.notes ? `<details><summary style="cursor:pointer;font-size:11.5px;color:#94a3b8">What's new</summary><pre class="sb-pre">${esc(u.notes)}</pre></details>` : ''}`;
            (info.blockers || []).forEach(b => { h += `<div class="sb-note err" style="margin-top:8px">${esc(b)}</div>`; });
            const can = info.allowed && !(info.blockers || []).length;
            if (p) h += `<button class="sb-btn ${armed === 'patch' ? 'armed' : 'go'}" data-act="patch" ${can ? '' : 'disabled'}>${armed === 'patch' ? 'Confirm: patch and restart?' : 'Apply patch'}</button>`;
            if (u) h += `<button class="sb-btn ${armed === 'upgrade' ? 'armed' : 'go'}" data-act="upgrade" ${can ? '' : 'disabled'}>${armed === 'upgrade' ? `Confirm: move to V${esc(u.version)}?` : `Upgrade to V${esc(u.version)}`}</button>`;
            if ((p || u) && can) h += `<div class="sb-note" style="margin:6px 0 0">Recording pauses while the dashboards restart${(info.services || []).length ? '' : ' (no services installed: you restart them yourself)'}.</div>`;
            if ((p || u) && !info.allowed) h += `<div class="sb-note warn" style="margin:6px 0 0">${esc(info.reason)}</div>`;
        }
        if (note) h += `<div class="sb-note ${note.cls}" style="margin-top:8px">${esc(note.text)}</div>`;
        h += `<button class="sb-btn" data-act="check" ${checking ? 'disabled' : ''}>${checking ? 'Checking GitHub…' : 'Check for updates'}</button>`;
        if (info && info.checked_at && !info.error) h += `<div class="sb-note" style="margin:6px 0 0;font-size:10.5px">Last checked ${ago(info.checked_at)}</div>`;
        upd.innerHTML = h;
    }
    async function loadUpdate(check) {
        try { info = await (await fetch('/api/system/update' + (check ? '?check=1' : ''), { cache: 'no-store' })).json(); }
        catch (e) { if (check) info = { error: 'No reply from the dashboard.' }; }
        const st = info && info.status;
        if (st && st.state === 'running' && !running) follow({ action: st.action, started: st.started });   // started from another tab
        paintUpdate();
    }
    function follow(run) {
        running = { ...run, status: null };
        paintUpdate();
        const tick = async () => {
            let st = null;
            try { st = (await (await fetch('/api/system/update', { cache: 'no-store' })).json()).status; } catch (e) {}
            if (!st) { running.status = { ...(running.status || {}), stage: 'Restarting the dashboards…' }; paintUpdate(); return setTimeout(tick, 2000); }
            if ((st.started || 0) < run.started - 5) return setTimeout(tick, 1500);          // the previous run's result
            running.status = st; paintUpdate();
            if (st.state === 'running') return setTimeout(tick, 2000);
            running = null;
            note = { cls: st.state === 'done' ? 'ok' : 'err', text: st.stage };
            await loadUpdate(false);
            if (st.state === 'done') setTimeout(() => location.reload(), 4000);
        };
        setTimeout(tick, 1200);
    }
    upd.addEventListener('click', async (e) => {
        const act = e.target.closest('button[data-act]')?.dataset.act;
        if (!act || running) return;
        if (act === 'check') { checking = true; note = null; paintUpdate(); await loadUpdate(true); checking = false; return paintUpdate(); }
        if (!arm(act, paintUpdate)) return;
        let res = null, out = {};
        try { res = await fetch('/api/system/update', { method: 'POST', headers: HEADER, body: JSON.stringify({ action: act }) }); out = await res.json(); }
        catch (err) { out = { error: 'No reply from the dashboard.' }; }
        if (res && res.ok) return follow({ action: act, started: out.started });
        note = { cls: 'err', text: out.error || 'Refused.' };
        paintUpdate();
    });
    paintUpdate();
    loadUpdate(false);

    // ---------------------------------------------------------------- power
    const pow = sb.querySelector('#sb-power');
    let pinfo = null, pdone = false, pnote = null;
    function paintPower() {
        if (pdone) return;
        sb.querySelector('#sb-foot').textContent = pinfo ? pinfo.host : '';
        const ok = pinfo && pinfo.allowed;
        const how = !pinfo ? 'Checking…' : pinfo.dry_run ? 'Dry run: nothing will actually happen' : ok ? '' : pinfo.reason;
        const label = (a) => armed === a ? 'Confirm?' : (a === 'shutdown' ? '⏻ Shut down' : '⟳ Reboot');
        pow.innerHTML = `${how ? `<div class="sb-note ${pinfo ? 'warn' : ''}">${esc(how)}</div>` : ''}
            <div class="sb-row">
                <button class="sb-btn ${armed === 'reboot' ? 'armed' : ''}" data-power="reboot" ${ok ? '' : 'disabled'}>${label('reboot')}</button>
                <button class="sb-btn ${armed === 'shutdown' ? 'armed' : ''}" data-power="shutdown" ${ok ? '' : 'disabled'}>${label('shutdown')}</button>
            </div>
            ${pinfo && pinfo.fix ? `<details style="margin-top:8px"><summary style="cursor:pointer;font-size:11.5px;color:#94a3b8">Allow it without a password</summary>
                <div class="sb-note" style="margin-top:6px">Run once on this computer:</div><pre class="sb-pre">${esc(pinfo.fix)}</pre></details>` : ''}
            ${pnote ? `<div class="sb-note ${pnote.cls}" style="margin:8px 0 0">${esc(pnote.text)}</div>` : ''}`;
    }
    async function loadPower() {
        try { pinfo = await (await fetch('/api/system/power', { cache: 'no-store' })).json(); } catch (e) {}
        paintPower();
    }
    pow.addEventListener('click', async (e) => {
        const action = e.target.closest('button[data-power]')?.dataset.power;
        if (!action || !pinfo || !pinfo.allowed || pdone) return;
        if (!arm(action, paintPower)) return;
        let res = null, out = {};
        try { res = await fetch('/api/system/power', { method: 'POST', headers: HEADER, body: JSON.stringify({ action }) }); out = await res.json(); }
        catch (err) { out = { error: 'No reply from the dashboard.' }; }
        if (res && res.ok) {
            if (out.dry_run) { pnote = { cls: 'warn', text: `Dry run: ${action} accepted (nothing happened).` }; return paintPower(); }
            pdone = true;
            pow.innerHTML = `<div class="sb-note warn" style="margin:0">${action === 'shutdown' ? '⏻ Shutting down' : '⟳ Rebooting'} ${esc(pinfo.host)} in 2 s…
                ${action === 'reboot' ? 'The dashboards come back on their own if they run as services.' : ''}</div>`;
        } else {
            if (out.fix && pinfo) pinfo.fix = out.fix;
            pnote = { cls: 'err', text: out.error || 'Refused.' };
            paintPower();
        }
    });
    paintPower();
    loadPower();
    setInterval(() => { if (document.body.classList.contains('sb-open') && !armed && !pdone) loadPower(); }, 30000);
})();
