// Shared by every device page: the page header with its live status, the "find your device"
// panel shown while it isn't connected, and the Device sheet (switch device, auto-reconnect,
// radio, restart, log).
import { dev, kindOf, personOf, icon, esc, pill, stateOf, rssiBars, api, dapi, sheet, confirm, toast, busy, ago } from './ui.js';
import { store } from './store.js';
import { Nick } from './nick.js';

// Whose device a page shows, when several people wear this kind: a switcher under the title.
function personSwitch(key) {
    const kind = kindOf(key), who = store.wearers(kind);
    if (!store.several() || !store.app.devices[kind].personal) return '';
    return `<nav class="seg people-seg" aria-label="Whose ${esc(dev(key).label.toLowerCase())}">${who.map(p =>
        `<a href="#/${kind}/${p.id}" ${p.id === personOf(key) ? 'aria-current="true"' : ''}>${esc(store.personName(p.id))}</a>`).join('')}</nav>`;
}

export function pageHead(key, extra = '') {
    const d = dev(key);
    return `<div class="page-head" style="--c:${d.c}">
      <div class="titles"><h1><span class="glyph">${icon(d.icon)}</span>${d.label}</h1>
        <p class="sub" data-devname>${esc(d.product)}</p></div>
      <div class="actions">${extra}<span data-pill>${pill(store.status[key])}</span>
        <button class="btn" data-device-sheet>${icon('bluetooth')}<span>Device</span></button></div>
    </div>${personSwitch(key)}`;
}

// who else (another person) has this address: connected to it, or saved for auto-reconnect
function ownerOf(key, address) {
    const kind = kindOf(key);
    for (const p of store.wearers(kind)) {
        const k = p.devices[kind].key, st = store.status[k] || {};
        if (k !== key && address && (st.address === address && st.state !== 'scanning' || (st.last_device || {}).address === address)) return p.id;
    }
    return null;
}

// keeps the header (pill + device name) and anything else in sync with the status
export function bindHead(root, key, onStatus) {
    const paint = () => {
        const st = store.status[key];
        const p = root.querySelector('[data-pill]');
        if (p) p.innerHTML = pill(st);
        const n = root.querySelector('[data-devname]');
        if (n) {
            const live = st && st.state === 'streaming' && st.name, ld = (st && st.last_device) || {};
            n.innerHTML = live ? esc(st.name) + Nick.tag(st.name, st.address)
                : esc(dev(key).product) + (ld.address ? Nick.tag(ld.name, ld.address) : '');
        }
        onStatus && onStatus(st, stateOf(st));
    };
    root.querySelector('[data-device-sheet]')?.addEventListener('click', () => deviceSheet(key));
    paint();
    const offNick = Nick.on(paint);
    const offStatus = store.on('status', (k) => { if (k === key) paint(); });
    return () => { offNick(); offStatus(); };
}

// ------------------------------------------------------------------ finder
// The list of devices the worker's last scan saw, with a Connect button each.
export function finder(el, key, { onlyLikely = true, extra = '' } = {}) {
    const d = dev(key);
    let showAll = !onlyLikely, timer = null, sig = '', alive = true, connectingTo = null;
    el.classList.add('finder');
    el.style.setProperty('--c', d.c);
    el.innerHTML = `<div data-list></div>
      ${key === 'water' ? `<label class="switch small" style="margin-top:12px"><input type="checkbox" data-all><span class="text"><b>Show everything nearby</b><span>Not only things that look like a bottle</span></span></label>` : ''}
      ${extra}`;
    const list = el.querySelector('[data-list]');
    el.querySelector('[data-all]')?.addEventListener('change', (e) => { showAll = e.target.checked; sig = ''; tick(); });

    async function tick() {
        if (!alive) return;
        const st = store.status[key] || {};
        if (st.state === 'connecting' || connectingTo) {
            const who = st.name || st.address || connectingTo;
            const tries = st.attempt > 1 ? ` — try ${st.attempt} of ${st.max_attempts || '?'}` : '';
            const html = `<div class="searching"><span class="radar"></span><div><b>Connecting to ${esc(who || 'your device')}…</b><div class="small">${tries ? esc(tries.slice(3)) : 'This usually takes a few seconds.'}</div></div></div>`;
            if (sig !== html) { list.innerHTML = html; sig = html; }
            if (st.state === 'streaming' || st.state === 'scanning' && connectingTo && Date.now() - connectingTo.at > 30000) connectingTo = null;
            return;
        }
        let devices = [];
        try { devices = await dapi(key, '/api/scan-results', { quiet: true }); } catch (e) { /* offline */ }
        if (!alive) return;
        if (key === 'water' && !showAll) devices = devices.filter(x => x.bottle);
        const ld = (st.last_device || {}).address;
        const html = devices.length ? devices.map(x => `
            <div class="found ${x.address === ld ? 'current' : ''}">
              ${rssiBars(x.rssi)}
              <div class="grow"><div class="title">${esc(x.name || 'Unnamed device')}${Nick.tag(x.name, x.address)}</div>
                <div class="desc">${x.address === ld ? 'Used last time · ' : ''}${ownerOf(key, x.address) ? `${esc(store.personName(ownerOf(key, x.address)))} has it · ` : ''}${x.rssi != null ? esc(x.rssi) + ' dBm' : ''}</div></div>
              <button class="btn sm ${ownerOf(key, x.address) ? '' : 'primary'}" data-connect="${esc(x.address)}" data-name="${esc(x.name || '')}">${ownerOf(key, x.address) ? 'Move here' : 'Connect'}</button>
            </div>`).join('')
            : `<div class="searching"><span class="radar"></span><div><b>Looking for your ${esc(d.product)}…</b><div class="small">${esc(d.find)}</div></div></div>
               <details class="more"><summary>Not showing up?</summary><div class="body"><ul class="small muted" style="margin:0;padding-left:18px">${d.notFound.map(t => `<li>${esc(t)}</li>`).join('')}</ul></div></details>`;
        const key2 = devices.map(x => x.address + x.name + (ownerOf(key, x.address) || '')).join('|') + (devices.length ? '' : 'none');
        if (key2 !== sig) { list.innerHTML = html; sig = key2; }
        else devices.forEach(x => { const b = list.querySelector(`[data-connect="${CSS.escape(x.address)}"]`); if (b) b.parentElement.querySelector('.bars').outerHTML = rssiBars(x.rssi); });
    }
    list.addEventListener('click', async (e) => {
        const b = e.target.closest('[data-connect]');
        if (!b) return;
        connectingTo = { address: b.dataset.connect, at: Date.now() };
        await busy(b, async () => {
            try {
                // a strap or ring is one person's at a time: whoever else had it lets go first
                if (store.several()) await api('/api/app/claim', { body: { device: key, address: b.dataset.connect } });
                await dapi(key, '/api/connect', { body: { address: b.dataset.connect } });
            } catch (err) { connectingTo = null; }
        });
        sig = ''; tick();
    });
    const unsub = store.on('status', (k) => { if (k === key) tick(); });
    tick();
    timer = setInterval(tick, 1500);
    return () => { alive = false; clearInterval(timer); unsub(); };
}

// The panel a device page shows instead of its readings while nothing is connected.
export function connectPanel(el, key, extra = '') {
    const d = dev(key);
    el.innerHTML = `<div class="card" style="--c:${d.c}">
        <div class="card-head"><h2>Connect your ${esc(d.product)}</h2></div>
        <div data-finder></div></div>`;
    return finder(el.querySelector('[data-finder]'), key, { extra });
}

// ------------------------------------------------------------------ the Device sheet
export async function deviceSheet(key) {
    const d = dev(key), kind = kindOf(key);
    const whose = store.several() && store.app.devices[kind].personal ? ` · ${esc(store.personName(personOf(key)))}` : '';
    const canDisconnect = kind === 'heart' || kind === 'brain';        // their workers take a "disconnect" command
    const s = sheet({ title: `${d.label}${whose} · device`, body: '<div data-body></div>' });
    const body = s.body.querySelector('[data-body]');
    let stopFinder = null, logTimer = null;
    const cleanup = () => { stopFinder && stopFinder(); clearInterval(logTimer); unsub(); };
    const unsub = store.on('status', (k) => { if (k === key) paintTop(); });
    const obs = new MutationObserver(() => { if (!s.el.isConnected) { cleanup(); obs.disconnect(); } });
    obs.observe(document.getElementById('layer'), { childList: true });

    body.innerHTML = `
      <div data-top></div>
      <div class="section-title">Find a device</div>
      <div data-finder></div>
      <div class="section-title">Bluetooth radio</div>
      <div data-radio class="small muted">Loading…</div>
      <details class="more"><summary>Troubleshooting</summary><div class="body stack">
        <div class="row"><button class="btn sm" data-restart>${icon('refresh')}Restart ${esc(d.label.toLowerCase())} connection</button>
          <button class="btn sm ghost" data-copy>Copy details</button></div>
        <dl class="kv" data-kv></dl>
        <div data-dbg></div>
        <div><div class="small muted" style="margin-bottom:6px">Recent messages</div><div class="log" data-log>…</div></div>
      </div></details>`;

    function paintTop() {
        const st = store.status[key] || {}, ss = stateOf(st), ld = st.last_device || {};
        body.querySelector('[data-top]').innerHTML = `
          <div class="list">
            <div class="item" style="--c:${d.c}">${icon(d.icon)}<div class="grow">
              <div class="title">${esc(ss.live ? (st.name || d.product) : (ld.name || d.product))}${(ss.live ? st.name : ld.name) || ld.address ? Nick.tag(ss.live ? st.name : ld.name, ss.live ? st.address : ld.address) : ''}</div>
              <div class="desc">${ss.live ? `Connected${st.rssi != null ? ' · signal ' + esc(st.rssi) + ' dBm' : ''}${st.radio && st.radio.hci ? ' · via ' + esc(st.radio.hci) : ''}`
                  : st.held ? 'Disconnected — pick a device below to connect again' : esc(ss.text)}</div></div>
              ${ss.live && canDisconnect ? `<button class="btn sm" data-disconnect title="End this session; reconnecting waits until you pick a device">${icon('eject')}Disconnect</button>` : pill(st)}</div>
            <label class="item switch" style="--c:${d.c}"><span class="text"><b>Reconnect automatically</b>
              <span>${ld.address ? `To ${esc(ld.name || ld.address)} whenever it’s in range` : 'Once you’ve connected a device'}</span></span>
              <input type="checkbox" data-auto ${ld.auto !== false ? 'checked' : ''} ${ld.address ? '' : 'disabled'}></label>
            ${ld.address ? `<div class="item"><div class="grow"><div class="title">Use a different ${esc(d.product)}</div>
              <div class="desc">Forgets ${esc(ld.name || 'this one')} and looks for others</div></div>
              <button class="btn sm" data-forget>Forget</button></div>` : ''}
          </div>`;
        body.querySelector('[data-auto]')?.addEventListener('change', async (e) => {
            await dapi(key, '/api/last-device', { body: { role: 'device', auto: e.target.checked } });
            toast(e.target.checked ? 'Will reconnect automatically' : 'Won’t reconnect automatically', 'ok');
        });
        body.querySelector('[data-disconnect]')?.addEventListener('click', async (e) => {
            await busy(e.currentTarget, () => dapi(key, '/api/disconnect', { method: 'POST' }));
            toast('Disconnected', 'ok');
        });
        body.querySelector('[data-forget]')?.addEventListener('click', async (e) => {
            const ok = await confirm({ title: `Forget ${esc(ld.name || 'this device')}?`, text: 'Bio-dash disconnects and starts looking for devices again. Your recordings stay.', ok: 'Forget' });
            if (!ok) return;
            await busy(e.target, async () => {
                await dapi(key, '/api/last-device', { body: { role: 'device', forget: true } });
                await api(`/api/app/restart/${key}`, { method: 'POST' });
            });
            toast('Looking for devices…', 'ok');
        });
    }
    paintTop();
    stopFinder = finder(body.querySelector('[data-finder]'), key);

    // radio choice
    (async () => {
        const box = body.querySelector('[data-radio]');
        try {
            const [{ radio }, radios] = await Promise.all([dapi(key, '/api/radio', { quiet: true }), api('/api/app/radios', { quiet: true })]);
            const ad = radios.adapters || [];
            if (!ad.length) { box.textContent = 'No Bluetooth radios found on this computer.'; return; }
            const used = ((store.status[key] || {}).radio || {}).hci;
            box.innerHTML = `<label class="field"><span>Use this radio</span>
              <select class="input" data-pin>${key === 'water' ? `<option value="any" ${radio.toUpperCase() === 'ANY' ? 'selected' : ''}>Any radio: whichever hears the bottle best${used ? ` (now ${esc(used)})` : ''}</option>` : ''}
              <option value="auto" ${radio.toUpperCase() === 'AUTO' ? 'selected' : ''}>Automatic${used && key !== 'water' ? ` (now ${esc(used)})` : ''}</option>
              ${ad.filter(a => a.address).map(a => `<option value="${esc(a.address)}" ${radio.toUpperCase() === (a.address || '').toUpperCase() ? 'selected' : ''}>${esc(a.name)} — ${esc([a.vendor, a.placement === 'external' ? 'USB dongle' : a.placement === 'internal' ? 'built in' : ''].filter(Boolean).join(', ') || a.alias || '')}</option>`).join('')}
              </select><span class="hint">${key === 'water' ? 'Any radio is right for almost everyone: every radio listens and the one that hears the bottle best connects; a radio that fails with it is passed over.' : 'Automatic is right for almost everyone.'} Pin a radio when two devices fight over one, or a USB dongle hears better.</span></label>
              ${(((store.status[key] || {}).radio || {}).avoided || []).map(x => `<p class="small warn" style="margin-top:6px">Passed over until ${esc(new Date(x.until * 1000).toLocaleString())}: ${esc(x.address)} (a connection through it failed)</p>`).join('')}`;
            box.querySelector('[data-pin]').addEventListener('change', async (e) => {
                await dapi(key, '/api/radio', { body: { address: e.target.value } });
                toast('Radio saved — it’s used from the next search', 'ok');
            });
        } catch (e) { box.textContent = 'Couldn’t list the radios.'; }
    })();

    // troubleshooting
    const kv = body.querySelector('[data-kv]'), log = body.querySelector('[data-log]');
    let since = 0;
    async function pollLog() {
        try {
            const r = await api(`/api/app/log/${key}?since=${since}`, { quiet: true });
            if (r.lines.length) {
                if (since === 0) log.textContent = '';
                since = r.lines[r.lines.length - 1][0];
                log.textContent += r.lines.map(([t, l]) => `${new Date(t * 1000).toLocaleTimeString()}  ${l}`).join('\n') + '\n';
                log.textContent = log.textContent.split('\n').slice(-300).join('\n');
                log.scrollTop = log.scrollHeight;
            } else if (since === 0) log.textContent = 'Nothing yet.';
            const st = store.status[key] || {}, w = r.worker || {};
            kv.innerHTML = [['Worker', `${w.state || 'off'}${w.pid ? ' (pid ' + w.pid + ')' : ''}${w.restarts ? ', restarted ' + w.restarts + '×' : ''}`],
                ['Running since', w.started_at ? ago(w.started_at).replace(' ago', '') : '—'],
                ['State', st.state || '—'], ['Address', st.address || '—'],
                ['Radio', st.radio ? `${st.radio.hci || 'system default'} — ${st.radio.reason || ''}` : '—'],
                ['Last exit', w.last_exit ? `code ${w.last_exit.code}, ${ago(w.last_exit.at)}` : '—']]
                .map(([a, b]) => `<dt>${a}</dt><dd>${esc(b)}</dd>`).join('');
            paintDebug();
        } catch (e) { /* offline */ }
    }
    // the worker's own telemetry: link, rates, counters (what V5's debug panels showed)
    const human = (k) => k.replace(/_/g, ' ').replace(/\bper s\b/, '/s');
    const val = (v) => v == null ? '—' : typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2))
        : typeof v === 'object' ? JSON.stringify(v) : String(v);
    async function paintDebug() {
        let st;
        try { st = await dapi(key, '/api/status', { quiet: true }); } catch (e) { return; }
        const dbg = st.debug || {}, rows = [];
        const add = (label, v) => { if (v !== undefined && v !== null && v !== '') rows.push(`<dt>${esc(label)}</dt><dd>${esc(val(v))}</dd>`); };
        if (dbg.device) add('Signal', dbg.device.rssi != null ? `${dbg.device.rssi} dBm` : null);
        add('Connected since', dbg.connected_at ? new Date(dbg.connected_at * 1000).toLocaleTimeString() : null);
        add('Saving to', dbg.recording);
        add('Last error', dbg.last_error);
        add('Handshake', dbg.handshake);
        if (dbg.gatt && dbg.gatt.mtu) add('MTU', dbg.gatt.mtu);
        for (const [k, v] of Object.entries(dbg.rates || {})) add(human(k), v);
        for (const [k, v] of Object.entries(dbg.counters || {})) add(human(k), v);
        if (dbg.parser) { add('Profile', dbg.parser.effective_profile || dbg.parser.profile); add('Columns from', dbg.parser.columns_from); }
        for (const [k, v] of Object.entries(dbg.last_frames || {})) add('Last frame · ' + k, v);
        for (const [k, v] of Object.entries(dbg.versions || {})) add(k, v);
        body.querySelector('[data-dbg]').innerHTML = rows.length ? `<div class="small muted" style="margin:10px 0 6px">From the ${esc(d.label.toLowerCase())} worker</div><dl class="kv">${rows.join('')}</dl>` : '';
    }
    body.querySelector('details').addEventListener('toggle', (e) => {
        clearInterval(logTimer);
        if (e.target.open) { pollLog(); logTimer = setInterval(pollLog, 2000); }
    });
    body.querySelector('[data-restart]').addEventListener('click', async (e) => {
        await busy(e.currentTarget, () => api(`/api/app/restart/${key}`, { method: 'POST' }));
        toast('Restarted', 'ok');
    });
    body.querySelector('[data-copy]').addEventListener('click', async () => {
        const text = `Bio-dash ${store.app.version} · ${d.label}\n` + kv.innerText + '\n\n' + log.textContent.split('\n').slice(-60).join('\n');
        try { await navigator.clipboard.writeText(text); toast('Copied', 'ok'); }
        catch (e) { toast('Couldn’t copy here — select the text instead', 'bad'); }
    });
    return s;
}
