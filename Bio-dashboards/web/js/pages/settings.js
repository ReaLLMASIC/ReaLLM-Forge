// Settings: people and devices, look, forecasts, Bluetooth radios, updates, this computer, about.
import { DEVICES, DEVICE_KEYS, PERSON_STYLE, dev, kindOf, personOf, icon, esc, api, dapi, pill, toast, busy, confirm, ask, ago, $ } from '../ui.js';
import { wakeSupported, wakeWanted, setWake } from '../wake.js';

const THEMES = [
    ['auto', 'Auto', ['#f3f5f9', '#0b1120']], ['dark', 'Dark', ['#0b1120', '#131c31']], ['light', 'Light', ['#f3f5f9', '#ffffff']],
    ['oled', 'OLED black', ['#000000', '#0a0a0c']], ['contrast', 'High contrast', ['#000000', '#ffffff']], ['warm', 'Warm night', ['#1a0f02', '#3b2307']],
];
const POWER = { 'X-Biodash-Power': '1' };

export async function mount(el, { store }) {
    const SECTIONS = [['devices', 'Devices'], ['people', 'People'], ['look', 'Look'], ...(store.app.forecast ? [['forecasts', 'Forecasts']] : []),
                      ['radios', 'Bluetooth'], ['updates', 'Updates'], ['system', 'This computer'], ['about', 'About']];
    el.innerHTML = `
      <div class="page-head"><div class="titles"><h1><span class="glyph" style="--c:var(--muted)">${icon('settings')}</span>Settings</h1></div></div>
      <nav class="jump" aria-label="Sections">${SECTIONS.map(([id, label]) => `<button data-jump="${id}">${label}</button>`).join('')}</nav>
      <div class="stack" style="max-width:860px">
        <section class="card" id="set-devices"><div class="card-head"><h2>Devices</h2><span class="spacer"></span><a class="btn sm" href="#/setup">${icon('refresh')}Run setup again</a></div>
          <p class="small muted" style="margin-bottom:8px">Turn on the devices you own. Bio-dash looks for each one it’s told about, and records whenever it’s connected.</p>
          <div class="list" data-devices></div></section>
        <section class="card" id="set-people"><div class="card-head"><h2>People</h2><span class="card-sub">up to ${store.app.max_people} at once</span><span class="spacer"></span>
            <button class="btn sm" data-add-person>${icon('plus')}Add a person</button></div>
          <div data-people></div></section>
        <section class="card" id="set-look"><div class="card-head"><h2>Look</h2><span class="card-sub">for every screen on this computer’s address</span></div><div class="themes" data-themes></div>
          <label class="switch" style="margin-top:14px" data-wake-row><span class="text"><b>Keep this screen on</b><span data-wake-note>While Bio-dash is showing — for a tablet on a stand. This screen only.</span></span>
            <input type="checkbox" data-wake></label></section>
        ${store.app.forecast ? `<section class="card" id="set-forecasts"><div class="card-head"><h2>Forecasts</h2><span class="card-sub">the Heart page’s look ahead</span><span class="spacer"></span>
            <a class="btn sm" href="/lab" target="_blank" rel="noopener">${icon('flask')}Model lab</a></div><div data-forecasts class="small muted">Loading…</div></section>` : ''}
        <section class="card" id="set-radios"><div class="card-head"><h2>Bluetooth radios</h2><span class="spacer"></span><button class="btn sm ghost" data-radios-refresh>${icon('refresh')}Refresh</button></div><div data-radios class="small muted">Loading…</div></section>
        <section class="card" id="set-updates"><div class="card-head"><h2>Updates</h2><span class="spacer"></span><button class="btn sm" data-check>${icon('refresh')}Check now</button></div><div data-updates class="small muted">Loading…</div></section>
        <section class="card" id="set-system"><div class="card-head"><h2>This computer</h2></div><div data-system class="stack"></div></section>
        <section class="card" id="set-about"><div class="card-head"><h2>About</h2></div><div data-about class="small"></div></section>
      </div>`;
    $('.jump', el).addEventListener('click', (e) => {
        const b = e.target.closest('[data-jump]');
        if (b) document.getElementById('set-' + b.dataset.jump).scrollIntoView({ behavior: 'smooth', block: 'start' });
    });

    // ---- look
    const paintThemes = () => {
        const cur = window.BioTheme.choice();
        $('[data-themes]', el).innerHTML = THEMES.map(([k, label, [a, b]]) => `<button class="theme-opt" data-theme-opt="${k}" aria-pressed="${cur === k}">
          <span class="sw" style="${k === 'warm' ? 'filter:sepia(1) saturate(1.8) hue-rotate(-22deg) brightness(.85)' : ''}"><i style="background:${a}"></i><i style="background:${b}"></i></span><b>${label}</b></button>`).join('');
    };
    $('[data-themes]', el).addEventListener('click', (e) => { const b = e.target.closest('[data-theme-opt]'); if (b) { window.BioTheme.set(b.dataset.themeOpt); paintThemes(); } });
    paintThemes();
    const wake = $('[data-wake]', el);
    wake.checked = wakeWanted();
    if (!wakeSupported()) {
        wake.disabled = true;
        $('[data-wake-note]', el).textContent = location.protocol === 'https:' ? 'This browser can’t keep the screen on.'
            : 'Browsers only allow this on a secure address (https, or on this computer itself).';
    }
    wake.addEventListener('change', () => setWake(wake.checked));

    // ---- devices and people
    const several = () => store.several();
    const shared = () => DEVICE_KEYS.filter(k => !store.app.devices[k].personal || !several());       // with one person: every kind
    const switchRow = (key, title, desc) => {
        const d = dev(key), on = store.enabled(key);
        return `<div class="item" style="--c:${d.c}">${icon(d.icon)}<div class="grow"><div class="title">${title}</div><div class="desc">${desc}</div></div>
          ${on ? `<span class="hide-sm">${pill(store.status[key])}</span>` : ''}
          <label class="switch" style="--c:${d.c}" title="${on ? 'Turn off' : 'Turn on'}"><input type="checkbox" data-dev="${key}" ${on ? 'checked' : ''} aria-label="${esc(d.label)}"></label></div>`;
    };
    const paintDevices = () => {
        $('[data-devices]', el).innerHTML = shared().map(k => switchRow(k, `${DEVICES[k].label} <span class="muted" style="font-weight:400">· ${esc(DEVICES[k].product)}</span>`, esc(DEVICES[k].what))).join('')
            + (several() ? `<p class="small muted" style="margin-top:10px">Straps and rings are each person’s own: see People below.</p>` : '');
        const people = store.people(), box = $('[data-people]', el);
        $('[data-add-person]', el).hidden = people.length >= store.app.max_people;
        if (!several()) {
            box.innerHTML = `<p class="small muted">Bio-dash is following one person. Add a second or third to give each their own heart strap and oxygen ring,
              and see them side by side on a <b>Compare</b> page. Each strap wants its own Bluetooth radio, so plug in a USB dongle per extra person.</p>`;
            return;
        }
        box.innerHTML = people.map((p, i) => `<div class="person" style="--c:${PERSON_STYLE[p.id].c}">
            <div class="person-head"><span class="pmark ${PERSON_STYLE[p.id].mark}"></span><b>${esc(store.personName(p.id))}</b>
              <button class="icon-btn sm" data-rename="${p.id}" aria-label="Rename ${esc(store.personName(p.id))}" title="Rename">${icon('edit')}</button><span class="spacer"></span>
              ${i ? `<button class="btn sm ghost" data-remove="${p.id}">${icon('trash')}Remove</button>` : ''}</div>
            <div class="list">${['heart', 'oxygen'].map(kind => switchRow(p.devices[kind].key, `${DEVICES[kind].label} <span class="muted" style="font-weight:400">· ${esc(DEVICES[kind].product)}</span>`,
                deviceLine(p.devices[kind].key))).join('')}</div></div>`).join('')
            + `<p class="small muted" style="margin-top:10px">To give someone a strap or ring, open <a href="#/heart">Heart</a> or <a href="#/oxygen">Oxygen</a>, pick the person, then <b>Device</b>. A device belongs to one person at a time.</p>`;
    };
    // which unit this person's worker is on (or was last)
    function deviceLine(key) {
        const st = store.status[key] || {}, ld = st.last_device || {};
        if (!store.enabled(key)) return 'Off';
        if (st.state === 'streaming' && st.name) return `Wearing ${esc(st.name)}`;
        return ld.name || ld.address ? `Last wore ${esc(ld.name || ld.address)}` : 'No device picked yet';
    }
    const onToggle = async (e) => {
        const k = e.target.dataset.dev;
        if (!k) return;
        const on = e.target.checked, d = dev(k);
        const who = several() && store.app.devices[kindOf(k)].personal ? `${store.personName(personOf(k))}’s ` : 'your ';
        if (!on && !await confirm({ title: `Turn off ${d.label}?`, text: `Bio-dash stops looking for ${esc(who)}${esc(d.product)} and stops recording it. Recordings so far are kept.`, ok: 'Turn off' })) { e.target.checked = true; return; }
        try {
            await api('/api/app/device', { body: { device: k, enabled: on } });
            await store.refreshApp();
            toast(on ? `${d.label} is on — looking for it` : `${d.label} is off`, 'ok');
        } catch (err) { e.target.checked = !on; }
        paintDevices();
    };
    $('[data-devices]', el).addEventListener('change', onToggle);
    $('[data-people]', el).addEventListener('change', onToggle);
    const nameOf = (title, value) => ask({ title, text: 'A name to tell people apart on the Compare page. It stays on this computer.', value, placeholder: 'Name', max: 24 });
    $('[data-add-person]', el).addEventListener('click', async (e) => {
        const btn = e.currentTarget;                       // gone from the event once we've awaited
        const name = await nameOf('Add a person', '');
        if (name === null) return;
        try {
            await busy(btn, () => api('/api/app/people', { body: { name } }));
            await store.refreshApp();
            toast('Added — now connect their strap and ring from Heart and Oxygen', 'ok');
        } catch (err) { /* the toast says why */ }
        paintDevices();
    });
    $('[data-people]', el).addEventListener('click', async (e) => {
        const ren = e.target.closest('[data-rename]'), rem = e.target.closest('[data-remove]');
        if (ren) {
            const name = await nameOf('Rename', (store.people().find(p => p.id === ren.dataset.rename) || {}).name || '');
            if (name === null) return;
            await api(`/api/app/people/${ren.dataset.rename}`, { body: { name } }).catch(() => {});
        } else if (rem) {
            const who = store.personName(rem.dataset.remove);
            if (!await confirm({ title: `Stop following ${esc(who)}?`, text: 'Their strap and ring are disconnected and forgotten. Their recordings stay.', ok: 'Remove', danger: true })) return;
            await busy(rem, () => api(`/api/app/people/${rem.dataset.remove}`, { method: 'DELETE' })).catch(() => {});
        } else return;
        await store.refreshApp();
        paintDevices();
    });
    paintDevices();
    const offStatus = store.on('status', paintDevices);

    // ---- forecasts: what the service is doing (the switches are on the Heart page, by each chart)
    let fcTimer = null;
    async function paintForecasts() {
        const box = $('[data-forecasts]', el);
        if (!box) return;
        let v;
        try { v = await dapi('heart', '/api/forecast/settings', { quiet: true }); } catch (e) { box.textContent = 'The Heart API isn’t answering.'; return; }
        const svc = v.service || {}, names = { ecg: ['ECG', 'ECG only'], acc: ['Motion', 'ECG + accelerometer'], rr: ['Beat-to-beat', 'ECG + R-R, a proof of concept'] };
        box.innerHTML = `<div class="callout ${svc.online ? (svc.state === 'unavailable' ? 'warn' : 'ok') : 'warn'}">${icon(svc.online && svc.state !== 'unavailable' ? 'check' : 'warn')}<span>${
            !svc.online ? 'The forecast service isn’t running. It starts with Bio-dash; its messages are under Troubleshooting on the Heart device sheet.'
            : svc.state === 'unavailable' ? esc(svc.detail || 'Forecasts are unavailable.') : `The forecast service is ${esc(svc.state || 'running')}${svc.device ? ', following ' + esc(svc.device) : ''}.`}</span></div>
          <div class="list" style="margin-top:10px">${Object.entries(names).map(([k, [label, model]]) => {
              const ov = (svc.overlays || {})[k] || {};
              return `<div class="item">${icon('sparkle')}<div class="grow"><div class="title">${label} <span class="muted" style="font-weight:400">· ${model}</span></div>
                <div class="desc">${ov.error ? esc(ov.error) : ov.loaded ? `Loaded${ov.compute_ms != null ? ` · ${Math.round(ov.compute_ms)} ms a forecast` : ''}${ov.forecasts ? ` · ${ov.forecasts.toLocaleString()} made` : ''}` : 'Loads the first time it’s switched on'}</div></div>
                <span class="badge ${v.settings[k] ? 'ok' : ''}">${v.settings[k] ? 'On' : 'Off'}</span></div>`;
          }).join('')}</div>
          <p class="small muted" style="margin-top:10px">Switch each one on from the <a href="#/heart">Heart</a> page, by its chart. They follow the first person’s strap, look ${esc(v.settings.horizon_s)} s ahead
            from ${esc(v.settings.samples)} samples, and are a research demo, not a medical device.</p>`;
    }
    if (store.app.forecast) { paintForecasts(); fcTimer = setInterval(paintForecasts, 3000); }

    // ---- radios
    async function paintRadios() {
        const box = $('[data-radios]', el);
        try {
            const r = await api('/api/app/radios', { quiet: true });
            if (!r.adapters.length) { box.innerHTML = 'No Bluetooth radios found. Is Bluetooth turned on (bluetoothctl show)?'; return; }
            const label = (k) => dev(k).label + (several() && store.app.devices[kindOf(k)].personal ? ` (${store.personName(personOf(k))})` : '');
            const users = (hci) => Object.entries(r.devices).filter(([, x]) => x && x.hci === hci).map(([k]) => label(k));
            const defaultUsers = Object.entries(r.devices).filter(([, x]) => x && !x.hci).map(([k]) => label(k));
            const straps = Object.keys(r.devices).filter(k => kindOf(k) === 'heart').length;
            box.innerHTML = `<div class="list">${r.adapters.map(a => {
                const u = users(a.name);
                return `<div class="item">${icon('radio')}<div class="grow"><div class="title">${esc(a.name)} <span class="muted" style="font-weight:400">· ${esc([a.vendor, a.placement === 'external' ? 'USB dongle' : a.placement === 'internal' ? 'built in' : ''].filter(Boolean).join(', ') || 'Bluetooth radio')}</span></div>
                  <div class="desc">${a.error ? esc(a.error) : `${a.powered === false ? 'Powered off · ' : ''}${esc(a.address || '')}${a.usb_id ? ' · USB ' + esc(a.usb_id) : ''}`}</div></div>
                  <span class="small muted">${u.length ? 'Used by ' + esc(u.join(', ')) : ''}</span></div>`;
            }).join('')}</div>
            ${defaultUsers.length ? `<p class="small muted" style="margin-top:8px">${esc(defaultUsers.join(', '))} use${defaultUsers.length === 1 ? 's' : ''} the system’s default radio.</p>` : ''}
            ${(() => {                                   // two straps on one radio while there's another free
                const on = Object.entries(r.devices).filter(([k, x]) => kindOf(k) === 'heart' && x && x.hci).map(([k, x]) => [label(k), x.hci]);
                const clash = on.filter(([, h], i) => on.some(([, h2], j) => j !== i && h2 === h));
                return clash.length && r.adapters.length >= on.length ? `<div class="callout warn" style="margin-top:8px">${icon('warn')}<span>${esc(clash.map(c => c[0]).join(' and '))} share ${esc(clash[0][1])}. A streaming strap can starve the other’s searching: unless one is pinned there, Bio-dash moves the one still searching to a free radio within a few seconds.</span></div>` : '';
            })()}
            ${straps > r.adapters.length ? `<div class="callout warn" style="margin-top:8px">${icon('warn')}<span>${straps} heart straps but ${r.adapters.length} radio${r.adapters.length > 1 ? 's' : ''}: a streaming strap can take a whole radio, so add a USB Bluetooth dongle per extra strap.</span></div>` : ''}
            <p class="small muted" style="margin-top:8px">${several() ? 'With several people the radios are shared out: a connected device stays on its radio, and a strap still searching gets a radio no other strap uses. ' : ''}To pin a device to a radio, open the device and tap <b>Device</b>.</p>`;
        } catch (e) { box.textContent = 'Couldn’t list the radios.'; }
    }
    $('[data-radios-refresh]', el).addEventListener('click', paintRadios);
    paintRadios();

    // ---- updates
    let updTimer = null;
    async function paintUpdates(check = false) {
        const box = $('[data-updates]', el);
        let u;
        try { u = await (await api(`/api/system/update${check ? '?check=1' : ''}`, { quiet: true, raw: true })).json(); }   // its "error" is news to show, not a failure
        catch (e) { box.textContent = 'Couldn’t check for updates.'; return; }
        const s = u.status || {};
        const running = s.state === 'running';
        const notes = (list) => list && list.length ? `<ul style="margin:6px 0 0;padding-left:18px">${list.map(n => `<li>${esc(n)}</li>`).join('')}</ul>` : '';
        let html = `<div class="row" style="margin-bottom:8px"><b style="font-size:1rem;color:var(--text)">Version ${esc(u.version || store.app.version)}</b>
            ${u.checked_at ? `<span class="muted">checked ${ago(u.checked_at)}</span>` : ''}</div>`;
        if (u.error) html += `<div class="callout warn">${icon('warn')}<span>${esc(u.error)}</span></div>`;
        else if (!u.patch && !u.upgrade) html += `<div class="callout ok">${icon('check')}<span>You’re up to date.</span></div>`;
        if (u.patch) html += `<div class="callout" style="margin-top:8px;display:block"><b>${u.patch.count} fix${u.patch.count > 1 ? 'es' : ''} for this version</b>
            ${u.patch.commits.slice(0, 5).map(c => `<div style="margin-top:6px">${esc(c.subject)} <span class="faint">${esc(c.date)}</span>${notes(c.notes)}</div>`).join('')}
            <div style="margin-top:10px"><button class="btn sm primary" data-apply="patch" ${u.allowed && !u.blockers.length && !running ? '' : 'disabled'}>Apply fixes</button></div></div>`;
        if (u.upgrade) html += `<div class="callout" style="margin-top:8px;display:block"><b>Version ${esc(u.upgrade.version)} is available</b>
            <div class="small" style="white-space:pre-wrap;max-height:180px;overflow:auto;margin-top:6px">${esc((u.upgrade.notes || '').slice(0, 1800))}</div>
            <div style="margin-top:10px"><button class="btn sm primary" data-apply="upgrade" ${u.allowed && !u.blockers.length && !running ? '' : 'disabled'}>Upgrade to ${esc(u.upgrade.version)}</button>
            <span class="tiny muted">Your settings come along; the old version stays on disk.</span></div></div>`;
        for (const b of u.blockers || []) html += `<div class="callout warn" style="margin-top:8px">${icon('warn')}<span>${esc(b)}</span></div>`;
        if ((u.patch || u.upgrade) && !u.allowed) html += `<p class="small muted" style="margin-top:8px">Updates can be started ${esc(u.reason || 'from this computer only')}.</p>`;
        if (s.state && s.state !== 'idle') html += `<div class="callout ${s.state === 'error' ? 'bad' : s.state === 'done' ? 'ok' : ''}" style="margin-top:8px;display:block">
            <b>${s.state === 'running' ? 'Updating…' : s.state === 'done' ? 'Last update finished' : 'Last update failed'}</b> <span class="muted">${esc(s.stage || '')}</span>
            ${s.log && s.log.length ? `<div class="log" style="margin-top:8px;max-height:140px">${esc(s.log.slice(-12).join('\n'))}</div>` : ''}</div>`;
        box.innerHTML = html;
        box.querySelectorAll('[data-apply]').forEach(b => b.addEventListener('click', async () => {
            const what = b.dataset.apply;
            if (!await confirm({ title: what === 'patch' ? 'Apply the fixes?' : `Upgrade to ${u.upgrade.version}?`, text: 'Bio-dash restarts when it’s done (about a minute). Recording pauses for that time.', ok: what === 'patch' ? 'Apply' : 'Upgrade' })) return;
            await busy(b, () => api('/api/system/update', { body: { action: what }, headers: POWER }));
            toast('Updating… this page reloads when Bio-dash is back', 'ok');
            paintUpdates();
        }));
        clearTimeout(updTimer);
        if (running) updTimer = setTimeout(() => paintUpdates(), 2500);
    }
    $('[data-check]', el).addEventListener('click', (e) => busy(e.currentTarget, () => paintUpdates(true)));
    paintUpdates();

    // ---- this computer
    (async () => {
        const box = $('[data-system]', el), a = store.app;
        let p = {};
        try { p = await api('/api/system/power', { quiet: true }); } catch (e) { /* offline */ }
        box.innerHTML = `
          <dl class="kv"><dt>Name</dt><dd>${esc(a.host)}</dd><dt>Recordings</dt><dd class="mono">${esc(a.data_root)}</dd>
            <dt>Open from a phone</dt><dd class="mono">http://${esc(a.host)}.local:${esc(location.port || '80')}</dd></dl>
          ${p.allowed ? `<div class="row"><button class="btn" data-power="reboot">${icon('refresh')}Restart computer</button><button class="btn danger" data-power="shutdown">${icon('power')}Shut down</button>
             ${p.dry_run ? '<span class="badge warn">test mode</span>' : ''}</div>`
            : `<p class="small muted">Restart and shut down: ${esc(p.reason || 'not available')}.</p>${p.fix ? `<p class="small muted">To allow it without a password, run once on this computer:</p><div class="log">${esc(p.fix)}</div>` : ''}`}`;
        let r = {};
        try { r = await api('/api/system/restart', { quiet: true }); } catch (e) { /* offline */ }
        if (r.allowed) {
            box.insertAdjacentHTML('beforeend', `
              <div style="margin-top:16px"><div class="row"><button class="btn" data-restart>${icon('refresh')}Restart Bio-dash</button>
                <label class="switch small"><input type="checkbox" data-restart-bt checked><span class="text">Reset Bluetooth too</span></label></div>
              <p class="small muted" style="margin-top:6px">Stops every Bio-dash process on this computer (this version and any older one still running, service or not) and its Bluetooth workers, resets the radios, then starts again. Recording pauses for about half a minute.</p>
              <div data-restart-out></div></div>`);
            $('[data-restart]', box).addEventListener('click', async (e) => {
                const btn = e.currentTarget, bt = $('[data-restart-bt]', box).checked;
                if (!await confirm({ title: 'Restart Bio-dash?', text: bt ? 'Everything stops, the Bluetooth radios are reset (headphones or a keyboard on Bluetooth drop for a few seconds), then it all starts again. This page reconnects by itself.' : 'Everything stops and starts again. This page reconnects by itself.', ok: 'Restart' })) return;
                let out;
                try { out = await busy(btn, () => api('/api/system/restart', { body: { bluetooth: bt }, headers: POWER })); } catch (err) { return; }
                const t0 = Date.now(), show = $('[data-restart-out]', box);
                toast('Restarting… this page reconnects when Bio-dash is back', 'ok');
                const poll = async () => {
                    let st = null;
                    try { const res = await fetch('/api/system/restart', { cache: 'no-store' }); st = (await res.json()).status; } catch (err) { /* stopped for now */ }
                    if (st && st.started >= out.started - 2) {
                        show.innerHTML = `<div class="log" style="margin-top:8px">${esc(st.log.slice(-6).join('\n'))}</div>`;
                        if (st.state !== 'running') {
                            if (!st.problems && st.state === 'done') { setTimeout(() => location.reload(), 2000); return; }
                            show.insertAdjacentHTML('afterbegin', `<p class="small ${st.problems ? 'warn' : ''}" style="margin-top:8px">${esc(st.stage)}</p>`
                                + (st.bt_fix ? `<p class="small muted">To let it restart the bluetooth service without a password, run once on this computer:</p><div class="log">${esc(st.bt_fix)}</div>` : ''));
                            return;
                        }
                    }
                    if (Date.now() - t0 < 180000) setTimeout(poll, 1500);
                    else show.innerHTML = '<p class="small warn">No word from the restart after 3 minutes; is Bio-dash still running?</p>';
                };
                setTimeout(poll, 1500);
            });
        }
        box.querySelectorAll('[data-power]').forEach(b => b.addEventListener('click', async () => {
            const act = b.dataset.power;
            if (!await confirm({ title: act === 'reboot' ? 'Restart this computer?' : 'Shut this computer down?', text: act === 'reboot' ? 'Recording stops for a minute or two, then carries on by itself.' : 'Recording stops until it’s switched on again.', ok: act === 'reboot' ? 'Restart' : 'Shut down', danger: act !== 'reboot' }) ) return;
            await busy(b, () => api('/api/system/power', { body: { action: act }, headers: POWER }));
            toast(act === 'reboot' ? 'Restarting…' : 'Shutting down…', 'ok');
        }));
    })();

    // ---- about
    $('[data-about]', el).innerHTML = `<p>Bio-dash ${esc(store.app.version)}${store.app.demo ? ' <span class="badge warn">Demo: made-up readings</span>' : ''} — heart, oxygen, brain, air and water, recorded on your own computer.</p>
      <p class="muted" style="margin-top:6px">Recordings are plain CSV files you own. Nothing is sent anywhere, except the weather lookup for the water goal (Open-Meteo) when you turn it on.</p>
      <p style="margin-top:10px"><a href="https://github.com/Meapy011/Bio-dash" target="_blank" rel="noopener">Project page ${icon('external').replace('<svg', '<svg style="width:14px;height:14px;display:inline;vertical-align:-2px"')}</a></p>`;

    return () => { offStatus(); clearTimeout(updTimer); clearInterval(fcTimer); };
}
