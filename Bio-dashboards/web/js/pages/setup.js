// First-run setup (and "Run setup again"): pick your devices, connect each one, done.
import { DEVICES, DEVICE_KEYS, icon, esc, api, stateOf, $ } from '../ui.js';
import { finder } from '../device.js';

export async function mount(el, { store }) {
    const wiz = document.createElement('div');
    wiz.className = 'wizard';
    document.body.appendChild(wiz);
    let chosen = DEVICE_KEYS.filter(k => store.enabled(k));
    let step = 0, stopFinder = null, offStatus = null;
    const first = !store.app.setup_done;

    const steps = () => ['welcome', 'pick', ...chosen.map(k => 'dev:' + k), 'done'];
    const bar = () => `<div class="steps">${steps().map((_, i) => `<i class="${i <= step ? 'on' : ''}"></i>`).join('')}</div>`;

    function render() {
        stopFinder && stopFinder(); stopFinder = null;
        offStatus && offStatus(); offStatus = null;
        const name = steps()[step];
        let html = '';
        if (name === 'welcome') {
            html = `${bar()}
              <div style="display:flex;gap:14px;align-items:center"><svg viewBox="0 0 32 32" style="width:52px;height:52px"><rect width="32" height="32" rx="9" fill="var(--brand)"/><path d="M5 17h5l2.5-6 4 12 3-9 2 3H27" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
                <h1>${first ? 'Welcome to Bio-dash' : 'Set up devices'}</h1></div>
              <p class="lead">Bio-dash connects to your health and home sensors over Bluetooth, shows them live, and records everything to plain files on this computer.</p>
              <div class="list card">
                <div class="item">${icon('bluetooth')}<div class="grow"><div class="title">Finds your devices</div><div class="desc">and reconnects by itself when they come back in range</div></div></div>
                <div class="item">${icon('folder')}<div class="grow"><div class="title">Keeps the recordings here</div><div class="desc mono">${esc(store.app.data_root)}</div></div></div>
                <div class="item">${icon('analysis')}<div class="grow"><div class="title">Looks back</div><div class="desc">days, nights and trends in Analysis</div></div></div>
              </div>
              <div class="nav-row">${first ? '<span></span>' : '<a class="btn ghost" href="#/settings">Cancel</a>'}<button class="btn primary lg" data-next>Get started</button></div>`;
        } else if (name === 'pick') {
            html = `${bar()}<h1>Which devices do you have?</h1><p class="lead">Pick all that apply. You can change this any time in Settings.</p>
              <div class="stack">${DEVICE_KEYS.map(k => { const d = DEVICES[k]; return `<button class="pick" data-pick="${k}" aria-pressed="${chosen.includes(k)}" style="--c:${d.c}">
                <span class="glyph">${icon(d.icon)}</span><span class="grow"><b>${esc(d.product)}</b><span>${esc(d.what)}</span></span><span class="check">${icon('check')}</span></button>`; }).join('')}</div>
              <div class="nav-row"><button class="btn ghost" data-back>Back</button><button class="btn primary lg" data-next>${chosen.length ? 'Next' : 'Skip — none yet'}</button></div>`;
        } else if (name.startsWith('dev:')) {
            const k = name.slice(4), d = DEVICES[k], i = chosen.indexOf(k);
            html = `${bar()}<div class="tiny muted" style="font-weight:700;letter-spacing:.06em;text-transform:uppercase">Device ${i + 1} of ${chosen.length}</div>
              <h1 style="display:flex;gap:12px;align-items:center"><span class="glyph" style="width:44px;height:44px;border-radius:12px;display:grid;place-items:center;background:color-mix(in srgb,${d.c} 15%,transparent);color:${d.c}">${icon(d.icon)}</span>Connect your ${esc(d.product)}</h1>
              <div class="callout">${icon('bulb')}<span>${esc(d.find)}</span></div>
              <div data-state></div>
              <div class="card" data-finder-card><div data-finder></div></div>
              <div class="nav-row"><button class="btn ghost" data-back>Back</button><button class="btn primary lg" data-next>Skip for now</button></div>`;
        } else {
            html = `${bar()}<h1>You’re all set</h1><p class="lead">${chosen.length ? 'Bio-dash keeps looking for anything that isn’t connected yet, and records whenever it is.' : 'Add devices any time from Settings.'}</p>
              ${chosen.length ? `<div class="card"><div class="list">${chosen.map(k => `<div class="item" style="--c:${DEVICES[k].c}">${icon(DEVICES[k].icon)}<div class="grow"><div class="title">${DEVICES[k].label}</div><div class="desc">${esc(DEVICES[k].product)}</div></div><span data-done-pill="${k}"></span></div>`).join('')}</div></div>` : ''}
              <div class="callout">${icon('info')}<span>Open Bio-dash from a phone or tablet on the same network at <b>http://${esc(store.app.host)}.local:${esc(location.port || '80')}</b> — and add it to your home screen.</span></div>
              <div class="nav-row"><button class="btn ghost" data-back>Back</button><button class="btn primary lg" data-finish>Go to Home</button></div>`;
        }
        wiz.innerHTML = `<div class="inner">${html}</div>`;
        wiz.scrollTop = 0;

        if (name.startsWith('dev:')) {
            const k = name.slice(4);
            stopFinder = finder($('[data-finder]', wiz), k);
            const paint = () => {
                const st = store.status[k], s = stateOf(st);
                const box = $('[data-state]', wiz);
                if (s.live) {
                    box.innerHTML = `<div class="callout ok" style="border-color:color-mix(in srgb,var(--ok) 40%,transparent)">${icon('check')}<span><b>Connected to ${esc((st && st.name) || DEVICES[k].product)}.</b> It’s recording now, and will reconnect by itself next time.</span></div>`;
                    $('[data-finder-card]', wiz).hidden = true;
                    $('[data-next]', wiz).textContent = 'Next';
                } else {
                    box.innerHTML = '';
                    $('[data-finder-card]', wiz).hidden = false;
                    $('[data-next]', wiz).textContent = 'Skip for now';
                }
            };
            paint();
            offStatus = store.on('status', (kk) => { if (kk === k) paint(); });
        }
        if (name === 'done') {
            const paint = () => chosen.forEach(k => { const p = wiz.querySelector(`[data-done-pill="${k}"]`); if (p) { const s = stateOf(store.status[k]); p.innerHTML = `<span class="pill ${s.tone}"><span class="dot ${s.tone}"></span>${s.live ? 'Recording' : 'Will connect when it’s near'}</span>`; } });
            paint();
            offStatus = store.on('status', paint);
        }
        $('[data-pick]', wiz) && wiz.querySelectorAll('[data-pick]').forEach(b => b.addEventListener('click', () => {
            const k = b.dataset.pick;
            chosen = chosen.includes(k) ? chosen.filter(x => x !== k) : DEVICE_KEYS.filter(x => x === k || chosen.includes(x));
            b.setAttribute('aria-pressed', String(chosen.includes(k)));
            $('[data-next]', wiz).textContent = chosen.length ? 'Next' : 'Skip — none yet';
            wiz.querySelector('.steps').outerHTML = bar();
        }));
        $('[data-back]', wiz)?.addEventListener('click', () => { step = Math.max(0, step - 1); render(); });
        $('[data-next]', wiz)?.addEventListener('click', async (e) => {
            if (name === 'pick') {
                // turn the chosen devices on now, so they're already being looked for on the next screen
                e.currentTarget.classList.add('busy');
                try { await api('/api/app/setup', { body: { devices: chosen, done: !first } }); await store.refreshApp(); } catch (err) { e.currentTarget.classList.remove('busy'); return; }
            }
            step++;
            render();
        });
        $('[data-finish]', wiz)?.addEventListener('click', async () => {
            await api('/api/app/setup', { body: { devices: chosen, done: true } });
            await store.refreshApp();
            location.hash = '#/';
        });
    }
    render();
    return () => { stopFinder && stopFinder(); offStatus && offStatus(); wiz.remove(); };
}
