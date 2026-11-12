// The app shell: navigation (side rail on wide screens, tab bar on phones), the router,
// and the first-run setup. Each page is a module in pages/ that exports mount(el, ctx) -> cleanup.
import { store, connect } from './store.js';
import { DEVICES, DEVICE_KEYS, icon, stateOf, esc, $ } from './ui.js';
import './wake.js';

const PAGES = {
    '': { title: 'Home', icon: 'home', module: 'overview' },
    heart: { title: 'Heart', icon: 'heart', module: 'heart', device: 'heart' },
    oxygen: { title: 'Oxygen', icon: 'oxygen', module: 'oxygen', device: 'oxygen' },
    brain: { title: 'Brain', icon: 'brain', module: 'brain', device: 'brain' },
    air: { title: 'Air', icon: 'air', module: 'air', device: 'air' },
    water: { title: 'Water', icon: 'water', module: 'water', device: 'water' },
    compare: { title: 'Compare', icon: 'people', module: 'compare' },
    analysis: { title: 'Analysis', icon: 'analysis', module: 'analysis' },
    settings: { title: 'Settings', icon: 'settings', module: 'settings' },
    setup: { title: 'Set up', icon: 'settings', module: 'setup' },
};

const main = $('#main');
let current = null;            // {name, cleanup}
let navSig = '';

function navItems() {
    const devs = DEVICE_KEYS.filter(k => store.kindOn(k));
    return [['', PAGES['']], ...(store.several() ? [['compare', PAGES.compare]] : []), ...devs.map(k => [k, PAGES[k]]), ['analysis', PAGES.analysis]];
}

// a kind's dot: its state, or with several wearers "recording" only when everyone's is
function kindState(kind) {
    const keys = DEVICES[kind] && store.app.devices[kind].personal ? store.wearers(kind).map(p => p.devices[kind].key) : [kind];
    const all = keys.map(k => stateOf(store.status[k] && { ...store.status[k], enabled: store.enabled(k) }));
    if (all.length < 2) return all[0] || stateOf(null);
    const live = all.filter(s => s.live).length;
    return live === all.length ? { tone: 'ok', text: 'Everyone’s is recording' }
        : { tone: all.some(s => s.tone === 'bad') ? 'bad' : 'wait', text: `${live} of ${all.length} recording` };
}
function dotFor(key) {
    const s = kindState(key);
    return `<span class="dot ${s.tone}" data-dot="${key}" title="${esc(s.text)}"></span>`;
}

function renderNav() {
    if (!store.app) return;
    const items = navItems();
    const sig = items.map(i => i[0]).join(',');
    if (sig !== navSig) {
        navSig = sig;
        const link = ([k, p]) => `<a href="#/${k}" data-page="${k}" ${p.device ? `style="--c:${DEVICES[k].c}"` : ''}>${icon(p.icon)}<span>${p.title}</span>${p.device ? dotFor(k) : ''}</a>`;
        const devs = items.filter(i => PAGES[i[0]].device);
        $('#nav').innerHTML = link(items[0]) + (store.several() ? link(['compare', PAGES.compare]) : '') + (devs.length ? '<div class="nav-sep"></div>' : '') + devs.map(link).join('')
            + '<div class="nav-sep"></div>' + link(['analysis', PAGES.analysis]) + link(['settings', PAGES.settings]);
        $('#tabbar').innerHTML = items.map(([k, p]) => `<a href="#/${k}" data-page="${k}" ${p.device ? `style="--c:${DEVICES[k].c}"` : ''}>${icon(p.icon)}<span>${p.title}</span>${p.device ? dotFor(k) : ''}</a>`).join('');
        $('#topbar-settings').innerHTML = icon('settings');
    }
    markCurrent();
    paintDots();
}
function paintDots() {
    if (!store.app) return;
    document.querySelectorAll('[data-dot]').forEach(el => {
        const s = kindState(el.dataset.dot);
        el.className = `dot ${s.tone}`;
        el.title = s.text;
    });
}
function markCurrent() {
    const name = current ? current.name : null;
    document.querySelectorAll('[data-page]').forEach(a => {
        if (a.dataset.page === name) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
}
function renderFoot() {
    const a = store.app;
    $('#rail-foot').innerHTML = a ? `${a.demo ? '<span class="badge warn">Demo</span> ' : ''}V${esc(a.version)} · ${esc(a.host)}` : '';
}

async function route() {
    const hash = location.hash.replace(/^#\/?/, '');
    const [name, ...rest] = hash.split('/');
    let pageName = PAGES[name] ? name : '';
    if (PAGES[pageName].device && !store.kindOn(pageName)) pageName = '';
    if (pageName === 'compare' && !store.several()) pageName = '';
    if (store.app && !store.app.setup_done && pageName !== 'setup') { location.replace('#/setup'); return; }
    const page = PAGES[pageName];
    // a worn device's page shows one person's: #/heart/p2 (the first wearer without it)
    let device = page.device, person = null;
    if (device && store.app.devices[device].personal) {
        const who = store.wearers(device);
        person = (who.find(p => p.id === rest[0]) || who[0]).id;
        device = store.keyFor(device, person);
    }
    const id = `${pageName}/${device || ''}`;
    if (current && current.id === id && pageName !== 'setup') { current.update && current.update(rest); return; }
    if (current) { try { current.cleanup && current.cleanup(); } catch (e) { console.error(e); } }
    current = { name: pageName, id };
    store.subscribe(device ? [device] : []);
    document.title = page.title === 'Home' ? 'Bio-dash' : `${page.title} · Bio-dash`;
    $('#topbar-title').textContent = page.title === 'Home' ? 'Bio-dash' : page.title;
    document.body.dataset.page = pageName || 'home';
    markCurrent();
    // each visit gets its own element: a page that finishes loading something after you've left
    // then paints into its own (detached) view, not into the page you're on now
    const view = document.createElement('div');
    view.className = 'view';
    main.replaceChildren(view);
    main.scrollTop = 0;
    window.scrollTo(0, 0);
    try {
        const mod = await import(`./pages/${page.module}.js`);
        if (current.id !== id) return;                  // navigated away while loading
        const out = await mod.mount(view, { store, params: rest, device, person });
        current.cleanup = typeof out === 'function' ? out : out && out.cleanup;
        current.update = out && out.update;
    } catch (e) {
        console.error(e);
        view.innerHTML = `<div class="empty-state">${icon('warn')}<h3>This page didn’t load</h3><p>${esc(e.message)}</p></div>`;
    }
}

store.on('status', () => { paintDots(); });
store.on('app', () => { renderNav(); renderFoot(); route(); });
store.on('online', (on) => { $('#conn').hidden = on; });
window.addEventListener('hashchange', route);

(async function start() {
    try {
        await store.refreshApp();
    } catch (e) {
        main.innerHTML = `<div class="empty-state">${icon('warn')}<h3>Can’t reach Bio-dash</h3><p>Is it running? This page retries every few seconds.</p></div>`;
        setTimeout(start, 3000);
        return;
    }
    renderNav();
    renderFoot();
    connect();
    route();
})();
