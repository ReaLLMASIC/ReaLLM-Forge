// Small shared helpers: icons, formatting, API calls, toasts, sheets and dialogs.

// ---------------------------------------------------------------- icons (24px, stroke)
const P = {
    home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5"/>',
    heart: '<path d="M12 20s-7.5-4.6-9.2-9.4C1.6 7.2 3.8 4 7.2 4c2 0 3.6 1.1 4.8 2.8C13.2 5.1 14.8 4 16.8 4c3.4 0 5.6 3.2 4.4 6.6C19.5 15.4 12 20 12 20z"/><path d="M3.5 12h4l1.5-3 2.5 6 2-4h7"/>',
    oxygen: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><path d="M12 3.5v2M12 18.5v2"/>',
    air: '<path d="M3 8h11a3 3 0 1 0-3-3"/><path d="M3 12h16a3 3 0 1 1-3 3"/><path d="M3 16h7"/>',
    water: '<path d="M12 3s6 6.4 6 11a6 6 0 0 1-12 0c0-4.6 6-11 6-11z"/><path d="M9 14.5a3 3 0 0 0 3 3"/>',
    analysis: '<path d="M4 20V4"/><path d="M4 20h16"/><path d="m7 15 4-5 3 3 5-7"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    bluetooth: '<path d="m7 7 10 10-5 4V3l5 4L7 17"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/>',
    warn: '<path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17v.5"/>',
    power: '<path d="M12 3v9"/><path d="M6.3 7.2a8 8 0 1 0 11.4 0"/>',
    refresh: '<path d="M20 11a8 8 0 0 0-14.3-4.9L4 8"/><path d="M4 4v4h4"/><path d="M4 13a8 8 0 0 0 14.3 4.9L20 16"/><path d="M20 20v-4h-4"/>',
    download: '<path d="M12 4v11"/><path d="m7 10 5 5 5-5"/><path d="M5 20h14"/>',
    chevron: '<path d="m9 6 6 6-6 6"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    battery: '<rect x="2.5" y="7" width="17" height="10" rx="2"/><path d="M22 10.5v3"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    external: '<path d="M14 4h6v6"/><path d="M20 4 11 13"/><path d="M19 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h5"/>',
    sound: '<path d="M11 5 6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7M18.5 5.5a9 9 0 0 1 0 13"/>',
    mute: '<path d="M11 5 6 9H3v6h3l5 4V5z"/><path d="m16 9 5 6M21 9l-5 6"/>',
    sparkle: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/>',
    radio: '<rect x="3" y="8" width="18" height="12" rx="2"/><path d="M7 8 16 3"/><circle cx="15.5" cy="14" r="2.5"/><path d="M6.5 12.5h3M6.5 15.5h3"/>',
    wifi: '<path d="M2 8.8a15 15 0 0 1 20 0"/><path d="M5 12.5a10.5 10.5 0 0 1 14 0"/><path d="M8.5 16a5.5 5.5 0 0 1 7 0"/><path d="M12 19.5v.01"/>',
    calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
    moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>',
    folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    terminal: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3M13 15h4"/>',
    bulb: '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3.5 10.9c.6.4 1 1.1 1 1.8V16h5v-.3c0-.7.4-1.4 1-1.8A6 6 0 0 0 12 3z"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    brain: '<path d="M12 5.2a3 3 0 0 0-5.5-1.4A3.1 3.1 0 0 0 3.8 8a3.2 3.2 0 0 0-.4 5.4A3.6 3.6 0 0 0 7 19.1a3 3 0 0 0 5 1.2z"/><path d="M12 5.2a3 3 0 0 1 5.5-1.4A3.1 3.1 0 0 1 20.2 8a3.2 3.2 0 0 1 .4 5.4 3.6 3.6 0 0 1-3.6 5.7 3 3 0 0 1-5 1.2z"/><path d="M12 5.2v15.1"/>',
    people: '<circle cx="9" cy="8" r="3.2"/><path d="M3 20a6 6 0 0 1 12 0"/><path d="M16 5.2a3.2 3.2 0 0 1 0 5.6"/><path d="M17.5 14.3A6 6 0 0 1 21 20"/>',
    person: '<circle cx="12" cy="8" r="3.5"/><path d="M5 20a7 7 0 0 1 14 0"/>',
    edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>',
    trash: '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6 7l1 13h10l1-13"/>',
    eject: '<path d="M5 15h14L12 6z"/><path d="M5 19h14"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    flask: '<path d="M9 3h6"/><path d="M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3"/><path d="M7.5 15h9"/>',
};
export const icon = (name, cls = '') =>
    `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name] || ''}</svg>`;

// ---------------------------------------------------------------- the devices
export const DEVICES = {
    heart: {
        label: 'Heart', product: 'Polar H10 chest strap', icon: 'heart', c: 'var(--heart)',
        what: 'Heart rate, heart-rate variability and a live ECG.',
        find: 'Wet the strap’s two electrode pads, then put it on. It wakes up when it touches skin.',
        notFound: ['Wet the pads again — dry pads are the usual reason it isn’t seen.',
                   'Close the Polar Beat / Flow app on your phone: the strap talks to one thing at a time.',
                   'Still nothing? Take the pod off the strap for 10 seconds, then clip it back on.'],
    },
    oxygen: {
        label: 'Oxygen', product: 'Checkme O2 ring', icon: 'oxygen', c: 'var(--oxygen)',
        what: 'Blood oxygen (SpO₂) and pulse, all night if you like.',
        find: 'Put the ring on a finger. It turns on and starts advertising after a few seconds.',
        notFound: ['Charge it — a flat ring stays silent.',
                   'Close the ViHealth app on your phone (the ring only talks to one thing at a time).',
                   'Take the ring off and on again to wake it.'],
    },
    brain: {
        label: 'Brain', product: 'Muse S Athena headband', icon: 'brain', c: 'var(--brain)',
        what: 'EEG and brain-wave bands, forehead blood oxygenation (fNIRS), pulse and head movement.',
        find: 'Hold the headband’s button until its lights move, then put it on.',
        notFound: ['Close the Muse app on your phone: the headband talks to one thing at a time.',
                   'Charge it — the lights stay dark on a flat battery.',
                   'Hold the button for 5 seconds to restart it, then switch it on again.'],
    },
    air: {
        label: 'Air', product: 'Atmos air monitor', icon: 'air', c: 'var(--air)',
        what: 'CO₂, particles, temperature, humidity and gases in the room.',
        find: 'Power the Atmos on. Bluetooth is found automatically; a Wi-Fi model can be added by its address.',
        notFound: ['Check it’s powered and its light is on.', 'Move it nearer this computer for the first connection.',
                   'A Wi-Fi Atmos-Sphere? Use “Connect over Wi-Fi” below.'],
    },
    water: {
        label: 'Water', product: 'HidrateSpark bottle', icon: 'water', c: 'var(--water)',
        what: 'Every sip, today’s total against your goal, and gentle glow reminders.',
        find: 'Pick up the bottle or open the cap to wake it.',
        notFound: ['Force-quit the HidrateSpark app on your phone — the bottle allows one connection at a time.',
                   'Tip the bottle or open the cap: it only advertises for a while after it moves.',
                   'Check the puck is charged (it glows when you tap it).'],
    },
};
export const DEVICE_KEYS = Object.keys(DEVICES);
// A worn device exists once per person: "heart" is the first person's, "heart-p2" the second's (server.py).
export const kindOf = (key) => key.split('-')[0];
export const personOf = (key) => key.split('-')[1] || 'p1';
export const dev = (key) => DEVICES[kindOf(key)];
// People are told apart by colour and by line style (so it also works without colour)
export const PERSON_STYLE = {
    p1: { c: 'var(--p1)', dash: null, mark: 'solid' },
    p2: { c: 'var(--p2)', dash: [7, 4], mark: 'dashed' },
    p3: { c: 'var(--p3)', dash: [2, 3], mark: 'dotted' },
};

// ---------------------------------------------------------------- formatting
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export function num(v, dp = 0, dash = '--') {
    if (v === null || v === undefined || v === '' || Number.isNaN(+v)) return dash;
    return (+v).toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
export const pad = (n) => String(n).padStart(2, '0');
export const hm = (ms) => { const d = new Date(ms); return `${pad(d.getHours())}:${pad(d.getMinutes())}`; };
export const hms = (ms) => { const d = new Date(ms); return `${hm(ms)}:${pad(d.getSeconds())}`; };
export function ago(tsSeconds) {
    if (!tsSeconds) return '—';
    const s = Math.max(0, Date.now() / 1000 - tsSeconds);
    if (s < 45) return 'just now';
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) { const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return m ? `${h} h ${m} min ago` : `${h} h ago`; }
    return `${Math.round(s / 86400)} days ago`;
}
export function duration(minutes) {
    if (minutes == null) return '--';
    if (minutes < 60) return `${Math.round(minutes)} min`;
    const h = Math.floor(minutes / 60), m = Math.round(minutes % 60);
    return m ? `${h} h ${m} min` : `${h} h`;
}
export const isoDay = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
export function dayLabel(iso, opts = { weekday: 'short', month: 'short', day: 'numeric' }) {
    const [y, m, d] = iso.split('-').map(Number);
    return new Date(y, m - 1, d).toLocaleDateString(undefined, opts);
}
export function rssiBars(rssi) {
    const s = rssi == null ? 0 : rssi > -60 ? 4 : rssi > -70 ? 3 : rssi > -80 ? 2 : 1;
    return `<span class="bars s${s}" title="${rssi == null ? 'signal unknown' : rssi + ' dBm'}"><i></i><i></i><i></i><i></i></span>`;
}
export const cssVar = (name, el = document.documentElement) => getComputedStyle(el).getPropertyValue(name).trim();

// ---------------------------------------------------------------- device state in plain words
export function stateOf(st) {
    // -> {text, tone: ok|wait|bad|off, live: bool}
    if (!st || st.enabled === false) return { text: 'Off', tone: 'off', live: false };
    const w = (st.worker || {}).state;
    if (w === 'restarting') return { text: 'Restarting…', tone: 'bad', live: false };
    if (w === 'starting' || w === 'off' && st.enabled) return { text: 'Starting…', tone: 'wait', live: false };
    if (st.state === 'streaming') {
        if (st.detail === 'Calibrating') return { text: 'Settling…', tone: 'wait', live: true };
        return { text: 'Recording', tone: 'ok', live: true };
    }
    if (st.held && st.state !== 'connecting') return { text: 'Disconnected', tone: 'off', live: false };
    if (st.state === 'connecting') {
        const tries = st.attempt > 1 ? ` (try ${st.attempt}${st.max_attempts ? ' of ' + st.max_attempts : ''})` : '';
        return { text: `Connecting${tries}`, tone: 'wait', live: false };
    }
    return { text: 'Looking…', tone: 'wait', live: false };
}
export const pill = (st) => { const s = stateOf(st); return `<span class="pill ${s.tone === 'off' ? '' : s.tone}"><span class="dot ${s.tone}"></span>${esc(s.text)}</span>`; };

// ---------------------------------------------------------------- API
export class ApiError extends Error {}
export async function api(path, { method, body, headers = {}, quiet = false, raw = false } = {}) {
    const opts = { method: method || (body !== undefined ? 'POST' : 'GET'), headers: { ...headers } };
    if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
    let res;
    try {
        res = await fetch(path, opts);
    } catch (e) {
        if (!quiet) toast('Can’t reach Bio-dash — is it still running?', 'bad');
        throw new ApiError('offline');
    }
    if (raw) return res;
    let data = null;
    try { data = await res.json(); } catch (e) { /* not JSON */ }
    const err = !res.ok ? (data && (data.error || data.detail)) || `${res.status} ${res.statusText}` : data && data.error;
    if (err) {
        if (!quiet) toast(typeof err === 'string' ? err : JSON.stringify(err), 'bad');
        throw new ApiError(err);
    }
    return data;
}
export const dapi = (dev, path, opts) => api(`/api/d/${dev}${path}`, opts);

// ---------------------------------------------------------------- toasts
export function toast(text, tone = '') {
    const box = document.getElementById('toasts');
    const t = document.createElement('div');
    t.className = `toast ${tone}`;
    t.textContent = text;
    box.appendChild(t);
    setTimeout(() => { t.style.transition = 'opacity .3s'; t.style.opacity = '0'; setTimeout(() => t.remove(), 320); }, tone === 'bad' ? 5200 : 2600);
}

// ---------------------------------------------------------------- sheets & dialogs
let openLayers = 0;
function layer(html, cls, onClose) {
    const host = document.getElementById('layer');
    const scrim = document.createElement('div');
    scrim.className = 'scrim';
    const box = document.createElement('div');
    box.className = cls;
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-modal', 'true');
    box.innerHTML = html;
    host.append(scrim, box);
    openLayers++;
    document.body.style.overflow = 'hidden';
    const prevFocus = document.activeElement;
    let closed = false;
    const close = () => {
        if (closed) return;
        closed = true;
        scrim.remove(); box.remove();
        document.removeEventListener('keydown', onKey);
        window.removeEventListener('hashchange', close);
        if (--openLayers === 0) document.body.style.overflow = '';
        onClose && onClose();
        prevFocus && prevFocus.focus && prevFocus.focus();
    };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    scrim.addEventListener('click', close);
    document.addEventListener('keydown', onKey);
    window.addEventListener('hashchange', close);          // going to another page (a link, Back) closes it
    box.addEventListener('click', (e) => { if (e.target.closest('[data-close]')) close(); });
    setTimeout(() => (box.querySelector('[autofocus]') || box.querySelector('.sheet-head .icon-btn') || box).focus?.(), 30);
    return { el: box, close };
}
export function sheet({ title, body = '', foot = '', onClose }) {
    const s = layer(`<div class="sheet-head"><h2>${title}</h2><button class="icon-btn" data-close aria-label="Close">${icon('x')}</button></div>
        <div class="sheet-body">${body}</div>${foot ? `<div class="sheet-foot">${foot}</div>` : ''}`, 'sheet', onClose);
    s.body = s.el.querySelector('.sheet-body');
    return s;
}
export function confirm({ title, text = '', ok = 'OK', cancel = 'Cancel', danger = false }) {
    return new Promise((resolve) => {
        let answer = false;
        const d = layer(`<h2>${title}</h2><p>${text}</p><div class="row"><button class="btn ghost" data-close>${cancel}</button>
            <button class="btn ${danger ? 'danger solid' : 'primary'}" data-ok autofocus>${ok}</button></div>`, 'dialog', () => resolve(answer));
        d.el.querySelector('[data-ok]').addEventListener('click', () => { answer = true; d.close(); });
    });
}

// a one-line question: resolves to the text, or null when cancelled
export function ask({ title, text = '', value = '', placeholder = '', ok = 'Save', cancel = 'Cancel', max = 40 }) {
    return new Promise((resolve) => {
        let answer = null;
        const d = layer(`<h2>${title}</h2>${text ? `<p>${text}</p>` : ''}
            <input class="input" data-ask maxlength="${max}" placeholder="${esc(placeholder)}" value="${esc(value)}" style="margin:10px 0 14px;width:100%" autofocus>
            <div class="row"><button class="btn ghost" data-close>${cancel}</button><button class="btn primary" data-ok>${ok}</button></div>`,
            'dialog', () => resolve(answer));
        const input = d.el.querySelector('[data-ask]');
        const done = () => { answer = input.value; d.close(); };
        d.el.querySelector('[data-ok]').addEventListener('click', done);
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') done(); });
        setTimeout(() => { input.focus(); input.select(); }, 40);
    });
}

// a button that shows it's busy while its promise runs
export async function busy(btn, fn) {
    btn.classList.add('busy');
    btn.disabled = true;
    try { return await fn(); } finally { btn.classList.remove('busy'); btn.disabled = false; }
}

// tiny helper for seg controls: <div class="seg" data-value="15"><button data-v="15">...</button></div>
export function seg(el, value, onChange) {
    const paint = (v) => el.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === String(v))));
    paint(value);
    el.addEventListener('click', (e) => {
        const b = e.target.closest('button[data-v]');
        if (!b) return;
        paint(b.dataset.v);
        onChange(b.dataset.v);
    });
    return paint;
}

// remembered per-viewer choices (window sizes, open sections): best effort
export const prefs = {
    get(k, d) { try { const v = localStorage.getItem('biodash.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('biodash.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
};
