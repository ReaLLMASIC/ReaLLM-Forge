// Live state for the whole app, fed by ONE websocket to the server:
//   status  – each device's connection state (pushed when it changes)
//   latest  – the newest reading of every device (once a second, for Home and the nav)
//   live    – a device's full stream (ECG, every air row ...) while a page shows that device
import { api } from './ui.js';

const listeners = new Map();
export const store = {
    app: null,
    status: {},
    latest: {},
    online: false,
    subs: [],

    on(evt, fn) {
        if (!listeners.has(evt)) listeners.set(evt, new Set());
        listeners.get(evt).add(fn);
        return () => listeners.get(evt).delete(fn);
    },
    emit(evt, ...args) {
        for (const fn of listeners.get(evt) || []) {
            try { fn(...args); } catch (e) { console.error(e); }
        }
    },
    // which devices' full streams this page wants
    subscribe(keys) {
        this.subs = keys;
        send({ sub: keys });
    },
    async refreshApp() {
        this.app = await api('/api/app', { quiet: true });
        this.emit('app', this.app);
        return this.app;
    },
    enabled(key) { return !!(this.app && this.app.instances[key] && this.app.instances[key].enabled); },

    // ---- people (up to three can wear a strap and a ring at once)
    people() { return (this.app && this.app.people) || []; },
    several() { return this.people().length > 1; },
    personName(pid) {
        const i = this.people().findIndex(p => p.id === pid);
        return i < 0 ? '' : this.people()[i].name || `Person ${i + 1}`;
    },
    keyFor(kind, pid) { return !pid || pid === 'p1' ? kind : `${kind}-${pid}`; },
    // the people who have this kind of device turned on
    wearers(kind) { return this.people().filter(p => p.devices[kind] && p.devices[kind].enabled); },
    // is this kind of device on for anyone
    kindOn(kind) {
        const d = this.app && this.app.devices[kind];
        return !!d && (d.personal ? this.wearers(kind).length > 0 : d.enabled);
    },
};

let ws = null, backoff = 1000, lastMsg = 0;
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function setOnline(v) {
    if (store.online === v) return;
    store.online = v;
    store.emit('online', v);
}

export function connect() {
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    ws.onopen = () => { backoff = 1000; setOnline(true); send({ sub: store.subs }); };
    ws.onmessage = (ev) => {
        lastMsg = Date.now();
        let m;
        try { m = JSON.parse(ev.data); } catch (e) { return; }
        if (m.t === 'live') store.emit('live:' + m.d, m.m);
        else if (m.t === 'latest') { store.latest = m.v; store.emit('latest', m.v); }
        else if (m.t === 'status') {
            store.status[m.d] = m.s;
            if (store.app && !store.app.instances[m.d]) store.refreshApp().catch(() => {});      // a person was added elsewhere
            else if (store.app && store.app.instances[m.d].enabled !== m.s.enabled) store.refreshApp().catch(() => {});
            store.emit('status', m.d, m.s);
        } else if (m.t === 'hello') {
            Object.assign(store.status, m.status || {});
            store.latest = m.latest || {};
            for (const k in m.status || {}) store.emit('status', k, m.status[k]);
            if (store.app && m.version && store.app.version !== m.version) location.reload();   // updated underneath us
        }
    };
    ws.onclose = () => {
        setOnline(false);
        setTimeout(connect, backoff);
        backoff = Math.min(8000, backoff * 1.6);
    };
}
// a socket that silently died (phone slept, Wi-Fi changed): the server sends something every second
setInterval(() => { if (ws && ws.readyState === 1 && Date.now() - lastMsg > 6000) ws.close(); }, 3000);
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && ws && ws.readyState === 1 && Date.now() - lastMsg > 3000) ws.close();
});
