// Names you give devices ("Mine", "Boss's H10"). Saved by the server in ~/.config/bio-dash/device_names.json,
// keyed by the name the device advertises (stable), or its address when it has none. Only what you see
// changes: recordings keep their folder names.
import { esc, api, ask, toast } from './ui.js';

const listeners = new Set();
export const Nick = {
    names: {},
    key(name, address) {
        name = String(name || '').trim();
        if (name && name !== '(no name)' && name !== 'unknown device' && name.toUpperCase() !== String(address || '').toUpperCase()) return name;
        return String(address || '').trim().toUpperCase();
    },
    of(name, address) { return Nick.names[Nick.key(name, address)] || ''; },
    // the tag shown next to a device: your name for it, or "Name it"; tapping it edits the name
    tag(name, address) {
        const k = Nick.key(name, address);
        if (!k) return '';
        const n = Nick.names[k];
        return `<button type="button" class="nick${n ? '' : ' empty'}" data-nick-key="${esc(k)}" title="${n ? 'Your name for this device, tap to change' : 'Give this device a name, e.g. Mine or Boss’s'}">${n ? esc(n) : 'Name it'}</button>`;
    },
    async load() {
        try { Nick.names = (await api('/api/app/device-names', { quiet: true })).names || {}; } catch (e) { /* offline */ }
        paint();
    },
    async edit(key) {
        const v = await ask({ title: 'Name this device', text: `${esc(key)}<br><span class="small muted">e.g. Mine, Boss’s, Spare. Empty removes the name. Shown everywhere Bio-dash lists it.</span>`,
                              value: Nick.names[key] || '', placeholder: 'Mine' });
        if (v === null) return;
        try { Nick.names = (await api('/api/app/device-names', { body: { key, name: v } })).names || {}; } catch (e) { return; }
        toast(v.trim() ? `Named “${v.trim()}”` : 'Name removed', 'ok');
        paint();
    },
    on(fn) { listeners.add(fn); return () => listeners.delete(fn); },
};

function paint() {
    document.querySelectorAll('[data-nick-key]').forEach(el => {
        const n = Nick.names[el.dataset.nickKey];
        el.classList.toggle('empty', !n);
        el.textContent = n || 'Name it';
    });
    listeners.forEach(fn => fn());
}

// a tap on any tag edits the name and never reaches the Connect button / row underneath
document.addEventListener('click', (e) => {
    const t = e.target.closest('[data-nick-key]');
    if (!t) return;
    e.preventDefault();
    e.stopPropagation();
    Nick.edit(t.dataset.nickKey);
}, true);

Nick.load();
setInterval(Nick.load, 60000);
