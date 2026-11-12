// Keep this screen on while Bio-dash is showing (Settings → Look), for a tablet on a stand or a
// monitor across the room. A per-screen choice; the browser drops the lock when the tab is hidden,
// so it's taken again when the tab comes back.
import { prefs } from './ui.js';

let lock = null;
export const wakeSupported = () => 'wakeLock' in navigator;
export const wakeWanted = () => !!prefs.get('wake', false);

async function take() {
    if (!wakeWanted() || !wakeSupported() || lock || document.hidden) return;
    try {
        lock = await navigator.wakeLock.request('screen');
        lock.addEventListener('release', () => { lock = null; });
    } catch (e) { lock = null; }             // not allowed right now (battery saver, no focus): try again when it's shown again
}

export async function setWake(on) {
    prefs.set('wake', !!on);
    if (on) return take();
    if (lock) { try { await lock.release(); } catch (e) { /* already gone */ } lock = null; }
}

document.addEventListener('visibilitychange', take);
take();
