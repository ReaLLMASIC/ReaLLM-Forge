// Theme choice: Auto · Dark · Light · OLED black · High contrast · Warm night.
// Runs in <head> before the page draws. The choice is a cookie (shared by every tab and port on
// this computer's address); "auto" follows the device's light/dark setting. The look is the
// tokens in css/app.css; pages listen for the "biodash-theme" event to redraw their canvases.
(() => {
    const NAMES = { auto: 'Auto', dark: 'Dark', light: 'Light', oled: 'OLED black', contrast: 'High contrast', warm: 'Warm night' };
    const root = document.documentElement;
    const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: light)') : null;
    const read = () => {
        const m = document.cookie.match(/(?:^|;\s*)biodash_theme=([a-z]+)/);
        return m && NAMES[m[1]] ? m[1] : 'auto';
    };
    let choice = read();
    const resolve = (c) => c === 'auto' ? (mq && mq.matches ? 'light' : 'dark') : c;

    function apply() {
        const theme = resolve(choice);
        if (root.dataset.theme === theme && root.dataset.themeChoice === choice) return;
        root.dataset.theme = theme;
        root.dataset.themeChoice = choice;
        const meta = document.querySelector('meta[name="theme-color"]');
        if (meta) meta.content = theme === 'light' ? '#f3f5f9' : (theme === 'dark' ? '#0b1120' : '#000000');
        window.dispatchEvent(new CustomEvent('biodash-theme', { detail: { choice, theme } }));
    }
    function set(c) {
        if (!NAMES[c]) return;
        choice = c;
        document.cookie = `biodash_theme=${c}; path=/; max-age=31536000; SameSite=Lax`;
        apply();
    }
    root.dataset.theme = '';
    apply();
    if (mq) (mq.addEventListener ? mq.addEventListener('change', apply) : mq.addListener(apply));
    setInterval(() => { const c = read(); if (c !== choice) { choice = c; apply(); } }, 3000);   // changed in another tab
    window.BioTheme = { NAMES, set, choice: () => choice, theme: () => root.dataset.theme };
})();
