// The desktop app (portable_polar/desktop, Electron) with the stand-in strap: opens the page from app://,
// records to ~/Documents/Bio-dash (a scratch HOME), the strap picker, the remembered strap, the forecast.
// Needs: npm i playwright-core (here or globally), the app's node_modules (cd desktop && npm install),
// desktop/page staged with a model (desktop/build.sh stages it), and a display (xvfb-run -a node test_desktop.mjs).
import { _electron as electron } from 'playwright-core';
import { pageUrl } from './locate.mjs';
import { fileURLToPath } from 'node:url';
import { readFileSync, readdirSync, mkdtempSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
const APP = join(fileURLToPath(pageUrl()), 'desktop');
const home = mkdtempSync(join(tmpdir(), 'pp-home-'));
let ok = 0;
const check = (name, cond, detail = '') => { console.log((cond ? '  ok  ' : '  FAIL ') + name + (cond ? '' : '  ' + detail)); if (!cond) { process.exitCode = 1; } else ok++; };

const app = await electron.launch({ executablePath: join(APP, 'node_modules/electron/dist/electron'), args: [APP, '--no-sandbox'],
                                    env: { ...process.env, HOME: home, XDG_CONFIG_HOME: join(home, '.config') } });
const fake = readFileSync(new URL('fake_h10.js', import.meta.url), 'utf8');
await app.context().addInitScript(fake);
const win = await app.firstWindow();
const errors = [];
win.on('pageerror', e => errors.push(String(e)));
win.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
await win.waitForLoadState();
await win.reload();                                  // so the init script (the stand-in strap) is in place
await win.waitForTimeout(1500);
check('opens the page from app://', win.url() === 'app://page/index.html', win.url());
check('title', (await win.title()) === 'Portable Polar');
const where = await win.textContent('#where');
check(`records to ~/Documents/Bio-dash by default (${where})`, where === join(home, 'Documents', 'Bio-dash'));
check('desktop buttons: Open + Change folder', await win.isVisible('#open-folder') && (await win.textContent('#pick')).includes('Change folder'));
check('cross-origin isolated (several CPU threads for the forecast)', await win.evaluate(() => self.crossOriginIsolated));
check('forecast model built in', (await win.textContent('#fc-btn')) === '🔮 Forecast');

// the strap picker, fed as the main process feeds it
await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.send('bt-devices',
    [{ id: 'a', name: 'Polar H10 TEST0001', last: true }, { id: 'b', name: 'Some Speaker', last: false }]));
await win.waitForTimeout(300);
const straps = await win.$$eval('#picker-list .strap b', els => els.map(e => e.textContent));
check('picker lists Polar straps only, the last one marked', straps.join() === 'Polar H10 TEST0001' && await win.isVisible('#picker-list .strap.last'));
// a name for the strap, typed on its card: saved where Bio-dash keeps them
await win.click('#picker-list .strap .nick');
await win.fill('#picker-list .nick-input', 'Mine');
await win.press('#picker-list .nick-input', 'Enter');
await win.waitForTimeout(400);
const namesFile = join(home, '.config', 'bio-dash', 'device_names.json');
const names = existsSync(namesFile) ? JSON.parse(readFileSync(namesFile, 'utf8')) : {};
check("naming a strap on its card saves it in Bio-dash's names file", names['Polar H10 TEST0001'] === 'Mine', JSON.stringify(names));
check('…and the card shows it', (await win.textContent('#picker-list .strap .nick')).includes('Mine'));
check('…without connecting', await win.isVisible('#picker'));
await win.click('#picker-cancel');
check('cancel closes it', !(await win.isVisible('#picker')));

// connect the stand-in strap, record, forecast
await win.click('#connect');
await win.waitForTimeout(4000);
check('connects', (await win.textContent('#status')).includes('Polar H10 TEST0001'));
check('the status shows your name for it', (await win.textContent('#status-nick')).includes('Mine'));
await win.keyboard.press('d');
await win.waitForTimeout(2500);
const dbg = (await win.textContent('#debug-panel')) || '';
check('debug panel: radios from BlueZ / sysfs, versions from Electron', /Bluetooth radio/i.test(dbg) && /Electron · Chromium\s*[\d.]+ · [\d.]+/.test(dbg), dbg.slice(0, 300));
await win.keyboard.press('d');
await win.click('#fc-btn');
await win.waitForFunction(() => document.querySelector('#fc-status').textContent.includes(' ms on '), null, { timeout: 120000 });
const st = await win.textContent('#fc-status');
check(`forecast runs inside the app: "${st.slice(0, 70)}"`, st.includes('samples'));
await win.waitForTimeout(3000);
await win.click('#disconnect');
await win.waitForTimeout(2500);
const dir = join(home, 'Documents', 'Bio-dash', 'Polar H10', 'Polar H10 TEST0001');
const day = existsSync(dir) ? readdirSync(dir)[0] : null;
const files = day ? readdirSync(join(dir, day)).sort() : [];
check(`CSVs on disk: ${files.join(', ')}`, files.length === 3 && files.every(f => /_(acc|ecg|rr)\.csv$/.test(f)));
if (files.length) {
    const ecg = readFileSync(join(dir, day, files.find(f => f.endsWith('_ecg.csv'))), 'utf8').trim().split('\n');
    check(`ECG file: header + ${ecg.length - 1} rows`, ecg[0] === 'Timestamp_Epoch_ms,ECG_mV,HR_BPM' && ecg.length > 500);
}
const cfg = readdirSync(join(home, '.config')).find(d => existsSync(join(home, '.config', d, 'settings.json')));
const settings = JSON.parse(readFileSync(join(home, '.config', cfg, 'settings.json'), 'utf8'));
console.log('  (settings in ~/.config/' + cfg + ')');
check('remembers the strap', settings.lastStrap === 'Polar H10 TEST0001', JSON.stringify(settings));
check('no script errors', errors.filter(e => !/onnxruntime|VerifyEachNode/.test(e)).length === 0, errors.join(' | '));
await app.close();
console.log(`${ok} checks passed`);
