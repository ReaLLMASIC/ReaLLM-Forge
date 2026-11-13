// Where the portable_polar page is (see locate.py): $PORTABLE_POLAR, else a portable_polar folder above here.
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
export function pageUrl() {
    const cands = [process.env.PORTABLE_POLAR];
    let d = dirname(fileURLToPath(import.meta.url));
    for (let i = 0; i < 6; i++) { d = dirname(d); cands.push(join(d, 'portable_polar')); }
    cands.push(join(homedir(), 'Forks', 'portable_polar'), join(homedir(), 'portable_polar'));
    const hit = cands.find(c => c && existsSync(join(c, 'h10.js')));
    if (!hit) { console.log("Can't find portable_polar: set PORTABLE_POLAR=/path/to/portable_polar"); process.exit(1); }
    return pathToFileURL(resolve(hit) + '/').href;
}
