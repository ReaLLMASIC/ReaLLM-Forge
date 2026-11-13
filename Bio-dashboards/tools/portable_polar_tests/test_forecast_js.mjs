// forecast.js against the Python it ports: the 8 ms rows (analysis_dashboard/waves.py LiveAligner, ECG only)
// and the line drawn (forecast_service.rhythm_pick + summarize).   node tools/portable_polar_tests/test_forecast_js.mjs
import { readFileSync } from 'node:fs';
import { pageUrl } from './locate.mjs';
const { GridAligner, rhythmPick, summarize, roundHalfEven } = await import(pageUrl() + 'forecast.js');

let ok = 0;
function check(name, cond, detail = '') {
    console.log((cond ? '  ok  ' : '  FAIL ') + name + (cond ? '' : `  ${detail}`));
    if (!cond) process.exit(1);
    ok++;
}
const fx = JSON.parse(readFileSync(new URL('forecast_parity.json', import.meta.url)));
check('round() as Python (half to even)', [0.5, 1.5, 2.5, -0.5, -1.5, 2.4, -2.6].map(roundHalfEven).join() === '0,2,2,0,-2,2,-3');

const al = new GridAligner();
for (const [t, mv] of fx.samples) al.push(t, mv);
const rows = al.rows();
const diff = rows.findIndex((r, i) => !fx.rows[i] || r[0] !== fx.rows[i][0] || r[1] !== fx.rows[i][1] || r[2] !== fx.rows[i][2]);
check(`grid rows: ${fx.rows.length} rows, identical to waves.LiveAligner (gaps, time going back, half-µV values)`,
      rows.length === fx.rows.length && diff === -1, `${rows.length} rows; first difference at ${diff}: ${JSON.stringify(rows[diff])} vs ${JSON.stringify(fx.rows[diff])}`);
check('…the same pieces start in the same places', rows.filter(r => r[2]).length === 3);

fx.cases.forEach((c, i) => {
    const pick = rhythmPick(c.ring, c.samples);
    check(`rhythm pick, case ${i + 1}: sample ${pick} as forecast_service.rhythm_pick`, pick === c.pick, `${pick} vs ${c.pick}`);
    check(`summarize, case ${i + 1}: draws sample ${c.rep_index}`, summarize(c.samples, pick).repIndex === c.rep_index);
});
console.log(`${ok} checks passed`);
