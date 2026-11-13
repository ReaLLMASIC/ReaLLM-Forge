// h10.js against polar-python (frames decoded by the library the Python worker uses) and against
// polar_dashboard/stream_clock.py (the same frame times in, the same stamps out).
//   node tools/portable_polar_tests/test_h10.mjs        (finds portable_polar: see locate.py)
import { readFileSync } from 'node:fs';
import { pageUrl } from './locate.mjs';
const { parsePmdFrame, parseHeartRate, parseControlResponse, StreamClock, ClockAnchor, START_ECG, START_ACC } = await import(pageUrl() + 'h10.js');

const here = new URL('.', import.meta.url);
let ok = 0;
function check(name, cond, detail = '') {
    console.log((cond ? '  ok  ' : '  FAIL ') + name + (cond ? '' : `  ${detail}`));
    if (!cond) process.exit(1);
    ok++;
}
const dv = hex => { const b = Uint8Array.from(hex.match(/../g).map(h => parseInt(h, 16))); return new DataView(b.buffer); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const fx = JSON.parse(readFileSync(new URL('polar_python_frames.json', here)));
for (const f of fx.frames) {
    const r = parsePmdFrame(dv(f.hex), f.factor === null ? {} : { 2: f.factor });
    check(`${f.what}: ${f.samples.length} samples, as polar-python`, r && same(r.samples, f.samples),
          r && JSON.stringify(r.samples.slice(0, 3)) + ' vs ' + JSON.stringify(f.samples.slice(0, 3)));
    check(`${f.what}: sensor timestamp`, r && r.tsNs === BigInt(f.ts));
}
for (const h of fx.hr) {
    const r = parseHeartRate(dv(h.hex));
    check(`heart rate 0x${h.hex.slice(0, 2)}: ${h.hr} bpm, ${h.rr.length} R-R`,
          r.hr === h.hr && r.rr.length === h.rr.length && r.rr.every((v, i) => Math.abs(v - h.rr[i]) < 1e-9), JSON.stringify(r));
}
const c = parseControlResponse(dv(fx.control.hex));
check('control response: type, no error, the scale factor', c.type === 2 && c.error === 0 && Math.abs(c.factor - fx.control.factor) < 1e-7);
check('start requests as the worker sends them (ECG 130 Hz / 14 bit; ACC 200 Hz / 16 bit / 8 G)',
      same(START_ECG, [2, 0, 0, 1, 130, 0, 1, 1, 14, 0]) && same(START_ACC, [2, 2, 0, 1, 200, 0, 1, 1, 16, 0, 2, 1, 8, 0]));

const calls = JSON.parse(readFileSync(new URL('stream_clock_calls.json', here)));
const anchor = new ClockAnchor(), clocks = { ecg: new StreamClock(130, anchor), acc: new StreamClock(200, anchor) };
let mism = 0, first = null;
for (const k of calls) {
    const out = clocks[k.s].stamps(BigInt(k.ts), k.n, k.now);
    if (!same(out, k.out)) { mism++; first ??= `${k.s} ${JSON.stringify(out.slice(0, 2))} vs ${JSON.stringify(k.out.slice(0, 2))}`; }
}
check(`timestamps: ${calls.length} frames (jitter, a stall, a late burst, the strap's clock going back) as stream_clock.py`, mism === 0, `${mism} differ; first ${first}`);
check('…including the restarts', anchor.reanchors === 2, anchor.reanchors);
console.log(`${ok} checks passed`);
