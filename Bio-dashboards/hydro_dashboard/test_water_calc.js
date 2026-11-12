// Checks for the pace maths in web/js/water_calc.js (the goal itself: test_water.py) with values worked out by hand. Run: node test_water_calc.js
const W = require('../web/js/water_calc.js');
let ok = 0;
function check(name, got, want) {
    if (JSON.stringify(got) !== JSON.stringify(want)) throw new Error(`${name}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
    ok++;
}

// pace: goal 3000 over 08:00-22:00 (14 h). At 15:00 half the window has gone -> 1500 expected
const at = (h, m = 0) => new Date(2026, 9, 11, h, m);
let p = W.pace(1000, 3000, at(15), '08:00', '22:00');
check('expected', p.expected, 1500);
check('behind', p.behind, 500);
check('left', p.left, 2000);
check('per hour', p.perHour, 286);            // 2000 over 7 h
check('within 5 % is on pace', W.pace(1400, 3000, at(15)).behind, 0);
check('ahead', W.pace(1800, 3000, at(15)).ahead, 300);
check('before the window', [W.pace(0, 3000, at(6)).before, W.pace(0, 3000, at(6)).behind], [true, 0]);
check('after the window', W.pace(2000, 3000, at(23)).after, true);
check('done', W.pace(3100, 3000, at(15)).done, true);
check('bad window falls back to 08-22', W.pace(1000, 3000, at(15), '22:00', '08:00').expected, 1500);

console.log(`${ok} checks passed`);
