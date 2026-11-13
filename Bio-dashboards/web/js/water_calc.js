// Pace through the day for the water goal. Pure functions (no page access), so they can be
// checked on their own:  node hydro_dashboard/test_water_calc.js
(function (root) {
    const minutesOf = (hhmm) => {
        const m = /^(\d{1,2}):(\d{2})$/.exec(hhmm || '');
        return m ? +m[1] * 60 + +m[2] : null;
    };

    /** Where today's intake stands against the goal, spread evenly over the drinking window
     *  (the reminder hours). now: a Date. */
    function pace(todayMl, goalMl, now, from = '08:00', to = '22:00') {
        const left = Math.max(0, goalMl - todayMl);
        if (!(goalMl > 0)) return null;
        if (left === 0) return { done: true, left: 0 };
        let a = minutesOf(from), b = minutesOf(to);
        if (a == null || b == null || b <= a) { a = 8 * 60; b = 22 * 60; }
        const t = now.getHours() * 60 + now.getMinutes() + now.getSeconds() / 60;
        const frac = Math.max(0, Math.min(1, (t - a) / (b - a)));
        const expected = goalMl * frac;
        const behind = Math.max(0, Math.round(expected - todayMl));
        const hoursLeft = Math.max(0, (b - Math.max(t, a)) / 60);
        return {
            done: false, left, expected: Math.round(expected),
            // within 5 % of the goal counts as on pace: one sip shouldn't flip the message
            behind: behind > goalMl * 0.05 ? behind : 0,
            ahead: todayMl > expected ? Math.round(todayMl - expected) : 0,
            perHour: hoursLeft > 0.25 ? Math.round(left / hoursLeft) : null,
            before: frac === 0, after: frac === 1,
        };
    }

    const api = { pace, minutesOf };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.WaterCalc = api;
})(typeof window !== 'undefined' ? window : this);
