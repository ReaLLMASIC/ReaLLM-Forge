// Water (HidrateSpark bottle): today against the goal, every sip, the week, and the bottle's
// settings (goal and its calculator, size, calibration, glow and reminders).
import { icon, esc, num, dapi, toast, busy, confirm, ago, hm, dayLabel, $, $$ } from '../ui.js';
import { Plot } from '../plot.js';
import { pageHead, bindHead, connectPanel } from '../device.js';
import '../water_calc.js';

const BOTTLE_SIZES = [[500, 17], [592, 20], [621, 21], [710, 24], [887, 30], [946, 32]];
const GLOW_STYLES = [['pulse', 'Pulse'], ['spin', 'Spin'], ['flash', 'Flash'], ['solid', 'Solid'], ['rainbow', 'Rainbow']];
const OZ = 29.5735;

export async function mount(el, { store }) {
    let settings = null, summary = null, st = {};
    const oz = () => settings && settings.units === 'oz';
    const vol = (ml, dp) => ml == null ? '--' : oz() ? num(ml / OZ, dp ?? 1) : num(ml);
    const unit = () => (oz() ? 'fl oz' : 'mL');
    const fromUnit = (v) => (oz() ? Math.round(+v * OZ) : Math.round(+v));
    const temp = (c) => c == null ? '--' : oz() ? `${Math.round(c * 9 / 5 + 32)} °F` : `${Math.round(c)} °C`;

    el.innerHTML = `${pageHead('water')}
      <div data-connect></div>
      <div class="callout warn" data-calwait hidden style="margin-bottom:16px;align-items:center;flex-wrap:wrap">${icon('warn')}
        <div style="flex:1;min-width:230px"><b>Now fill the bottle and tap “It’s full now”</b><br>You told the bottle it’s empty<span data-calwait-since></span>.
          It takes empty and full as one calibration and <b>won’t count sips until it has both</b>. Fill it to the top, put the cap on, then tap the button.</div>
        <button class="btn primary" data-calwait-btn style="flex:none">It’s full now</button></div>
      <div class="card" style="--c:var(--water)">
        <div class="row" style="gap:28px;align-items:center">
          <div class="ring" data-ring><svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="52" fill="none" stroke-width="12"/>
            <circle class="fill" cx="60" cy="60" r="52" fill="none" stroke-width="12" stroke-linecap="round" stroke-dasharray="326.7" stroke-dashoffset="326.7"/></svg>
            <div class="center"><b data-today>--</b><span data-of>today</span></div></div>
          <div style="flex:1;min-width:220px">
            <h2 data-pace style="font-size:1.25rem">…</h2><p class="muted" data-pace2 style="margin-top:4px"></p>
            <div class="grid tiles" style="margin-top:16px;grid-template-columns:repeat(auto-fill,minmax(120px,1fr))" data-stats></div>
          </div>
        </div>
      </div>
      <div class="grid cols-2" style="margin-top:16px">
        <div class="card"><div class="card-head"><h2>Today</h2><span class="card-sub" data-today-sub></span></div>
          <div class="legend" style="margin-bottom:8px"><span><i style="background:var(--water)"></i>Drunk so far</span><span><i style="background:var(--muted)"></i>On pace for the goal</span></div>
          <div class="chart" data-cum></div></div>
        <div class="card"><div class="card-head"><h2>By hour</h2></div><div class="chart" data-hourly></div></div>
        <div class="card"><div class="card-head"><h2>Last 7 days</h2><span class="card-sub" data-week-sub></span></div><div class="chart" data-daily></div></div>
        <div class="card"><div class="card-head"><h2>Today’s sips</h2></div><div class="list" data-sips></div></div>
      </div>
      <div class="section-title">Settings</div>
      <div class="stack" data-settings>
        <div class="card"><div class="card-head"><h2>Daily goal</h2><span class="spacer"></span><span class="badge" data-goal-badge></span></div>
          <div class="form-grid" style="max-width:420px">
            <label class="field"><span>Goal</span><div class="input-group"><input class="input" type="number" min="1" data-goal><select class="input" data-units style="width:96px"><option value="ml">mL</option><option value="oz">fl oz</option></select></div></label>
          </div>
          <details class="more" data-calc-wrap><summary>How much should I drink?</summary><div class="body" data-calc></div></details>
        </div>
        <div class="card"><div class="card-head"><h2>Bottle</h2></div>
          <div class="form-grid" style="max-width:420px">
            <label class="field"><span>Size</span><select class="input" data-cap></select><span class="hint" data-cap-note></span></label>
          </div>
          <details class="more"><summary>Calibrate the level</summary><div class="body">
            <p class="small muted" style="margin-bottom:10px">If “in the bottle” is off, fill it to the top (or empty it), stand it on a flat surface, then tell it which:</p>
            <div class="row"><button class="btn" data-cal="full">It’s full now</button><button class="btn" data-cal="empty">It’s empty now</button></div></div></details>
        </div>
        <div class="card"><div class="card-head"><h2>Glow &amp; reminders</h2><span class="spacer"></span><button class="btn sm" data-glow-now>${icon('sparkle')}Glow now</button></div>
          <div class="stack" data-glow></div>
        </div>
      </div>`;

    // ------------------------------------------------------------ charts
    const dayStart = () => new Date().setHours(0, 0, 0, 0);
    const cum = new Plot($('[data-cum]', el), {
        series: [{ id: 'pace', label: 'On pace', color: 'var(--muted)', dash: [5, 5], width: 1.5 },
                 { id: 'drunk', label: 'Drunk', color: 'var(--water)', type: 'step', fill: true, width: 2.4 }],
        x: { min: dayStart(), max: dayStart() + 86400e3 }, y: { zero: true }, gap: 86400e3, empty: 'No sips yet today',
    });
    const hourly = new Plot($('[data-hourly]', el), {
        series: [{ id: 'ml', label: 'Drunk', color: 'var(--water)', type: 'bar' }],
        x: { kind: 'value', step: 1, min: 0, max: 24, labelStep: 3, format: (h) => `${String(h).padStart(2, '0')}:00`, tipFormat: (h) => `${String(h).padStart(2, '0')}:00–${String(h + 1).padStart(2, '0')}:00` },
        y: { zero: true }, empty: 'No sips yet today',
    });
    const daily = new Plot($('[data-daily]', el), {
        series: [{ id: 'ml', label: 'Drunk', color: 'var(--water)', type: 'bar' }],
        x: { kind: 'value', step: 1, labelStep: 1 }, y: { zero: true }, empty: 'No days yet',
    });

    function paintSummary() {
        if (!summary || !settings) return;
        const s = summary, goal = settings.goal_ml, today = s.today_ml;
        const pct = goal ? Math.min(1, today / goal) : 0;
        $('[data-ring] .fill', el).style.strokeDashoffset = String(326.7 * (1 - pct));
        $('[data-today]', el).textContent = vol(today, 0);
        $('[data-of]', el).textContent = goal ? `of ${vol(goal, 0)} ${unit()}` : unit();
        const r = settings.reminders || {};
        const p = window.WaterCalc.pace(today, goal, new Date(), r.from, r.to);
        let head = '', sub = '';
        if (!p) head = 'Set a goal below';
        else if (p.done) { head = 'Goal reached — nice work'; sub = `${vol(today - goal, 0)} ${unit()} over. Keep sipping when you’re thirsty.`; }
        else if (p.before) { head = today ? 'An early start' : 'A fresh day'; sub = `${vol(p.left, 0)} ${unit()} to go; reminders start at ${r.from || '08:00'}.`; }
        else if (p.after) { head = `${vol(p.left, 0)} ${unit()} short today`; sub = 'Tomorrow is a new goal.'; }
        else if (p.behind) { head = `About ${vol(p.behind, 0)} ${unit()} behind`; sub = p.perHour ? `${vol(p.perHour, 0)} ${unit()} an hour gets you there by ${r.to || '22:00'}.` : ''; }
        else if (p.ahead) { head = 'Ahead of pace'; sub = `${vol(p.left, 0)} ${unit()} left for today.`; }
        else { head = 'Right on pace'; sub = `${vol(p.left, 0)} ${unit()} left for today.`; }
        $('[data-pace]', el).textContent = head;
        $('[data-pace2]', el).textContent = sub;
        paintStats();
        // today: cumulative vs pace
        cum.o.x.min = dayStart(); cum.o.x.max = dayStart() + 86400e3;
        const pts = s.cumulative.map(p => [p.x, p.y]);
        if (pts.length) pts.unshift([dayStart(), 0]);
        if (pts.length) pts.push([Date.now(), today]);
        cum.set('drunk', pts.map(([x, y]) => [x, oz() ? y / OZ : y]));
        const a = minutes(r.from || '08:00'), b = minutes(r.to || '22:00');
        cum.set('pace', goal ? [[dayStart(), 0], [dayStart() + a * 60e3, 0], [dayStart() + b * 60e3, oz() ? goal / OZ : goal], [dayStart() + 86400e3, oz() ? goal / OZ : goal]] : []);
        $('[data-today-sub]', el).textContent = `${s.sips_today} sip${s.sips_today === 1 ? '' : 's'}`;
        hourly.set('ml', s.hourly.map((v, h) => [h, oz() ? v / OZ : v]).filter(p => p[1] > 0));
        hourly.get('ml').unit = unit(); cum.get('drunk').unit = unit(); cum.get('pace').unit = unit();
        // week
        const days = s.daily;
        daily.o.x.min = 0; daily.o.x.max = days.length;
        daily.o.x.format = (i) => days[i] ? dayLabel(days[i].date, { weekday: 'short' }) : '';
        daily.o.x.tipFormat = (i) => days[i] ? dayLabel(days[i].date) : '';
        daily.o.refs = goal ? [{ y: oz() ? goal / OZ : goal, label: 'goal', color: 'var(--muted)' }] : [];
        const d = daily.get('ml');
        d.unit = unit();
        d.alpha = (i, v) => (goal && v * (oz() ? OZ : 1) >= goal ? 1 : .55);
        daily.set('ml', days.map((x, i) => [i, oz() ? x.ml / OZ : x.ml]));
        const hit = days.filter(x => goal && x.ml >= goal).length;
        $('[data-week-sub]', el).textContent = goal ? `goal reached ${hit} of ${days.length} days` : '';
        // sips
        $('[data-sips]', el).innerHTML = s.recent.length ? s.recent.map(x => `<div class="item"><span class="mono muted">${hm(x.ts * 1000)}</span>
            <div class="grow"><b>${vol(x.ml)} ${unit()}</b>${x.source === 'replayed' ? ' <span class="badge" title="Sipped while Bio-dash wasn’t connected; the bottle kept it and sent it later">caught up</span>' : ''}</div>
            <span class="small muted">${ago(x.ts)}</span></div>`).join('')
            : '<div class="empty-state" style="padding:20px">No sips yet today.</div>';
    }
    const minutes = (s) => { const m = /^(\d+):(\d+)$/.exec(s); return m ? +m[1] * 60 + +m[2] : 0; };

    function paintStats() {
        const live = st.live || {}, connected = st.state === 'streaming';
        const t = (label, value, note = '') => `<div class="tile"><div class="label">${label}</div><div class="value" style="font-size:1.35rem">${value}</div><div class="note">${note}</div></div>`;
        const last = (summary && summary.last_sip) || live.last_sip;
        $('[data-stats]', el).innerHTML =
            t('In the bottle', connected && live.fill_ml != null ? `${vol(live.fill_ml, 0)}<small>${unit()}</small>` : '--', connected && live.fill_pct != null ? `${live.fill_pct}% full` : '')
            + t('Last sip', last ? `${vol(last.ml, 0)}<small>${unit()}</small>` : '--', last ? ago(last.ts) : '')
            + t('Sips today', summary ? summary.sips_today : '--', summary && summary.refills_today ? `${summary.refills_today} refill${summary.refills_today > 1 ? 's' : ''}` : '')
            + t('Battery', connected && live.battery != null ? `${live.battery}<small>%</small>` : '--', live.battery != null && live.battery < 20 ? 'charge it soon' : '');
    }

    async function loadSummary() {
        try { summary = await dapi('water', '/api/summary?days=7', { quiet: true }); paintSummary(); } catch (e) { /* offline */ }
    }

    // ------------------------------------------------------------ settings
    async function save(body, msg = 'Saved') {
        settings = await dapi('water', '/api/settings', { body });
        toast(msg, 'ok');
        fillForms();
        paintSummary();
    }
    function fillForms() {
        const g = $('[data-goal]', el);
        if (document.activeElement !== g) g.value = oz() ? Math.round(settings.goal_ml / OZ) : settings.goal_ml;
        $('[data-units]', el).value = settings.units || 'ml';
        const follow = (settings.goal_calc || {}).follow;
        $('[data-goal-badge]', el).textContent = follow ? 'updates each morning' : '';
        // bottle size
        const box = $('[data-cap]', el), capBottle = (st.live || {}).capacity_bottle;
        const want = settings.capacity_auto !== false ? 'auto' : String(settings.capacity_ml);
        const opts = [['auto', `Automatic${capBottle ? ` (${oz() ? Math.round(capBottle / OZ) + ' fl oz' : capBottle + ' mL'})` : ''}`]]
            .concat(BOTTLE_SIZES.map(([ml, o]) => [String(ml), `${o} fl oz · ${ml} mL`]));
        if (want !== 'auto' && !opts.some(o => o[0] === want)) opts.push([want, `${want} mL (custom)`]);
        box.innerHTML = opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('');
        box.value = want;
        $('[data-cap-note]', el).textContent = settings.capacity_auto !== false ? 'Uses the size the bottle reports.' : '';
        paintGlow();
    }
    $('[data-goal]', el).addEventListener('change', (e) => {
        const ml = fromUnit(e.target.value);
        if (!(ml >= 250 && ml <= 8000)) { toast(`Pick a goal between ${vol(250, 0)} and ${vol(8000, 0)} ${unit()}`, 'bad'); fillForms(); return; }
        save({ goal_ml: ml }, 'Goal saved and sent to the bottle');
    });
    $('[data-units]', el).addEventListener('change', (e) => save({ units: e.target.value }));
    $('[data-cap]', el).addEventListener('change', (e) => save(e.target.value === 'auto' ? { capacity_auto: true } : { capacity_ml: +e.target.value }, 'Bottle size saved'));
    $$('[data-cal]', el).forEach(b => b.addEventListener('click', async () => {
        const which = b.dataset.cal;
        if (st.state !== 'streaming') { toast('Connect the bottle first', 'bad'); return; }
        if (!await confirm({ title: `Is the bottle ${which} right now?`, text: `It should be ${which === 'full' ? 'filled to the top' : 'completely empty'}, cap on, standing on something flat.`, ok: `Yes, it’s ${which}` })) return;
        await busy(b, () => dapi('water', '/api/calibrate', { body: { which } }));
        toast(which === 'empty' ? 'Now fill it and tap “It’s full now”: it counts no sips until then' : 'Calibrated', 'ok');
    }));

    // after "It's empty now" the bottle counts no sips until "It's full now": keep saying so
    function paintCalWait() {
        const p = (st.live || {}).cal_pending, box = $('[data-calwait]', el);
        box.hidden = !(p && p.which === 'full');
        if (box.hidden) return;
        const mins = Math.round((Date.now() / 1000 - (p.since || 0)) / 60);
        $('[data-calwait-since]', el).textContent = !p.since ? '' : mins < 1 ? ' just now' : mins < 90 ? ` ${mins} min ago` : ` ${Math.round(mins / 60)} h ago`;
    }
    $('[data-calwait-btn]', el).addEventListener('click', async (e) => {
        const b = e.currentTarget;
        if (st.state !== 'streaming') { toast('Connect the bottle first: tip it to wake it, and close the HidrateSpark phone app', 'bad'); return; }
        if (!await confirm({ title: 'Is the bottle full right now?', text: 'Filled to the top, cap on, standing on something flat.', ok: 'Yes, it’s full' })) return;
        await busy(b, () => dapi('water', '/api/calibrate', { body: { which: 'full' } }));
        toast('Calibrated: sips count again from the next one', 'ok');
    });

    // goal calculator
    let calc = null, calcSeq = 0, calcTimer = null, place = null;
    function calcInputs() {
        const c = $('[data-calc]', el);
        const w = c.querySelector('[name=weight]').value;
        return { weight: w === '' ? null : +w, weight_unit: c.querySelector('[name=wunit]').value, reference: c.querySelector('[name=ref]').value,
                 exercise_min: +c.querySelector('[name=exercise]').value || 0, weather: c.querySelector('[name=weather]').value };
    }
    function buildCalc() {
        const g = settings.goal_calc || {};
        place = g.place || null;
        $('[data-calc]', el).innerHTML = `
          <div class="form-grid">
            <label class="field"><span>Body weight</span><div class="input-group"><input class="input" name="weight" type="number" min="20" max="700" value="${g.weight ?? ''}" placeholder="optional">
              <select class="input" name="wunit" style="width:80px"><option value="lb">lb</option><option value="kg">kg</option></select></div></label>
            <label class="field"><span>Starting point</span><select class="input" name="ref"><option value="men">3.7 L a day (men)</option><option value="women">2.7 L a day (women)</option></select></label>
            <label class="field"><span>Exercise today</span><div class="input-group"><input class="input" name="exercise" type="number" min="0" max="600" value="${g.exercise_min || 0}"><span class="input" style="width:auto;flex:none;color:var(--muted)">min</span></div></label>
            <label class="field"><span>Weather</span><select class="input" name="weather"><option value="auto">Today’s local forecast</option><option value="hot">It’s hot (by hand)</option><option value="off">Leave it out</option></select></label>
          </div>
          <div data-place-row style="margin-top:12px"></div>
          <div class="callout" style="margin-top:14px;display:block" data-result>Working it out…</div>
          <div class="row" style="margin-top:12px"><button class="btn primary" data-use>Use as my daily goal</button>
            <label class="switch" style="--c:var(--water)"><input type="checkbox" data-follow ${g.follow ? 'checked' : ''}><span class="text"><b>Update it every morning</b><span>With that day’s weather</span></span></label></div>
          <p class="tiny faint" style="margin-top:12px">Reference intakes: US National Academies (total water, ~80 % from drinks). Weather: Open-Meteo. Nothing is sent to the bottle until you use the goal.</p>`;
        const c = $('[data-calc]', el);
        c.querySelector('[name=wunit]').value = g.weight_unit || 'lb';
        c.querySelector('[name=ref]').value = g.reference || 'men';
        c.querySelector('[name=weather]').value = g.weather || 'off';
        c.addEventListener('input', () => { clearTimeout(calcTimer); calcTimer = setTimeout(() => { suggest(); saveCalc(); }, 400); });
        c.querySelector('[name=weather]').addEventListener('change', () => { paintPlace(); suggest(); saveCalc(); });
        c.querySelector('[data-use]').addEventListener('click', (e) => calc && busy(e.currentTarget, () => save({ goal_ml: calc.goal }, 'Goal saved and sent to the bottle')));
        c.querySelector('[data-follow]').addEventListener('change', (e) => save({ goal_calc: { ...calcInputs(), follow: e.target.checked } },
            e.target.checked ? 'Your goal will follow the weather each morning' : 'Morning updates off'));
        paintPlace();
        suggest();
    }
    function paintPlace() {
        const row = $('[data-place-row]', el), auto = calcInputs().weather === 'auto';
        row.hidden = !auto;
        if (!auto) return;
        row.innerHTML = place
            ? `<div class="row"><span>${icon('target').replace('<svg', '<svg style="width:18px;height:18px;color:var(--muted)"')}</span><b>${esc(place.label || place.name)}</b><button class="btn sm ghost" data-change>Change town</button></div>`
            : `<form class="row" data-find style="align-items:flex-end"><label class="field" style="flex:1;min-width:180px"><span>Your town (for the forecast)</span><input class="input" name="q" placeholder="e.g. Portland" autocomplete="off"></label><button class="btn" type="submit">${icon('search')}Find</button></form><div data-places style="margin-top:8px"></div>`;
        row.querySelector('[data-change]')?.addEventListener('click', () => { place = null; paintPlace(); });
        row.querySelector('[data-find]')?.addEventListener('submit', async (e) => {
            e.preventDefault();
            const q = e.target.q.value.trim();
            if (q.length < 2) return;
            const r = await dapi('water', `/api/goal/places?q=${encodeURIComponent(q)}`, { quiet: true }).catch(() => ({ places: [] }));
            const box = row.querySelector('[data-places]');
            box.innerHTML = r.error ? `<p class="small muted">${esc(r.error)}</p>` : r.places.length ? `<div class="list">${r.places.map((p, i) => `<div class="item"><div class="grow">${esc(p.label || p.name)}</div><button class="btn sm" data-pick="${i}">Pick</button></div>`).join('')}</div>` : '<p class="small muted">No town by that name.</p>';
            box.querySelectorAll('[data-pick]').forEach(b => b.addEventListener('click', async () => {
                place = r.places[+b.dataset.pick];
                await dapi('water', '/api/settings', { body: { goal_calc: { ...calcInputs(), place } } }).then(s => { settings = s; });
                paintPlace(); suggest();
            }));
        });
    }
    async function saveCalc() {
        try { settings = await dapi('water', '/api/settings', { body: { goal_calc: calcInputs() }, quiet: true }); } catch (e) { /* keeps the form */ }
    }
    async function suggest() {
        const seq = ++calcSeq, box = $('[data-result]', el);
        let c;
        try { c = await dapi('water', '/api/goal/suggest', { body: { goal_calc: calcInputs() }, quiet: true }); } catch (e) { box.textContent = 'Couldn’t work it out right now.'; return; }
        if (seq !== calcSeq) return;
        calc = c;
        const u = unit(), lines = [];
        lines.push([`${vol(c.reference, 0)} ${u}`, `starting point (${calcInputs().reference === 'women' ? '2.7' : '3.7'} L total, ~80 % from drinks)`]);
        if (c.by_weight) { lines.push([`${vol(c.by_weight, 0)} ${u}`, `by your weight (${c.kg} kg × 33 mL)`]); lines.push([`${vol(c.base, 0)} ${u}`, 'halfway between the two']); }
        if (c.exercise) lines.push([`+ ${vol(c.exercise, 0)} ${u}`, 'for exercise (≈ 0.7 L an hour)']);
        if (c.weather_ml) lines.push([`+ ${vol(c.weather_ml, 0)} ${u}`, c.weather ? `${c.weather_word} day: feels like ${temp(c.weather.feels_max_c)} at the warmest` : c.weather_word]);
        else if (c.weather_word === 'cold' || c.weather_word === 'mild') lines.push(['+ 0', `${c.weather_word} day`]);
        const wx = c.weather;
        box.innerHTML = `<div class="row" style="align-items:baseline;margin-bottom:8px"><span class="big" style="font-size:2rem;color:var(--water)">${vol(c.goal, 0)}<small>${u}</small></span>
            <span class="muted small">${settings.capacity_ml ? `≈ ${(c.goal / settings.capacity_ml).toFixed(1)} bottles` : ''}${c.clamped ? ' · kept within 1.5–5 L' : ''}</span></div>
          <table class="t" style="font-size:.84rem"><tbody>${lines.map(([a, b]) => `<tr><td class="r" style="width:1%"><b>${a}</b></td><td class="muted" style="white-space:normal">${esc(b)}</td></tr>`).join('')}</tbody></table>
          ${c.weather_note ? `<p class="small" style="margin-top:8px;color:var(--warn)">${esc(c.weather_note)}</p>` : wx ? `<p class="small muted" style="margin-top:8px">Today${place ? ' in ' + esc(place.name) : ''}: high ${temp(wx.max_c)} (feels like ${temp(wx.feels_max_c)}), low ${temp(wx.min_c)}.</p>` : ''}`;
        const use = $('[data-use]', el), same = c.goal === settings.goal_ml;
        use.disabled = same;
        use.textContent = same ? 'This is your goal' : 'Use as my daily goal';
    }
    $('[data-calc-wrap]', el).addEventListener('toggle', (e) => { if (e.target.open && !$('[data-calc] [name]', el)) buildCalc(); });

    // glow + reminders
    let glowSig = '';
    function paintGlow() {
        const box = $('[data-glow]', el), lv = st.live || {};
        // redraw only when something it shows changed, so half-made edits aren't wiped
        const sig = JSON.stringify([settings.reminders, settings.glow, settings.goal_glow, settings.sip_glow, lv.glow_upload, lv.synced_at, lv.reminders_set, lv.can_upload_glow]);
        if (sig === glowSig || box.contains(document.activeElement)) return;
        glowSig = sig;
        const r = settings.reminders || {}, g = settings.glow || {}, live = st.live || {};
        const sip = settings.sip_glow === true ? 'on' : settings.sip_glow === false ? 'off' : 'leave';
        const up = live.glow_upload;
        box.innerHTML = `
          <label class="switch" style="--c:var(--water)"><input type="checkbox" name="enabled" ${r.enabled !== false ? 'checked' : ''}><span class="text"><b>Glow reminders</b><span>The bottle glows to remind you to drink</span></span></label>
          <div class="form-grid" data-rem ${r.enabled === false ? 'hidden' : ''}>
            <label class="field"><span>From</span><input class="input" type="time" name="from" value="${esc(r.from || '08:00')}"></label>
            <label class="field"><span>Until</span><input class="input" type="time" name="to" value="${esc(r.to || '22:00')}"></label>
            <label class="field"><span>Every</span><select class="input" name="every">${[15, 30, 45, 60, 90, 120, 180, 240].map(m => `<option value="${m}" ${+r.every_min === m ? 'selected' : ''}>${m < 60 ? m + ' min' : m / 60 + ' h'}</option>`).join('')}</select></label>
            <label class="field"><span>When</span><select class="input" name="always"><option value="behind">Only when I’m behind</option><option value="always" ${r.always ? 'selected' : ''}>At every reminder</option></select></label>
          </div>
          <label class="switch" style="--c:var(--water)"><input type="checkbox" name="goal_glow" ${settings.goal_glow !== false ? 'checked' : ''}><span class="text"><b>Celebrate the goal</b><span>A glow when you reach it</span></span></label>
          <label class="field"><span>Glow on each sip</span><select class="input" name="sip_glow" style="max-width:260px"><option value="leave">Leave it as the bottle has it</option><option value="on">On</option><option value="off">Off</option></select></label>
          <div class="row"><button class="btn primary" data-save-rem>Save &amp; send to bottle</button>${live.synced_at ? `<span class="small muted">Bottle last updated ${ago(live.synced_at)}${live.reminders_set != null ? ` · ${live.reminders_set} reminders set` : ''}</span>` : ''}</div>
          <details class="more"><summary>Glow style</summary><div class="body stack">
            <div class="form-grid"><label class="field"><span>Style</span><select class="input" name="style">${GLOW_STYLES.map(([k, t]) => `<option value="${k}" ${g.style === k ? 'selected' : ''}>${t}</option>`).join('')}</select></label>
              <label class="field"><span>Colours</span><div class="row tight"><input class="input" type="color" name="c1" value="${esc(g.color1 || '#2bff00')}"><input class="input" type="color" name="c2" value="${esc(g.color2 || '#1499ff')}"></div></label></div>
            <div class="row"><button class="btn" data-send-glow ${live.can_upload_glow === false ? 'disabled title="This bottle can’t take a custom glow"' : ''}>Send glow to bottle</button>
              ${up ? `<span class="small ${up.state === 'failed' ? '' : 'muted'}" style="${up.state === 'failed' ? 'color:var(--bad)' : ''}">${up.state === 'done' ? 'Sent ' + ago(up.at) : up.state === 'sending' ? 'Sending…' : 'Didn’t send: ' + esc(up.error || '')}</span>` : ''}</div>
          </div></details>`;
        box.querySelector('[name=sip_glow]').value = sip;
        box.querySelector('[name=enabled]').addEventListener('change', (e) => { box.querySelector('[data-rem]').hidden = !e.target.checked; });
        box.querySelector('[data-save-rem]').addEventListener('click', (e) => busy(e.currentTarget, () => save({
            reminders: { enabled: box.querySelector('[name=enabled]').checked, from: box.querySelector('[name=from]').value, to: box.querySelector('[name=to]').value,
                         every_min: +box.querySelector('[name=every]').value, always: box.querySelector('[name=always]').value === 'always' },
            goal_glow: box.querySelector('[name=goal_glow]').checked, sip_glow: box.querySelector('[name=sip_glow]').value,
        }, st.state === 'streaming' ? 'Saved and sent to the bottle' : 'Saved — sent when the bottle next connects')));
        box.querySelector('[data-send-glow]').addEventListener('click', (e) => busy(e.currentTarget, async () => {
            if (st.state !== 'streaming') { toast('Connect the bottle first', 'bad'); return; }
            await dapi('water', '/api/glow/pattern', { body: { style: box.querySelector('[name=style]').value, color1: box.querySelector('[name=c1]').value, color2: box.querySelector('[name=c2]').value } });
            toast('Sending the glow to the bottle…', 'ok');
        }));
    }
    $('[data-glow-now]', el).addEventListener('click', (e) => busy(e.currentTarget, async () => {
        if (st.state !== 'streaming') { toast('Connect the bottle first', 'bad'); return; }
        await dapi('water', '/api/glow', { method: 'POST' });
        toast('Glowing', 'ok');
    }));

    // ------------------------------------------------------------ start
    try { settings = await dapi('water', '/api/settings', { quiet: true }); } catch (e) { settings = { goal_ml: 2500, units: 'ml', reminders: {} }; }
    st = store.status.water || {};
    fillForms();
    await loadSummary();
    let lastSipTs = null;
    let stopFinder = null;
    const connectBox = $('[data-connect]', el);
    const offHead = bindHead(el, 'water', (s2, s) => {
        st = s2 || {};
        if (!s.live && !stopFinder) { stopFinder = connectPanel(connectBox, 'water'); connectBox.hidden = false; }
        if (s.live && stopFinder) { stopFinder(); stopFinder = null; connectBox.innerHTML = ''; connectBox.hidden = true; }
        const ls = (st.live || {}).last_sip;
        if (ls && ls.ts !== lastSipTs) { if (lastSipTs !== null) loadSummary(); lastSipTs = ls.ts; }
        paintStats();
        paintCalWait();
        if (settings) paintGlow();
    });
    const t = setInterval(loadSummary, 60e3);
    return () => { offHead(); stopFinder && stopFinder(); clearInterval(t); cum.destroy(); hourly.destroy(); daily.destroy(); };
}
