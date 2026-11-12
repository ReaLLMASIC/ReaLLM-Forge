// What the Atmos readings mean: names, units, colours, and plain-language bands.
// Series ids are a metric, optionally tagged with the sensor: "co2", "temp@scd30", "rh@sen55".

// Same priority as atmos_parser.py: SEN5x/SEN6x temperature & RH are compensated, the SCD30
// runs warm from its IR source, SFA3X last.
export const SOURCE_ORDER = ['sen69c', 'sen68', 'sen66', 'sen55', 'sen54', 'sen5x', 'sen44', 'sht4x',
                             'scd30', 'scd41', 'scd40', 'scd4x', 'sfa3x', 'sfa30', 'sgp41', ''];
export const METRICS = {
    co2:  { label: 'CO₂', long: 'Carbon dioxide', unit: 'ppm', dp: 0, color: '#f59e0b' },
    pm25: { label: 'PM2.5', long: 'Fine particles', unit: 'µg/m³', dp: 1, color: '#22d3ee' },
    temp: { label: 'Temperature', long: 'Temperature', unit: '°C', dp: 1, color: '#fb7185' },
    rh:   { label: 'Humidity', long: 'Relative humidity', unit: '%', dp: 0, color: '#60a5fa' },
    voc:  { label: 'VOC index', long: 'Volatile organic compounds', unit: '', dp: 0, color: '#a78bfa' },
    nox:  { label: 'NOx index', long: 'Nitrogen oxides', unit: '', dp: 0, color: '#818cf8' },
    hcho: { label: 'Formaldehyde', long: 'Formaldehyde (HCHO)', unit: 'ppb', dp: 1, color: '#f87171' },
    pm1:  { label: 'PM1', long: 'Ultrafine particles', unit: 'µg/m³', dp: 1, color: '#a3e635' },
    pm4:  { label: 'PM4', long: 'Particles up to 4 µm', unit: 'µg/m³', dp: 1, color: '#e879f9' },
    pm10: { label: 'PM10', long: 'Coarse particles', unit: 'µg/m³', dp: 1, color: '#fb923c' },
};
export const CANONICAL = Object.keys(METRICS);
export const metricOf = (id) => id.split('@')[0];
export const sourceOf = (id) => id.split('@')[1] || '';
const rank = (id) => { const i = SOURCE_ORDER.indexOf(sourceOf(id)); return i < 0 ? SOURCE_ORDER.length : i; };

// best available value of a metric in one row's values
export function pick(v, metric) {
    let best = null, bestRank = Infinity;
    for (const id in v || {}) {
        if (metricOf(id) !== metric) continue;
        const r = rank(id);
        if (r < bestRank) { best = v[id]; bestRank = r; }
    }
    return best;
}

// -> {word, tone: ok|warn|bad, advice}
export function co2Band(ppm) {
    if (ppm == null) return null;
    if (ppm < 800) return { word: 'Fresh', tone: 'ok', advice: 'Well ventilated.' };
    if (ppm < 1000) return { word: 'OK', tone: 'ok', advice: 'Getting a little stuffy.' };
    if (ppm < 1500) return { word: 'Stuffy', tone: 'warn', advice: 'Open a window or door for a few minutes.' };
    return { word: 'Very stuffy', tone: 'bad', advice: 'Ventilate now — high CO₂ makes it hard to concentrate.' };
}
// US EPA PM2.5 categories (2024 breakpoints; the official AQI uses a 24 h average)
export function pmBand(v) {
    if (v == null) return null;
    if (v <= 9.0) return { word: 'Good', tone: 'ok', advice: 'Clean air.' };
    if (v <= 35.4) return { word: 'Moderate', tone: 'warn', advice: 'Fine for most people.' };
    if (v <= 55.4) return { word: 'Unhealthy for some', tone: 'bad', advice: 'Sensitive people should limit exposure.' };
    if (v <= 125.4) return { word: 'Unhealthy', tone: 'bad', advice: 'Close windows; run a purifier if you have one.' };
    return { word: 'Very unhealthy', tone: 'bad', advice: 'Reduce exposure; run a purifier.' };
}
export function vocBand(v) {      // Sensirion index: 100 = this room's learned baseline
    if (v == null) return null;
    if (v > 250) return { word: 'High', tone: 'bad', advice: 'Something is giving off fumes.' };
    if (v > 150) return { word: 'Above usual', tone: 'warn', advice: 'Cooking, cleaning products or paint?' };
    return { word: 'Usual', tone: 'ok', advice: '' };
}
export function overall(v) {
    // the worst of CO₂ and particles, as one line for the top of the page
    const bands = [co2Band(pick(v, 'co2')), pmBand(pick(v, 'pm25'))].filter(Boolean);
    if (!bands.length) return null;
    const order = { ok: 0, warn: 1, bad: 2 };
    return bands.sort((a, b) => order[b.tone] - order[a.tone])[0];
}
