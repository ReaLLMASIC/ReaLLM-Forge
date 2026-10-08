// Switches the ECG card between the bedside monitor (sweep) and the classic
// scrolling Chart.js line. The monitor always receives the data, so beat detection
// and the QRS beep keep working in both views. Choice is remembered per dashboard.
//
//   const ecgView = new EcgView({ monitor, storageKey: 'polar-ecg-view' });
//   ecgView.push(rows)   // rows = [[ts_ms, mV, ...], ...]  (use instead of monitor.push)

class EcgView {
    constructor({ monitor, storageKey }) {
        this.monitor = monitor;
        this.key = storageKey;
        const $ = (id) => document.getElementById(id);
        this.el = {
            sweep: $('ecg-sweep'), hrPanel: $('ecg-hr-panel'), classic: $('ecg-classic'),
            specs: $('ecg-specs'), gain: $('gain-btn'), toggle: $('ecg-view-toggle'),
        };

        // Same look as the original dashboard's ECG chart
        this.chart = new Chart($('ecgClassicChart').getContext('2d'), {
            type: 'line',
            data: { datasets: [{ borderColor: '#f43f5e', borderWidth: 2, pointRadius: 0, data: [] }] },
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                scales: {
                    x: { type: 'realtime', realtime: { duration: 5000, refresh: 40, delay: 1000 },
                         ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } },
                    y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                },
                plugins: { legend: { display: false } },
            },
        });
        this.data = this.chart.data.datasets[0].data;

        this.el.toggle.addEventListener('click', (e) => {
            const b = e.target.closest('button[data-view]');
            if (b) this.set(b.dataset.view);
        });
        let saved = 'monitor';
        try { saved = localStorage.getItem(this.key) || 'monitor'; } catch (e) {}
        this.set(saved);
    }

    set(mode) {
        this.mode = mode === 'classic' ? 'classic' : 'monitor';
        try { localStorage.setItem(this.key, this.mode); } catch (e) {}
        const classic = this.mode === 'classic';
        this.el.sweep.classList.toggle('hidden', classic);
        this.el.hrPanel.classList.toggle('hidden', classic);
        this.el.specs.classList.toggle('hidden', classic);
        this.el.gain.classList.toggle('hidden', classic);      // gain only applies to the monitor
        this.el.classic.classList.toggle('hidden', !classic);
        this.el.toggle.querySelectorAll('button[data-view]').forEach(b => setPillActive(b, b.dataset.view === this.mode));
        if (!classic) this.data.length = 0;                     // don't let a hidden chart accumulate
    }

    push(rows) {
        this.monitor.push(rows);
        if (this.mode === 'classic') for (const r of rows) this.data.push({ x: r[0], y: r[1] });
    }
}
