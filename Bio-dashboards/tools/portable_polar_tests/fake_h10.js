// A stand-in Polar H10 for Web Bluetooth (navigator.bluetooth), for the browser test: answers the PMD start
// requests and sends ECG (130 Hz, 73-sample frames), accelerometer (~205.6 Hz, raw 16-bit) and heart rate
// at the strap's pace, with its own clock (ns since 2000) a few seconds off the computer's.
(() => {
    const PMD = 'fb005c80-02e7-f387-1cad-8acd2d8df0c8', CP = 'fb005c81-02e7-f387-1cad-8acd2d8df0c8', DATA = 'fb005c82-02e7-f387-1cad-8acd2d8df0c8';
    class Char extends EventTarget {
        constructor(uuid, dev) { super(); this.uuid = uuid; this.dev = dev; this.value = null; }
        async startNotifications() { this.notifying = true; return this; }
        emit(bytes) { if (!this.notifying || !this.dev.gatt.connected) return; this.value = new DataView(new Uint8Array(bytes).buffer); this.dispatchEvent(new Event('characteristicvaluechanged')); }
        async writeValueWithResponse(v) { this.dev.onWrite(this, new Uint8Array(v.buffer || v)); }
    }
    class Dev extends EventTarget {
        constructor() {
            super(); this.name = 'Polar H10 TEST0001'; this.id = 'fake';
            this.chars = { hr: new Char('heart_rate_measurement', this), cp: new Char(CP, this), data: new Char(DATA, this) };
            const dev = this;
            this.gatt = {
                connected: false, device: this,
                async connect() { this.connected = true; dev.start(); return this; },
                disconnect() { if (!this.connected) return; this.connected = false; dev.stop(); dev.dispatchEvent(new Event('gattserverdisconnected')); },
                async getPrimaryService(s) {
                    if (s === 'heart_rate') return { getCharacteristic: async () => dev.chars.hr };
                    if (s === PMD) return { getCharacteristic: async c => (c === CP ? dev.chars.cp : dev.chars.data) };
                    throw new Error('no service ' + s);
                },
            };
            this.offsetNs = (BigInt(Date.now()) - 946684800000n + 3456n) * 1000000n;     // strap clock: ns since 2000, a bit off
            window.__fake = this;
        }
        sensorNs() { return BigInt(Date.now()) * 1000000n - 946684800000n * 1000000n - 3456000000n; }
        onWrite(ch, b) {
            if (b[0] === 2) {
                const reply = [0xf0, 0x02, b[1], 0x00, 0x00];
                if (b[1] === 2) { const f = new DataView(new ArrayBuffer(4)); f.setFloat32(0, 1, true); reply.push(5, 1, ...new Uint8Array(f.buffer)); }
                setTimeout(() => ch.emit(reply), 30);
                if (b[1] === 0) this.ecgOn = true; else this.accOn = true;
            }
        }
        start() {
            this.ecgN = 0; this.accN = 0; const t0 = Date.now();
            this.timers = [
                setInterval(() => this.frameEcg(t0), 561),
                setInterval(() => this.frameAcc(), 175),
                setInterval(() => { const rr = 820 * 1024 / 1000 | 0; this.chars.hr.emit([0x10, 73, rr & 255, rr >> 8]); }, 1000),
            ];
        }
        stop() { (this.timers || []).forEach(clearInterval); }
        frame(type, ft, payload) {
            const h = new DataView(new ArrayBuffer(10)); h.setUint8(0, type); h.setBigUint64(1, this.sensorNs(), true); h.setUint8(9, ft);
            this.chars.data.emit([...new Uint8Array(h.buffer), ...payload]);
        }
        frameEcg() {
            if (!this.ecgOn) return;
            const out = [];
            for (let i = 0; i < 73; i++, this.ecgN++) {
                const p = (this.ecgN % 107) - 30;                                  // a beat every 107 samples (~73 bpm)
                let uv = Math.round(1000 * Math.exp(-((p / 1.5) ** 2)) + 250 * Math.exp(-(((p - 35) / 6) ** 2)) - 60);
                uv &= 0xffffff; out.push(uv & 255, (uv >> 8) & 255, (uv >> 16) & 255);
            }
            this.frame(0, 0x00, out);
        }
        frameAcc() {
            if (!this.accOn) return;
            const out = [];
            for (let i = 0; i < 36; i++, this.accN++) for (const v of [Math.round(30 * Math.sin(this.accN / 20)), -980, 15]) { const u = v & 0xffff; out.push(u & 255, u >> 8); }
            this.frame(2, 0x01, out);
        }
    }
    Object.defineProperty(navigator, 'bluetooth', { value: { requestDevice: async () => new Dev() } });
})();
