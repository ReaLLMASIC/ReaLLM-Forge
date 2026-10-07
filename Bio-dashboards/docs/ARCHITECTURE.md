# How Bio-dash works

Flow diagrams for the whole suite, from the devices to the files on disk. GitHub renders the
diagrams below; elsewhere, paste a block into <https://mermaid.live>.

1. [The big picture](#1-the-big-picture) — devices, radios, workers, web apps, recordings
2. [Inside one dashboard](#2-inside-one-dashboard) — how a worker and its web app talk
3. [A device's life cycle](#3-a-devices-life-cycle) — scan, connect, record, reconnect
4. [Hydro-dash: the bottle protocol](#4-hydro-dash-the-bottle-protocol)
5. [Atmos: from any sensor format to tiles](#5-atmos-from-any-sensor-format-to-tiles)
6. [Starting and running everything](#6-starting-and-running-everything)

**Colour key** (same in every diagram):

```mermaid
flowchart LR
    a["Device"]:::device ~~~ b["Radio / BLE link"]:::radio ~~~ c["Worker / processing"]:::worker ~~~ d["Web app / feature"]:::app ~~~ e["Shared file / script"]:::file
    f["Recording (CSV)"]:::store ~~~ g["Browser / you run"]:::ui ~~~ h{"Decision"}:::decision ~~~ i["Problem / dropped"]:::warn ~~~ j(["Start / end"]):::start
    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
```

---

## 1. The big picture

Each device has its own dashboard: a **worker** that talks Bluetooth (or Wi-Fi) and records, and
a **web app** that serves the page. Omni runs both body sensors in one process.

```mermaid
flowchart LR
    subgraph DEV["Devices"]
        H10["Polar H10<br/>ECG · ACC · HR · R-R"]
        O2["Viatom O2 band<br/>SpO₂ · pulse · battery"]
        ATM["Atmos sensors<br/>Mini · Sphere S4/S5 · SEN69C"]
        HS["HidrateSpark PRO 2<br/>sips · weight · cap"]
    end

    subgraph RAD["Bluetooth radios (BlueZ)"]
        EXT["External USB dongle<br/>preferred by Polar"]
        INT["Internal radio<br/>preferred by the others"]
    end

    subgraph SUITE["Bio-dash (one process pair per dashboard)"]
        PW["Polar worker"] --- PA["Polar app :5001"]
        VW["Viatom worker"] --- VA["Viatom app :5003"]
        AW["Atmos worker"] --- AA["Atmos app :5002"]
        HW["Hydro worker"] --- HA["Hydro app :5004"]
        OM["Omni :5000<br/>Polar + O2 together"]
    end

    H10 --> EXT
    O2 --> INT
    ATM -->|BLE| INT
    ATM -.->|Wi-Fi TCP :8080| AW
    HS --> INT
    EXT --> PW & OM
    INT --> VW & AW & HW & OM

    PA & VA & AA & HA & OM --> BR["Browser<br/>localhost or phone"]
    PW & VW & AW & HW & OM --> REC[("~/Documents/Bio-dash<br/>CSV per connection")]

    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
    class H10,O2,ATM,HS device
    class EXT,INT radio
    class PW,VW,AW,HW,OM worker
    class PA,VA,AA,HA app
    class BR ui
    class REC store
    style DEV fill:none,stroke:#F59E0B,stroke-width:1.5px
    style RAD fill:none,stroke:#8B5CF6,stroke-width:1.5px
    style SUITE fill:none,stroke:#94A3B8,stroke-width:1.5px
```

- **Radio routing:** each dashboard picks a radio by its *address* (stable across reboots) from
  `radio_config.json`, or automatically: Polar prefers the external dongle, the rest the internal radio.
- **Omni and the standalone Polar / Viatom dashboards are alternatives** — both would try to own
  the same devices (`biodash.py` won't install them together).

---

## 2. Inside one dashboard

The worker and the web app are separate processes that share a few small files, so either can
restart without taking the other down.

```mermaid
flowchart TB
    DEV(("Device")) <-->|Bluetooth / TCP| W

    subgraph D["e.g. hydro_dashboard/"]
        W["Worker<br/>hydro_worker.py · advanced_worker.py<br/>ble_worker.py · sensor_worker.py"]
        S[/"status.json<br/>live values · debug info"/]
        DV[/"devices.json<br/>last scan results"/]
        C[/"command.json<br/>connect · calibrate"/]
        CFG[/"radio_config.json · last_device.json<br/>settings / profile"/]
        A["Web app<br/>app.py (FastAPI / Flask)"]
        W -->|every second| S
        W -->|after each scan| DV
        C -->|picked up within 1 s| W
        CFG <--> W
        S & DV --> A
        A -->|Connect, Calibrate…| C
        A <--> CFG
    end

    A <-->|"/api/… · WebSocket or SSE"| P["Page in the browser<br/>tiles · charts · debug panel (D)"]
    W -->|"CSV rows"| R[("Documents/Bio-dash/&lt;dashboard&gt;/&lt;device&gt;/&lt;date&gt;/")]
    R -->|"/api/history"| A

    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
    class DEV device
    class W worker
    class S,DV,C,CFG file
    class A app
    class P ui
    class R store
    style D fill:none,stroke:#94A3B8,stroke-width:1.5px
```

- **Live data:** Polar and Atmos push over a WebSocket, Viatom and Omni over Server-Sent Events,
  Hydro-dash polls (its data changes a few times an hour).
- **History:** tile trends and Hydro-dash's totals are read back from the CSV recordings, so they
  survive restarts.
- **Shared code** (identical copies in every dashboard): `bt_debug.py` (radios, auto-reconnect,
  recordings, shutdown/reboot) and the page scripts `nav.js`, `tile_trends.js`, `reconnect.js`, `power.js`.

---

## 3. A device's life cycle

The same loop in every worker. Hydro-dash's connect step differs; see section 4.

```mermaid
flowchart TD
    START(["Worker starts"]) --> SCAN["Scan ~4 s<br/>(on the pinned or auto radio)"]
    SCAN --> LIST["Write devices.json<br/>→ scanner panel on the page"]
    LIST --> Q{"Connect clicked,<br/>or remembered device seen<br/>with auto-reconnect on?"}
    Q -- no --> SCAN
    Q -- yes --> CONN["Connect<br/>(up to 3 attempts)"]
    CONN -->|"all attempts failed"| SCAN
    CONN -->|connected| OPEN["Open a new set of CSVs<br/>HH-MM-SS_*.csv"]
    OPEN --> REM["Remember the device<br/>(last_device.json)"]
    REM --> STREAM["Stream: decode → status.json → page<br/>write rows · flush"]
    STREAM -->|"link drops / device sleeps"| CLOSE["Close that CSV set"]
    CLOSE --> SCAN

    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
    class START start
    class SCAN,LIST,CONN,STREAM worker
    class Q decision
    class OPEN,CLOSE store
    class REM file
```

- **One CSV set per connection.** A dropout ends the set; the reconnect starts a new one in the
  same date folder. Timestamps are absolute, so sets line up.
- **Auto-reconnect** can be paused or forgotten from the reconnect bar on each page
  (`BIODASH_AUTORECONNECT=0` turns it off by default).

---

## 4. Hydro-dash: the bottle protocol

The PRO 2 sleeps, advertises only briefly after being moved, **changes its address each wake-up**,
and is dropped by BlueZ the moment scanning stops — hence the unusual connect step.

```mermaid
flowchart TB
    subgraph S1["1 · Find and connect"]
        direction LR
        WAKE(["Bottle moved"]) --> ADV["Advertises h2o0000xxxx<br/>new random address"] --> FIND["Match by NAME<br/>while scanning"] --> CONN["Connect with the scan<br/>still running"] --> MAP["Read info,<br/>subscribe to notifies"]
    end
    subgraph S2["2 · Handshake and drain"]
        direction LR
        HS["Handshake<br/>13 writes"] --> WAIT["Wait 1.5 s"] --> DRAIN["Ask for one record<br/>(0x57, one at a time)"] --> FRAME{"Records<br/>queued?"}
        FRAME -->|"01 00 00…"| DRAIN
    end
    subgraph S3["3 · Each record"]
        direction LR
        DEC{"Layout?"} -->|"PRO 2 / older"| P2["sip · time ·<br/>weight before→after"] --> DUP{"Seen<br/>already?"} -->|no| SIP[("_sips.csv")]
        DEC -->|none fits| UNK["Undecoded<br/>(kept in _raw.csv)"]
        DUP -->|yes| SKIPD["Dropped as<br/>duplicate"]
        NEXT["Next record → 2"]
        SIP & SKIPD & UNK --> NEXT
    end
    subgraph S4["4 · Connected, idle"]
        direction LR
        W["Weight ~2 s"] --> FILL["Fill level"]
        CAP["Cap open / close"] --> REF["Refill detection"]
        NEW["New sip, or every 30 s<br/>when idle → back to 2"]
        GONE(["Bottle sleeps →<br/>scan again"])
        NEW ~~~ GONE
    end
    S1 --> S2
    S2 -->|"each record"| S3
    S3 -->|"queue empty"| S4
    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
    class WAKE device
    class ADV device
    class FIND,CONN,MAP radio
    class HS,WAIT,DRAIN,NEW worker
    class FRAME,DEC,DUP decision
    class NEXT worker
    class P2,W,CAP,FILL,REF app
    class UNK,SKIPD warn
    class SIP store
    class GONE start
    style S1 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style S2 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style S3 fill:none,stroke:#CBD5E1,stroke-width:1.5px
    style S4 fill:none,stroke:#CBD5E1,stroke-width:1.5px
```

- **Totals, charts, "today"** are computed from `_sips.csv` across all sessions, so replayed sips
  land on the right day and nothing double-counts.
- **Everything raw** goes to `_raw.csv` (every notification, hex) — the basis for decoding new firmware.
- Protocol details and sources: `hydro_dashboard/PROTOCOL.md`.

---

## 5. Atmos: from any sensor format to tiles

One parser handles every Atmos board, over Bluetooth (Nordic UART) or Wi-Fi (TCP).

```mermaid
flowchart TD
    IN1["BLE UART<br/>(chunks of ~20 bytes)"] --> RX["handle_rx():<br/>reassemble lines"]
    IN2["Wi-Fi TCP :8080"] --> RX
    RX --> KIND{"Line format?"}
    KIND -->|"JSON"| NORM
    KIND -->|"key=value"| NORM
    KIND -->|"header line<br/>co2@scd30,…"| HDR["Remember column names"] --> NORM
    KIND -->|"bare CSV"| PROF{"Columns from?"}
    PROF -->|"header seen"| NORM
    PROF -->|"profile: auto / mini / s4 / s5 / custom"| NORM
    NORM["Normalise names<br/>CO2_ppm → co2 · SCD30 Temp → temp@scd30<br/>NaN / sentinels → gap"]
    NORM --> PRI["Pick one source per metric<br/>temp/RH: SEN69C > SEN55 > SCD30 > SFA3X"]
    PRI --> TILES["Tiles shown only for metrics the device reports<br/>unknown fields → 'Other readings'"]
    PRI --> CSV[("_readings.csv<br/>new _partN if fields change")]

    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
    class IN1,IN2 radio
    class RX,HDR,NORM,PRI worker
    class KIND,PROF decision
    class TILES ui
    class CSV store
```

- The firmware for the S4 and SEN69C boards sends a **header line** first, so columns are mapped
  by name and no profile guessing is needed.

---

## 6. Starting and running everything

```mermaid
flowchart LR
    U(["You"]) --> L["./launch.sh polar atmos …<br/>foreground, Ctrl+C stops all"]
    U --> B["./biodash.py<br/>menu or CLI"]
    U --> R["&lt;dashboard&gt;/run_dashboard.sh<br/>one dashboard"]
    B -->|install| SVC["systemd user services<br/>biodash-&lt;name&gt;.service"]
    SVC --> SR["service_run.sh"]
    L --> RD["run_dashboard.sh (each)"]
    SR --> RD
    R --> RD
    RD --> VENV{".venv present?"}
    VENV -->|yes| PY1["use .venv/bin/python3"]
    VENV -->|no| PY2["system python3"]
    PY1 & PY2 --> PROC["start worker + web app"]
    L & SR -.->|"systemd-inhibit"| AWAKE["keep the computer awake<br/>(optional: lid close too)"]
    SVC -.->|"--boot (linger)"| BOOT["start at power-on, no login"]

    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef ui fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
    class U start
    class L,B,R ui
    class SVC,SR,RD file
    class VENV decision
    class PY1,PY2,PROC worker
    class AWAKE,BOOT app
```

- **`./setup_env.sh`** creates the `.venv` once; everything else uses it automatically.
- **Service options** (`biodash.py`): keep awake, lid close, auto-reconnect, shutdown/reboot from
  other devices, start at boot.
- **On the page:** ☀ *Keep screen on* (localhost or HTTPS), 🐞 debug panel with the System card
  (shutdown / reboot), header links between dashboards.
