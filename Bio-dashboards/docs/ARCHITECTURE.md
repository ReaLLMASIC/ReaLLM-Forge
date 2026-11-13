# How Bio-dash works

Flow diagrams for the whole suite, from the devices to the files on disk. GitHub renders the
diagrams below; elsewhere, paste a block into <https://mermaid.live>.

1. [The big picture](#1-the-big-picture) — devices, radios, the one server, recordings
2. [Inside the server](#2-inside-the-server) — how a worker, its API and the page talk; several people; forecasts
3. [A device's life cycle](#3-a-devices-life-cycle) — scan, connect, record, reconnect
4. [Hydro-dash: the bottle protocol](#4-hydro-dash-the-bottle-protocol)
5. [Atmos: from any sensor format to tiles](#5-atmos-from-any-sensor-format-to-tiles)
6. [Starting and running everything](#6-starting-and-running-everything)
7. [Hydro-dash: which Bluetooth radio](#7-hydro-dash-which-bluetooth-radio) — the old-dongle catch
8. [Analysis: from recordings to one clock](#8-analysis-from-recordings-to-one-clock)
9. [Radios on the Orin: what plugs in where](#9-radios-on-the-orin-what-plugs-in-where) — every adapter tried, and the boot catch
10. [When something's wrong](#10-when-somethings-wrong) — a troubleshooting flowchart

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

**One app.** A single server (`server.py`, one port, one service) runs a Bluetooth **worker** for
each device you turned on, serves each device's API, and sends live readings to every open page
over one WebSocket. The workers are the same programs as in the separate dashboards (V5 – V7) and
write the same recordings.

```mermaid
flowchart TB
    subgraph DEV["Devices"]
        direction LR
        H10["Polar H10<br/>ECG · ACC · HR · R-R"]
        O2["Checkme O2 ring<br/>SpO₂ · pulse · battery"]
        MU["Muse S Athena<br/>EEG · fNIRS · PPG · motion"]
        HS["HidrateSpark<br/>sips · weight · cap"]
        ATM["Atmos sensors<br/>Mini · Sphere S4/S5"]
    end

    subgraph RAD["Bluetooth radios (BlueZ)"]
        direction LR
        EXT["USB dongle<br/>preferred by the Polar"]
        INT["Built-in radio<br/>preferred by the others"]
    end

    subgraph SRV["server.py · port 5000"]
        subgraph WK[" "]
            direction LR
            PW["Heart worker<br/>advanced_worker.py"]
            VW["Oxygen worker<br/>ble_worker.py"]
            MW["Brain worker<br/>muse_worker.py"]
            HW["Water worker<br/>hydro_worker.py"]
            AW["Air worker<br/>sensor_worker.py"]
        end
        API["Device APIs<br/>/api/d/heart · oxygen · brain · air · water · analysis"]
        FC["forecast_service.py<br/>(optional: PyTorch)"]
        HUB["Live hub<br/>one WebSocket /ws"]
    end

    H10 --> EXT --> PW
    O2 & HS & MU --> INT
    ATM -->|BLE| INT
    INT --> VW & HW & AW & MW
    ATM -.->|Wi-Fi TCP :8080| AW
    WK --> REC[("~/Documents/Bio-dash<br/>CSV per connection")]
    REC --> API --> HUB
    REC --> FC -->|"the next second"| API
    API & HUB --> BR["The app in a browser<br/>computer · phone (installable)"]

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
    class H10,O2,ATM,HS,MU device
    class EXT,INT radio
    class PW,VW,AW,HW,MW,FC worker
    class API,HUB app
    class BR ui
    class REC store
    style DEV fill:none,stroke:#F59E0B,stroke-width:1.5px
    style RAD fill:none,stroke:#8B5CF6,stroke-width:1.5px
    style SRV fill:none,stroke:#94A3B8,stroke-width:1.5px
    style WK fill:none,stroke:#2563EB,stroke-width:1px,stroke-dasharray:4 3
```

- **Only the devices you own run.** The first-run setup (or Settings → Devices) turns each one on
  or off; that starts or stops its worker. The choice lives in `~/.config/bio-dash/app.json`.
- **A worker that crashes is restarted** after 2 s, then 4, 8 … up to 60 s; a run that lasted two
  minutes resets the pause. Its last 400 lines are kept for the Device sheet's *Troubleshooting*.
- **Radio routing:** each device picks a radio by its *address* (stable across reboots) from
  `radio_config.json`, or automatically: the Polar prefers a USB dongle, the rest the built-in radio.
- **Several people** (up to three) each get a Heart and an Oxygen worker of their own; see section 2.
  This replaces Omni, whose old recordings still show up in Analysis.
- **Forecasts** are a separate process (`polar_dashboard/forecast_service.py`), started when a model
  is installed: it reads the first person's ECG as it is recorded and posts the next second to the
  Heart API, which passes it to the open Heart pages.

---

## 2. Inside the server

Workers are separate processes that share a few small files with the server, so a stuck Bluetooth
stack can be restarted without taking the app down.

```mermaid
flowchart TB
    DEV(("Device")) <-->|Bluetooth / TCP| W

    subgraph D["e.g. hydro_dashboard/"]
        W["Worker<br/>hydro_worker.py"]
        S[/"logs_hydro/status.json<br/>state · live values · debug"/]
        DV[/"devices.json<br/>last scan"/]
        C[/"command.json<br/>connect · calibrate · glow"/]
        CFG[/"radio_config.json · last_device.json"/]
        A["api.py<br/>(FastAPI, mounted at /api/d/water)"]
        W -->|every second| S
        W -->|after each scan| DV
        C -->|picked up within 1 s| W
        CFG <--> W
        S & DV --> A
        A -->|Connect, Calibrate…| C
        A <--> CFG
    end

    S -->|"read once a second;<br/>pushed only when it changes"| HUB["Live hub (server.py)"]
    A -->|"Heart, Air: rows tailed from the CSV<br/>Oxygen: data.json"| HUB
    HUB <-->|"/ws: status · latest (1/s) · live (subscribed)"| P["The app<br/>web/ (no build step)"]
    A <-->|"/api/d/water/api/…"| P
    W -->|"CSV rows"| R[("Documents/Bio-dash/&lt;device&gt;/&lt;unit&gt;/&lt;date&gt;/")]
    R -->|"history · summaries"| A

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
    class A,HUB app
    class P ui
    class R store
    style D fill:none,stroke:#94A3B8,stroke-width:1.5px
```

- **One WebSocket per page.** Every page gets each device's status when it changes and a summary
  of the newest readings once a second (Home and the nav dots need nothing else). A device page
  *subscribes* to that device's full stream (130 Hz ECG, every air row) and unsubscribes when you
  leave, so a phone on Home isn't sent the ECG.
- **Nothing is parsed for nobody:** with no page open, the server stops tailing the ECG.
- **Analysis runs apart:** its summaries, charts and export run in a separate process that exits
  a minute after the last request, so a week of parsed recordings doesn't stay in the server.
- **History** (trends, today's water, Analysis) is read back from the recordings, so it survives
  restarts. Finding the newest recording only looks in the newest date folders.
- **Several people.** A worn device (strap, ring) exists once per person. The first person's are
  `heart` and `oxygen`, with their state files in the device folder as always. Each extra person
  (`p2`, `p3`) gets one more worker and one more copy of the same `api.py`, as `heart-p2`,
  `oxygen-p2` …, with `BIODASH_STATE_DIR` pointing at `~/.config/bio-dash/people/<id>/<device>/`
  for their status, command, last-device and radio files (`bt_debug.state_dir`). Nothing else
  differs: the same scan → connect → record loop, the same recordings by device.
  - A worker only ever connects to the device it was told to (or last used), so two workers never
    grab the same strap. Connecting a device for one person first makes whoever had it let go
    (`POST /api/app/claim`).
  - An API follows *its own worker's* recording (the worker lists its files in its status), not
    "the newest file", so two people's live charts and history never cross.
  - With two or more people the server shares the radios out every 10 s: straps take the USB
    dongles first, rings start from the other end; each worker reads its share from its
    `radio_config.json` and follows it unless you pinned a radio.
- **Shared code:** one `bt_debug.py` in the release folder (radios, auto-reconnect, recordings,
  shutdown/reboot, updates). The page is plain ES modules in `web/`: a 7 kB (compressed) canvas chart
  (`plot.js`) replaces Chart.js + luxon + the streaming plugin.

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
- **Auto-reconnect** can be switched off, or the device forgotten, from each device's **Device** sheet
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
    U(["You"]) --> L["./launch.sh<br/>in a terminal, Ctrl+C stops"]
    U --> D["./launch.sh --demo<br/>made-up readings, no Bluetooth"]
    U --> B["./biodash.py<br/>a few questions, or CLI"]
    B -->|install| SVC["systemd user service<br/>biodash-app.service"]
    SVC --> SR["service_run.sh"]
    L & D --> SR
    SR --> VENV{".venv present?"}
    VENV -->|yes| PY1["use .venv/bin/python3"]
    VENV -->|no| PY2["system python3"]
    PY1 & PY2 --> PROC["server.py<br/>→ a worker per device that's on"]
    PROC -.->|"systemd-inhibit, held while it runs"| AWAKE["keep the computer awake<br/>(optional: lid close too)"]
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
    class L,D,B ui
    class SVC,SR file
    class VENV decision
    class PY1,PY2,PROC worker
    class AWAKE,BOOT app
```

- **`./setup_env.sh`** creates the `.venv` once; everything else uses it automatically.
- **Service options** (`biodash.py install`): `--port`, keep awake, `--lid`, `--no-reconnect`,
  `--allow-power` (shut down / update from other devices), `--boot`.
- **Stopping** sends the server SIGTERM (`KillMode=mixed`); it gives each worker Ctrl+C so the
  recordings are closed properly, then exits.
- **From the separate dashboards (V5 – V7):** `biodash.py install` removes the old `biodash-<dashboard>` services and turns on the
  devices they ran, which is exactly what their *Upgrade* button runs.

---

## 7. Hydro-dash: which Bluetooth radio

The bottle is the one device in the suite that is picky about the radio. It has to be **heard**
(it only advertises for a few seconds after it's moved) and it has to get through **reading its
service list**, and the two radios a laptop usually has each fail a different half.

```mermaid
flowchart TB
    B(("HidrateSpark<br/>bottle")) --> Q{"Which radio is<br/>Hydro-dash using?"}

    Q -->|"laptop's built-in<br/>(shared with Wi-Fi)"| I["Built-in radio"]
    Q -->|"old USB dongle<br/>(Bluetooth 4.0)"| O["Old dongle"]
    Q -->|"newer USB dongle<br/>(Bluetooth 5)"| N["Bluetooth 5 dongle"]

    I --> I1["Often can't hear the bottle:<br/>'not seen', connect timeouts"]
    I1 -->|when it does connect| OK

    O --> O1["Hears the bottle straight away"]
    O1 --> M{"Message size the<br/>computer asks for"}
    M -->|"default (large)"| X["Bottle agrees to 200 bytes, then never<br/>sends its first large reply.<br/>'Connected' with no services,<br/>firmware unknown"]
    M -->|"capped: ExchangeMTU = 64<br/>in /etc/bluetooth/main.conf"| OK
    X -.->|add the cap,<br/>restart bluetooth| M

    N --> N1["Hears it, carries large replies whole,<br/>no cap needed. UB500 Plus on the Orin:<br/>10/10 cold connects, −60 dBm"]
    N1 --> OK

    OK(["Services read:<br/>sips, fill level and glow work"])
    CAP[/"The cap applies to every device on this<br/>computer. Check the Polar H10's ECG<br/>with it on before keeping it."/]
    M -.-> CAP

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
    class B device
    class I,O,N radio
    class Q,M decision
    class I1,X warn
    class O1,N1 worker
    class OK start
    class CAP file
```

- **What it looks like:** the scanner finds the bottle, the page says connected, and nothing else
  arrives: no firmware version, no battery, no sips.
- **Why:** an old dongle sends 27 bytes per radio packet, so a large reply has to be cut into many
  pieces, and this bottle doesn't send it at all. With small messages every reply fits.
- **The bottle's address changes** from time to time. Each change makes the computer read the
  service list afresh, so this can appear on a setup that worked earlier the same day.
- **Choosing the radio:** the device's **Device** sheet → Bluetooth radio. The choice is saved by the radio's
  address, so it survives reboots (when hci numbers shuffle) and upgrades. Without a choice,
  Hydro-dash prefers a built-in radio; on boards whose USB ports don't say which radios are built in
  (the Orin Nano dev kit), that falls back to whichever radio is hci0.
- Step by step, with the commands: `hydro_dashboard/TROUBLESHOOTING.md`, sections 11 to 15.
- Every adapter tried so far, and why it did or didn't work: [section 9](#9-radios-on-the-orin-what-plugs-in-where).

---

## 8. Analysis: from recordings to one clock

Analysis has no worker and no radio: it reads the CSV recordings the other
dashboards wrote, so it runs anywhere those files are.

```mermaid
flowchart LR
    R[("Documents/Bio-dash/<br/>&lt;dashboard&gt;/&lt;device&gt;/&lt;date&gt;/*.csv")] --> L["biodata.py<br/>read · clean · merge"]

    L --> SIG["Named signals<br/>heart.hr · oxygen.spo2 · air.co2@… ·<br/>water.fill · water.sips · water.total · raw.*"]
    L --> SUM["Summaries<br/>recording strips · device cards · day by day"]

    SIG --> K{"What kind<br/>of signal?"}
    K -->|"readings<br/>HR, SpO₂, CO₂"| M["mean · min · max per step<br/>straight line across short gaps"]
    K -->|"state<br/>fill level"| H["last value<br/>carried forward"]
    K -->|"events<br/>sips"| E["mL per step<br/>+ running daily total"]

    M & H & E --> G["One clock<br/>each value + n real readings"]
    G --> P["Charts<br/>one per signal, shared time axis"]
    G --> C[/"CSV export<br/>page button or tools/interpolate.py"/]
    G --> MC[/"Integer CSV for ReaLLM-Forge csv_mc_int<br/>page button or tools/export_mcint.py"/]
    SUM --> P2["Analysis page<br/>/api/d/analysis"]
    P --> P2
    R --> N["tools/night_report.py"] --> NR[/"Overnight report<br/>linked from the Oxygen card"/]

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
    class R store
    class L,SIG worker
    class K decision
    class M,H,E worker
    class G app
    class P,P2,SUM ui
    class C,MC,NR file
    class N worker
```

- **Nothing is changed on disk:** the dashboard and the tool only read the recordings (exports and
  reports are written to their own `Exports/` and `Reports/` folders).
- **Signals are merged across dashboards:** heart rate recorded by Polar-dash in the day and by
  Omni at night is one signal.
- **Estimates are marked:** every value carries `n`, the number of real readings behind it;
  0 means interpolated, carried forward, or "no sip in this step" (exports keep the count in `.n` columns).
- **The ReaLLM-Forge export** (`analysis_dashboard/mcint.py`) writes the same table as whole numbers:
  a fixed scale and range per signal, `minute_of_day` / `second` for the time, and one value below
  each range (`-1`) for "no reading", because `csv_mc_int` allows no blank cells.

### Just the ECG (`tools/export_ecg.py`), step by step

The raw file keeps every sample as recorded; the 200 Hz file is resampled stretch by stretch and
never across a gap. With `--with-acc` the accelerometer (`_acc.csv`, X/Y/Z mg, 200 Hz) goes through
the same steps 1-3 on its own clock, is interpolated at each 200 Hz ECG row's time and added as
`ACC_X_mg,ACC_Y_mg,ACC_Z_mg`; a row with no unbroken motion either side ends the stretch there.

```mermaid
flowchart TB
    R[("ECG recordings on disk<br/>Polar H10/&lt;strap&gt;/&lt;date&gt;/&lt;time&gt;#95;ecg.csv<br/>Omni/&lt;strap&gt;/&lt;date&gt;/&lt;time&gt;#95;ecg.csv<br/>Timestamp_Epoch_ms, ECG_mV · 130 Hz · never changed")]

    subgraph S1["1 · Read and merge"]
        direction TB
        A1["Pick the files<br/>day or --from/--to · --device"]
        A2["Read each file<br/>zero bytes stripped · bad rows skipped<br/>only the ECG_mV column"]
        A3["Merge in time order<br/>(heapq merge of all files)"]
        A4{"Less than half a period<br/>(3.8 ms) after the<br/>last kept sample?"}
        A5["Keep"]
        X1["Dropped<br/>same strap recorded twice<br/>(Polar-dash + Omni)"]
        A1 --> A2 --> A3 --> A4
        A4 -->|no| A5
        A4 -->|yes| X1
    end

    RAW[/"&lt;name&gt;_raw.csv<br/>Timestamp_Epoch_ms, ECG_mV<br/>every sample, values unchanged"/]

    subgraph S2["2 · Find unbroken stretches"]
        direction TB
        B1{"Step to the next sample<br/>&gt; 3 periods (~23 ms)<br/>or backwards?"}
        B2["Same stretch"]
        B3["New stretch<br/>gap noted"]
        B4{"Stretch shorter than<br/>--min-stretch (10 s)?"}
        X2["Left out"]
        B1 -->|no| B2
        B1 -->|yes| B3
        B2 & B3 --> B4
        B4 -->|yes| X2
    end

    subgraph S3["3 · Resample each stretch to 200 Hz"]
        direction TB
        C1["Sample times from position<br/>t_i = first + i × measured spacing<br/>(removes the 7/8 ms rounding jitter)"]
        C2{"--method"}
        C3["linear<br/>never overshoots"]
        C4["cubic (Catmull-Rom)<br/>~3× closer on QRS peaks"]
        C5["Every 5 ms from the stretch start<br/>nothing invented across gaps"]
        C1 --> C2
        C2 --> C3 & C4
        C3 & C4 --> C5
    end

    U{"--uv?"}
    U1["mV, up to 4 decimals"]
    U2["whole µV, clipped ±10000<br/>prints csv_mc_int --range"]

    HZ[/"&lt;name&gt;_200hz.csv<br/>ECG only, no timestamp<br/>row k = k/200 s into its stretch"/]
    ST[/"&lt;name&gt;_200hz_stretches.csv<br/>first row · rows · start time<br/>gap before · recorded Hz"/]
    SP[/"--split: one file per stretch<br/>&lt;name&gt;_200hz_001.csv …"/]

    R --> A1
    A5 --> RAW
    A5 --> B1
    B4 -->|no| C1
    C5 --> U
    RAW -.->|"--uv applies here too"| U
    U -->|no| U1 --> HZ
    U -->|yes| U2 --> HZ
    C5 --> ST
    HZ -.->|--split| SP

    W1["⚠ the 200 Hz file has no timestamps:<br/>the stretches file is the only record<br/>of where the gaps were"]
    ST -.- W1

    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef gone fill:#F8FAFC,stroke:#94A3B8,color:#64748B,stroke-dasharray:4 3
    class R store
    class A1,A2,A3,A5,B2,B3,C1,C5 worker
    class A4,B1,B4,C2,U decision
    class C3,C4,U1,U2 app
    class RAW,HZ,ST,SP file
    class W1 warn
    class X1,X2 gone
    style S1 fill:#F8FAFC,stroke:#CBD5E1
    style S2 fill:#F8FAFC,stroke:#CBD5E1
    style S3 fill:#F8FAFC,stroke:#CBD5E1
```

### The ReaLLM-Forge export, step by step

Everything happens when you export; the recordings are never changed. Red notes are the things to
know before training on it (`--max-gap 0 --hold 0 --real-flags` keeps it to real readings).

```mermaid
flowchart TB
    R[("Recordings on disk<br/>&lt;dashboard&gt;/&lt;device&gt;/&lt;date&gt;/*.csv<br/>never changed")]

    subgraph S1["1 · Read: load_signals"]
        direction TB
        A1["Clean<br/>zero bytes stripped · bad rows skipped<br/>timestamps → ms"]
        A2{"Sane range?<br/>HR 25–230 · RR 300–2000<br/>SpO₂ 50–100 · sip 1–2000 mL"}
        A3["Merge dashboards<br/>Polar + Omni = one heart rate<br/>one value per timestamp"]
        A4["Derived signals<br/>HR = 60000/RR (Omni)<br/>RMSSD per minute (skip &gt;20 % jumps)<br/>sips de-duplicated (±10 mL, 2 s) + daily total"]
        A1 --> A2
        A2 -->|yes| A3 --> A4
        A2 -->|no| X1["dropped"]
    end

    subgraph S2["2 · One clock: resample"]
        direction TB
        K{"Kind of signal<br/>(one value per --step)"}
        M["Readings<br/>mean of the step<br/>empty step: straight line<br/>if gap ≤ --max-gap (60 s)"]
        H["Levels (bottle fill)<br/>last value carried<br/>up to 30 min"]
        E["Sips<br/>mL per step · 0 while recording<br/>daily total held to midnight"]
        K --> M & H & E
    end

    subgraph S3["3 · Integers: mcint.build"]
        direction TB
        B1["Scale · round · clip to a fixed range<br/>temp_c_x10 −400…850, hr_bpm 0…250 …"]
        B2["Hold readings across gaps<br/>≤ --hold (2 min)"]
        B3{"Still empty?"}
        B4["−1  (one below the range)<br/>or --missing drop: row left out"]
        B5["minute_of_day (+ second)<br/>rows with no device at all left out"]
        B1 --> B2 --> B3
        B3 -->|yes| B4 --> B5
        B3 -->|no| B5
    end

    OUT[/"Export folder<br/>&lt;name&gt;.csv · .columns.json · .prepare.sh"/]

    subgraph S4["4 · Forge: get_dataset.sh"]
        direction TB
        F1["token = value − int_min<br/>one vocabulary per column"]
        F2["split by row order<br/>first 90 % train · last 10 % val"]
        F1 --> F2
    end

    BIN[("train.bin · val.bin · meta.pkl<br/>per column + manifest.json")]

    R --> A1
    A4 --> K
    M & H & E --> B1
    B5 --> OUT --> F1
    F2 --> BIN

    W1["⚠ up to ~3 min of each gap is made up<br/>(60 s line + 2 min hold) · --real-flags marks it<br/>--max-gap 0 --hold 0 for real data only"]
    W2["⚠ averaging hides beat-to-beat R-R<br/>at 10 s steps · use 1 s or hrv_rmssd_ms"]
    W3["⚠ time jumps where rows are left out<br/>no date column · DST repeats/skips an hour"]
    M -.- W1
    B2 -.- W1
    M -.- W2
    B5 -.- W3

    classDef store fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    classDef file fill:#F1F5F9,stroke:#64748B,color:#1E293B,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef gone fill:#F8FAFC,stroke:#94A3B8,color:#64748B,stroke-dasharray:4 3
    class R,BIN store
    class A1,A3,A4,M,H,E,B2,B5,F1,F2 worker
    class A2,K,B3 decision
    class B1,B4 app
    class OUT file
    class W1,W2,W3 warn
    class X1 gone
    style S1 fill:#F8FAFC,stroke:#CBD5E1
    style S2 fill:#F8FAFC,stroke:#CBD5E1
    style S3 fill:#F8FAFC,stroke:#CBD5E1
    style S4 fill:#F8FAFC,stroke:#CBD5E1
```

---

## 9. Radios on the Orin: what plugs in where

Every Bluetooth device gets its own radio (one radio streaming a busy device, like the Polar's ECG,
can starve the others). On the Jetson Orin Nano that's four radios for four devices:

```mermaid
flowchart LR
    BOT(("HidrateSpark<br/>bottle")) --> UB["TP-Link UB500 Plus<br/>Bluetooth 5.1 · USB"]
    POL(("Polar H10")) --> INT["Orin built-in<br/>Realtek · Bluetooth 5.1"]
    O2(("Viatom O2 ring")) --> B1["ASUS USB-BT400 #1<br/>Broadcom · Bluetooth 4.0"]
    ATM(("Atmos monitor")) --> B2["ASUS USB-BT400 #2<br/>Broadcom · Bluetooth 4.0"]
    ATM -.->|"or Wi-Fi,<br/>freeing a radio"| WIFI["Wi-Fi (TCP)"]

    UB --> HY["Hydro-dash"]
    INT --> PO["Heart (Polar)"]
    B1 --> VI["Oxygen (O2 ring)"]
    B2 --> AT["Atmos-dash"]
    WIFI --> AT


    classDef device fill:#FEF3C7,stroke:#D97706,color:#78350F,stroke-width:2px
    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef app fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2px
    class BOT,POL,O2,ATM device
    class UB,INT,B1,B2,WIFI radio
    class HY,PO,VI,AT app
```

- **The bottle goes on the UB500 Plus.** It's the one device that needs a Bluetooth 5 radio
  (section 7). The O2 ring and the Atmos send small, slow packets that a 4.0 dongle handles fine.
- **The Polar gets a modern radio to itself** because its ECG is the busiest stream.
- **Each device pins its radio by address** in its **Device** sheet → Bluetooth radio
  (saved in `radio_config.json`, carried across upgrades). The hci numbers shuffle whenever
  dongles are added or the Orin boots, and the address is what stays put.

### The boot catch: what's plugged in at power-on

```mermaid
flowchart TB
    P(["Power on"]) --> FW["Jetson boot firmware<br/>(before Linux, before the screen)<br/>scans every USB device"]
    FW --> Q{"A UB500 Plus AND an<br/>ASUS BT400 attached?"}
    Q -->|yes| HANG["Hangs: no NVIDIA logo, no screen,<br/>never reaches Linux<br/>(charger shows a few watts)"]
    Q -->|"no: UB500 alone,<br/>or ASUS dongles alone"| LNX["Linux starts"]
    HANG -.->|"fix: plug the ASUS dongles in<br/>after boot (or a switched hub)"| LNX
    LNX --> FWL["Linux loads each dongle's firmware into its RAM<br/>UB500: rtl_bt/rtl8761bu_fw.bin<br/>ASUS: brcm/BCM20702A1-0b05-17cb.hcd → build 1467"]
    FWL --> RUN(["Dashboards pin their radios by address"])

    classDef radio fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px
    classDef worker fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px
    classDef decision fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px
    classDef warn fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D,stroke-width:2px
    classDef start fill:#1E293B,stroke:#94A3B8,color:#F8FAFC,stroke-width:2px
    class P,RUN start
    class FW,LNX worker
    class FWL radio
    class Q decision
    class HANG warn
```

- **Not power:** two ASUS dongles together boot fine, and all four dongles add about 1 W once
  running (7–8 W total, no over-current messages). It's that particular pair at power-on.
- **The dongles' firmware can't fix it.** Neither chip has flash for its firmware: they run what's
  built into the chip, and Linux uploads the improved firmware into their RAM at every power-up,
  after the boot firmware has already met them.
- The Orin's boot firmware here was L4T 36.5.2, the newest of the JetPack 6 line (August 2026).

### Every adapter tried, and why it did or didn't work

| Adapter | Chip · USB ID | Bluetooth | Firmware | With the bottle | Notes |
|---|---|---|---|---|---|
| **TP-Link UB500 Plus** | Realtek RTL8761BU · `2357:0604` (same ID as the plain UB500) | reports 5.1 (box says 5.3) | `rtl_bt/rtl8761bu_fw.bin` + `_config.bin`, loaded by the kernel itself on JetPack 6 (kernel 5.15); no `btusb new_id` needed | ✅ **Works.** 10 of 10 cold connects with all 5 HidrateSpark characteristics; heard 8 times in 25 s at −60 dBm; Hydro-dash picked it by itself after a reboot; no `ExchangeMTU` cap | Antenna plugs in with an **MHF4** connector (the small one, not U.FL): use an MHF4-to-SMA pigtail to move it outside a case |
| **ASUS USB-BT400** | Broadcom BCM20702A1 · `0b05:17cb` | 4.0 | built-in "build 0000"; Linux needs `brcm/BCM20702A1-0b05-17cb.hcd` to bring it to **build 1467** (TROUBLESHOOTING 15) | ⚠ **Hears it instantly, then "connected, no services"**: 4.0 sends 27 bytes per packet and the bottle never sends its first large reply. Works only with `ExchangeMTU = 64`, which affects every device | Fine for the O2 ring and Atmos (small packets). **Stops the Orin booting** if attached together with a UB500 Plus at power-on (above) |
| **Orin Nano built-in** | Realtek 8822-series combo card, Realtek's own `rtk_btusb` driver | 5.1 | built in | not tried with the bottle this round | Its USB port doesn't say it's built in, so the app can't tell it's built in |
| **Laptop built-in** (FW13) | combo card shared with Wi-Fi | 5.x | built in | ❌ **Often doesn't hear it**: the bottle only advertises for a few seconds after it moves, and the shared antenna misses those bursts | TROUBLESHOOTING 11 |

**Powering the Orin:** a 12 V USB-C "trigger" board works only with chargers that offer 12 V,
which USB Power Delivery doesn't require. Any charger over 45 W must offer 20 V, the Orin's normal
input, so a 20 V trigger with a 65 W charger is the dependable choice.

---

## 10. When something's wrong

Start at the top and follow what you see. Each answer points at the fix, and at the place in the
app or the docs that says more.

```mermaid
flowchart TD
    START(["Something isn't working"]) --> PAGE{"Does the app open?<br/>http://&lt;computer&gt;.local:5000"}
    PAGE -- no --> SRV["Bio-dash isn't running<br/>./biodash.py status · ./biodash.py logs<br/>missing packages: ./setup_env.sh"]
    PAGE -- yes --> LIVE{"Does anything update live?<br/>(top bar says Reconnecting…)"}
    LIVE -- no --> WS["No live connection<br/>WebSocket package missing: ./setup_env.sh<br/>(installs uvicorn[standard])"]
    LIVE -- yes --> PILL{"What does the device's status say?"}

    PILL -- "Off" --> OFF["Turn it on<br/>Settings → Devices"]
    PILL -- "Restarting…" --> CRASH["Its Bluetooth worker keeps stopping<br/>Device → Troubleshooting: recent messages<br/>often BlueZ: sudo systemctl restart bluetooth"]
    PILL -- "Looking… (never found)" --> FIND{"Is it in the<br/>Find a device list?"}
    FIND -- no --> AWAKE["Wake it and free it<br/>strap: wet the pads · ring: on a finger · bottle: tip it<br/>close its phone app (one connection at a time)<br/>still nothing: tools/ble_scan.py, try the other radio"]
    FIND -- yes --> TAP["Tap Connect<br/>or turn on Reconnect automatically"]
    PILL -- "Connecting (try N of 3) fails" --> LINK["Weak or busy radio<br/>signal under −85 dBm: move closer<br/>Device → Bluetooth radio: pin the USB dongle<br/>sudo systemctl restart bluetooth"]
    PILL -- "Recording, with a warning" --> WARN{"What does the warning say?"}
    WARN -- "invalid MTU" --> MTU["Packet size capped<br/>comment out ExchangeMTU in /etc/bluetooth/main.conf<br/>restart bluetooth · Device → Restart"]
    WARN -- "busy" --> BUSY["Something else is streaming from the strap<br/>old V5 service: systemctl --user stop 'biodash-*'<br/>Polar Beat app, another computer"]
    PILL -- "Recording, numbers look wrong" --> WRONG{"Which device?"}
    WRONG -- Air --> PROF["Readings in the wrong boxes<br/>Air → Device options → pick the Atmos model"]
    WRONG -- Water --> CAL["Fill level off<br/>Water → Bottle → Calibrate the level"]
    WRONG -- "Heart: ECG signal low" --> PADS["Poor contact: wet the pads,<br/>tighten the strap"]

    START --> BOOT{"Orin doesn't power on<br/>(no logo)?"}
    BOOT -- yes --> CATCH["Unplug the ASUS dongles, boot,<br/>plug them in after · see the boot catch (section 9)"]

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
    class PAGE,LIVE,PILL,FIND,WARN,WRONG,BOOT decision
    class SRV,WS,CRASH,LINK,MTU,BUSY,CATCH warn
    class OFF,TAP,PROF,CAL,PADS,AWAKE app
```

- **Every fix in the app is one or two taps from the device's page:** the status pill, the
  *Connect your …* panel with its *Not showing up?* tips, and the **Device** sheet (radio,
  reconnect, *Troubleshooting* with the worker's recent messages and *Copy details*).
- **Device-specific detail:** the bottle has its own guide, `hydro_dashboard/TROUBLESHOOTING.md`;
  the radios and the Orin's boot catch are in section 9.
