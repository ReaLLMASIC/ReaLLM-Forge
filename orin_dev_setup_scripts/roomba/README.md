# Orin Nano Roomba Controller

Web-dashboard teleop + autonomous line-following control for an iRobot
Roomba, driven from a Jetson Orin Nano over USB-serial using the real
iRobot Open Interface (OI) protocol. Also logs downsampled camera frames
+ telemetry to CSV for behavior-cloning / imitation-learning datasets.

## Hardware required

- **Roomba with Open Interface support** (Create, or a consumer Roomba
  with an accessible Mini-DIN serial port)
- **CP2102 USB-to-serial adapter**, wired to the Roomba's Mini-DIN port —
  this is also what the hardware wake-pulse feature uses (grounds pin 5 /
  BRC via the adapter's RTS line to wake a sleeping docked Roomba)
- **Jetson CSI camera module** (the video pipeline uses `nvarguscamerasrc`
  specifically — a USB webcam will not work without rewriting the
  GStreamer pipeline string in the code)

## Setup

```bash
./roomba-bootstrap.sh
```
Checks/installs `python3`/`pip3` themselves, the Python package deps
(`flask`, `numpy`, `pyserial`), checks OpenCV has the GStreamer support
this needs, and adds your user to the `dialout` group for serial port
access. See that script's own output for anything it flags as needing
attention — some of this (OpenCV/GStreamer in particular) it can only
check and report on, not fully auto-fix.

## Running it

```bash
./control-roomba
```
Then open `http://<orin-ip>:5000` in a browser on the same network.
`control-roomba` resolves its own location before running, so it works
the same way regardless of your current directory when you launch it —
via a relative path, an absolute path, a desktop shortcut, whatever.
It just launches `roomba-webui/wake-controller.py` — that's the one real
entry point, everything else in this folder is either an older version
or build-up history (see below).

## Files

| File | What it is |
|---|---|
| `control-roomba` | Entry point — launches `wake-controller.py` |
| `roomba-bootstrap.sh` | Environment setup — see Setup above |
| `roomba-webui/wake-controller.py` | **Current version.** Full dashboard: teleop, blue-line-follower autonomy, dataset logging, macro record/playback, closed-loop degree turns, hardware wake-pulse. Sensor polling runs on its own background thread (not inline per video frame), JPEG encoding for the video feed happens outside the frame lock, and the dataset CSV flushes periodically rather than only at clean shutdown — see comments in the file for why each of those matters |
| `roomba-webui/v2-controller.py` | Previous version — same core feature set, no wake-pulse, 32×32 pixel grid instead of 25×25. Kept for reference, not actively used |
| `roomba-webui/examples/` | Build-up history, roughly in order of complexity: `teleop_streamer.py` → `overlay_streamer.py` → `chase_streamer.py` → `simon-says.py` → `uni-controller.py`. Useful for understanding how a specific feature was added, not meant to be run as-is |
| `roomba-webui/v1-code/` | Older snapshots of `overlay_streamer.py` and `uni-controller.py` |

## Controls

| Key(s) | Action |
|---|---|
| `W` / `S` | Forward / backward |
| `A` / `D` | Spin axis (turn while driving) |
| `Space` | Emergency stop |
| `Q` / `E` | Decrease / increase target speed (50–600 mm/s) |
| `V` | Toggle vacuum |

Dashboard buttons cover: autonomous blue-line-follower mode, grayscale
"AI view" (shows exactly what the downsampled dataset frames look like),
manual hardware wake-pulse trigger, macro record/playback, fixed-degree
turns (90°/180°, closed-loop via wheel encoders), sticky-spin utilities,
and dock/cancel-dock.

## Dataset logging

While running, every frame gets logged to
`~/Documents/roomba_dataset_<timestamp>/dataset_25x25.csv`:

```
timestamp, frame_id, action, pressed_keys, bump_left, bump_right,
wheel_dropped, total_distance_mm, heading_deg, speed_mm_s,
left_wheel_v, right_wheel_v, battery_percent, p0, p1, ..., p624
```

`p0`–`p624` are a flattened 25×25 grayscale downsample of the camera
frame (625 pixel values, 0–255) — small enough to be a lightweight
per-frame feature vector for a small imitation-learning model, at the
cost of losing most spatial detail. The "AI Matrix View" dashboard toggle
shows you exactly this downsampled view live, so you can sanity-check
what the model would actually be trained on before collecting a real run.

## Safety behavior

- **Wheel-drop cutoff**: if a wheel-drop sensor stays triggered for 5+
  continuous seconds, the robot force-stops and drops out of autonomous
  mode — treat this as "pick the robot up" safety, not a soft warning.
- **Bump-triggered recovery**: in autonomous line-follower mode, a bump
  sensor triggers a stop, a short reverse, then a 180° turn in a random
  direction before resuming the search for the line.
- The hardware wake-pulse (RTS line toggle) is timed to iRobot's
  documented BRC pulse-width requirement (>500ms low) — don't shorten
  that timing without checking the OI spec, a too-short pulse won't
  reliably wake a sleeping Roomba.

## Known gotchas

- **OpenCV must have GStreamer support compiled in.** A `pip install
  opencv-python` will silently shadow the correct NVIDIA-provided system
  OpenCV and break the CSI camera pipeline. The script checks for this
  itself at startup and exits with a clear fix message if it's wrong —
  `roomba-bootstrap.sh` checks for this too, up front, so you find out
  before you're mid-session rather than after.
- **Serial port permissions**: your user needs to be in the `dialout`
  group to open `/dev/ttyUSB0` without `sudo`. `roomba-bootstrap.sh`
  handles this, but group membership changes need a fresh login (or
  `newgrp dialout`) to take effect in your current shell.
- The Roomba's serial device may not always enumerate as exactly
  `/dev/ttyUSB0` if other USB-serial adapters are plugged in — check
  `ls /dev/ttyUSB*` if the wake-pulse or drive commands silently do
  nothing.
