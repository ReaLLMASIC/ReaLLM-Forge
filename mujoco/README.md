# Roomba MuJoCo demo pack

This folder contains the Roomba-like MuJoCo collector plus small scripts that demonstrate each major feature: randomized wandering, bump recovery, first-person rendering, viewer modes, GPU/offscreen rendering, camera mount hyperparameters, parallel collection, video output, PNG frame dumping, and compressed 16×16 grayscale CSV datasets.

## Contents

```text
roomba_mujoco_collect.py        Main simulator / data collector
requirements.txt                Python dependencies
scripts/                        Feature demo scripts
runs/                           Created by demos; output videos, CSVs, XML, frames
```

## Install

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/00_check_install.py
```

On macOS, the interactive MuJoCo viewer may need `mjpython` instead of `python`:

```bash
PYTHON=mjpython scripts/02_interactive_third_person_viewer.sh
```

## Run a basic demo

```bash
scripts/01_basic_headless_video_csv.sh
```

Expected outputs:

```text
runs/01_basic_headless_video_csv/
  roomba_room.xml       Generated MuJoCo XML
  roomba_fp.mp4         First-person annotated video
  dataset.csv.gz        Compressed CSV with 16x16 uint8 grayscale images
```

## Demo scripts

| Script | Demonstrates |
| --- | --- |
| `scripts/00_check_install.py` | Dependency/import check. |
| `scripts/01_basic_headless_video_csv.sh` | Default wandering, bump behavior, first-person MP4, compressed CSV, generated MJCF XML. |
| `scripts/02_interactive_third_person_viewer.sh` | Interactive MuJoCo viewer with a free third-person camera. |
| `scripts/03_interactive_first_person_viewer.sh` | Interactive viewer locked to the Roomba first-person camera. |
| `scripts/04_gpu_egl_headless.sh` | Linux headless GPU/offscreen rendering with `--gl egl`. |
| `scripts/05_camera_height_pitch_sweep.sh` | First-person camera height and pitch hyperparameters. |
| `scripts/06_gaussian_random_wander.sh` | More frequent Gaussian-randomized turns and command noise. |
| `scripts/07_bump_reverse_180.sh` | Forced wall bump, reverse, then roughly 180° turn. |
| `scripts/08_parallel_batch_collection.sh` | Multi-episode multiprocessing data collection. |
| `scripts/09_compressed_hex_csv.sh` | Compact `dataset.csv.gz` with one hex image field per row. |
| `scripts/10_wide_csv_pixels.sh` | `dataset.csv.gz` with `px_000` through `px_255` pixel columns. |
| `scripts/11_png_frame_dump.sh` | Full-resolution first-person PNG snapshots. |
| `scripts/12_csv_only_fast_collection.sh` | Faster CSV-only collection: lower render size, lower FPS, no MP4. |
| `scripts/13_custom_room_robot_geometry.sh` | Room, wall, robot, camera FOV, camera pose geometry settings. |
| `scripts/14_motion_control_hyperparams.sh` | Speed, reverse speed, turn speed, controller force/torque settings. |
| `scripts/15_no_annotations_video.sh` | Clean MP4 frames without action/timestamp overlay. |
| `scripts/16_quick_headless_suite.sh` | A short non-viewer smoke-test suite. |

## GPU/headless rendering

For a Linux GPU machine, run:

```bash
scripts/04_gpu_egl_headless.sh
```

or directly:

```bash
python roomba_mujoco_collect.py --gl egl --duration 60 --output-dir runs/gpu_run
```

`--gl auto` chooses EGL on Linux when not using the viewer. `--gl glfw` is usually the right choice for interactive viewer windows. `--gl osmesa` can be useful on CPU-only Linux systems if OSMesa is installed.

## Camera height and angle

The first-person camera is mounted at the middle of the robot top. These are the main camera hyperparameters:

```bash
--camera-height-above-top 0.06   # meters above robot top center
--camera-height 0.06             # alias for the same setting
--camera-pitch-deg 20            # positive tilts downward toward the floor
--camera-fovy 90                 # vertical field of view in degrees
```

Example:

```bash
python roomba_mujoco_collect.py \
  --gl egl \
  --duration 30 \
  --camera-height-above-top 0.08 \
  --camera-pitch-deg 25 \
  --output-dir runs/custom_camera
```

The camera convention is:

```text
0 degrees    looks forward horizontally along the Roomba heading
20 degrees   looks forward and downward
-10 degrees  looks slightly upward
```

## Behavior hyperparameters

Random turns are Gaussian-randomized:

```bash
--turn-interval-mean 4.0
--turn-interval-std 1.0
--wander-turn-mean-deg 35
--wander-turn-std-deg 18
--min-wander-turn-deg 8
```

Bump recovery backs up and turns around:

```bash
--reverse-seconds 0.70
--bump-turn-std-deg 8       # Gaussian noise around 180 degrees
--bump-debounce 0.45
```

Motion/control settings:

```bash
--speed 0.35
--reverse-speed 0.25
--turn-speed 1.35
--cmd-noise-v-std 0.015
--cmd-noise-omega-std 0.035
```

## Dataset format

Every recorded row contains:

```text
episode, frame, time_s, action_id, action, bumped, x_m, y_m, yaw_rad,
cmd_v_mps, cmd_omega_radps, image data
```

Default image format is compact hex:

```bash
--csv-image-format hex
```

That creates one `image16x16_gray_u8_hex` field per row. It is 512 hex characters representing 256 uint8 grayscale pixels in row-major order.

Use wide columns instead with:

```bash
--csv-image-format wide
```

That creates `px_000` through `px_255` columns.

Example decode for the default hex format:

```python
import gzip
import csv
import numpy as np

with gzip.open("runs/01_basic_headless_video_csv/dataset.csv.gz", "rt", newline="") as f:
    row = next(csv.DictReader(f))

img16 = np.frombuffer(bytes.fromhex(row["image16x16_gray_u8_hex"]), dtype=np.uint8).reshape(16, 16)
print(row["action"], img16.shape, img16.dtype)
```

## Parallel collection

```bash
scripts/08_parallel_batch_collection.sh
```

Equivalent direct command:

```bash
python roomba_mujoco_collect.py \
  --gl egl \
  --num-episodes 32 \
  --num-workers 8 \
  --duration 120 \
  --video-every 8 \
  --output-dir runs/parallel_batch
```

The script writes per-episode CSV shards during collection, then merges them into `dataset.csv.gz`. By default, shards are deleted after merging. Keep them with `--keep-shards`.

## Environment overrides for demo scripts

Most shell demos support simple environment overrides:

```bash
DURATION=5 scripts/01_basic_headless_video_csv.sh
GL=osmesa scripts/09_compressed_hex_csv.sh
OUT=/tmp/roomba_demo scripts/07_bump_reverse_180.sh
PYTHON=/path/to/python scripts/04_gpu_egl_headless.sh
```

Parallel demos also accept:

```bash
EPISODES=64 WORKERS=8 scripts/08_parallel_batch_collection.sh
```

## Troubleshooting

**`mujoco.FatalError` or OpenGL context errors on a server:** try `--gl egl` on a Linux GPU node, or `--gl osmesa` on CPU-only machines with OSMesa installed.

**Viewer opens but script cannot run on macOS:** use `mjpython` for the viewer scripts.

**MP4 writer errors:** confirm `imageio[ffmpeg]` is installed from `requirements.txt`.

**Large CSVs:** lower `--record-fps`, lower `--duration`, use `--csv-image-format hex`, or use `--no-video` for data-only collection.
