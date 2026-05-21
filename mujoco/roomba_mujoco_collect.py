#!/usr/bin/env python3
"""
Roomba-like MuJoCo room simulation + first-person dataset collector.

Features
--------
- Planar Roomba 500-series-like robot in a rectangular room.
- Wanders forward, makes Gaussian-randomized turns every few seconds.
- On wall bump: backs up, then rotates about 180 degrees with Gaussian noise.
- Optional interactive MuJoCo viewer.
- First-person camera mounted at the middle of the robot top.
- Saves annotated first-person MP4 video.
- Saves compressed CSV/GZip dataset with 16x16 uint8 grayscale observations.
- Multi-process episode collection for throughput.

Install
-------
    pip install mujoco numpy pillow imageio[ffmpeg]

Headless GPU rendering on Linux, typical NVIDIA machine:
    python roomba_mujoco_collect.py --gl egl --duration 60 --output-dir runs/roomba

Interactive viewer:
    python roomba_mujoco_collect.py --view --gl glfw --duration 60

On macOS, MuJoCo's passive viewer may need:
    mjpython roomba_mujoco_collect.py --view --duration 60
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class Action:
    FORWARD = "forward"
    BACKWARD = "backward"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"


ACTION_ID = {
    Action.FORWARD: 0,
    Action.BACKWARD: 1,
    Action.ROTATE_LEFT: 2,
    Action.ROTATE_RIGHT: 3,
}


@dataclasses.dataclass(frozen=True)
class Config:
    # Runtime / output.
    output_dir: str
    csv_path: str
    seed: int
    duration: float
    num_episodes: int
    num_workers: int
    keep_shards: bool
    print_every: int

    # MuJoCo / rendering.
    gl_backend: str
    dt: float
    width: int
    height: int
    record_fps: float
    view: bool
    view_camera: str
    viewer_fps: float
    real_time: bool
    realtime_factor: float
    save_video: bool
    video_path: str
    video_every: int
    video_codec: str
    video_quality: int
    annotate_video: bool
    save_frame_images_every: int
    save_xml_path: str

    # CSV dataset.
    save_csv: bool
    csv_image_format: str

    # World and robot geometry.
    room_size: float
    wall_height: float
    wall_thickness: float
    robot_radius: float
    robot_height: float
    robot_mass: float
    camera_fovy: float
    camera_height_above_top: float
    camera_pitch_deg: float
    randomize_start: bool

    # Motion / controller.
    speed: float
    reverse_speed: float
    turn_speed: float
    control_kv: float
    max_force: float
    max_torque: float
    cmd_noise_v_std: float
    cmd_noise_omega_std: float

    # Behavior randomization.
    turn_interval_mean: float
    turn_interval_std: float
    wander_turn_mean_deg: float
    wander_turn_std_deg: float
    min_wander_turn_deg: float
    bump_turn_std_deg: float
    reverse_seconds: float
    bump_debounce: float


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(
        description="MuJoCo Roomba-like room simulation, video, and 16x16 grayscale CSV data collection."
    )

    # Runtime / output.
    p.add_argument("--output-dir", default="roomba_mujoco_runs", help="Directory for videos, CSV shards, XML, frames.")
    p.add_argument("--csv-path", default="", help="Final merged CSV path. Default: OUTPUT_DIR/dataset.csv.gz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=60.0, help="Seconds per episode.")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=1, help="Parallel worker processes. Headless collection only.")
    p.add_argument("--keep-shards", action="store_true", help="Keep per-episode CSV shards after merging.")
    p.add_argument("--print-every", type=int, default=1, help="Print one line every N completed episodes. 0 disables.")

    # MuJoCo / rendering.
    p.add_argument(
        "--gl",
        choices=["auto", "egl", "glfw", "osmesa"],
        default="auto",
        help="OpenGL backend. Use egl for GPU-accelerated headless rendering on Linux; glfw for viewer.",
    )
    p.add_argument("--dt", type=float, default=0.005, help="MuJoCo physics timestep.")
    p.add_argument("--width", type=int, default=320, help="First-person camera image width.")
    p.add_argument("--height", type=int, default=240, help="First-person camera image height.")
    p.add_argument("--record-fps", type=float, default=30.0, help="FPS for video and CSV image samples.")
    p.add_argument("--view", action="store_true", help="Open interactive MuJoCo viewer for one episode.")
    p.add_argument(
        "--view-camera",
        choices=["free", "roomba_fp"],
        default="free",
        help="Viewer camera. free gives a third-person room view; roomba_fp mirrors the top-mounted robot camera.",
    )
    p.add_argument("--viewer-fps", type=float, default=60.0)
    p.add_argument("--real-time", action="store_true", help="Throttle simulation to wall-clock time. Automatically used with --view.")
    p.add_argument("--realtime-factor", type=float, default=1.0, help="1.0 is real-time, 2.0 runs twice real-time when throttled.")
    p.add_argument("--no-video", dest="save_video", action="store_false", help="Disable MP4 output.")
    p.set_defaults(save_video=True)
    p.add_argument("--video-path", default="", help="Base MP4 path. Default: OUTPUT_DIR/roomba_fp.mp4")
    p.add_argument("--video-every", type=int, default=1, help="Save video for episodes where ep %% video_every == 0.")
    p.add_argument("--video-codec", default="libx264")
    p.add_argument("--video-quality", type=int, default=8, help="imageio/ffmpeg quality, usually 0-10.")
    p.add_argument("--no-annotate-video", dest="annotate_video", action="store_false", help="Do not draw action text on video frames.")
    p.set_defaults(annotate_video=True)
    p.add_argument(
        "--save-frame-images-every",
        type=int,
        default=0,
        help="Save full RGB first-person PNG every N recorded frames. 0 disables.",
    )
    p.add_argument("--save-xml", default="", help="Optional path to write the generated MJCF XML.")

    # CSV dataset.
    p.add_argument("--no-csv", dest="save_csv", action="store_false", help="Disable CSV data collection.")
    p.set_defaults(save_csv=True)
    p.add_argument(
        "--csv-image-format",
        choices=["hex", "wide"],
        default="hex",
        help="hex stores one compact 512-char hex string. wide stores 256 uint8 pixel columns px_000..px_255.",
    )

    # World and robot geometry.
    p.add_argument("--room-size", type=float, default=4.0, help="Interior square room size in meters.")
    p.add_argument("--wall-height", type=float, default=0.35)
    p.add_argument("--wall-thickness", type=float, default=0.06)
    p.add_argument("--robot-radius", type=float, default=0.17, help="Roomba 500-series-like radius in meters.")
    p.add_argument("--robot-height", type=float, default=0.09)
    p.add_argument("--robot-mass", type=float, default=3.6)
    p.add_argument("--camera-fovy", type=float, default=90.0, help="Vertical field of view of the first-person camera in degrees.")
    p.add_argument(
        "--camera-height-above-top",
        "--camera-height",
        dest="camera_height_above_top",
        type=float,
        default=0.025,
        help="First-person camera mount height in meters above the robot top center. --camera-height is an alias.",
    )
    p.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=0.0,
        help="First-person camera pitch in degrees. 0 looks forward horizontally; positive tilts downward; negative tilts upward.",
    )
    p.add_argument("--fixed-start", dest="randomize_start", action="store_false", help="Start at x=y=yaw=0 instead of random pose.")
    p.set_defaults(randomize_start=True)

    # Motion / controller.
    p.add_argument("--speed", type=float, default=0.35, help="Forward speed target in m/s.")
    p.add_argument("--reverse-speed", type=float, default=0.25, help="Backing speed target magnitude in m/s.")
    p.add_argument("--turn-speed", type=float, default=1.35, help="In-place angular speed target in rad/s.")
    p.add_argument("--control-kv", type=float, default=140.0, help="P gain on generalized velocity error.")
    p.add_argument("--max-force", type=float, default=80.0, help="Max slide joint force in Newtons.")
    p.add_argument("--max-torque", type=float, default=18.0, help="Max yaw joint torque in N*m.")
    p.add_argument("--cmd-noise-v-std", type=float, default=0.015, help="Gaussian per-step speed command noise std.")
    p.add_argument("--cmd-noise-omega-std", type=float, default=0.035, help="Gaussian per-step angular command noise std.")

    # Behavior randomization.
    p.add_argument("--turn-interval-mean", type=float, default=4.0, help="Mean seconds between spontaneous turns.")
    p.add_argument("--turn-interval-std", type=float, default=1.0, help="Gaussian std for spontaneous-turn interval.")
    p.add_argument("--wander-turn-mean-deg", type=float, default=35.0)
    p.add_argument("--wander-turn-std-deg", type=float, default=18.0)
    p.add_argument("--min-wander-turn-deg", type=float, default=8.0)
    p.add_argument("--bump-turn-std-deg", type=float, default=8.0, help="Gaussian std around 180 degree bump turn.")
    p.add_argument("--reverse-seconds", type=float, default=0.70)
    p.add_argument("--bump-debounce", type=float, default=0.45)

    args = p.parse_args(argv)

    if args.record_fps <= 0:
        raise SystemExit("--record-fps must be > 0")
    if args.dt <= 0:
        raise SystemExit("--dt must be > 0")
    if args.duration <= 0:
        raise SystemExit("--duration must be > 0")
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be >= 1")
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be >= 1")
    if args.video_every < 1:
        raise SystemExit("--video-every must be >= 1")

    output_dir = Path(args.output_dir)
    default_csv_path = output_dir / "dataset.csv.gz"
    default_video_path = output_dir / "roomba_fp.mp4"

    gl_backend = choose_gl_backend(args.gl, args.view)
    if args.view and args.num_workers > 1:
        print("[info] --view uses a single interactive episode; forcing --num-workers 1 and --num-episodes 1.", file=sys.stderr)
        args.num_workers = 1
        args.num_episodes = 1
    if args.view:
        args.real_time = True

    return Config(
        output_dir=str(output_dir),
        csv_path=args.csv_path or str(default_csv_path),
        seed=args.seed,
        duration=args.duration,
        num_episodes=args.num_episodes,
        num_workers=args.num_workers,
        keep_shards=args.keep_shards,
        print_every=args.print_every,
        gl_backend=gl_backend,
        dt=args.dt,
        width=args.width,
        height=args.height,
        record_fps=args.record_fps,
        view=args.view,
        view_camera=args.view_camera,
        viewer_fps=args.viewer_fps,
        real_time=args.real_time,
        realtime_factor=args.realtime_factor,
        save_video=args.save_video,
        video_path=args.video_path or str(default_video_path),
        video_every=args.video_every,
        video_codec=args.video_codec,
        video_quality=args.video_quality,
        annotate_video=args.annotate_video,
        save_frame_images_every=args.save_frame_images_every,
        save_xml_path=args.save_xml,
        save_csv=args.save_csv,
        csv_image_format=args.csv_image_format,
        room_size=args.room_size,
        wall_height=args.wall_height,
        wall_thickness=args.wall_thickness,
        robot_radius=args.robot_radius,
        robot_height=args.robot_height,
        robot_mass=args.robot_mass,
        camera_fovy=args.camera_fovy,
        camera_height_above_top=args.camera_height_above_top,
        camera_pitch_deg=args.camera_pitch_deg,
        randomize_start=args.randomize_start,
        speed=args.speed,
        reverse_speed=args.reverse_speed,
        turn_speed=args.turn_speed,
        control_kv=args.control_kv,
        max_force=args.max_force,
        max_torque=args.max_torque,
        cmd_noise_v_std=args.cmd_noise_v_std,
        cmd_noise_omega_std=args.cmd_noise_omega_std,
        turn_interval_mean=args.turn_interval_mean,
        turn_interval_std=args.turn_interval_std,
        wander_turn_mean_deg=args.wander_turn_mean_deg,
        wander_turn_std_deg=args.wander_turn_std_deg,
        min_wander_turn_deg=args.min_wander_turn_deg,
        bump_turn_std_deg=args.bump_turn_std_deg,
        reverse_seconds=args.reverse_seconds,
        bump_debounce=args.bump_debounce,
    )


def choose_gl_backend(requested: str, view: bool) -> str:
    if requested != "auto":
        return requested
    if view:
        return "glfw"
    if sys.platform.startswith("linux"):
        return "egl"
    # On macOS/Windows, let MuJoCo/GLFW choose the normal windowing backend.
    return "glfw"


def configure_mujoco_env(gl_backend: str) -> None:
    # Must happen before importing mujoco.
    if gl_backend:
        os.environ.setdefault("MUJOCO_GL", gl_backend)


def import_mujoco(gl_backend: str):
    configure_mujoco_env(gl_backend)
    import mujoco  # type: ignore

    return mujoco


def make_xml(cfg: Config) -> str:
    half = cfg.room_size / 2.0
    wall_t = cfg.wall_thickness
    wall_z = cfg.wall_height / 2.0
    r = cfg.robot_radius
    h = cfg.robot_height
    base_z = h / 2.0 + 0.01
    top_cam_z = h / 2.0 + cfg.camera_height_above_top
    wheel_r = min(0.055, h * 0.58)
    wheel_w_half = 0.018
    wheel_y = r * 0.82
    wheel_z = -h * 0.18

    # MuJoCo cameras look along their local -Z axis.  At zero pitch,
    # -Z points along robot +X and image-up points along robot/world +Z.
    # Positive pitch tilts the view downward toward the floor.
    pitch = math.radians(cfg.camera_pitch_deg)
    cam_x_axis = (0.0, -1.0, 0.0)
    cam_y_axis = (math.sin(pitch), 0.0, math.cos(pitch))
    camera_xyaxes = (
        f"{cam_x_axis[0]:.9f} {cam_x_axis[1]:.9f} {cam_x_axis[2]:.9f} "
        f"{cam_y_axis[0]:.9f} {cam_y_axis[1]:.9f} {cam_y_axis[2]:.9f}"
    )

    return f"""
<mujoco model="roomba500_like_room">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="{cfg.dt:.8f}" integrator="Euler" solver="Newton" iterations="50" tolerance="1e-8" cone="pyramidal"/>
  <size njmax="1000" nconmax="300"/>
  <statistic center="0 0 0" extent="{cfg.room_size:.6f}"/>

  <visual>
    <global offwidth="{cfg.width}" offheight="{cfg.height}"/>
    <quality shadowsize="1024" numslices="24" numstacks="12"/>
  </visual>

  <default>
    <joint damping="0.02" armature="0.001"/>
    <geom condim="3" friction="1.0 0.03 0.003" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>

  <asset>
    <texture name="floor_checker" type="2d" builtin="checker" width="512" height="512" rgb1="0.72 0.72 0.72" rgb2="0.88 0.88 0.88"/>
    <material name="floor_mat" texture="floor_checker" texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light name="ceiling" pos="0 0 3" dir="0 0 -1" diffuse="0.95 0.95 0.95" specular="0.15 0.15 0.15"/>
    <light name="front_fill" pos="-1.5 -1.5 1.2" dir="1 1 -0.5" diffuse="0.35 0.35 0.35"/>

    <geom name="floor" type="plane" pos="0 0 0" size="{half + 0.5:.6f} {half + 0.5:.6f} 0.05" material="floor_mat" contype="0" conaffinity="0"/>

    <geom name="wall_x_pos" type="box" pos="{half + wall_t:.6f} 0 {wall_z:.6f}" size="{wall_t:.6f} {half + wall_t:.6f} {wall_z:.6f}" rgba="0.78 0.80 0.84 1"/>
    <geom name="wall_x_neg" type="box" pos="{-half - wall_t:.6f} 0 {wall_z:.6f}" size="{wall_t:.6f} {half + wall_t:.6f} {wall_z:.6f}" rgba="0.78 0.80 0.84 1"/>
    <geom name="wall_y_pos" type="box" pos="0 {half + wall_t:.6f} {wall_z:.6f}" size="{half + wall_t:.6f} {wall_t:.6f} {wall_z:.6f}" rgba="0.74 0.76 0.80 1"/>
    <geom name="wall_y_neg" type="box" pos="0 {-half - wall_t:.6f} {wall_z:.6f}" size="{half + wall_t:.6f} {wall_t:.6f} {wall_z:.6f}" rgba="0.74 0.76 0.80 1"/>

    <body name="roomba" pos="0 0 {base_z:.6f}">
      <!-- Planar base: x/y translation and yaw only. The visual wheels are cosmetic;
           generalized velocity control implements differential-drive-like motion. -->
      <joint name="root_x" type="slide" axis="1 0 0"/>
      <joint name="root_y" type="slide" axis="0 1 0"/>
      <joint name="root_yaw" type="hinge" axis="0 0 1"/>

      <geom name="roomba_bumper" type="cylinder" size="{r:.6f} {h / 2.0:.6f}" mass="{cfg.robot_mass:.6f}" rgba="0.05 0.05 0.055 1"/>
      <geom name="roomba_silver_top" type="cylinder" pos="0 0 {h / 2.0 + 0.006:.6f}" size="{r * 0.88:.6f} 0.012" contype="0" conaffinity="0" rgba="0.56 0.58 0.60 1"/>
      <geom name="roomba_front_panel" type="box" pos="{r * 0.56:.6f} 0 {h / 2.0 + 0.021:.6f}" size="{r * 0.19:.6f} {r * 0.52:.6f} 0.007" contype="0" conaffinity="0" rgba="0.04 0.18 0.28 1"/>
      <site name="front_led" pos="{r * 0.93:.6f} 0 {h / 2.0 + 0.034:.6f}" size="0.014" rgba="0.0 0.9 0.25 1"/>

      <geom name="left_wheel_visual" type="cylinder" pos="0 {wheel_y:.6f} {wheel_z:.6f}" euler="90 0 0" size="{wheel_r:.6f} {wheel_w_half:.6f}" contype="0" conaffinity="0" rgba="0.015 0.015 0.018 1"/>
      <geom name="right_wheel_visual" type="cylinder" pos="0 {-wheel_y:.6f} {wheel_z:.6f}" euler="90 0 0" size="{wheel_r:.6f} {wheel_w_half:.6f}" contype="0" conaffinity="0" rgba="0.015 0.015 0.018 1"/>

      <!-- MuJoCo cameras look along local -Z. At camera-pitch=0, -Z points along robot +X; positive pitch tilts downward. -->
      <camera name="roomba_fp" mode="fixed" pos="0 0 {top_cam_z:.6f}" xyaxes="{camera_xyaxes}" fovy="{cfg.camera_fovy:.6f}"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor_x" joint="root_x" gear="1" ctrllimited="true" ctrlrange="{-cfg.max_force:.6f} {cfg.max_force:.6f}"/>
    <motor name="motor_y" joint="root_y" gear="1" ctrllimited="true" ctrlrange="{-cfg.max_force:.6f} {cfg.max_force:.6f}"/>
    <motor name="motor_yaw" joint="root_yaw" gear="1" ctrllimited="true" ctrlrange="{-cfg.max_torque:.6f} {cfg.max_torque:.6f}"/>
  </actuator>
</mujoco>
""".strip()


def normal_positive(rng: np.random.Generator, mean: float, std: float, min_value: float) -> float:
    if std <= 0:
        return max(min_value, mean)
    for _ in range(32):
        v = float(rng.normal(mean, std))
        if v >= min_value:
            return v
    return max(min_value, mean)


class RoombaBrain:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.action = Action.FORWARD
        self.action_until = math.inf
        self.pending_bump_turn = False
        self.last_bump_t = -1e9
        self.next_wander_turn_t = 0.0
        self._schedule_next_wander(0.0)

    def _schedule_next_wander(self, t: float) -> None:
        interval = normal_positive(
            self.rng,
            self.cfg.turn_interval_mean,
            self.cfg.turn_interval_std,
            min_value=0.25,
        )
        self.next_wander_turn_t = t + interval

    def _start_forward(self, t: float) -> None:
        self.action = Action.FORWARD
        self.action_until = math.inf
        self.pending_bump_turn = False
        self._schedule_next_wander(t)

    def _start_turn(self, t: float, angle_rad: float) -> None:
        direction = 1.0 if self.rng.random() < 0.5 else -1.0
        self.action = Action.ROTATE_LEFT if direction > 0 else Action.ROTATE_RIGHT
        self.action_until = t + max(0.02, abs(angle_rad) / max(1e-6, self.cfg.turn_speed))

    def update(self, t: float, bumped: bool) -> str:
        # Bumps only matter while driving forward. During backup/turn, old contacts are ignored.
        if self.action == Action.FORWARD and bumped and (t - self.last_bump_t) >= self.cfg.bump_debounce:
            self.last_bump_t = t
            self.action = Action.BACKWARD
            self.action_until = t + max(0.0, self.cfg.reverse_seconds)
            self.pending_bump_turn = True
            return self.action

        if t >= self.action_until:
            if self.action == Action.BACKWARD and self.pending_bump_turn:
                noisy_180 = math.pi + float(self.rng.normal(0.0, math.radians(self.cfg.bump_turn_std_deg)))
                self.pending_bump_turn = False
                self._start_turn(t, noisy_180)
                return self.action
            if self.action in (Action.ROTATE_LEFT, Action.ROTATE_RIGHT):
                self._start_forward(t)
                return self.action

        if self.action == Action.FORWARD and t >= self.next_wander_turn_t:
            mean = math.radians(self.cfg.wander_turn_mean_deg)
            std = math.radians(self.cfg.wander_turn_std_deg)
            min_angle = math.radians(self.cfg.min_wander_turn_deg)
            angle = normal_positive(self.rng, mean, std, min_angle)
            self._start_turn(t, angle)
            return self.action

        return self.action


def command_for_action(cfg: Config, action: str) -> Tuple[float, float]:
    if action == Action.FORWARD:
        return cfg.speed, 0.0
    if action == Action.BACKWARD:
        return -cfg.reverse_speed, 0.0
    if action == Action.ROTATE_LEFT:
        return 0.0, cfg.turn_speed
    if action == Action.ROTATE_RIGHT:
        return 0.0, -cfg.turn_speed
    raise ValueError(f"Unknown action: {action}")


@dataclasses.dataclass(frozen=True)
class ModelIndices:
    x_qpos: int
    y_qpos: int
    yaw_qpos: int
    x_dof: int
    y_dof: int
    yaw_dof: int
    motor_x: int
    motor_y: int
    motor_yaw: int
    camera_id: int
    wall_geom_ids: frozenset[int]
    robot_geom_ids: frozenset[int]


def get_indices(mujoco: Any, model: Any) -> ModelIndices:
    def jid(name: str) -> int:
        return int(model.joint(name).id)

    def aid(name: str) -> int:
        return int(model.actuator(name).id)

    wall_ids: set[int] = set()
    robot_ids: set[int] = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("wall_"):
            wall_ids.add(gid)
        if name == "roomba_bumper":
            robot_ids.add(gid)

    jx, jy, jyaw = jid("root_x"), jid("root_y"), jid("root_yaw")
    return ModelIndices(
        x_qpos=int(model.jnt_qposadr[jx]),
        y_qpos=int(model.jnt_qposadr[jy]),
        yaw_qpos=int(model.jnt_qposadr[jyaw]),
        x_dof=int(model.jnt_dofadr[jx]),
        y_dof=int(model.jnt_dofadr[jy]),
        yaw_dof=int(model.jnt_dofadr[jyaw]),
        motor_x=aid("motor_x"),
        motor_y=aid("motor_y"),
        motor_yaw=aid("motor_yaw"),
        camera_id=int(model.camera("roomba_fp").id),
        wall_geom_ids=frozenset(wall_ids),
        robot_geom_ids=frozenset(robot_ids),
    )


def reset_episode(mujoco: Any, model: Any, data: Any, idx: ModelIndices, cfg: Config, rng: np.random.Generator) -> None:
    mujoco.mj_resetData(model, data)
    if cfg.randomize_start:
        margin = max(0.05, cfg.room_size / 2.0 - cfg.robot_radius - 0.35)
        data.qpos[idx.x_qpos] = float(rng.uniform(-margin, margin))
        data.qpos[idx.y_qpos] = float(rng.uniform(-margin, margin))
        data.qpos[idx.yaw_qpos] = float(rng.uniform(-math.pi, math.pi))
    else:
        data.qpos[idx.x_qpos] = 0.0
        data.qpos[idx.y_qpos] = 0.0
        data.qpos[idx.yaw_qpos] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def normalize_angle_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wall_bumped(data: Any, idx: ModelIndices) -> bool:
    walls = idx.wall_geom_ids
    robots = idx.robot_geom_ids
    for c_i in range(data.ncon):
        c = data.contact[c_i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in walls and g2 in robots) or (g2 in walls and g1 in robots):
            return True
    return False


def apply_velocity_servo(
    data: Any,
    idx: ModelIndices,
    cfg: Config,
    yaw: float,
    v_body: float,
    omega: float,
) -> None:
    desired_x = v_body * math.cos(yaw)
    desired_y = v_body * math.sin(yaw)
    desired_yaw = omega

    err_x = desired_x - float(data.qvel[idx.x_dof])
    err_y = desired_y - float(data.qvel[idx.y_dof])
    err_yaw = desired_yaw - float(data.qvel[idx.yaw_dof])

    ctrl_x = float(np.clip(cfg.control_kv * err_x, -cfg.max_force, cfg.max_force))
    ctrl_y = float(np.clip(cfg.control_kv * err_y, -cfg.max_force, cfg.max_force))
    ctrl_yaw = float(np.clip(cfg.control_kv * err_yaw, -cfg.max_torque, cfg.max_torque))

    data.ctrl[idx.motor_x] = ctrl_x
    data.ctrl[idx.motor_y] = ctrl_y
    data.ctrl[idx.motor_yaw] = ctrl_yaw


def open_csv_text(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="")
    return open(path, mode, newline="")


def csv_header(cfg: Config) -> List[str]:
    base = [
        "episode",
        "frame",
        "time_s",
        "action_id",
        "action",
        "bumped",
        "x_m",
        "y_m",
        "yaw_rad",
        "cmd_v_mps",
        "cmd_omega_radps",
    ]
    if cfg.csv_image_format == "wide":
        return base + [f"px_{i:03d}" for i in range(16 * 16)]
    return base + ["image16x16_gray_u8_hex"]


def rgb_to_gray16(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image

    if rgb.dtype != np.uint8:
        rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    im = Image.fromarray(gray, mode="L")
    # Pillow >= 9 uses Image.Resampling; older versions expose constants at top-level.
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    im16 = im.resize((16, 16), resample=resample)
    return np.asarray(im16, dtype=np.uint8)


def annotate_frame(rgb: np.ndarray, action: str, t: float, bumped: bool) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    label = f"action={action}  t={t:6.2f}s  bump={int(bumped)}"
    # Use a black strip to make text readable without external font files.
    strip_h = 24
    draw.rectangle([0, 0, im.width, strip_h], fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255), font=font)
    return np.asarray(im, dtype=np.uint8)


def render_rgb(renderer: Any, data: Any, camera_id: int) -> np.ndarray:
    renderer.update_scene(data, camera=camera_id)
    rgb = renderer.render()
    return np.asarray(rgb, dtype=np.uint8)


def episode_csv_shard_path(cfg: Config, episode: int) -> Path:
    shard_dir = Path(cfg.output_dir) / "csv_shards"
    return shard_dir / f"episode_{episode:05d}.csv.gz"


def episode_video_path(cfg: Config, episode: int) -> Optional[Path]:
    if not cfg.save_video or episode % cfg.video_every != 0:
        return None
    base = Path(cfg.video_path)
    if not base.suffix:
        base = base.with_suffix(".mp4")
    base.parent.mkdir(parents=True, exist_ok=True)
    if cfg.num_episodes == 1:
        return base
    return base.with_name(f"{base.stem}_ep{episode:05d}{base.suffix}")


def write_csv_row(
    writer: csv.writer,
    cfg: Config,
    episode: int,
    frame_idx: int,
    t: float,
    action: str,
    bumped: bool,
    x: float,
    y: float,
    yaw: float,
    v_cmd: float,
    omega_cmd: float,
    gray16: np.ndarray,
) -> None:
    row: List[Any] = [
        episode,
        frame_idx,
        f"{t:.6f}",
        ACTION_ID[action],
        action,
        int(bumped),
        f"{x:.6f}",
        f"{y:.6f}",
        f"{yaw:.6f}",
        f"{v_cmd:.6f}",
        f"{omega_cmd:.6f}",
    ]
    if cfg.csv_image_format == "wide":
        row.extend(int(v) for v in gray16.reshape(-1))
    else:
        row.append(gray16.tobytes().hex())
    writer.writerow(row)


def run_episode(cfg: Config, episode: int) -> Dict[str, Any]:
    mujoco = import_mujoco(cfg.gl_backend)
    xml = make_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    idx = get_indices(mujoco, model)
    rng = np.random.default_rng(cfg.seed + 10007 * episode)
    reset_episode(mujoco, model, data, idx, cfg, rng)
    brain = RoombaBrain(cfg, rng)

    # Lazy imports: avoid pulling rendering libraries into processes that do not render.
    need_render = cfg.save_csv or cfg.save_video or cfg.save_frame_images_every > 0
    renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width) if need_render else None

    video_path = episode_video_path(cfg, episode)
    video_writer = None
    if video_path is not None:
        import imageio.v2 as imageio

        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(
            str(video_path),
            fps=cfg.record_fps,
            codec=cfg.video_codec,
            quality=cfg.video_quality,
            macro_block_size=None,
        )

    csv_path = episode_csv_shard_path(cfg, episode)
    csv_file = None
    csv_writer = None
    if cfg.save_csv:
        csv_file = open_csv_text(csv_path, "w")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header(cfg))

    frames_dir: Optional[Path] = None
    if cfg.save_frame_images_every > 0:
        frames_dir = Path(cfg.output_dir) / "frames" / f"episode_{episode:05d}"
        frames_dir.mkdir(parents=True, exist_ok=True)

    viewer_cm = None
    viewer = None
    if cfg.view:
        import mujoco.viewer  # type: ignore

        viewer_cm = mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True)
        viewer = viewer_cm.__enter__()
        configure_viewer_camera(mujoco, model, viewer, idx, cfg)
        sync_viewer(viewer)

    frame_idx = 0
    next_record_t = 0.0
    next_viewer_t = 0.0
    wall_start = time.perf_counter()
    start_cpu = time.perf_counter()
    bump_count = 0
    prev_bumped = False

    try:
        while data.time < cfg.duration:
            if viewer is not None and not viewer.is_running():
                break

            t = float(data.time)
            bumped = wall_bumped(data, idx)
            if bumped and not prev_bumped:
                bump_count += 1
            prev_bumped = bumped

            action = brain.update(t, bumped)
            v_cmd, omega_cmd = command_for_action(cfg, action)
            if cfg.cmd_noise_v_std > 0:
                v_cmd += float(rng.normal(0.0, cfg.cmd_noise_v_std))
            if cfg.cmd_noise_omega_std > 0:
                omega_cmd += float(rng.normal(0.0, cfg.cmd_noise_omega_std))

            # Record current first-person image/action at fixed FPS before stepping to the next physics state.
            if need_render and renderer is not None and t + 1e-12 >= next_record_t:
                rgb = render_rgb(renderer, data, idx.camera_id)
                gray16 = rgb_to_gray16(rgb)
                x = float(data.qpos[idx.x_qpos])
                y = float(data.qpos[idx.y_qpos])
                yaw = normalize_angle_rad(float(data.qpos[idx.yaw_qpos]))

                if csv_writer is not None:
                    write_csv_row(
                        csv_writer,
                        cfg,
                        episode,
                        frame_idx,
                        t,
                        action,
                        bumped,
                        x,
                        y,
                        yaw,
                        v_cmd,
                        omega_cmd,
                        gray16,
                    )

                if video_writer is not None:
                    out_rgb = annotate_frame(rgb, action, t, bumped) if cfg.annotate_video else rgb
                    video_writer.append_data(out_rgb)

                if frames_dir is not None and cfg.save_frame_images_every > 0 and frame_idx % cfg.save_frame_images_every == 0:
                    from PIL import Image

                    Image.fromarray(rgb, mode="RGB").save(frames_dir / f"frame_{frame_idx:06d}.png")

                frame_idx += 1
                next_record_t = frame_idx / cfg.record_fps

            yaw = float(data.qpos[idx.yaw_qpos])
            apply_velocity_servo(data, idx, cfg, yaw, v_cmd, omega_cmd)
            mujoco.mj_step(model, data)

            if viewer is not None and data.time + 1e-12 >= next_viewer_t:
                sync_viewer(viewer)
                next_viewer_t += 1.0 / max(1e-6, cfg.viewer_fps)

            if cfg.real_time:
                target_wall = wall_start + float(data.time) / max(1e-6, cfg.realtime_factor)
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(min(sleep_s, 0.05))
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        if video_writer is not None:
            video_writer.close()
        if csv_file is not None:
            csv_file.close()
        # Renderer exposes close() in recent MuJoCo versions; harmless if absent.
        if renderer is not None and hasattr(renderer, "close"):
            renderer.close()

    elapsed_cpu = time.perf_counter() - start_cpu
    return {
        "episode": episode,
        "csv_path": str(csv_path) if cfg.save_csv else "",
        "video_path": str(video_path) if video_path is not None else "",
        "frames": frame_idx,
        "sim_time": float(data.time),
        "wall_time": elapsed_cpu,
        "bump_count": bump_count,
    }


def configure_viewer_camera(mujoco: Any, model: Any, viewer: Any, idx: ModelIndices, cfg: Config) -> None:
    if cfg.view_camera == "roomba_fp":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = idx.camera_id
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])
        viewer.cam.distance = max(3.0, cfg.room_size * 1.25)
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -60.0


def sync_viewer(viewer: Any) -> None:
    # state_only was added after the original passive viewer API. Use it when available.
    try:
        viewer.sync(state_only=True)
    except TypeError:
        viewer.sync()


def merge_csv_shards(shards: Sequence[Path], output_path: Path) -> None:
    if not shards:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_csv_text(output_path, "w") as fout:
        writer = csv.writer(fout)
        wrote_header = False
        for shard in sorted(shards):
            with open_csv_text(shard, "r") as fin:
                reader = csv.reader(fin)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if not wrote_header:
                    writer.writerow(header)
                    wrote_header = True
                for row in reader:
                    writer.writerow(row)


def run_all(cfg: Config) -> List[Dict[str, Any]]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.save_xml_path:
        xml_path = Path(cfg.save_xml_path)
    else:
        xml_path = out / "roomba_room.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(make_xml(cfg), encoding="utf-8")

    if cfg.num_workers == 1:
        results = []
        for ep in range(cfg.num_episodes):
            result = run_episode(cfg, ep)
            results.append(result)
            maybe_print_progress(cfg, result, len(results))
    else:
        # Use spawn so each process creates its own OpenGL context cleanly.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=cfg.num_workers) as pool:
            tasks = [(cfg, ep) for ep in range(cfg.num_episodes)]
            results = []
            for result in pool.imap_unordered(_run_episode_star, tasks):
                results.append(result)
                maybe_print_progress(cfg, result, len(results))
        results.sort(key=lambda r: r["episode"])

    if cfg.save_csv:
        shards = [Path(r["csv_path"]) for r in results if r.get("csv_path")]
        final_csv = Path(cfg.csv_path)
        same_single_file = False
        if len(shards) == 1:
            try:
                same_single_file = shards[0].resolve() == final_csv.resolve()
            except FileNotFoundError:
                same_single_file = False
        if not same_single_file:
            merge_csv_shards(shards, final_csv)
        if not cfg.keep_shards and not same_single_file:
            for s in shards:
                try:
                    s.unlink()
                except FileNotFoundError:
                    pass
            # Remove shard dir if empty.
            shard_dir = Path(cfg.output_dir) / "csv_shards"
            try:
                shard_dir.rmdir()
            except OSError:
                pass

    return results


def _run_episode_star(args: Tuple[Config, int]) -> Dict[str, Any]:
    cfg, ep = args
    return run_episode(cfg, ep)


def maybe_print_progress(cfg: Config, result: Dict[str, Any], completed: int) -> None:
    if cfg.print_every <= 0 or completed % cfg.print_every != 0:
        return
    sim_time = result["sim_time"]
    wall_time = max(1e-9, result["wall_time"])
    rate = sim_time / wall_time
    print(
        f"[episode {result['episode']:05d}] frames={result['frames']} "
        f"sim={sim_time:.2f}s wall={wall_time:.2f}s rate={rate:.2f}x "
        f"bumps={result['bump_count']} video={result.get('video_path') or '-'}",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    configure_mujoco_env(cfg.gl_backend)
    results = run_all(cfg)

    print("\nDone.")
    xml_out = Path(cfg.save_xml_path) if cfg.save_xml_path else Path(cfg.output_dir) / "roomba_room.xml"
    print(f"  XML:       {xml_out}")
    if cfg.save_csv:
        print(f"  CSV:       {cfg.csv_path}")
    videos = [r["video_path"] for r in results if r.get("video_path")]
    if videos:
        print("  Video(s):")
        for v in videos[:10]:
            print(f"    {v}")
        if len(videos) > 10:
            print(f"    ... {len(videos) - 10} more")


if __name__ == "__main__":
    main()
