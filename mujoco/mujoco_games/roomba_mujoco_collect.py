#!/usr/bin/env python3
"""
Roomba-like MuJoCo room simulation + first-person dataset collector.
Updated: Pink target seeking with dynamic teleportation + parallel generation optimizations.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class Action:
    FORWARD = "w"
    BACKWARD = "s"
    ROTATE_LEFT = "a"
    ROTATE_RIGHT = "d"
    STOP = "p"


ACTION_ID = {
    Action.FORWARD: 0,
    Action.BACKWARD: 1,
    Action.ROTATE_LEFT: 2,
    Action.ROTATE_RIGHT: 3,
    Action.STOP: 4,
}


@dataclasses.dataclass(frozen=True)
class Config:
    output_dir: str
    csv_path: str
    seed: int
    duration: float
    num_episodes: int
    num_workers: int
    keep_shards: bool
    print_every: int

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

    save_csv: bool
    csv_image_format: str
    nanogpt_csv: bool

    room_size: float
    ceiling_height: float
    wall_thickness: float
    robot_radius: float
    robot_height: float
    robot_mass: float
    camera_fovy: float
    camera_height_above_top: float
    camera_pitch_deg: float
    randomize_start: bool

    speed: float
    reverse_speed: float
    turn_speed: float
    control_kv: float
    max_force: float
    max_torque: float
    cmd_noise_v_std: float
    cmd_noise_omega_std: float

    turn_interval_mean: float
    turn_interval_std: float
    wander_turn_mean_deg: float
    wander_turn_std_deg: float
    min_wander_turn_deg: float
    bump_turn_std_deg: float
    reverse_seconds: float
    bump_debounce: float


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="roomba_mujoco_runs")
    p.add_argument("--csv-path", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--keep-shards", action="store_true")
    p.add_argument("--print-every", type=int, default=1)

    p.add_argument("--gl", choices=["auto", "egl", "glfw", "osmesa"], default="auto")
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--record-fps", type=float, default=30.0)
    p.add_argument("--view", action="store_true")
    p.add_argument("--view-camera", choices=["free", "roomba_fp"], default="free")
    p.add_argument("--viewer-fps", type=float, default=60.0)
    p.add_argument("--real-time", action="store_true")
    p.add_argument("--realtime-factor", type=float, default=1.0)
    p.add_argument("--no-video", dest="save_video", action="store_false")
    p.set_defaults(save_video=True)
    p.add_argument("--video-path", default="")
    p.add_argument("--video-every", type=int, default=1)
    p.add_argument("--video-codec", default="libx264")
    p.add_argument("--video-quality", type=int, default=8)
    p.add_argument("--no-annotate-video", dest="annotate_video", action="store_false")
    p.set_defaults(annotate_video=True)
    p.add_argument("--save-frame-images-every", type=int, default=0)
    p.add_argument("--save-xml", default="")

    p.add_argument("--no-csv", dest="save_csv", action="store_false")
    p.set_defaults(save_csv=True)
    p.add_argument("--csv-image-format", choices=["hex", "wide"], default="hex")
    p.add_argument("--nanogpt-csv", action="store_true")

    p.add_argument("--room-size", type=float, default=4.0)
    p.add_argument("--ceiling-height", type=float, default=3.048)
    p.add_argument("--wall-thickness", type=float, default=0.06)
    p.add_argument("--robot-radius", type=float, default=0.17)
    p.add_argument("--robot-height", type=float, default=0.09)
    p.add_argument("--robot-mass", type=float, default=3.6)
    p.add_argument("--camera-fovy", type=float, default=90.0)
    p.add_argument("--camera-height-above-top", type=float, default=0.025)
    p.add_argument("--camera-pitch-deg", type=float, default=0.0)
    p.add_argument("--fixed-start", dest="randomize_start", action="store_false")
    p.set_defaults(randomize_start=True)

    p.add_argument("--speed", type=float, default=0.35)
    p.add_argument("--reverse-speed", type=float, default=0.25)
    p.add_argument("--turn-speed", type=float, default=1.35)
    p.add_argument("--control-kv", type=float, default=140.0)
    p.add_argument("--max-force", type=float, default=80.0)
    p.add_argument("--max-torque", type=float, default=18.0)
    p.add_argument("--cmd-noise-v-std", type=float, default=0.015)
    p.add_argument("--cmd-noise-omega-std", type=float, default=0.035)

    p.add_argument("--turn-interval-mean", type=float, default=4.0)
    p.add_argument("--turn-interval-std", type=float, default=1.0)
    p.add_argument("--wander-turn-mean-deg", type=float, default=35.0)
    p.add_argument("--wander-turn-std-deg", type=float, default=18.0)
    p.add_argument("--min-wander-turn-deg", type=float, default=8.0)
    p.add_argument("--bump-turn-std-deg", type=float, default=8.0)
    p.add_argument("--reverse-seconds", type=float, default=0.70)
    p.add_argument("--bump-debounce", type=float, default=0.45)

    args = p.parse_args(argv)
    if args.record_fps <= 0: raise SystemExit("--record-fps must be > 0")

    output_dir = Path(args.output_dir)
    default_csv_path = output_dir / ("dataset.csv" if args.nanogpt_csv else "dataset.csv.gz")
    default_video_path = output_dir / "roomba_fp.mp4"

    gl_backend = choose_gl_backend(args.gl, args.view)
    if args.view and args.num_workers > 1:
        args.num_workers = 1
        args.num_episodes = 1
    if args.view: args.real_time = True

    return Config(
        output_dir=str(output_dir), csv_path=args.csv_path or str(default_csv_path),
        seed=args.seed, duration=args.duration, num_episodes=args.num_episodes,
        num_workers=args.num_workers, keep_shards=args.keep_shards, print_every=args.print_every,
        gl_backend=gl_backend, dt=args.dt, width=args.width, height=args.height,
        record_fps=args.record_fps, view=args.view, view_camera=args.view_camera,
        viewer_fps=args.viewer_fps, real_time=args.real_time, realtime_factor=args.realtime_factor,
        save_video=args.save_video, video_path=args.video_path or str(default_video_path),
        video_every=args.video_every, video_codec=args.video_codec, video_quality=args.video_quality,
        annotate_video=args.annotate_video, save_frame_images_every=args.save_frame_images_every,
        save_xml_path=args.save_xml, save_csv=args.save_csv, csv_image_format=args.csv_image_format,
        nanogpt_csv=args.nanogpt_csv, room_size=args.room_size, ceiling_height=args.ceiling_height,
        wall_thickness=args.wall_thickness, robot_radius=args.robot_radius, robot_height=args.robot_height,
        robot_mass=args.robot_mass, camera_fovy=args.camera_fovy, camera_height_above_top=args.camera_height_above_top,
        camera_pitch_deg=args.camera_pitch_deg, randomize_start=args.randomize_start, speed=args.speed,
        reverse_speed=args.reverse_speed, turn_speed=args.turn_speed, control_kv=args.control_kv,
        max_force=args.max_force, max_torque=args.max_torque, cmd_noise_v_std=args.cmd_noise_v_std,
        cmd_noise_omega_std=args.cmd_noise_omega_std, turn_interval_mean=args.turn_interval_mean,
        turn_interval_std=args.turn_interval_std, wander_turn_mean_deg=args.wander_turn_mean_deg,
        wander_turn_std_deg=args.wander_turn_std_deg, min_wander_turn_deg=args.min_wander_turn_deg,
        bump_turn_std_deg=args.bump_turn_std_deg, reverse_seconds=args.reverse_seconds, bump_debounce=args.bump_debounce,
    )


def choose_gl_backend(requested: str, view: bool) -> str:
    if requested != "auto": return requested
    if view: return "glfw"
    if sys.platform.startswith("linux"): return "egl"
    return "glfw"


def configure_mujoco_env(gl_backend: str) -> None:
    if gl_backend: os.environ.setdefault("MUJOCO_GL", gl_backend)


def import_mujoco(gl_backend: str):
    configure_mujoco_env(gl_backend)
    import mujoco  # type: ignore
    return mujoco


def make_xml(cfg: Config) -> str:
    half = cfg.room_size / 2.0
    wall_t = cfg.wall_thickness
    wall_z = cfg.ceiling_height / 2.0
    r = cfg.robot_radius
    h = cfg.robot_height
    base_z = h / 2.0 + 0.01
    top_cam_z = h / 2.0 + cfg.camera_height_above_top
    wheel_r = min(0.055, h * 0.58)
    wheel_w_half = 0.018
    wheel_y = r * 0.82
    wheel_z = -h * 0.18

    pitch = math.radians(cfg.camera_pitch_deg)
    cam_x_axis = (0.0, -1.0, 0.0)
    cam_y_axis = (math.sin(pitch), 0.0, math.cos(pitch))
    camera_xyaxes = (f"{cam_x_axis[0]:.9f} {cam_x_axis[1]:.9f} {cam_x_axis[2]:.9f} "
                     f"{cam_y_axis[0]:.9f} {cam_y_axis[1]:.9f} {cam_y_axis[2]:.9f}")

    light_z = max(1.0, cfg.ceiling_height - 0.1)

    return f"""
<mujoco model="roomba_target_room">
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
    <light name="ceiling_light" pos="0 0 {light_z:.6f}" dir="0 0 -1" diffuse="0.95 0.95 0.95" specular="0.15 0.15 0.15"/>
    <light name="front_fill" pos="-1.5 -1.5 1.2" dir="1 1 -0.5" diffuse="0.35 0.35 0.35"/>

    <geom name="floor" type="plane" pos="0 0 0" size="{half + 0.5:.6f} {half + 0.5:.6f} 0.05" material="floor_mat" contype="0" conaffinity="0"/>
    <geom name="ceiling" type="plane" pos="0 0 {cfg.ceiling_height:.6f}" euler="180 0 0" size="{half + 0.5:.6f} {half + 0.5:.6f} 0.05" rgba="0.8 0.8 0.8 1" contype="0" conaffinity="0"/>

    <geom name="wall_x_pos" type="box" pos="{half + wall_t:.6f} 0 {wall_z:.6f}" size="{wall_t:.6f} {half + wall_t:.6f} {wall_z:.6f}" rgba="0.78 0.80 0.84 1"/>
    <geom name="wall_x_neg" type="box" pos="{-half - wall_t:.6f} 0 {wall_z:.6f}" size="{wall_t:.6f} {half + wall_t:.6f} {wall_z:.6f}" rgba="0.78 0.80 0.84 1"/>
    <geom name="wall_y_pos" type="box" pos="0 {half + wall_t:.6f} {wall_z:.6f}" size="{half + wall_t:.6f} {wall_t:.6f} {wall_z:.6f}" rgba="0.74 0.76 0.80 1"/>
    <geom name="wall_y_neg" type="box" pos="0 {-half - wall_t:.6f} {wall_z:.6f}" size="{half + wall_t:.6f} {wall_t:.6f} {wall_z:.6f}" rgba="0.74 0.76 0.80 1"/>

    <body name="target" pos="0 0 0.05" mocap="true">
      <geom name="target_geom" type="sphere" size="0.05" rgba="1.0 0.4 0.7 1" contype="0" conaffinity="0"/>
    </body>

    <body name="roomba" pos="0 0 {base_z:.6f}">
      <joint name="root_x" type="slide" axis="1 0 0"/>
      <joint name="root_y" type="slide" axis="0 1 0"/>
      <joint name="root_yaw" type="hinge" axis="0 0 1"/>

      <geom name="roomba_bumper" type="cylinder" size="{r:.6f} {h / 2.0:.6f}" mass="{cfg.robot_mass:.6f}" rgba="0.05 0.05 0.055 1"/>
      <geom name="roomba_silver_top" type="cylinder" pos="0 0 {h / 2.0 + 0.006:.6f}" size="{r * 0.88:.6f} 0.012" contype="0" conaffinity="0" rgba="0.56 0.58 0.60 1"/>
      <geom name="roomba_front_panel" type="box" pos="{r * 0.56:.6f} 0 {h / 2.0 + 0.021:.6f}" size="{r * 0.19:.6f} {r * 0.52:.6f} 0.007" contype="0" conaffinity="0" rgba="0.04 0.18 0.28 1"/>

      <geom name="left_wheel_visual" type="cylinder" pos="0 {wheel_y:.6f} {wheel_z:.6f}" euler="90 0 0" size="{wheel_r:.6f} {wheel_w_half:.6f}" contype="0" conaffinity="0" rgba="0.015 0.015 0.018 1"/>
      <geom name="right_wheel_visual" type="cylinder" pos="0 {-wheel_y:.6f} {wheel_z:.6f}" euler="90 0 0" size="{wheel_r:.6f} {wheel_w_half:.6f}" contype="0" conaffinity="0" rgba="0.015 0.015 0.018 1"/>

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


def normalize_angle_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def normal_positive(rng: np.random.Generator, mean: float, std: float, min_value: float) -> float:
    if std <= 0: return max(min_value, mean)
    for _ in range(32):
        v = float(rng.normal(mean, std))
        if v >= min_value: return v
    return max(min_value, mean)


class RoombaBrain:
    """Traditional random wandering module"""
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
        interval = normal_positive(self.rng, self.cfg.turn_interval_mean, self.cfg.turn_interval_std, min_value=0.25)
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

    def update(self, t: float, bumped: bool, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> str:
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


class SeekerBrain:
    """Intelligently searches for the pink ball, rotates until seen, approaches, and handles bumps."""
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.action = Action.ROTATE_LEFT
        self.action_until = 0.0
        self.last_bump_t = -1e9
        self.search_direction = Action.ROTATE_LEFT

    def update(self, t_sim: float, bumped: bool, x: float, y: float, yaw: float, tx: float, ty: float) -> str:
        if bumped and (t_sim - self.last_bump_t) >= self.cfg.bump_debounce:
            self.last_bump_t = t_sim
            self.action = Action.BACKWARD
            self.action_until = t_sim + self.cfg.reverse_seconds
            # Reverse search direction upon bumping to prevent endless cyclic loops
            self.search_direction = Action.ROTATE_LEFT if self.rng.random() < 0.5 else Action.ROTATE_RIGHT
            return self.action

        if t_sim < self.action_until:
            return self.action

        target_yaw = math.atan2(ty - y, tx - x)
        yaw_err = normalize_angle_rad(target_yaw - yaw)
        
        # Determine if target is roughly within the camera's horizontal FOV (~80 degrees or +/- 40 deg)
        visible = abs(yaw_err) < math.radians(40)

        # Inject occasional stop action for dataset diversity
        if self.rng.random() < 0.02:
            self.action = Action.STOP
            self.action_until = t_sim + self.rng.uniform(0.1, 0.4)
            return self.action

        if not visible:
            # Can't see it, spin and search
            self.action = self.search_direction
            self.action_until = t_sim + 0.1
        else:
            # We see it! Remember direction incase we lose it
            self.search_direction = Action.ROTATE_LEFT if yaw_err > 0 else Action.ROTATE_RIGHT
            
            # Close the distance, centering as we go
            if yaw_err > 0.15:
                self.action = Action.ROTATE_LEFT
            elif yaw_err < -0.15:
                self.action = Action.ROTATE_RIGHT
            else:
                self.action = Action.FORWARD
            self.action_until = t_sim + 0.1

        return self.action


def command_for_action(cfg: Config, action: str) -> Tuple[float, float]:
    if action == Action.FORWARD: return cfg.speed, 0.0
    if action == Action.BACKWARD: return -cfg.reverse_speed, 0.0
    if action == Action.ROTATE_LEFT: return 0.0, cfg.turn_speed
    if action == Action.ROTATE_RIGHT: return 0.0, -cfg.turn_speed
    if action == Action.STOP: return 0.0, 0.0
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
    target_mocap_id: int
    wall_geom_ids: frozenset[int]
    robot_geom_ids: frozenset[int]


def get_indices(mujoco: Any, model: Any) -> ModelIndices:
    def jid(name: str) -> int: return int(model.joint(name).id)
    def aid(name: str) -> int: return int(model.actuator(name).id)

    wall_ids: set[int] = set()
    robot_ids: set[int] = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("wall_"): wall_ids.add(gid)
        if name == "roomba_bumper": robot_ids.add(gid)

    jx, jy, jyaw = jid("root_x"), jid("root_y"), jid("root_yaw")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    mocap_id = model.body_mocapid[target_body_id]

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
        target_mocap_id=int(mocap_id),
        wall_geom_ids=frozenset(wall_ids),
        robot_geom_ids=frozenset(robot_ids),
    )


def spawn_target(cfg: Config, rng: np.random.Generator, rx: float, ry: float) -> Tuple[float, float]:
    """Generates a random coordinate for the ball, enforcing a minimum distance from the robot."""
    margin = cfg.robot_radius + 0.1
    limit = max(0.1, cfg.room_size / 2.0 - margin)
    for _ in range(100):
        tx = float(rng.uniform(-limit, limit))
        ty = float(rng.uniform(-limit, limit))
        if math.hypot(tx - rx, ty - ry) > 0.8:
            return tx, ty
    # Fallback bounds if geometry scaling fails
    return float(rng.uniform(-limit, limit)), float(rng.uniform(-limit, limit))


def reset_episode(mujoco: Any, model: Any, data: Any, idx: ModelIndices, cfg: Config, rng: np.random.Generator) -> None:
    mujoco.mj_resetData(model, data)
    margin = max(0.05, cfg.room_size / 2.0 - cfg.robot_radius - 0.35)
    
    if cfg.randomize_start:
        rx = float(rng.uniform(-margin, margin))
        ry = float(rng.uniform(-margin, margin))
        ryaw = float(rng.uniform(-math.pi, math.pi))
    else:
        rx, ry, ryaw = 0.0, 0.0, 0.0

    data.qpos[idx.x_qpos] = rx
    data.qpos[idx.y_qpos] = ry
    data.qpos[idx.yaw_qpos] = ryaw
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    # Initialize Mocap target safely away from robot
    tx, ty = spawn_target(cfg, rng, rx, ry)
    data.mocap_pos[idx.target_mocap_id][0] = tx
    data.mocap_pos[idx.target_mocap_id][1] = ty
    data.mocap_pos[idx.target_mocap_id][2] = 0.05

    mujoco.mj_forward(model, data)


def wall_bumped(data: Any, idx: ModelIndices) -> bool:
    walls = idx.wall_geom_ids
    robots = idx.robot_geom_ids
    for c_i in range(data.ncon):
        c = data.contact[c_i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in walls and g2 in robots) or (g2 in walls and g1 in robots):
            return True
    return False


def apply_velocity_servo(data: Any, idx: ModelIndices, cfg: Config, yaw: float, v_body: float, omega: float) -> None:
    err_x = (v_body * math.cos(yaw)) - float(data.qvel[idx.x_dof])
    err_y = (v_body * math.sin(yaw)) - float(data.qvel[idx.y_dof])
    err_yaw = omega - float(data.qvel[idx.yaw_dof])

    data.ctrl[idx.motor_x] = float(np.clip(cfg.control_kv * err_x, -cfg.max_force, cfg.max_force))
    data.ctrl[idx.motor_y] = float(np.clip(cfg.control_kv * err_y, -cfg.max_force, cfg.max_force))
    data.ctrl[idx.motor_yaw] = float(np.clip(cfg.control_kv * err_yaw, -cfg.max_torque, cfg.max_torque))


def open_csv_text(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="")
    return open(path, mode, newline="")


def csv_header(cfg: Config) -> List[str]:
    if cfg.nanogpt_csv:
        return [f"p{i}" for i in range(9)] + ["action"]

    base = ["episode", "frame", "time_s", "action_id", "action", "bumped", "x_m", "y_m", "yaw_rad", "cmd_v_mps", "cmd_omega_radps"]
    if cfg.csv_image_format == "wide":
        return base + [f"px_{i:03d}" for i in range(16 * 16)]
    return base + ["image16x16_gray_u8_hex"]


def rgb_to_gray3x3(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image
    if rgb.dtype != np.uint8:
        rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    im = Image.fromarray(gray, mode="L")
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", getattr(Image, "BILINEAR", 2))
    return np.asarray(im.resize((3, 3), resample=resample), dtype=np.uint8)


def rgb_to_gray16(rgb: np.ndarray) -> np.ndarray:
    from PIL import Image
    if rgb.dtype != np.uint8:
        rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    im = Image.fromarray(gray, mode="L")
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", getattr(Image, "BILINEAR", 2))
    return np.asarray(im.resize((16, 16), resample=resample), dtype=np.uint8)


def annotate_frame(rgb: np.ndarray, action: str, t: float, bumped: bool, targets: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    label = f"action={action}  t={t:6.2f}s  bump={int(bumped)} targets={targets}"
    draw.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255), font=font)
    return np.asarray(im, dtype=np.uint8)


def render_rgb(renderer: Any, data: Any, camera_id: int) -> np.ndarray:
    renderer.update_scene(data, camera=camera_id)
    return np.asarray(renderer.render(), dtype=np.uint8)


def episode_csv_shard_path(cfg: Config, episode: int) -> Path:
    shard_dir = Path(cfg.output_dir) / "csv_shards"
    suffix = ".csv" if cfg.nanogpt_csv else ".csv.gz"
    return shard_dir / f"episode_{episode:05d}{suffix}"


def episode_video_path(cfg: Config, episode: int) -> Optional[Path]:
    if not cfg.save_video or episode % cfg.video_every != 0: return None
    base = Path(cfg.video_path)
    if not base.suffix: base = base.with_suffix(".mp4")
    base.parent.mkdir(parents=True, exist_ok=True)
    if cfg.num_episodes == 1: return base
    return base.with_name(f"{base.stem}_ep{episode:05d}{base.suffix}")


def write_csv_row(
    writer: csv.writer, cfg: Config, episode: int, frame_idx: int, t: float, action: str,
    bumped: bool, x: float, y: float, yaw: float, v_cmd: float, omega_cmd: float,
    gray16: np.ndarray, gray3: Optional[np.ndarray] = None
) -> None:
    if cfg.nanogpt_csv and gray3 is not None:
        row: List[Any] = [int(v) for v in gray3.reshape(-1)]
        row.append(action)
        writer.writerow(row)
        return

    row: List[Any] = [
        episode, frame_idx, f"{t:.6f}", ACTION_ID.get(action, -1), action,
        int(bumped), f"{x:.6f}", f"{y:.6f}", f"{yaw:.6f}", f"{v_cmd:.6f}", f"{omega_cmd:.6f}",
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

    brain = SeekerBrain(cfg, rng) if cfg.nanogpt_csv else RoombaBrain(cfg, rng)

    need_render = cfg.save_csv or cfg.save_video or cfg.save_frame_images_every > 0
    renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width) if need_render else None

    video_path = episode_video_path(cfg, episode)
    video_writer = None
    if video_path is not None:
        try:
            import imageio.v2 as imageio
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(str(video_path), fps=cfg.record_fps, codec=cfg.video_codec, quality=cfg.video_quality)
        except ImportError:
            pass # Failed to capture video due to missing FFMPEG, falling back gracefully

    csv_path = episode_csv_shard_path(cfg, episode)
    csv_file, csv_writer = None, None
    if cfg.save_csv:
        csv_file = open_csv_text(csv_path, "w")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header(cfg))

    frames_dir: Optional[Path] = None
    if cfg.save_frame_images_every > 0:
        frames_dir = Path(cfg.output_dir) / "frames" / f"episode_{episode:05d}"
        frames_dir.mkdir(parents=True, exist_ok=True)

    viewer_cm, viewer = None, None
    if cfg.view:
        import mujoco.viewer  # type: ignore
        viewer_cm = mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True)
        viewer = viewer_cm.__enter__()
        configure_viewer_camera(mujoco, model, viewer, idx, cfg)
        sync_viewer(viewer)

    frame_idx, next_record_t, next_viewer_t = 0, 0.0, 0.0
    wall_start = time.perf_counter()
    start_cpu = time.perf_counter()
    bump_count, prev_bumped = 0, False
    collected_targets = 0

    try:
        while data.time < cfg.duration:
            if viewer is not None and not viewer.is_running(): break
            
            t = float(data.time)
            bumped = wall_bumped(data, idx)
            if bumped and not prev_bumped: bump_count += 1
            prev_bumped = bumped
            
            x = float(data.qpos[idx.x_qpos])
            y = float(data.qpos[idx.y_qpos])
            yaw = normalize_angle_rad(float(data.qpos[idx.yaw_qpos]))

            # Read target's current mocap position
            tx = data.mocap_pos[idx.target_mocap_id][0]
            ty = data.mocap_pos[idx.target_mocap_id][1]

            # Collect ball if within physical radius + a tiny threshold
            if math.hypot(tx - x, ty - y) < (cfg.robot_radius + 0.15):
                collected_targets += 1
                tx, ty = spawn_target(cfg, rng, x, y)
                data.mocap_pos[idx.target_mocap_id][0] = tx
                data.mocap_pos[idx.target_mocap_id][1] = ty
                data.mocap_pos[idx.target_mocap_id][2] = 0.05

            action = brain.update(t, bumped, x, y, yaw, tx, ty)
            v_cmd, omega_cmd = command_for_action(cfg, action)
            
            if cfg.cmd_noise_v_std > 0 and action != Action.STOP: v_cmd += float(rng.normal(0.0, cfg.cmd_noise_v_std))
            if cfg.cmd_noise_omega_std > 0 and action != Action.STOP: omega_cmd += float(rng.normal(0.0, cfg.cmd_noise_omega_std))

            if need_render and renderer is not None and t + 1e-12 >= next_record_t:
                rgb = render_rgb(renderer, data, idx.camera_id)
                gray16 = np.zeros((16, 16), dtype=np.uint8) if cfg.nanogpt_csv else rgb_to_gray16(rgb)
                gray3 = rgb_to_gray3x3(rgb) if cfg.nanogpt_csv else None

                if csv_writer is not None:
                    write_csv_row(csv_writer, cfg, episode, frame_idx, t, action, bumped, x, y, yaw, v_cmd, omega_cmd, gray16, gray3)

                if video_writer is not None:
                    video_writer.append_data(annotate_frame(rgb, action, t, bumped, collected_targets) if cfg.annotate_video else rgb)

                if frames_dir is not None and cfg.save_frame_images_every > 0 and frame_idx % cfg.save_frame_images_every == 0:
                    from PIL import Image
                    Image.fromarray(rgb, mode="RGB").save(frames_dir / f"frame_{frame_idx:06d}.png")

                frame_idx += 1
                next_record_t = frame_idx / cfg.record_fps

            apply_velocity_servo(data, idx, cfg, yaw, v_cmd, omega_cmd)
            mujoco.mj_step(model, data)

            if viewer is not None and data.time + 1e-12 >= next_viewer_t:
                sync_viewer(viewer)
                next_viewer_t += 1.0 / max(1e-6, cfg.viewer_fps)

            if cfg.real_time:
                target_wall = wall_start + float(data.time) / max(1e-6, cfg.realtime_factor)
                sleep_s = target_wall - time.perf_counter()
                if sleep_s > 0: time.sleep(min(sleep_s, 0.05))
    finally:
        if viewer_cm is not None: viewer_cm.__exit__(None, None, None)
        if video_writer is not None: video_writer.close()
        if csv_file is not None: csv_file.close()
        if renderer is not None and hasattr(renderer, "close"): renderer.close()

    elapsed_cpu = time.perf_counter() - start_cpu
    return {
        "episode": episode, "csv_path": str(csv_path) if cfg.save_csv else "",
        "video_path": str(video_path) if video_path is not None else "",
        "frames": frame_idx, "sim_time": float(data.time),
        "wall_time": elapsed_cpu, "bump_count": bump_count, "collected_targets": collected_targets,
    }


def configure_viewer_camera(mujoco: Any, model: Any, viewer: Any, idx: ModelIndices, cfg: Config) -> None:
    if cfg.view_camera == "roomba_fp":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = idx.camera_id
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0])
        viewer.cam.distance = max(3.0, cfg.room_size * 1.25)
        viewer.cam.azimuth, viewer.cam.elevation = 90.0, -60.0


def sync_viewer(viewer: Any) -> None:
    try:
        viewer.sync(state_only=True)
    except TypeError:
        viewer.sync()


def merge_csv_shards(shards: Sequence[Path], output_path: Path) -> None:
    if not shards: return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_csv_text(output_path, "w") as fout:
        writer = csv.writer(fout)
        wrote_header = False
        for shard in sorted(shards):
            with open_csv_text(shard, "r") as fin:
                reader = csv.reader(fin)
                try: header = next(reader)
                except StopIteration: continue
                if not wrote_header:
                    writer.writerow(header)
                    wrote_header = True
                for row in reader: writer.writerow(row)


def run_all(cfg: Config) -> List[Dict[str, Any]]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    xml_path = Path(cfg.save_xml_path) if cfg.save_xml_path else out / "roomba_room.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(make_xml(cfg), encoding="utf-8")

    if cfg.num_workers == 1:
        results = []
        for ep in range(cfg.num_episodes):
            res = run_episode(cfg, ep)
            results.append(res)
            maybe_print_progress(cfg, res, len(results))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=cfg.num_workers) as pool:
            tasks = [(cfg, ep) for ep in range(cfg.num_episodes)]
            results = []
            for res in pool.imap_unordered(_run_episode_star, tasks):
                results.append(res)
                maybe_print_progress(cfg, res, len(results))
        results.sort(key=lambda r: r["episode"])

    if cfg.save_csv:
        shards = [Path(r["csv_path"]) for r in results if r.get("csv_path")]
        final_csv = Path(cfg.csv_path)
        same_single_file = False
        if len(shards) == 1:
            try: same_single_file = shards[0].resolve() == final_csv.resolve()
            except FileNotFoundError: same_single_file = False
        if not same_single_file:
            merge_csv_shards(shards, final_csv)
        if not cfg.keep_shards and not same_single_file:
            for s in shards:
                try: s.unlink()
                except OSError: pass
            shard_dir = Path(cfg.output_dir) / "csv_shards"
            try: shard_dir.rmdir()
            except OSError: pass
    return results


def _run_episode_star(args: Tuple[Config, int]) -> Dict[str, Any]:
    return run_episode(*args)


def maybe_print_progress(cfg: Config, result: Dict[str, Any], completed: int) -> None:
    if cfg.print_every <= 0 or completed % cfg.print_every != 0: return
    sim_time, wall_time = result["sim_time"], max(1e-9, result["wall_time"])
    rate = sim_time / wall_time
    print(
        f"[episode {result['episode']:05d}] targets={result.get('collected_targets', 0)} frames={result['frames']} "
        f"sim={sim_time:.2f}s wall={wall_time:.2f}s rate={rate:.2f}x "
        f"bumps={result['bump_count']}",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    configure_mujoco_env(cfg.gl_backend)
    results = run_all(cfg)

    print("\nDone.")
    print(f"  XML:       {Path(cfg.save_xml_path) if cfg.save_xml_path else Path(cfg.output_dir) / 'roomba_room.xml'}")
    if cfg.save_csv: print(f"  CSV:       {cfg.csv_path}")
    videos = [r["video_path"] for r in results if r.get("video_path")]
    if videos:
        print("  Video(s):")
        for v in videos[:10]: print(f"    {v}")
        if len(videos) > 10: print(f"    ... {len(videos) - 10} more")


if __name__ == "__main__":
    main()
