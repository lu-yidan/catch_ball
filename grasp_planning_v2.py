#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绳组逆运动学规划 v2（不用 GP transport / 演示帧模板）。

臂系 (用户标定):
  +X 侧向, +Y 重力向下/倒立臂主轴, -Z 朝相机; 基座在 -Y.
倒立软体臂弯曲平面是 XZ（垂直于主轴 +Y），勿把 +Y 当弯曲方向。

- 第 1–2 节: phi=atan2(Z,X) in XZ, motor6/3 & 5/2
- 第 3 节: motor1/4 在可行域内朝球 (XZ)

用法见 grasp_plan_execute_v2.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from grasp_excute import (
    DEFAULT_ARM_BASE_MM,
    DEFAULT_VISION_TO_ARM_SCALE,
    SECTION12_PHI_OFFSET_DEG,
    apply_section12_bend_xz,
    apply_section3_bend_xy,
    make_rope_home_pose,
    motor14_in_grasp_range,
    project_motor14_to_grasp_range,
    section12_bend_xz,
    section3_bend_xz,
    section3_pose_from_grasp_motor14,
    section3_rope_steps_from_home,
    solve_section3_grasp_motor14,
    solve_sections_12_inverse_kinematics,
    vision_center_to_arm_mm,
)

from paths import DEFAULT_ARM_AXES_CONFIG, DEFAULT_DEMO_CSV

_ROOT = Path(__file__).resolve().parent
DEFAULT_BALL_RADIUS_MM = 35.0
DEFAULT_OUTPUT_V2 = _ROOT / "grasp_plan_v2.json"
PLAN_VERSION = 2
GRASP_BLEND_START_V2 = 0.45


@dataclass
class PlanResultV2:
    approach: str
    final_pose: np.ndarray
    trajectory: np.ndarray
    info: dict


def load_arm_axes_config(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_ARM_AXES_CONFIG
    if not path.is_file():
        return {
            "vision_to_arm_scale": DEFAULT_VISION_TO_ARM_SCALE.tolist(),
            "arm_base_mm": DEFAULT_ARM_BASE_MM.tolist(),
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_inversej_first_pose(csv_path: Path) -> np.ndarray:
    """读 inverse-J CSV 首帧 [Plane1..3, Theta1..3]，不调用 GP。"""
    df = pd.read_csv(csv_path)
    required = [
        "Plane1vsYOZ",
        "Plane2vsPlane1",
        "Plane3vsPlane2",
        "PCC_Theta_Plane1",
        "PCC_Theta_Plane2",
        "PCC_Theta_Plane3",
    ]
    if all(col in df.columns for col in required):
        phi = df[["Plane1vsYOZ", "Plane2vsPlane1", "Plane3vsPlane2"]].to_numpy(dtype=float)
        theta = df[["PCC_Theta_Plane1", "PCC_Theta_Plane2", "PCC_Theta_Plane3"]].to_numpy(dtype=float)
    else:
        phi = df.iloc[:, 1:4].to_numpy(dtype=float)
        theta = df.iloc[:, 4:7].to_numpy(dtype=float)
    return np.hstack([phi[0], theta[0]]).astype(float)


def plan_grasp_v2(
    ball_center_mm: np.ndarray,
    radius: float = DEFAULT_BALL_RADIUS_MM,
    demo_csv: Path | None = None,
    n_waypoints: int = 40,
    motor1_steps: int | None = None,
    motor4_steps: int | None = None,
    robot_base_mm: np.ndarray | None = None,
    approach: str = "side",
    phi_offset_deg: float = SECTION12_PHI_OFFSET_DEG,
    arm_axes_config: Path | None = DEFAULT_ARM_AXES_CONFIG,
) -> PlanResultV2:
    """Rope inverse kinematics for inverted arm frame (XZ bend plane)."""
    if demo_csv is None:
        demo_csv = DEFAULT_DEMO_CSV

    axes_cfg = load_arm_axes_config(arm_axes_config)
    vision_scale = np.asarray(
        axes_cfg.get("vision_to_arm_scale", DEFAULT_VISION_TO_ARM_SCALE),
        dtype=float,
    )
    if robot_base_mm is None:
        robot_base_mm = np.asarray(
            axes_cfg.get("arm_base_mm", DEFAULT_ARM_BASE_MM.tolist()),
            dtype=float,
        )
    else:
        robot_base_mm = np.asarray(robot_base_mm, dtype=float)

    ball_center_mm = np.asarray(ball_center_mm, dtype=float).reshape(3)
    ball_arm_mm = vision_center_to_arm_mm(ball_center_mm, vision_scale)

    neutral_pose = load_inversej_first_pose(demo_csv)
    rope_home = make_rope_home_pose(neutral_pose)

    approach_pose = solve_sections_12_inverse_kinematics(
        rope_home,
        ball_center_mm,
        base_mm=robot_base_mm,
        phi_offset_deg=phi_offset_deg,
        vision_to_arm_scale=vision_scale,
    )

    motor14_auto = motor1_steps is None or motor4_steps is None
    if motor14_auto:
        m1, m4, sec3_pose = solve_section3_grasp_motor14(
            rope_home,
            ball_center_mm,
            robot_base_mm,
            approach=approach,
            vision_to_arm_scale=vision_scale,
        )
    else:
        m1, m4 = project_motor14_to_grasp_range(int(motor1_steps), int(motor4_steps))
        sec3_pose = section3_pose_from_grasp_motor14(rope_home, m1, m4)

    final_pose = np.asarray(sec3_pose, dtype=float).copy()
    final_pose[0] = approach_pose[0]
    final_pose[1] = approach_pose[1]
    final_pose[3] = approach_pose[3]
    final_pose[4] = approach_pose[4]

    m1, m4 = section3_rope_steps_from_home(rope_home, final_pose)
    m1, m4 = project_motor14_to_grasp_range(m1, m4)

    idx = np.linspace(0, 1, max(2, int(n_waypoints)))
    bend12_home_s1 = section12_bend_xz(rope_home, 1)
    bend12_home_s2 = section12_bend_xz(rope_home, 2)
    bend12_goal_s1 = section12_bend_xz(approach_pose, 1)
    bend12_goal_s2 = section12_bend_xz(approach_pose, 2)
    final_bend_s3 = section3_bend_xz(final_pose)
    rope_bend_s3 = section3_bend_xz(rope_home)

    trajectory = []
    for t in idx:
        waypoint = np.asarray(rope_home, dtype=float).copy()
        b1 = bend12_home_s1 + float(t) * (bend12_goal_s1 - bend12_home_s1)
        b2 = bend12_home_s2 + float(t) * (bend12_goal_s2 - bend12_home_s2)
        waypoint = apply_section12_bend_xz(waypoint, 1, b1)
        waypoint = apply_section12_bend_xz(waypoint, 2, b2)
        if t >= GRASP_BLEND_START_V2:
            blend = (t - GRASP_BLEND_START_V2) / max(1e-6, 1.0 - GRASP_BLEND_START_V2)
            bend = (1.0 - blend) * rope_bend_s3 + blend * final_bend_s3
            waypoint = apply_section3_bend_xy(waypoint, bend)
        trajectory.append(waypoint)
    trajectory = np.asarray(trajectory, dtype=float)
    trajectory[-1] = final_pose

    safe = motor14_in_grasp_range(m1, m4)
    bend_s3 = section3_bend_xz(final_pose)
    base_arm_mm = vision_center_to_arm_mm(robot_base_mm, vision_scale)
    to_xz = np.array(
        [ball_arm_mm[0] - base_arm_mm[0], ball_arm_mm[2] - base_arm_mm[2]],
        dtype=float,
    )
    info = {
        "planner": "grasp_planning_v2_inverse_kinematics",
        "frame": "soft_arm_center",
        "arm_frame": "+X lateral, +Y down/main axis, -Z camera; bend plane XZ",
        "vision_to_arm_scale": vision_scale.tolist(),
        "ball_center_mm": ball_center_mm.tolist(),
        "ball_arm_mm": ball_arm_mm.tolist(),
        "robot_base_mm": robot_base_mm.tolist(),
        "to_ball_xz_mm": to_xz.tolist(),
        "neutral_pose_deg": neutral_pose.tolist(),
        "rope_home_pose_deg": rope_home.tolist(),
        "home_pose_deg": rope_home.tolist(),
        "final_pose_deg": final_pose.tolist(),
        "motor14_auto": motor14_auto,
        "motor1_steps_cmd": m1,
        "motor4_steps_cmd": m4,
        "motor1_steps_from_home": m1,
        "motor4_steps_from_home": m4,
        "motor14_sum_abs": abs(m1) + abs(m4),
        "motor14_in_range": safe,
        "success": safe,
        "section3_phi_deg": float(final_pose[2]),
        "section3_theta_deg": float(final_pose[5]),
        "section3_bend_xz": bend_s3.tolist(),
        "section12_phi_ik_deg": float(final_pose[0]),
        "section12_phi_offset_deg": float(phi_offset_deg),
        "approach_dist_xz_mm": float(np.linalg.norm(to_xz)),
    }
    return PlanResultV2("inverse_kinematics", final_pose, trajectory, info)


def save_plan_v2(
    path: Path,
    result: PlanResultV2,
    ball_center_mm: np.ndarray,
    radius: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PLAN_VERSION,
        "created_by": "grasp_planning_v2.py",
        "ball_center_mm": np.asarray(ball_center_mm, dtype=float).tolist(),
        "ball_radius_mm": float(radius),
        "approach": result.approach,
        "success": bool(result.info["success"]),
        "safe_to_execute": bool(result.info["motor14_in_range"]),
        "home_pose_deg": result.trajectory[0].tolist(),
        "final_pose_deg": result.final_pose.tolist(),
        "trajectory_deg": result.trajectory.tolist(),
        "metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in result.info.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
