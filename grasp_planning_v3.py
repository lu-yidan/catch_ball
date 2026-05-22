#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绳组逆运动学规划 v3（逻辑同 v2，输出 grasp_plan_v3.json）。

坐标系与相机 / soft_arm_center 一致:
  +X 侧向, +Y 向下(主轴), -Z 朝相机; 球多在 base 正 +X.

用法见 grasp_plan_execute_v3.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grasp_planning_v2 import (
    DEFAULT_ARM_AXES_CONFIG,
    DEFAULT_BALL_RADIUS_MM,
    DEFAULT_DEMO_CSV,
    GRASP_BLEND_START_V2,
    PlanResultV2,
    load_arm_axes_config,
    load_inversej_first_pose,
    plan_grasp_v2,
)

_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_V3 = _ROOT / "grasp_plan_v3.json"
PLAN_VERSION = 3

# v3 复用 v2 规划与混合参数
GRASP_BLEND_START_V3 = GRASP_BLEND_START_V2


@dataclass
class PlanResultV3:
    approach: str
    final_pose: np.ndarray
    trajectory: np.ndarray
    info: dict


def plan_grasp_v3(
    ball_center_mm: np.ndarray,
    radius: float = DEFAULT_BALL_RADIUS_MM,
    demo_csv: Path | None = None,
    n_waypoints: int = 40,
    motor1_steps: int | None = None,
    motor4_steps: int | None = None,
    robot_base_mm: np.ndarray | None = None,
    approach: str = "side",
    phi_offset_deg: float = 0.0,
    arm_axes_config: Path | None = DEFAULT_ARM_AXES_CONFIG,
) -> PlanResultV3:
    """与 v2 相同绳组 IK；返回类型为 v3。"""
    r2 = plan_grasp_v2(
        ball_center_mm=ball_center_mm,
        radius=radius,
        demo_csv=demo_csv,
        n_waypoints=n_waypoints,
        motor1_steps=motor1_steps,
        motor4_steps=motor4_steps,
        robot_base_mm=robot_base_mm,
        approach=approach,
        phi_offset_deg=phi_offset_deg,
        arm_axes_config=arm_axes_config,
    )
    info = dict(r2.info)
    info["planner"] = "grasp_planning_v3_inverse_kinematics"
    info["plan_version"] = PLAN_VERSION
    return PlanResultV3(r2.approach, r2.final_pose, r2.trajectory, info)


def save_plan_v3(
    path: Path,
    result: PlanResultV3,
    ball_center_mm: np.ndarray,
    radius: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PLAN_VERSION,
        "created_by": "grasp_planning_v3.py",
        "ball_center_mm": np.asarray(ball_center_mm, dtype=float).tolist(),
        "ball_radius_mm": float(radius),
        "approach": result.approach,
        "success": bool(result.info.get("success")),
        "safe_to_execute": bool(result.info.get("motor14_in_range")),
        "home_pose_deg": result.trajectory[0].tolist(),
        "final_pose_deg": result.final_pose.tolist(),
        "trajectory_deg": result.trajectory.tolist(),
        "metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in result.info.items()
        },
    }
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
