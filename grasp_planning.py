#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan a tennis-ball grasp with the inverse PCC model from GP_tennis.py.

Inputs can be either:
  1) a ball center in robot/base coordinates (--x --y --z), or
  2) an external JSON coordinate file from detect_tennis_blue_disk_apriltag.py, or
  3) an optional Intel RealSense D435/D435i detection (--detect-realsense).

The output JSON is consumed by grasp_excute.py.  It contains the planned
six-angle trajectory in degrees plus metadata for converting it to motor steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _ROOT.parent
_ALGORITHM_ROOT = _PROJECT_ROOT / "algorithm_May_2026"
for _path in (_ROOT, _PROJECT_ROOT, _ALGORITHM_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import GP_tennis as gp  # noqa: E402
from grasp_excute import (  # noqa: E402
    MAX_SECTION12_POSE_DELTA,
    apply_section3_bend_xy,
    motor14_grasp_penalty,
    motor14_in_grasp_range,
    project_motor14_to_grasp_range,
    section3_bend_xy,
    section3_pose_from_grasp_motor14,
    section3_rope_steps_from_home,
    solve_section3_grasp_motor14,
)

DEFAULT_BALL_RADIUS_MM = 35.0
# Grasp closing uses section-3 ropes only (motor1 phi180 + motor4 phi270).
GRASP_BLEND_START = 0.45
MAX_SECTION3_POSE_DELTA = np.array([0.0, 0.0, 36.0, 0.0, 0.0, 42.0], dtype=float)
DEFAULT_OUTPUT = _ROOT / "grasp_plan.json"
PLAN_VERSION = 1


@dataclass
class PlanResult:
    approach: str
    final_pose: np.ndarray
    trajectory: np.ndarray
    info: dict


def _as_vec3(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 numbers")
    return arr


def _resolve_radius_mm(args: argparse.Namespace, fallback_mm: float | None = None) -> float:
    if args.radius is not None:
        return float(args.radius)
    if args.diameter is not None:
        return float(args.diameter) * 0.5
    if fallback_mm is not None:
        return float(fallback_mm)
    return DEFAULT_BALL_RADIUS_MM


def load_external_ball_coordinate(path: Path) -> tuple[np.ndarray, float | None, dict]:
    """Read a ball target JSON written by the AprilTag detector."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return _as_vec3(data, "external ball center"), None, {"coord_file": str(path)}

    if not isinstance(data, dict):
        raise ValueError("external coordinate file must be a JSON object or [x, y, z] list")
    if data.get("valid") is False:
        reason = data.get("reason", "unknown reason")
        raise ValueError(f"external coordinate file is marked invalid: {path} ({reason})")
    if data.get("depth_src") == "—" or data.get("depth_m") == 0:
        raise ValueError(f"external coordinate file has no valid ball depth: {path}")

    center_mm = None
    for key in ("tennis_center_mm", "ball_center_mm", "center_mm"):
        if data.get(key) is not None:
            center_mm = _as_vec3(data[key], key)
            break
    if center_mm is None:
        for key in ("tennis_center_m", "ball_center_m", "center_m"):
            if data.get(key) is not None:
                center_mm = _as_vec3(data[key], key) * 1000.0
                break
    if center_mm is None:
        units = str(data.get("units", "mm")).lower()
        for key in ("tennis_center", "ball_center", "center"):
            if data.get(key) is not None:
                center_mm = _as_vec3(data[key], key)
                if units in {"m", "meter", "meters"}:
                    center_mm *= 1000.0
                break
    if center_mm is None:
        raise ValueError(
            "external coordinate file must contain tennis_center_mm or tennis_center_m"
        )

    radius_mm = data.get("ball_radius_mm")
    if radius_mm is None and data.get("ball_radius_m") is not None:
        radius_mm = float(data["ball_radius_m"]) * 1000.0
    if radius_mm is None and data.get("ball_diameter_m") is not None:
        radius_mm = float(data["ball_diameter_m"]) * 500.0
    if radius_mm is None and data.get("ball_diameter_mm") is not None:
        radius_mm = float(data["ball_diameter_mm"]) * 0.5

    meta = {
        "coord_file": str(path),
        "source_schema": data.get("schema"),
        "source_frame": data.get("frame"),
        "source_timestamp_s": data.get("timestamp_s"),
        "tag_status": data.get("tag_status"),
        "depth_src": data.get("depth_src"),
        "tennis_tag_mm": data.get("tennis_tag_mm"),
        "tennis_tag_m": data.get("tennis_tag_m"),
        "center_origin_in_tag_mm": data.get("center_origin_in_tag_mm"),
        "center_origin_in_tag_m": data.get("center_origin_in_tag_m"),
    }
    return center_mm, (float(radius_mm) if radius_mm is not None else None), meta


DEFAULT_CENTER_TO_ROBOT = _ROOT / "config" / "center_to_robot.json"
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
DEFAULT_TAG_AXIS_SCALES = [1.0, -1.0, 1.0]


def load_center_to_robot(path: Path | None = None) -> dict | None:
    """Load soft_arm_center -> GP robot frame calibration (mm)."""
    candidates = [path, DEFAULT_CENTER_TO_ROBOT]
    for candidate in candidates:
        if candidate is None or not Path(candidate).is_file():
            continue
        with Path(candidate).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{candidate} must be a JSON object")
        data["calib_file"] = str(candidate)
        return data
    return None


def _calib_center_origin_tag_mm(calib: dict) -> np.ndarray:
    if "center_origin_in_tag_mm" in calib:
        return _as_vec3(calib["center_origin_in_tag_mm"], "center_origin_in_tag_mm")
    if "center_origin_in_tag_m" in calib:
        return _as_vec3(calib["center_origin_in_tag_m"], "center_origin_in_tag_m") * 1000.0
    if "center_origin_in_tag_cm" in calib:
        return np.asarray(calib["center_origin_in_tag_cm"], dtype=float) * 10.0
    raise ValueError(
        "tag calibration needs center_origin_in_tag_mm (same as run_d435_vision.py)"
    )


def center_mm_to_tag_mm(center_mm: np.ndarray, calib: dict) -> np.ndarray:
    """p_tag = p_center + center_origin_in_tag (parallel axes, metres->mm)."""
    center_mm = np.asarray(center_mm, dtype=float)
    return center_mm + _calib_center_origin_tag_mm(calib)


def tag_mm_to_robot(tag_mm: np.ndarray, calib: dict) -> np.ndarray:
    """Apply per-axis map on AprilTag-frame coordinates (mm)."""
    tag_mm = np.asarray(tag_mm, dtype=float)
    out = np.zeros(3, dtype=float)
    for axis in calib.get("axes", []):
        ri = _AXIS_INDEX[str(axis["robot"]).lower()]
        src_key = "tag" if "tag" in axis else "center"
        ti = _AXIS_INDEX[str(axis[src_key]).lower()]
        out[ri] = float(axis.get("scale", 1.0)) * tag_mm[ti] + float(axis.get("offset", 0.0))
    return out


def fit_tag_to_robot_axes(
    tag_mm: np.ndarray,
    robot_mm: np.ndarray,
    scales: list[float] | None = None,
) -> list[dict]:
    """Fit tag->robot axis_map with fixed scales (default X flip on Y)."""
    tag_mm = np.asarray(tag_mm, dtype=float)
    robot_mm = np.asarray(robot_mm, dtype=float)
    if scales is None:
        scales = list(DEFAULT_TAG_AXIS_SCALES)
    axes = []
    for i, name in enumerate(("x", "y", "z")):
        axes.append(
            {
                "tag": name,
                "robot": name,
                "scale": float(scales[i]),
                "offset": float(robot_mm[i] - scales[i] * tag_mm[i]),
            }
        )
    return axes


def save_center_to_robot_calib(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def center_point_to_robot(
    point_mm: np.ndarray,
    calib: dict,
    demo_robot_center_mm: np.ndarray | None = None,
    tag_mm: np.ndarray | None = None,
) -> np.ndarray:
    """
    Map a point from soft_arm_center (vision JSON) to GP robot coordinates.

    Modes:
      - tag: AprilTag frame (preferred; uses tennis_tag_mm if provided)
      - axis_map: per-axis on center frame (legacy)
      - delta: demo_robot + rotation @ (point - demo_ball_center)
      - matrix: 4x4 center_to_robot
    """
    point_mm = np.asarray(point_mm, dtype=float)
    mode = str(calib.get("mode", "tag")).lower()

    if mode == "tag":
        if tag_mm is None:
            tag_mm = center_mm_to_tag_mm(point_mm, calib)
        return tag_mm_to_robot(tag_mm, calib)

    if mode == "matrix":
        mat = np.asarray(calib.get("center_to_robot", calib.get("matrix")), dtype=float)
        if mat.shape != (4, 4):
            raise ValueError("center_to_robot matrix must be 4x4")
        return (mat @ np.r_[point_mm, 1.0])[:3]

    if mode == "delta":
        anchor = _as_vec3(calib["demo_ball_center_mm"], "demo_ball_center_mm")
        if demo_robot_center_mm is None:
            demo_robot = _as_vec3(
                calib.get("demo_robot_center_mm", [0.0, 0.0, 0.0]),
                "demo_robot_center_mm",
            )
        else:
            demo_robot = np.asarray(demo_robot_center_mm, dtype=float)
        rot = np.asarray(calib.get("rotation", np.eye(3)), dtype=float)
        if rot.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        return demo_robot + rot @ (point_mm - anchor)

    if mode == "axis_map":
        out = np.zeros(3, dtype=float)
        for axis in calib.get("axes", []):
            ri = _AXIS_INDEX[str(axis["robot"]).lower()]
            ci = _AXIS_INDEX[str(axis["center"]).lower()]
            out[ri] = float(axis.get("scale", 1.0)) * point_mm[ci] + float(
                axis.get("offset", 0.0)
            )
        return out

    raise ValueError(f"unsupported center_to_robot mode: {mode}")


def resolve_ball_center_for_planning(
    ball_center_mm: np.ndarray,
    source_frame: str | None,
    calib: dict | None,
    demo_robot_center_mm: np.ndarray | None = None,
    tag_mm: np.ndarray | None = None,
    vision_meta: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Return ball center in GP robot frame plus transform metadata."""
    meta: dict = {
        "input_frame": source_frame,
        "input_center_mm": np.asarray(ball_center_mm, dtype=float).tolist(),
    }
    frame = (source_frame or "").lower()
    if frame in {"", "robot", "gp_robot"}:
        meta["plan_frame"] = "robot"
        meta["plan_center_mm"] = np.asarray(ball_center_mm, dtype=float).tolist()
        return np.asarray(ball_center_mm, dtype=float), meta

    if calib is None:
        raise ValueError(
            "ball_target.json 使用 soft_arm_center 坐标系，但缺少 "
            f"{DEFAULT_CENTER_TO_ROBOT}。\n"
            "请创建标定文件或传 --center-to-robot <path>（见 config/center_to_robot.json）。"
        )

    if tag_mm is None and vision_meta is not None:
        if vision_meta.get("tennis_tag_mm") is not None:
            tag_mm = _as_vec3(vision_meta["tennis_tag_mm"], "tennis_tag_mm")
        elif vision_meta.get("tennis_tag_m") is not None:
            tag_mm = _as_vec3(vision_meta["tennis_tag_m"], "tennis_tag_m") * 1000.0

    if tag_mm is None and str(calib.get("mode", "tag")).lower() == "tag":
        origin = vision_meta or {}
        if origin.get("center_origin_in_tag_mm") is not None:
            calib = dict(calib)
            calib["center_origin_in_tag_mm"] = origin["center_origin_in_tag_mm"]
        tag_mm = center_mm_to_tag_mm(ball_center_mm, calib)

    plan_center = center_point_to_robot(
        ball_center_mm, calib, demo_robot_center_mm, tag_mm=tag_mm
    )
    meta.update(
        {
            "plan_frame": "robot",
            "plan_center_mm": plan_center.tolist(),
            "center_to_robot_mode": calib.get("mode"),
            "center_to_robot_file": calib.get("calib_file"),
        }
    )
    if tag_mm is not None:
        meta["tennis_tag_mm"] = np.asarray(tag_mm, dtype=float).tolist()
    return plan_center, meta


def sphere_keypoints(center: np.ndarray, radius: float) -> np.ndarray:
    """Environment keypoints for GP policy transport of a sphere."""
    center = np.asarray(center, dtype=float)
    r = float(radius)
    return np.array(
        [
            center,
            center + np.array([r, 0.0, 0.0]),
            center - np.array([r, 0.0, 0.0]),
            center + np.array([0.0, r, 0.0]),
            center - np.array([0.0, r, 0.0]),
            center + np.array([0.0, 0.0, r]),
            center - np.array([0.0, 0.0, r]),
        ],
        dtype=float,
    )


def _surface_gap(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    pts = np.atleast_2d(points)
    return np.abs(np.linalg.norm(pts - center, axis=1) - float(radius))


def _tip_target(center: np.ndarray, radius: float, transported_tip: np.ndarray, approach: str) -> np.ndarray:
    if approach == "top-down":
        direction = np.array([0.0, 0.0, 1.0])
    elif approach == "below":
        direction = np.array([0.0, 0.0, -1.0])
    else:
        direction = transported_tip - center
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = center - gp.ROBOT_BASE
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
    return center + direction * (float(radius) + 1.5)


def _approach_score(points: np.ndarray, center: np.ndarray, radius: float, approach: str) -> float:
    distal = points[34:]
    gaps = _surface_gap(distal, center, radius)
    near = distal[gaps <= max(10.0, radius * 0.35)]
    if len(near) == 0:
        near = distal[np.argsort(gaps)[: max(3, min(8, len(distal)))]]

    if approach == "top-down":
        return float(np.mean(near[:, 2] >= center[2] + radius * 0.10))
    if approach == "below":
        return float(np.mean(near[:, 2] <= center[2] - radius * 0.10))

    horizontal = np.linalg.norm(near[:, :2] - center[:2], axis=1)
    side_height = np.abs(near[:, 2] - center[2])
    side_ratio = np.mean(side_height <= radius * 0.85)
    horizontal_ratio = np.mean(horizontal >= radius * 0.55)
    return float(0.55 * side_ratio + 0.45 * horizontal_ratio)


def _plan_angle_bounds(demo_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    _, phi_data, theta_data = gp.load_inversej_robot_data(demo_csv)
    traj_full = np.hstack([phi_data, theta_data])
    extra_margin = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=float)
    lower = np.maximum(traj_full.min(axis=0) - extra_margin, [0, 0, 0, 0, 0, 0])
    upper = np.minimum(traj_full.max(axis=0) + extra_margin, [360, 360, 360, 170, 170, 170])
    lower[0:3] = 0.0
    upper[0:3] = 360.0
    return lower, upper


def _clip_final_pose(
    candidate: np.ndarray,
    source_pose: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    delta = candidate - source_pose
    per_dim_cap = np.minimum(gp.MAX_POSE_DELTA, MAX_SECTION12_POSE_DELTA + MAX_SECTION3_POSE_DELTA)
    delta = np.clip(delta, -per_dim_cap, per_dim_cap)
    return np.clip(source_pose + delta, lower, upper)


def tune_ball_final_pose(
    source_pose: np.ndarray,
    target_center: np.ndarray,
    radius: float,
    transported_tip: np.ndarray,
    angle_bounds: tuple[np.ndarray, np.ndarray],
    approach: str,
    home_pose: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Tune grasp pose: sections 1-2 adjust approach; section 3 closes via rope
    kinematics (motor1/4 in 6500..8000 each, |m1|+|m4| < 15000).
    """
    lower, upper = angle_bounds
    source_pose = np.clip(np.asarray(source_pose, dtype=float), lower, upper)
    home_pose = np.asarray(home_pose, dtype=float)

    pose = source_pose.copy()
    pose = np.clip(pose, lower, upper)

    source_tip_ref = gp.pose_to_points(source_pose, gp.SEGMENT_LENGTHS, gp.ROBOT_BASE)[-1]
    source_reach = float(np.linalg.norm(source_tip_ref - gp.ROBOT_BASE))
    tip_target = _tip_target(target_center, radius, transported_tip, approach)

    def clip_pose(candidate: np.ndarray) -> np.ndarray:
        out = _clip_final_pose(candidate, source_pose, lower, upper)
        out[2] = np.clip(candidate[2], lower[2], upper[2])
        out[5] = np.clip(candidate[5], lower[5], upper[5])
        return out

    def evaluate(candidate: np.ndarray) -> tuple[float, dict]:
        candidate = clip_pose(candidate)
        points = gp.pose_to_points(candidate, gp.SEGMENT_LENGTHS, gp.ROBOT_BASE)
        curl_center = gp.get_curl_center_from_points(points, gp.ROBOT_BASE)
        tip = points[-1]
        distal = points[34:]

        center_error = float(np.linalg.norm(curl_center - target_center))
        tip_target_error = float(np.linalg.norm(tip - tip_target))
        tip_surface_gap = float(_surface_gap(tip, target_center, radius)[0])
        surface_gaps = _surface_gap(distal, target_center, radius)
        wrap_gap = float(np.min(surface_gaps))
        wrap_penalty = float(5.0 * np.mean(np.clip(surface_gaps - 4.0, 0.0, None)))
        approach_score = _approach_score(points, target_center, radius, approach)
        approach_penalty = 55.0 * max(0.0, 0.45 - approach_score)

        motor1_steps, motor4_steps = section3_rope_steps_from_home(home_pose, candidate)
        rope_limit_penalty = motor14_grasp_penalty(motor1_steps, motor4_steps)

        pose_delta = candidate - source_pose
        dir_delta = pose_delta[[0, 1, 3, 4]]
        dir_penalty = float(2.5 * np.linalg.norm(dir_delta / np.array([24.0, 24.0, 32.0, 32.0])))
        sec3_delta = pose_delta[[2, 5]]
        sec3_penalty = float(1.5 * np.linalg.norm(sec3_delta / np.array([30.0, 36.0])))

        tip_reach = float(np.linalg.norm(tip - gp.ROBOT_BASE))
        reach_penalty = 120.0 * max(0.0, tip_reach - source_reach * gp.MAX_TIP_REACH_SCALE)

        score = (
            0.35 * center_error
            + 0.55 * tip_target_error
            + 1.20 * tip_surface_gap
            + wrap_penalty
            + approach_penalty
            + reach_penalty
            + rope_limit_penalty
            + dir_penalty
            + sec3_penalty
        )
        return score, {
            "points": points,
            "curl_center": curl_center,
            "tip": tip,
            "center_error": center_error,
            "tip_target_error": tip_target_error,
            "tip_surface_gap": tip_surface_gap,
            "wrap_gap": wrap_gap,
            "approach_score": approach_score,
            "tip_reach": tip_reach,
            "source_reach": source_reach,
            "pose_delta": pose_delta,
            "motor1_steps": motor1_steps,
            "motor4_steps": motor4_steps,
            "section3_phi_deg": float(candidate[2]),
            "section3_theta_deg": float(candidate[5]),
            "score": score,
        }

    best_score, best_info = evaluate(pose)
    section12_steps = [6.0, 3.0, 1.5]
    for step in section12_steps:
        improved = True
        while improved:
            improved = False
            for dim in (0, 1, 3, 4):
                for direction in (-1.0, 1.0):
                    candidate = pose.copy()
                    candidate[dim] += direction * step
                    score, info = evaluate(candidate)
                    if score + 1e-6 < best_score:
                        pose = clip_pose(candidate)
                        best_score = score
                        best_info = info
                        improved = True

    motor1_steps, motor4_steps, sec3_pose = solve_section3_grasp_motor14(
        home_pose, target_center, gp.ROBOT_BASE, approach
    )
    pose = clip_pose(sec3_pose)
    motor1_steps, motor4_steps = section3_rope_steps_from_home(home_pose, pose)
    motor1_steps, motor4_steps = project_motor14_to_grasp_range(motor1_steps, motor4_steps)
    pose = clip_pose(section3_pose_from_grasp_motor14(home_pose, motor1_steps, motor4_steps))

    best_info = evaluate(pose)[1]
    best_info["pose_delta"] = pose - source_pose
    best_info["max_pose_delta"] = float(np.max(np.abs(best_info["pose_delta"])))
    best_info["motor1_steps"] = motor1_steps
    best_info["motor4_steps"] = motor4_steps
    best_info["motor14_sum_abs"] = abs(motor1_steps) + abs(motor4_steps)
    best_info["motor14_in_range"] = motor14_in_grasp_range(motor1_steps, motor4_steps)
    best_info["success"] = (
        best_info["center_error"] <= max(55.0, radius * 1.6)
        and best_info["wrap_gap"] <= max(18.0, radius * 0.55)
        and best_info["tip_surface_gap"] <= max(20.0, radius * 0.55)
        and best_info["motor14_in_range"]
        and best_info["tip_reach"] <= best_info["source_reach"] * gp.MAX_TIP_REACH_SCALE + 1e-6
    )
    return pose, best_info


def plan_grasp(
    ball_center: np.ndarray,
    radius: float = DEFAULT_BALL_RADIUS_MM,
    approach: str = "auto",
    demo_csv: Path | None = None,
    n_waypoints: int = 80,
) -> PlanResult:
    if demo_csv is None:
        demo_csv = _ROOT / gp.SOURCE_FILENAME

    source = gp._load_source_demo(demo_csv)
    angle_bounds = _plan_angle_bounds(demo_csv)

    # ----- GP transport（试验请改用 grasp_planning_v2.py，勿删） -----
    transporter = gp.GaussianProcessPolicyTransporter(
        length_scale=120.0, signal_variance=1.0, noise_variance=1e-6
    )
    transporter.fit(
        sphere_keypoints(source["source_center"], radius),
        sphere_keypoints(ball_center, radius),
    )
    transported_tip = transporter.transform(source["source_tip"][-1])
    # ----- /GP transport -----

    home_pose = source["curl_slice"][0]
    rope_home_pose = home_pose.copy()
    rope_home_pose[2] = 0.0
    rope_home_pose[5] = 0.0
    approaches = ["side"] if approach == "auto" else [approach]
    candidates: list[PlanResult] = []
    for candidate_approach in approaches:
        final_pose, info = tune_ball_final_pose(
            source["final_pose"],
            ball_center,
            radius,
            transported_tip,
            angle_bounds,
            candidate_approach,
            rope_home_pose,
        )
        candidates.append(PlanResult(candidate_approach, final_pose, np.empty((0, 6)), info))

    candidates.sort(key=lambda r: (not r.info["success"], r.info["score"]))
    best = candidates[0]

    source_traj = source["curl_slice"]
    idx = np.linspace(0, len(source_traj) - 1, max(2, int(n_waypoints))).astype(int)
    base_traj = source_traj[idx].copy()
    phase = np.linspace(0.0, 1.0, len(base_traj))
    pose_delta_full = best.final_pose - source["final_pose"]
    dir_delta = np.zeros(6, dtype=float)
    dir_delta[[0, 1, 3, 4]] = pose_delta_full[[0, 1, 3, 4]]
    final_bend_s3 = section3_bend_xy(best.final_pose)
    rope_home_bend_s3 = section3_bend_xy(rope_home_pose)

    trajectory = []
    for row, t in zip(base_traj, phase):
        waypoint = row + t * dir_delta
        if t >= GRASP_BLEND_START:
            blend = (t - GRASP_BLEND_START) / max(1e-6, 1.0 - GRASP_BLEND_START)
            bend = (1.0 - blend) * rope_home_bend_s3 + blend * final_bend_s3
            waypoint = apply_section3_bend_xy(waypoint, bend)
        trajectory.append(waypoint)
    best.trajectory = np.asarray(trajectory, dtype=float)
    best.trajectory[-1] = best.final_pose
    m1, m4 = section3_rope_steps_from_home(rope_home_pose, best.final_pose)
    best.info["source_center"] = source["source_center"]
    best.info["transported_tip"] = transported_tip
    best.info["demo"] = str(demo_csv)
    best.info["home_pose_deg"] = home_pose.tolist()
    best.info["rope_home_pose_deg"] = rope_home_pose.tolist()
    best.info["motor1_steps_from_home"] = m1
    best.info["motor4_steps_from_home"] = m4
    best.info["motor14_sum_abs"] = abs(m1) + abs(m4)
    best.info["motor14_in_range"] = motor14_in_grasp_range(m1, m4)
    return best


def detect_tennis_ball_catch_ball(
    vision_cfg_path: Path | None = None,
) -> tuple[np.ndarray, float, dict]:
    """Detect tennis ball via catch_ball HSV + AprilTag + centre frame."""
    from vision_catch_ball import detect_tennis_once, load_vision_config

    cfg_path = vision_cfg_path or (_ROOT / "config" / "vision_local.json")
    det = detect_tennis_once(vision_cfg=load_vision_config(cfg_path), no_viz=True)
    meta = {
        "tag_detected": det.tag_detected,
        "tennis_center_m": det.tennis_center_m.tolist() if det.tennis_center_m is not None else None,
        "depth_src": det.depth_src,
        "vision_config": str(cfg_path),
    }
    return det.ball_center_mm, det.ball_radius_mm, meta


def load_camera_to_robot(path: Path | None) -> dict:
    """
    Load vision calibration.

    Supported JSON formats:
      1) {"camera_to_robot": [[... 4x4 ...]]}
      2) {
           "base_origin_camera_mm": [x, y, z],
           "metal_rod_z_camera_mm": z0,
           "z_sign": -1
         }

    Format 2 uses the soft-arm base center as robot x/y origin and the
    horizontal metal rod as the robot z=0 plane.
    """
    if path is None:
        return {"camera_to_robot": np.eye(4)}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) or "camera_to_robot" in data:
        mat = np.asarray(data.get("camera_to_robot", data), dtype=float)
        if mat.shape != (4, 4):
            raise ValueError("camera_to_robot transform must be a 4x4 matrix")
        return {"camera_to_robot": mat}

    base_origin = np.asarray(data["base_origin_camera_mm"], dtype=float)
    if base_origin.shape != (3,):
        raise ValueError("base_origin_camera_mm must contain 3 numbers")
    return {
        "base_origin_camera_mm": base_origin,
        "metal_rod_z_camera_mm": float(data["metal_rod_z_camera_mm"]),
        "z_sign": float(data.get("z_sign", -1.0)),
    }


def camera_point_to_robot(point_mm: np.ndarray, calibration: dict) -> np.ndarray:
    """Convert D435 camera point to robot frame."""
    point_mm = np.asarray(point_mm, dtype=float)
    if "camera_to_robot" in calibration:
        return (calibration["camera_to_robot"] @ np.r_[point_mm, 1.0])[:3]

    origin = calibration["base_origin_camera_mm"]
    rod_z = calibration["metal_rod_z_camera_mm"]
    z_sign = calibration.get("z_sign", -1.0)
    return np.array(
        [
            point_mm[0] - origin[0],
            point_mm[1] - origin[1],
            z_sign * (point_mm[2] - rod_z),
        ],
        dtype=float,
    )


def detect_tennis_ball_realsense(
    radius_mm: float,
    camera_to_robot: dict | np.ndarray,
    warmup_frames: int = 20,
) -> np.ndarray:
    """Detect a yellow/green tennis ball and return center in robot coordinates."""
    try:
        import cv2
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("RealSense detection needs pyrealsense2 and opencv-python") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("No aligned RealSense frames")

        color = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 55, 45]), np.array([85, 255, 255]))
        mask = cv2.medianBlur(mask, 7)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No tennis-ball colored blob detected")

        contour = max(contours, key=cv2.contourArea)
        (u, v), pixel_radius = cv2.minEnclosingCircle(contour)
        if pixel_radius < 6:
            raise RuntimeError("Detected blob is too small")

        depth_m = depth_frame.get_distance(int(round(u)), int(round(v)))
        if depth_m <= 0.0:
            raise RuntimeError("Invalid depth at detected ball center")

        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        camera_xyz_m = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], depth_m)
        point_mm = np.array(camera_xyz_m, dtype=float) * 1000.0
        if isinstance(camera_to_robot, np.ndarray):
            calibration = {"camera_to_robot": camera_to_robot}
        else:
            calibration = camera_to_robot
        center = camera_point_to_robot(point_mm, calibration)
        print(
            f"Detected ball pixel=({u:.1f}, {v:.1f}), pixel_radius={pixel_radius:.1f}, "
            f"center_robot_mm={center.round(2).tolist()}, radius_mm={radius_mm:.1f}"
        )
        return center
    finally:
        pipeline.stop()


def save_plan(path: Path, result: PlanResult, ball_center: np.ndarray, radius: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PLAN_VERSION,
        "created_by": "grasp_planning.py",
        "ball_center_mm": ball_center.tolist(),
        "ball_radius_mm": float(radius),
        "approach": result.approach,
        "success": bool(result.info["success"]),
        "home_pose_deg": result.trajectory[0].tolist(),
        "final_pose_deg": result.final_pose.tolist(),
        "trajectory_deg": result.trajectory.tolist(),
        "metrics": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in result.info.items()
            if key != "points"
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a tennis-ball grasp trajectory.")
    parser.add_argument("--x", type=float, help="ball center X in robot/base frame, mm")
    parser.add_argument("--y", type=float, help="ball center Y in robot/base frame, mm")
    parser.add_argument("--z", type=float, help="ball center Z in robot/base frame, mm")
    parser.add_argument("--radius", type=float, help="ball radius in mm")
    parser.add_argument("--diameter", type=float, help="ball diameter in mm; 70 for a 7 cm ball")
    parser.add_argument("--approach", choices=["auto", "side", "top-down", "below"], default="auto")
    parser.add_argument("--demo", type=Path, default=_ROOT / gp.SOURCE_FILENAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-waypoints", type=int, default=80)
    parser.add_argument("--strict", action="store_true", help="exit non-zero if planner marks the grasp unsafe")
    parser.add_argument(
        "--coord-file",
        type=Path,
        help="external ball coordinate JSON in soft-arm/base frame from detect_tennis_blue_disk_apriltag.py",
    )
    parser.add_argument(
        "--center-to-robot",
        type=Path,
        default=None,
        help="soft_arm_center -> robot calibration JSON (default config/center_to_robot.json)",
    )
    parser.add_argument(
        "--detect-realsense",
        action="store_true",
        help="use catch_ball HSV + D435/D455 + AprilTag (recommended)",
    )
    parser.add_argument("--camera-to-robot", type=Path, help="legacy JSON 4x4 camera_to_robot transform, mm")
    parser.add_argument(
        "--vision-calib",
        type=Path,
        help="legacy JSON with base_origin_camera_mm and metal_rod_z_camera_mm",
    )
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=_ROOT / "config" / "vision_local.json",
        help="catch_ball vision config (HSV, AprilTag, centre origin)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vision_meta: dict | None = None
    if args.detect_realsense:
        ball_center, detected_radius, vision_meta = detect_tennis_ball_catch_ball(args.vision_config)
        radius = _resolve_radius_mm(args, detected_radius)
    elif args.coord_file is not None:
        ball_center, file_radius, vision_meta = load_external_ball_coordinate(args.coord_file)
        radius = _resolve_radius_mm(args, file_radius)
        calib = load_center_to_robot(args.center_to_robot)
        demo_source = gp._load_source_demo(args.demo)["source_center"]
        ball_center, frame_meta = resolve_ball_center_for_planning(
            ball_center,
            vision_meta.get("source_frame"),
            calib,
            demo_source,
            vision_meta=vision_meta,
        )
        vision_meta.update(frame_meta)
    else:
        radius = _resolve_radius_mm(args)
        missing = [name for name in ("x", "y", "z") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"Missing ball coordinate(s): {', '.join(missing)}")
        ball_center = _as_vec3([args.x, args.y, args.z], "ball center")

    if not args.demo.is_file():
        raise SystemExit(
            "Missing GP demo CSV: "
            f"{args.demo}\n"
            "Pass the recorded inverse-J demo with --demo <path-to-csv>. "
            "The planner needs this source trajectory for GP transport."
        )

    result = plan_grasp(
        ball_center=ball_center,
        radius=radius,
        approach=args.approach,
        demo_csv=args.demo,
        n_waypoints=args.n_waypoints,
    )
    save_plan(args.output, result, ball_center, radius)
    if vision_meta is not None:
        with args.output.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["vision"] = vision_meta
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"Saved plan: {args.output}\n"
        f"  approach={result.approach}, success={result.info['success']}, "
        f"radius={radius:.1f} mm, "
        f"center_error={result.info['center_error']:.1f} mm, "
        f"wrap_gap={result.info['wrap_gap']:.1f} mm, "
        f"tip_surface_gap={result.info['tip_surface_gap']:.1f} mm, "
        f"motor1_steps={result.info.get('motor1_steps_from_home', result.info.get('motor1_steps'))}, "
        f"motor4_steps={result.info.get('motor4_steps_from_home', result.info.get('motor4_steps'))}"
    )
    if args.strict and not result.info["success"]:
        raise SystemExit("Planner did not meet success thresholds.")


if __name__ == "__main__":
    main()
