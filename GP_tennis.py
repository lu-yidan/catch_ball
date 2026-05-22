#!/usr/bin/env python
"""
GP_IL_inverseJ — J-shape imitation learning + v4-style generalization figures

约束：
  - 使用 2020_03_13_chishuru0016_4_angles_data.csv 复现 J-shape 卷取
  - 固定阶段帧：0 initial, 120 approaching, 190 start curling, 350 curling, 530 grasped
  - 输出 v4 风格泛化图，同时补充 v3 风格原始演示数据图

输出：
  - GP_IL_inverseJ.png / GP_IL_inverseJ_transparent.png
  - GP_IL_inverseJ_posture_grid.png
  - GP_IL_inverseJ_overview.png
  - GP_IL_inverseJ_demo_keyframes.png / GP_IL_inverseJ_demo_snapshots.png

Usage:
    python GP_IL_inverseJ.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 复用 GP_IL 运动学 / GP 运输 / 微调
_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from GP_IL import (  # noqa: E402
        GaussianProcessPolicyTransporter,
        compute_tracking_error,
        create_cylinder,
        generate_fixed_base_curl_trajectory,
        get_curl_center_from_points,
        pcc_transformation_matrix,
        SimpleDMP_V0,
    )
except ModuleNotFoundError:
    class GaussianProcessPolicyTransporter:
        """Minimal GP policy transport used by grasp_planning.py."""

        def __init__(
            self,
            length_scale: float = 120.0,
            signal_variance: float = 1.0,
            noise_variance: float = 1e-6,
        ):
            self.length_scale = float(length_scale)
            self.signal_variance = float(signal_variance)
            self.noise_variance = float(noise_variance)
            self.source_points: np.ndarray | None = None
            self.weights: np.ndarray | None = None

        def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a = np.atleast_2d(np.asarray(a, dtype=float))
            b = np.atleast_2d(np.asarray(b, dtype=float))
            sqdist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
            return self.signal_variance * np.exp(-0.5 * sqdist / (self.length_scale**2))

        def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> None:
            source_points = np.asarray(source_points, dtype=float)
            target_points = np.asarray(target_points, dtype=float)
            displacement = target_points - source_points
            k_mat = self._kernel(source_points, source_points)
            k_mat += self.noise_variance * np.eye(len(source_points))
            self.source_points = source_points
            self.weights = np.linalg.solve(k_mat, displacement)

        def transform(self, points: np.ndarray) -> np.ndarray:
            if self.source_points is None or self.weights is None:
                raise RuntimeError("GaussianProcessPolicyTransporter.fit() must be called first")
            points_arr = np.asarray(points, dtype=float)
            original_shape = points_arr.shape
            flat = np.atleast_2d(points_arr.reshape(-1, 3))
            transported = flat + self._kernel(flat, self.source_points) @ self.weights
            return transported.reshape(original_shape)

    def get_curl_center_from_points(points: np.ndarray, robot_base: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        distal = points[34:] if len(points) > 34 else points
        return np.mean(distal, axis=0)

    def compute_tracking_error(reference: np.ndarray, reproduced: np.ndarray) -> dict:
        diff = np.asarray(reference, dtype=float) - np.asarray(reproduced, dtype=float)
        return {"total_rmse": float(np.sqrt(np.mean(diff**2)))}

    def create_cylinder(center, radius, height, axis="z", n=32):
        center = np.asarray(center, dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        circle = np.c_[np.cos(angles) * radius, np.sin(angles) * radius]
        faces = []
        for i in range(n):
            j = (i + 1) % n
            if axis == "x":
                p0 = center + np.array([-height / 2, circle[i, 0], circle[i, 1]])
                p1 = center + np.array([-height / 2, circle[j, 0], circle[j, 1]])
                p2 = center + np.array([height / 2, circle[j, 0], circle[j, 1]])
                p3 = center + np.array([height / 2, circle[i, 0], circle[i, 1]])
            elif axis == "y":
                p0 = center + np.array([circle[i, 0], -height / 2, circle[i, 1]])
                p1 = center + np.array([circle[j, 0], -height / 2, circle[j, 1]])
                p2 = center + np.array([circle[j, 0], height / 2, circle[j, 1]])
                p3 = center + np.array([circle[i, 0], height / 2, circle[i, 1]])
            else:
                p0 = center + np.array([circle[i, 0], circle[i, 1], -height / 2])
                p1 = center + np.array([circle[j, 0], circle[j, 1], -height / 2])
                p2 = center + np.array([circle[j, 0], circle[j, 1], height / 2])
                p3 = center + np.array([circle[i, 0], circle[i, 1], height / 2])
            faces.append([p0, p1, p2, p3])
        return None, faces

    def generate_fixed_base_curl_trajectory(*args, **kwargs):
        raise NotImplementedError("generate_fixed_base_curl_trajectory requires the original GP_IL.py")

    def pcc_transformation_matrix(alpha: float, beta: float, l0: float) -> np.ndarray:
        beta = float(beta)
        alpha = float(alpha)
        rot_z = np.array(
            [
                [np.cos(beta), -np.sin(beta), 0.0],
                [np.sin(beta), np.cos(beta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        if abs(alpha) < 1e-9:
            translation = np.array([0.0, 0.0, float(l0)])
        else:
            radius = float(l0) / alpha
            translation = rot_z @ np.array(
                [radius * (1.0 - np.cos(alpha)), 0.0, radius * np.sin(alpha)]
            )
        out = np.eye(4)
        out[:3, :3] = rot_z
        out[:3, 3] = translation
        return out

    class SimpleDMP_V0:
        def imitate(self, demo, *args, **kwargs):
            self.demo = np.asarray(demo, dtype=float)

        def generate(self):
            return self.demo.copy(), None

# Marker CSV -> Part 3 基座系 B（pcc_kinematics.tex + PC_control_feete）
INVERTED_PCC_THETA_SIGN = -1.0
INVERTED_PCC_PHI_SIGN = -1.0

OUT_DIR = _ROOT / "output_gp_il_inverseJ"
SOURCE_FILENAME = str(
    _ROOT / "data" / "demo" / "2020_03_13_chishuru0016_4_angles_data.csv"
)
PHASE_FRAMES = {
    "initial": 0,
    "approaching": 120,
    "start_curling": 190,
    "curling": 350,
    "grasped": 530,
}
SEGMENT_LENGTHS = np.array([150.0, 150.0, 150.0])
ROBOT_BASE = np.array([0.0, 0.0, 1000.0])
# CSV 的 Plane1/2/3 对应 base->tip 三段 marker 平面。
PCC_SECTION_ORDER = np.array([0, 1, 2], dtype=int)
# inverse-J 实验中木头为空中横放圆柱：沿 X 轴放置，半径加大以体现抓紧。
CYL_R, CYL_H = 38.0, 105.0
CYL_AXIS = "x"
SEG_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
N_SUCCESS = 16
N_FROM_BELOW_MAX = 2
# 相对 base 的水平散布（mm）
SPREAD_X = (-135.0, 135.0)
SPREAD_Y = (-320.0, 90.0)
MIN_XY_FROM_BASE = 42.0
MIN_TARGET_SEP_XY = 25.0
MAX_PER_QUADRANT = 8
RES_4K = (3840, 2160)
WOOD_AXIS_ELEV = 0.0
WOOD_AXIS_AZIM = 0.0

SIDE_TIP_RADIUS = CYL_R + 0.5
SIDE_TIP_Z_OFFSET = 0.0
# 相对演示最后一帧的最大角度偏移（禁止为够到目标而把臂“拉直/拉长”）
MAX_POSE_DELTA = np.array([34.0, 34.0, 34.0, 40.0, 40.0, 36.0], dtype=float)
MAX_TIP_REACH_SCALE = 1.0
N_TOP_DOWN_MIN = 16
N_PER_MIRROR_SIDE = 8
# 连续 Part 3 链式运动学下的可达池偏向源中心一侧，镜像分割线略向内侧移动。
MIRROR_AXIS_Y_OFFSET = 60.0


def _marker_to_part3_angles_deg(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marker 正放 CSV 角 -> Part 3 基座系 (phi, theta) 度。"""
    pose = np.asarray(pose, dtype=float)
    phi_deg = pose[0:3][PCC_SECTION_ORDER] * INVERTED_PCC_PHI_SIGN
    theta_deg = pose[3:6][PCC_SECTION_ORDER] * INVERTED_PCC_THETA_SIGN
    return phi_deg, theta_deg


def pcc_transformation_matrix_inverted(alpha: float, beta: float, l0: float) -> np.ndarray:
    """
    单节倒立 PCC 变换（tex Part 3 截图公式）。

    该矩阵只用于把第一节基座从正立 PCC 基座翻到世界系倒立基座；
    多节串联时，后续节仍必须写在上一节末端系下，不能每节重复左乘 R_x(pi)。
    """
    g_part2 = pcc_transformation_matrix(alpha, beta, l0)
    rx_pi = np.diag([1.0, -1.0, -1.0])
    g_inv = np.eye(4)
    g_inv[:3, :3] = rx_pi @ g_part2[:3, :3]
    g_inv[:3, 3] = rx_pi @ g_part2[:3, 3]
    return g_inv


def _section_world_plane_direction(beta_deg: float, tangent: np.ndarray) -> np.ndarray:
    """取 marker 弯曲平面方向在当前切线法平面内的投影，保证段间切线连续。"""
    beta = np.deg2rad(beta_deg)
    plane_dir = np.array([np.sin(beta), np.cos(beta), 0.0], dtype=float)
    bend_dir = plane_dir - float(np.dot(plane_dir, tangent)) * tangent
    norm = float(np.linalg.norm(bend_dir))
    if norm < 1e-9:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        bend_dir = fallback - float(np.dot(fallback, tangent)) * tangent
        norm = float(np.linalg.norm(bend_dir))
    return bend_dir / max(norm, 1e-12)


def pose_to_points(
    pose: np.ndarray,
    segment_lengths: np.ndarray,
    robot_base: np.ndarray,
    num_points_per_segment: int = 20,
) -> np.ndarray:
    """
    倒立 PCC 正运动学（marker 平面连续圆弧版）。

    参考图中的 Plane1/2/3 是每段 marker 拟合出的弯曲平面。这里从倒立基座
    的向下切线出发，每节在该 marker 平面内积分 PCC 圆弧，并把下一节初始
    切线设为上一节圆弧末端切线，保证三节连续柔顺。
    """
    phi_deg, theta_deg = _marker_to_part3_angles_deg(pose)
    segment_lengths = np.asarray(segment_lengths, dtype=float)[PCC_SECTION_ORDER]
    robot_base = np.asarray(robot_base, dtype=float)

    points: list[np.ndarray] = [robot_base.copy()]
    position = robot_base.copy()
    tangent = np.array([0.0, 0.0, -1.0], dtype=float)

    for seg_i in range(3):
        alpha = np.deg2rad(theta_deg[seg_i])
        length = float(segment_lengths[seg_i])
        bend_dir = _section_world_plane_direction(phi_deg[seg_i], tangent)

        for i in range(1, num_points_per_segment + 1):
            frac = i / num_points_per_segment
            alpha_i = alpha * frac
            length_i = length * frac
            if abs(alpha) < 1e-9:
                point = position + tangent * length_i
            else:
                radius = length / alpha
                point = (
                    position
                    + radius * np.sin(alpha_i) * tangent
                    + radius * (1.0 - np.cos(alpha_i)) * bend_dir
                )
            points.append(point.copy())

        if abs(alpha) < 1e-9:
            position = position + tangent * length
        else:
            radius = length / alpha
            position = (
                position
                + radius * np.sin(alpha) * tangent
                + radius * (1.0 - np.cos(alpha)) * bend_dir
            )
            tangent = np.cos(alpha) * tangent + np.sin(alpha) * bend_dir
            tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)

    return np.array(points)


def compute_tip_trajectory(
    trajectory: np.ndarray,
    segment_lengths: np.ndarray,
    robot_base: np.ndarray,
) -> np.ndarray:
    tips = np.zeros((len(trajectory), 3), dtype=float)
    for idx, pose in enumerate(trajectory):
        tips[idx] = pose_to_points(pose, segment_lengths, robot_base)[-1]
    return tips


def load_inversej_robot_data(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 inverse-J CSV：Plane* 是 beta，PCC_Theta* 才是 PCC 弯曲角 alpha。"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    frames = df.iloc[:, 0].to_numpy()

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
        # 兼容旧的 6 维 phi/theta 文件。
        phi = df.iloc[:, 1:4].to_numpy(dtype=float)
        theta = df.iloc[:, 4:7].to_numpy(dtype=float)

    print(f"  加载 {len(frames)} 帧 inverse-J 数据")
    print(f"  Phi/Plane 范围: [{phi.min():.2f}, {phi.max():.2f}] 度")
    print(f"  PCC Theta 范围: [{theta.min():.2f}, {theta.max():.2f}] 度")
    return frames, phi, theta


def _load_source_demo(demo_csv: Path, grasp_time_txt: Path | None = None):
    filename = demo_csv.name
    frames, phi, theta = load_inversej_robot_data(demo_csv)
    traj = np.hstack([phi, theta])

    grasp_start = min(PHASE_FRAMES["start_curling"], len(traj) - 1)
    grasp_end = min(PHASE_FRAMES["grasped"], len(traj) - 1)
    approach_frame = min(PHASE_FRAMES["approaching"], grasp_start)
    curl_mid_frame = min(PHASE_FRAMES["curling"], grasp_end)
    curl_slice = traj[: grasp_end + 1]
    points_final = pose_to_points(traj[grasp_end], SEGMENT_LENGTHS, ROBOT_BASE)
    source_center = get_curl_center_from_points(points_final, ROBOT_BASE)
    source_tip = compute_tip_trajectory(curl_slice, SEGMENT_LENGTHS, ROBOT_BASE)
    return {
        "filename": filename,
        "trajectory": traj,
        "frames": frames,
        "curl_slice": curl_slice,
        "source_center": source_center,
        "source_tip": source_tip,
        "final_pose": curl_slice[-1],
        "phase_frames": {
            "initial": 0,
            "approaching": approach_frame,
            "start_curling": grasp_start,
            "curling": curl_mid_frame,
            "grasped": grasp_end,
        },
    }


def _quadrant(xy: np.ndarray, origin: np.ndarray) -> str:
    """相对 base 的象限标签 (+x+y, +x-y, -x+y, -x-y, axis)。"""
    d = xy - origin
    if abs(d[0]) < 18 and abs(d[1]) < 18:
        return "axis"
    qx = "+" if d[0] >= 0 else "-"
    qy = "+" if d[1] >= 0 else "-"
    return f"{qx}x{qy}y"


def _object_keypoints(center: np.ndarray) -> np.ndarray:
    """横放圆柱的环境关键点：长度沿 X，圆截面在 YZ 平面。"""
    center = np.asarray(center, dtype=float)
    return np.array(
        [
            center,
            center + np.array([CYL_H / 2.0, 0.0, 0.0]),
            center - np.array([CYL_H / 2.0, 0.0, 0.0]),
            center + np.array([0.0, CYL_R, 0.0]),
            center - np.array([0.0, CYL_R, 0.0]),
            center + np.array([0.0, 0.0, CYL_R]),
            center - np.array([0.0, 0.0, CYL_R]),
        ],
        dtype=float,
    )


def _horizontal_surface_gap(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """点到有限长横放圆柱表面的近似间隙，半径在 YZ 平面，长度沿 X。"""
    pts = np.atleast_2d(points)
    radial = np.linalg.norm(pts[:, 1:3] - center[1:3], axis=1)
    radial_gap = np.abs(radial - CYL_R)
    x_over = np.clip(np.abs(pts[:, 0] - center[0]) - CYL_H / 2.0, 0.0, None)
    return np.sqrt(radial_gap**2 + x_over**2)


def _side_direction(target_center: np.ndarray) -> np.ndarray:
    """从 base 指向木块的水平单位向量，作为侧面接触方向。"""
    direction = target_center[:2] - ROBOT_BASE[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return direction / norm


def _side_tip_target(target_center: np.ndarray, transported_tip: np.ndarray) -> np.ndarray:
    """末端贴在横放圆柱的 YZ 圆截面外表面。"""
    direction = transported_tip[1:3] - target_center[1:3]
    norm = float(np.linalg.norm(direction))
    if norm < 12.0:
        direction = ROBOT_BASE[1:3] - target_center[1:3]
        norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([0.0, 1.0])
    else:
        direction = direction / norm
    # 参考实验是从圆柱上方往下卷，末端目标约束在横放圆柱上半侧。
    direction[1] = max(direction[1], 0.35)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    x = np.clip(transported_tip[0], target_center[0] - CYL_H / 2.0, target_center[0] + CYL_H / 2.0)
    return target_center + np.array(
        [
            x - target_center[0],
            direction[0] * SIDE_TIP_RADIUS,
            direction[1] * SIDE_TIP_RADIUS + SIDE_TIP_Z_OFFSET,
        ]
    )


def below_wrap_score(final_points: np.ndarray, target_center: np.ndarray) -> float:
    """0=侧面卷取，1=自下而上（末端/卷曲段在横放圆柱底部以下且接近）。"""
    bottom_z = target_center[2] - CYL_R
    distal = final_points[34:]
    tip = final_points[-1]

    tip_h = float(np.linalg.norm(tip[1:3] - target_center[1:3]))
    tip_under = max(0.0, bottom_z - tip[2]) / 35.0
    tip_under *= max(0.0, 1.0 - tip_h / (CYL_R * 2.2))

    near = distal[np.linalg.norm(distal[:, 1:3] - target_center[1:3], axis=1) < CYL_R * 2.5]
    if len(near) == 0:
        near_under = 0.0
    else:
        near_under = float(np.mean(np.clip((bottom_z - near[:, 2]) / 35.0, 0.0, 1.0)))

    return float(np.clip(0.65 * tip_under + 0.35 * near_under, 0.0, 1.0))


def top_down_wrap_score(final_points: np.ndarray, target_center: np.ndarray) -> float:
    """越接近 1 表示远端从横放圆柱上方接触并向下包覆。"""
    distal = final_points[34:]
    surface_gap = _horizontal_surface_gap(distal, target_center)
    near = distal[surface_gap <= max(12.0, CYL_R * 0.38)]
    if len(near) == 0:
        near = distal[np.argsort(surface_gap)[: max(3, min(8, len(distal)))]]

    top_ratio = float(np.mean(near[:, 2] >= target_center[2] + CYL_R * 0.10))
    high_pass = 1.0 if float(np.max(distal[:, 2])) >= target_center[2] + CYL_R * 0.45 else 0.0
    not_under = 1.0 - float(np.mean(near[:, 2] < target_center[2] - CYL_R * 0.45))
    tip_not_under = 1.0 if final_points[-1, 2] >= target_center[2] - CYL_R * 0.80 else 0.0
    return float(np.clip(0.45 * top_ratio + 0.25 * high_pass + 0.20 * not_under + 0.10 * tip_not_under, 0.0, 1.0))


def classify_wrap_approach(final_points: np.ndarray, target_center: np.ndarray) -> str:
    """优先区分 top-down；below 表示从圆柱底部以下接近。"""
    if top_down_wrap_score(final_points, target_center) >= 0.52:
        return "top-down"
    return "below" if below_wrap_score(final_points, target_center) >= 0.35 else "side"


def tune_side_final_pose(
    source_pose: np.ndarray,
    target_center: np.ndarray,
    transported_tip: np.ndarray,
    segment_lengths: np.ndarray,
    robot_base: np.ndarray,
    angle_bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict]:
    """
    侧面抓取专用微调。

    与 GP_IL 的通用微调不同，这里把末端目标放到圆柱侧面，并显式惩罚
    末端或卷曲段落到圆柱底面以下。第一、二节的正则略放宽，使整体能
    像参考图那样从侧面弯过去。
    """
    lower, upper = angle_bounds
    pose = np.clip(source_pose.copy(), lower, upper)
    source_pose = pose.copy()
    source_tip_ref = pose_to_points(source_pose, segment_lengths, robot_base)[-1]
    source_reach = float(np.linalg.norm(source_tip_ref - robot_base))
    side_tip_target = _side_tip_target(target_center, transported_tip)
    bottom_z = target_center[2] - CYL_R

    def _clip_pose_delta(candidate: np.ndarray) -> np.ndarray:
        delta = np.clip(candidate - source_pose, -MAX_POSE_DELTA, MAX_POSE_DELTA)
        return np.clip(source_pose + delta, lower, upper)

    def evaluate(candidate: np.ndarray) -> tuple[float, dict]:
        candidate = _clip_pose_delta(candidate)
        points = pose_to_points(candidate, segment_lengths, robot_base)
        curl_center = get_curl_center_from_points(points, robot_base)
        tip = points[-1]

        center_error = float(np.linalg.norm(curl_center - target_center))
        side_tip_error = float(np.linalg.norm(tip - side_tip_target))
        transported_tip_error = float(np.linalg.norm(tip - transported_tip))

        distal = points[34:]
        below_amount = np.clip(bottom_z - distal[:, 2], 0.0, None)
        below_penalty = float(6.0 * np.mean(below_amount) + 80.0 * np.mean(below_amount > 4.0))

        tip_surface_gap = float(_horizontal_surface_gap(tip, target_center)[0])
        tip_h = float(np.linalg.norm(tip[1:3] - target_center[1:3]))
        side_radius_error = abs(tip_h - SIDE_TIP_RADIUS)
        side_height_penalty = 3.0 * max(0.0, bottom_z - 3.0 - tip[2])

        surface_gap = _horizontal_surface_gap(distal, target_center)
        wrap_gap = float(np.min(surface_gap))
        wrap_contact_penalty = float(6.0 * np.mean(np.clip(surface_gap - 3.0, 0.0, None)))
        top_down_score = top_down_wrap_score(points, target_center)
        top_down_penalty = 48.0 * max(0.0, 0.68 - top_down_score)

        pose_delta = candidate - source_pose
        regularization = float(
            np.linalg.norm(pose_delta / np.array([45.0, 45.0, 45.0, 60.0, 60.0, 60.0]))
        )
        tip_reach = float(np.linalg.norm(tip - robot_base))
        reach_penalty = 120.0 * max(0.0, tip_reach - source_reach * MAX_TIP_REACH_SCALE)
        elongation_penalty = 25.0 * float(np.sum(np.maximum(np.abs(pose_delta) - MAX_POSE_DELTA, 0.0)))

        score = (
            0.65 * center_error
            + 0.85 * side_tip_error
            + 0.10 * transported_tip_error
            + 1.10 * side_radius_error
            + 2.20 * tip_surface_gap
            + side_height_penalty
            + below_penalty
            + wrap_contact_penalty
            + top_down_penalty
            + reach_penalty
            + elongation_penalty
            + 8.0 * regularization
        )
        return score, {
            "points": points,
            "curl_center": curl_center,
            "tip": tip,
            "center_error": center_error,
            "side_tip_error": side_tip_error,
            "tip_error": transported_tip_error,
            "wrap_gap": wrap_gap,
            "tip_surface_gap": tip_surface_gap,
            "top_down_score": top_down_score,
            "tip_reach": tip_reach,
            "pose_delta": pose_delta,
            "score": score,
        }

    best_score, best_info = evaluate(pose)
    for step in [12.0, 6.0, 3.0, 1.5, 0.75, 0.35]:
        improved = True
        while improved:
            improved = False
            for dim in range(6):
                for direction in [-1.0, 1.0]:
                    candidate = pose.copy()
                    candidate[dim] += direction * step
                    score, info = evaluate(candidate)
                    if score + 1e-6 < best_score:
                        pose = candidate
                        best_score = score
                        best_info = info
                        improved = True

    below_score = below_wrap_score(best_info["points"], target_center)
    best_info["pose_delta"] = pose - source_pose
    best_info["below_score"] = below_score
    max_delta = float(np.max(np.abs(best_info["pose_delta"])))
    best_info["max_pose_delta"] = max_delta
    best_info["source_reach"] = source_reach
    best_info["can_wrap"] = (
        best_info["center_error"] <= 32.0
        and best_info["wrap_gap"] <= 10.0
        and best_info["tip_surface_gap"] <= 12.0
        and best_info.get("top_down_score", 0.0) >= 0.52
        and below_score <= 0.35
        and max_delta <= float(np.max(MAX_POSE_DELTA)) + 1e-6
        and best_info["tip_reach"] <= source_reach * MAX_TIP_REACH_SCALE + 1e-6
    )
    return pose, best_info


def _candidate_targets(source_center: np.ndarray, seed: int = 7) -> list[np.ndarray]:
    """以 base 为原点，在 ±x/±y 地面附近分散采样候选木块中心。"""
    rng = np.random.default_rng(seed)
    base_xy = ROBOT_BASE[:2].copy()
    ground_z = float(source_center[2])
    offsets: list[np.ndarray] = []

    # 规则网格：四象限都要覆盖
    for dx in np.linspace(SPREAD_X[0], SPREAD_X[1], 11):
        for dy in np.linspace(SPREAD_Y[0], SPREAD_Y[1], 11):
            if abs(dx) < MIN_XY_FROM_BASE and abs(dy) < MIN_XY_FROM_BASE:
                continue
            for dz in (-12, -6, 0, 6, 10):
                offsets.append(np.array([dx, dy, dz], dtype=float))

    # 象限锚点：保证 +/-x、+/-y 都有
    anchors = [
        (95, 85), (110, -75), (-100, 90), (-115, -80),
        (125, 25), (-130, 30), (40, 115), (35, -120),
        (-55, 105), (60, -105), (-90, -45), (88, -40),
        (-120, -95), (-105, -110), (-80, -70), (-135, -55),
        (-70, -100), (100, -110), (-125, 40),
    ]
    for ax, ay in anchors:
        offsets.append(np.array([ax, ay, 0.0]))
        offsets.append(np.array([ax, ay, rng.uniform(-8, 8)]))

    # inverse-J 0016_4 的可达包络主要在 source 下方/内侧，补充更密的局部候选点。
    for ax in np.linspace(45, 145, 6):
        for ay in np.linspace(65, 145, 5):
            for dz in (-8, 0, 8):
                offsets.append(np.array([ax, ay, dz], dtype=float))

    # 论文图用：围绕源木头中心生成左右镜像的目标，8 对候选用于一侧/另一侧对称展示。
    mirror_dx = np.linspace(-72, 72, N_PER_MIRROR_SIDE)
    mirror_dy = np.array([62, 78, 94, 110, 110, 94, 78, 62], dtype=float)
    mirror_dz = np.array([-8, -4, 0, 4, 8, 4, 0, -4], dtype=float)
    for dx, dy, dz in zip(mirror_dx, mirror_dy, mirror_dz):
        for sign in (-1.0, 1.0):
            xy = source_center[:2] + np.array([dx, sign * dy], dtype=float)
            offsets.append(np.array([xy[0] - base_xy[0], xy[1] - base_xy[1], dz], dtype=float))

    extra = rng.uniform(-1, 1, size=(80, 3)) * np.array([SPREAD_X[1], SPREAD_Y[1], 14])
    offsets.extend(extra.tolist())
    rng.shuffle(offsets)

    seen: set[tuple] = set()
    out: list[np.ndarray] = []
    for off in offsets:
        xy = base_xy + off[:2]
        if np.linalg.norm(xy - base_xy) < MIN_XY_FROM_BASE:
            continue
        c = np.array([xy[0], xy[1], ground_z + off[2]], dtype=float)
        key = tuple(np.round(c, 0))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _select_diverse_successes(
    candidates: list[dict],
    source_center: np.ndarray,
    n: int = N_SUCCESS,
) -> list[dict]:
    """优先选择 top-down，并让目标在木头两侧各 8 个，形成镜像对称布局。"""
    origin = ROBOT_BASE[:2]
    mirror_axis_y = float(source_center[1] + MIRROR_AXIS_Y_OFFSET)
    top_downs = sorted(
        [c for c in candidates if c.get("approach") == "top-down"],
        key=lambda item: (
            -item.get("top_down_score", 0.0),
            item.get("tip_surface_gap", 999.0),
            item.get("wrap_gap", 999.0),
            item["center_error"],
        ),
    )
    sides = sorted(
        [c for c in candidates if c.get("approach") in {"top-down", "side"}],
        key=lambda item: (
            0 if item.get("approach") == "top-down" else 1,
            -item.get("top_down_score", 0.0),
            -np.linalg.norm(item["target_center"][:2] - origin),
            item.get("tip_surface_gap", 999.0),
            item.get("wrap_gap", 999.0),
            item["center_error"],
        ),
    )
    belows = sorted(
        [c for c in candidates if c["approach"] == "below"],
        key=lambda item: (-item["below_score"], item["center_error"]),
    )

    chosen: list[dict] = []
    chosen_ids: set[int] = set()
    n_below = 0

    def far_enough(item: dict) -> bool:
        xy = item["target_center"][:2]
        for other in chosen:
            if np.linalg.norm(xy - other["target_center"][:2]) < MIN_TARGET_SEP_XY:
                return False
        return True

    quad_used = {"+x+y": 0, "+x-y": 0, "-x+y": 0, "-x-y": 0}

    def add_item(item: dict, force_below: bool = False) -> bool:
        nonlocal n_below
        iid = id(item)
        if iid in chosen_ids or len(chosen) >= n:
            return False
        q = _quadrant(item["target_center"][:2], origin)
        if q == "axis" or not far_enough(item):
            return False
        if q in quad_used and quad_used[q] >= MAX_PER_QUADRANT:
            return False
        is_below = force_below or item["approach"] == "below"
        if is_below and n_below >= N_FROM_BELOW_MAX:
            return False
        if is_below:
            n_below += 1
        chosen.append({**item, "approach": "below" if is_below else item.get("approach", "side")})
        chosen_ids.add(iid)
        if q in quad_used:
            quad_used[q] += 1
        return True

    def side_of(item: dict) -> str:
        return "pos_y_side" if item["target_center"][1] >= mirror_axis_y else "neg_y_side"

    def side_count(side: str) -> int:
        return sum(1 for c in chosen if side_of(c) == side)

    # 先按木头轴向两侧各选 8 个 top-down 结果，保证最终图左右镜像好看。
    for side in ("neg_y_side", "pos_y_side"):
        side_pool = [item for item in top_downs if side_of(item) == side]
        for item in side_pool:
            if side_count(side) >= N_PER_MIRROR_SIDE or len(chosen) >= n:
                break
            add_item(item)

    for item in top_downs:
        if sum(1 for c in chosen if c.get("approach") == "top-down") >= N_TOP_DOWN_MIN:
            break
        add_item(item)

    # 每象限至少 2 个（优先贴柱、再考虑远离 base）
    per_quad = 2
    for q in ("+x+y", "+x-y", "-x+y", "-x-y"):
        n_q = 0
        pool_q = sorted(
            [c for c in sides if _quadrant(c["target_center"][:2], origin) == q],
            key=lambda item: (
                item.get("tip_surface_gap", 999.0),
                item.get("wrap_gap", 999.0),
                -np.linalg.norm(item["target_center"][:2] - origin),
            ),
        )
        for item in pool_q:
            if n_q >= per_quad or len(chosen) >= n:
                break
            if add_item(item):
                n_q += 1

    for item in sides:
        if len(chosen) >= n:
            break
        add_item(item)

    for item in sides + belows:
        if len(chosen) >= n:
            break
        add_item(item)

    def quad_counts() -> dict[str, int]:
        out = {"+x+y": 0, "+x-y": 0, "-x+y": 0, "-x-y": 0}
        for c in chosen:
            q = _quadrant(c["target_center"][:2], origin)
            if q in out:
                out[q] += 1
        return out

    # 缺失象限优先补齐（可略放宽间距）
    qc = quad_counts()
    for q in ("+x+y", "+x-y", "-x+y", "-x-y"):
        pool_q = [c for c in sides if _quadrant(c["target_center"][:2], origin) == q]
        while qc[q] < 2 and len(chosen) < n and pool_q:
            added = False
            for item in pool_q:
                if id(item) in chosen_ids:
                    continue
                xy = item["target_center"][:2]
                if any(np.linalg.norm(xy - o["target_center"][:2]) < 32 for o in chosen):
                    continue
                if add_item(item):
                    qc[q] += 1
                    added = True
                    break
            if not added:
                break

    if len(chosen) < n:
        for item in sorted(candidates, key=lambda x: x["center_error"]):
            if len(chosen) >= n:
                break
            if id(item) in chosen_ids:
                continue
            if not far_enough(item):
                continue
            is_below = item["approach"] == "below" and n_below < N_FROM_BELOW_MAX
            if is_below:
                n_below += 1
            chosen.append({**item, "approach": "below" if is_below else item.get("approach", "side")})
            chosen_ids.add(id(item))

    if len(chosen) < n:
        for item in sorted(candidates, key=lambda x: x["center_error"]):
            if len(chosen) >= n:
                break
            if id(item) in chosen_ids:
                continue
            xy = item["target_center"][:2]
            if any(np.linalg.norm(xy - o["target_center"][:2]) < 22.0 for o in chosen):
                continue
            chosen.append({**item, "approach": item.get("approach", "side")})
            chosen_ids.add(id(item))

    if len(chosen) < n:
        for item in sorted(candidates, key=lambda x: x["center_error"]):
            if len(chosen) >= n:
                break
            if id(item) in chosen_ids:
                continue
            chosen.append({**item, "approach": item.get("approach", "side")})
            chosen_ids.add(id(item))

    return chosen[:n]


def _inversej_can_wrap(info: dict) -> bool:
    """
    J-shape 卷取的接触点偏在圆柱侧边，curl center 不必像 C-shape 那样落在圆柱中心。
    因此保留贴柱/侧面接触约束，适当放宽中心误差。
    """
    return (
        info["center_error"] <= 75.0
        and info["wrap_gap"] <= 45.0
        and info["tip_surface_gap"] <= 55.0
        and info.get("top_down_score", 0.0) >= 0.48
        and info.get("below_score", 0.0) <= 0.45
        and info.get("max_pose_delta", 0.0) <= float(np.max(MAX_POSE_DELTA)) + 1e-6
        and info.get("tip_reach", np.inf) <= info.get("source_reach", 0.0) * MAX_TIP_REACH_SCALE + 1e-6
    )


def _run_generalization(source: dict, angle_bounds) -> list[dict]:
    """收集 can_wrap 成功结果，再按侧面优先 + 四象限分散选出 N_SUCCESS 组。"""
    pool: list[dict] = []
    transporter = GaussianProcessPolicyTransporter(
        length_scale=120.0, signal_variance=1.0, noise_variance=1e-6
    )

    for target_center in _candidate_targets(source["source_center"]):
        target_keypoints = _object_keypoints(target_center)
        transporter.fit(_object_keypoints(source["source_center"]), target_keypoints)
        transported_tip = transporter.transform(source["source_tip"][-1])

        tuned_pose, info = tune_side_final_pose(
            source["final_pose"],
            target_center,
            transported_tip,
            SEGMENT_LENGTHS,
            ROBOT_BASE,
            angle_bounds,
        )
        if not (info["can_wrap"] or _inversej_can_wrap(info)):
            continue

        final_points = pose_to_points(tuned_pose, SEGMENT_LENGTHS, ROBOT_BASE)
        approach = classify_wrap_approach(final_points, target_center)

        bscore = info["below_score"]
        pool.append(
            {
                "target_center": target_center,
                "tuned_pose": tuned_pose,
                "final_points": final_points,
                "curl_center": info["curl_center"],
                "center_error": info["center_error"],
                "pose_delta": info["pose_delta"],
                "approach": approach,
                "below_score": bscore,
                "side_tip_error": info["side_tip_error"],
                "wrap_gap": info["wrap_gap"],
                "tip_surface_gap": info["tip_surface_gap"],
                "top_down_score": info.get("top_down_score", top_down_wrap_score(final_points, target_center)),
                "tip_reach": info["tip_reach"],
                "source_reach": info["source_reach"],
            }
        )

    return _select_diverse_successes(pool, source["source_center"], N_SUCCESS)


def _draw_small_cylinder_2d(ax, xy, color="#8d6e63", alpha=0.34, scale: float = 1.0):
    """俯视图：横放圆柱投影为沿 X 轴的圆角木块。"""
    cx, cy = xy
    width = CYL_H * scale
    height = 2.0 * CYL_R * scale
    block = patches.FancyBboxPatch(
        (cx - width / 2.0, cy - height / 2.0),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={height / 2.0}",
        facecolor=color,
        edgecolor="#5d4037",
        lw=1.1,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(block)


def _add_cylinder_3d(ax, center, facecolor="#8d6e63", alpha=0.65):
    alpha = min(alpha, 0.30)
    _, faces = create_cylinder(center, CYL_R, CYL_H, axis=CYL_AXIS)
    poly = Poly3DCollection(
        faces[:-2], alpha=alpha, facecolor=facecolor, edgecolor="#5d4037", linewidth=0.5
    )
    ax.add_collection3d(poly)


def _plot_arm_final_frame(ax, points: np.ndarray, seg_colors=SEG_COLORS, lw: float = 4.0, alpha: float = 0.92):
    """绘制最后一帧 PCC 曲线（三段配色，与 GP_IL 一致）。"""
    for seg_idx in range(3):
        start = seg_idx * 20 + (1 if seg_idx > 0 else 0)
        end = min(start + 21, len(points))
        ax.plot(
            points[start:end, 0],
            points[start:end, 1],
            points[start:end, 2],
            color=seg_colors[seg_idx],
            linewidth=lw,
            alpha=alpha,
        )
    ax.scatter(*points[-1], color="red", s=55, marker="o", zorder=10)
    ax.scatter(*ROBOT_BASE, color="black", s=90, marker="^", zorder=11)


def _set_3d_limits(
    ax,
    point_list: list[np.ndarray],
    margin: float = 70.0,
    elev: float = 22.0,
    azim: float = 45.0,
    set_view: bool = True,
):
    all_pts = np.vstack(point_list + [ROBOT_BASE])
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    ax.set_zlim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)
    if set_view:
        ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.72))
    except AttributeError:
        pass
    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Y (mm)", fontsize=9)
    ax.set_zlabel("Z (mm)", fontsize=9)


def _set_wood_axis_view(ax, point_list: list[np.ndarray], margin: float = 70.0) -> None:
    """沿横放木头轴向平视，突出从上往下包覆圆柱截面。"""
    _set_3d_limits(ax, point_list, margin=margin, elev=WOOD_AXIS_ELEV, azim=WOOD_AXIS_AZIM)


def _plot_one_experiment(ax, result: dict, exp_id: int, cmap_color=None):
    """单实验子图：最后一帧卷曲 + 加长圆柱。"""
    pts = result["final_points"]
    if cmap_color is None:
        _plot_arm_final_frame(ax, pts)
    else:
        # 蓝→黄：整臂用渐变色近似（按弧长参数）
        n = len(pts)
        for i in range(n - 1):
            t = i / max(n - 2, 1)
            c = plt.colormaps["jet"](0.15 + 0.75 * t)
            ax.plot(pts[i : i + 2, 0], pts[i : i + 2, 1], pts[i : i + 2, 2], color=c, linewidth=3.8, alpha=0.95)
        ax.scatter(*pts[-1], color="red", s=50, marker="o")
        ax.scatter(*ROBOT_BASE, color="black", s=80, marker="^")

    _add_cylinder_3d(ax, result["target_center"], facecolor="#a1887f", alpha=0.62)
    ax.scatter(*result["curl_center"], color="purple", s=35, marker="x", linewidths=1.2)
    _set_wood_axis_view(ax, [pts, result["target_center"]])
    approach = result.get("approach", "side")
    if approach == "top-down":
        tag = "TOP-DOWN wrap"
    elif approach == "side":
        tag = "SIDE wrap"
    else:
        tag = "FROM-BELOW (allowed)"
    ax.set_title(
        f"Exp {exp_id}: {tag}\n"
        f"err={result['center_error']:.1f} mm, "
        f"|dangle|={np.max(np.abs(result['pose_delta'])):.1f} deg",
        fontsize=8,
        fontweight="bold",
    )


def plot_gp_il_v4(
    source: dict,
    results: list[dict],
    output_png: Path,
    dpi: int = 200,
    transparent: bool = True,
    figure_title: str | None = None,
) -> None:
    """论文双面板：A 最后一帧卷曲姿态叠图；B 俯视运输。"""
    results = results[:N_SUCCESS]
    ref = source["source_center"].copy()
    demo_final = pose_to_points(source["final_pose"], SEGMENT_LENGTHS, ROBOT_BASE)

    fig = plt.figure(figsize=(RES_4K[0] / dpi, RES_4K[1] / dpi), dpi=dpi, facecolor="none")
    bold = FontProperties(family="Arial", weight="bold", size=14)
    panel_fp = FontProperties(family="Arial", weight="bold", size=22)

    ax_a = fig.add_axes([0.05, 0.12, 0.42, 0.78], projection="3d", facecolor="none")

    for seg_idx in range(3):
        start = seg_idx * 20 + (1 if seg_idx > 0 else 0)
        end = min(start + 21, len(demo_final))
        seg = demo_final[start:end]
        ax_a.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="black", linestyle="--", linewidth=2.2, alpha=0.55)

    jet = plt.colormaps["jet"]
    for i, res in enumerate(results):
        pts = res["final_points"]
        n = len(pts)
        for k in range(n - 1):
            t = k / max(n - 2, 1)
            c = jet(0.15 + 0.75 * t)
            ax_a.plot(pts[k : k + 2, 0], pts[k : k + 2, 1], pts[k : k + 2, 2], color=c, linewidth=2.8, alpha=0.92)
        ax_a.scatter(*pts[-1], color="red", s=45, marker="o", zorder=12)
        _add_cylinder_3d(ax_a, res["target_center"], facecolor="#a1887f", alpha=0.55)

    ax_a.scatter(*ROBOT_BASE, color="black", s=100, marker="^")
    _set_3d_limits(
        ax_a,
        [r["final_points"] for r in results] + [demo_final],
        margin=80,
        elev=WOOD_AXIS_ELEV,
        azim=WOOD_AXIS_AZIM,
        set_view=True,
    )
    ax_a.set_xlabel("X (mm)", fontproperties=bold, labelpad=8)
    ax_a.set_ylabel("Y (mm)", fontproperties=bold, labelpad=8)
    ax_a.set_zlabel("Z (mm)", fontproperties=bold, labelpad=8)
    ax_a.tick_params(labelsize=10)
    ax_a.text2D(0.02, 0.98, "A  Wood-axis view", transform=ax_a.transAxes, fontproperties=panel_fp, va="top")

    ax_b = fig.add_axes([0.55, 0.12, 0.40, 0.78])
    src_xy = source["source_center"][:2]
    _draw_small_cylinder_2d(ax_b, src_xy)
    for i, res in enumerate(results):
        c = jet(0.15 + 0.85 * i / max(N_SUCCESS - 1, 1))
        tgt_xy = res["target_center"][:2]
        _draw_small_cylinder_2d(ax_b, tgt_xy, color="#8d6e63", alpha=0.9)
        ax_b.annotate(
            "",
            xy=tgt_xy,
            xytext=src_xy,
            arrowprops=dict(arrowstyle="->", color=c, lw=2.0),
            zorder=2,
        )
    ax_b.set_xlabel("X (mm)", fontproperties=bold)
    ax_b.set_ylabel("Y (mm)", fontproperties=bold)
    ax_b.grid(True, alpha=0.25)
    margin = 85
    xs = [r["target_center"][0] for r in results] + [source["source_center"][0]]
    ys = [r["target_center"][1] for r in results] + [source["source_center"][1]]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) + margin
    ax_b.set_xlim(cx - half, cx + half)
    ax_b.set_ylim(cy - half, cy + half)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.tick_params(labelsize=10)
    ax_b.text(0.02, 0.98, "B  Top view", transform=ax_b.transAxes, fontproperties=panel_fp, va="top")

    fig.text(
        0.5,
        0.97,
        figure_title or "GP-IL: final-frame wrap (16 targets, mirrored top-down)",
        ha="center",
        fontproperties=FontProperties(family="Arial", weight="bold", size=18),
        color="#111111",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, transparent=transparent, facecolor="none", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_posture_grid(results: list[dict], source_filename: str, output_png: Path, dpi: int = 150):
    """4×4 网格：16 组最后一帧，沿木头轴向平视。"""
    results = results[:N_SUCCESS]
    n_below = sum(1 for r in results if r.get("approach") == "below")
    n_top_down = sum(1 for r in results if r.get("approach") == "top-down")
    fig = plt.figure(figsize=(26, 22))
    fig.suptitle(
        f"Fixed-base GP-IL: mirrored final curl (16 blocks, {n_top_down} top-down, {n_below} from-below)\n"
        f"Source: {source_filename}",
        fontsize=14,
        fontweight="bold",
    )

    for idx in range(N_SUCCESS):
        ax = fig.add_subplot(4, 4, idx + 1, projection="3d")
        _plot_one_experiment(ax, results[idx], idx + 1, cmap_color=False)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overview_only(results: list[dict], output_png: Path, dpi: int = 200, transparent: bool = True):
    """仅合览：10 条最后一帧卷曲 + 加长圆柱。"""
    fig = plt.figure(figsize=(12, 10), dpi=dpi, facecolor="none")
    ax = fig.add_subplot(111, projection="3d", facecolor="none")
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    overview_pts = [ROBOT_BASE]
    for idx, (res, color) in enumerate(zip(results, colors)):
        pts = res["final_points"]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=3.0, alpha=0.9, label=f"Exp {idx+1}")
        ax.scatter(*pts[-1], color="red", s=40, marker="o", zorder=12)
        _add_cylinder_3d(ax, res["target_center"], facecolor=color, alpha=0.28)
        overview_pts.extend([pts, res["target_center"]])
    ax.scatter(*ROBOT_BASE, color="black", s=120, marker="^")
    _set_wood_axis_view(ax, overview_pts, margin=85)
    n_below = sum(1 for r in results if r.get("approach") == "below")
    n_top_down = sum(1 for r in results if r.get("approach") == "top-down")
    ax.set_title(
        f"16 mirrored final curl postures ({n_top_down} top-down, {n_below} from-below)",
        fontweight="bold",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(output_png, dpi=dpi, transparent=transparent, bbox_inches="tight", facecolor="none")
    plt.close(fig)


def plot_inversej_demo_keyframes(source: dict, output_png: Path, dpi: int = 150) -> None:
    """v3 风格原始演示关键帧图，但严格使用 inverse-J 标注帧。"""
    phase_order = [
        ("initial", "Initial"),
        ("approaching", "Approaching"),
        ("start_curling", "Start curling"),
        ("curling", "Curling"),
        ("grasped", "Grasped finish"),
    ]
    trajectory = source["trajectory"]
    phase_frames = source["phase_frames"]
    cylinder_center = source["source_center"]

    fig = plt.figure(figsize=(18, 10), dpi=dpi)
    all_points: list[np.ndarray] = []
    for key, title in phase_order:
        frame = min(int(phase_frames[key]), len(trajectory) - 1)
        points = pose_to_points(trajectory[frame], SEGMENT_LENGTHS, ROBOT_BASE)
        all_points.append(points)
        all_points.append(cylinder_center[None, :])

        ax = fig.add_subplot(2, 3, len(all_points) // 2, projection="3d")
        _plot_arm_final_frame(ax, points, lw=4.2)
        _add_cylinder_3d(ax, cylinder_center, facecolor="#8d6e63", alpha=0.66)
        _set_wood_axis_view(ax, [points, cylinder_center], margin=85)
        ax.set_title(f"{title}\nFrame {frame}", fontsize=11, fontweight="bold")

    ax_angles = fig.add_subplot(2, 3, 6)
    t = np.arange(len(source["curl_slice"]))
    curl_slice = source["curl_slice"]
    theta_labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    for dim, color, label in zip((3, 4, 5), ("#FF6B6B", "#4ECDC4", "#45B7D1"), theta_labels):
        ax_angles.plot(t, curl_slice[:, dim], color=color, linewidth=2.0, label=label)
    for key, title in phase_order:
        frame = min(int(phase_frames[key]), len(curl_slice) - 1)
        ax_angles.axvline(frame, linestyle="--", linewidth=1.4, alpha=0.75, label=f"{title} ({frame})")
    ax_angles.set_xlabel("Frame")
    ax_angles.set_ylabel("Bending angle theta (deg)")
    ax_angles.set_title("J-shape demo phases", fontsize=11, fontweight="bold")
    ax_angles.grid(True, alpha=0.28)
    ax_angles.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"Inverse-J imitation demo keyframes - {source['filename']}\n"
        "0 initial, 120 approaching, 190 start curling, 350 curling, 530 grasped finish",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_inversej_demo_snapshots(source: dict, output_png: Path, dpi: int = 150) -> None:
    """v3 风格过程快照，使用横放大圆柱。"""
    start_frame = source["phase_frames"]["start_curling"]
    end_frame = source["phase_frames"]["grasped"]
    frames = np.linspace(start_frame, end_frame, 8, dtype=int)
    trajectory = source["trajectory"]
    cylinder_center = source["source_center"]

    fig = plt.figure(figsize=(16, 14), dpi=dpi)
    cmap = plt.cm.Reds
    snapshot_points = [pose_to_points(trajectory[min(f, len(trajectory) - 1)], SEGMENT_LENGTHS, ROBOT_BASE) for f in frames]
    final_points = pose_to_points(trajectory[end_frame], SEGMENT_LENGTHS, ROBOT_BASE)

    views = [
        ("Wood-axis view: top-down curl", WOOD_AXIS_ELEV, WOOD_AXIS_AZIM),
        ("Oblique view", 22, -55),
        ("Top view", 82, -90),
    ]
    for plot_idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, plot_idx, projection="3d")
        for i, points in enumerate(snapshot_points):
            color = cmap(0.25 + 0.70 * i / max(len(snapshot_points) - 1, 1))
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2.8, alpha=0.45 + 0.5 * i / 7)
            ax.scatter(*points[-1], color=color, s=35)
        _plot_arm_final_frame(ax, final_points, lw=5.0)
        _add_cylinder_3d(ax, cylinder_center, facecolor="#8d6e63", alpha=0.70)
        _set_3d_limits(ax, snapshot_points + [final_points, cylinder_center], margin=90, elev=elev, azim=azim)
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax = fig.add_subplot(2, 2, 4)
    t = np.arange(len(source["curl_slice"]))
    for dim, color, label in zip((3, 4, 5), ("#FF6B6B", "#4ECDC4", "#45B7D1"), (r"$\theta_1$", r"$\theta_2$", r"$\theta_3$")):
        ax.plot(t, source["curl_slice"][:, dim], color=color, linewidth=2.0, label=label)
    for frame in frames:
        ax.axvline(frame, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(start_frame, color="orange", linestyle="--", linewidth=2.0, label="Start curling")
    ax.axvline(end_frame, color="purple", linestyle="--", linewidth=2.0, label="Grasped finish")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Bending angle theta (deg)")
    ax.set_title("Bending angles during inverse-J curl", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Inverse-J demo snapshots with horizontal enlarged cylinder - {source['filename']}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_v3_style_original_demo_outputs(source: dict, output_dir: Path, dpi: int = 150) -> dict[str, Path]:
    """保存 v3 风格原始数据图和阶段信息。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_subset = source["curl_slice"]
    start_frame = source["phase_frames"]["start_curling"]
    end_frame = source["phase_frames"]["grasped"]

    keyframes_path = output_dir / "GP_IL_inverseJ_demo_keyframes.png"
    snapshots_path = output_dir / "GP_IL_inverseJ_demo_snapshots.png"
    phase_csv_path = output_dir / "GP_IL_inverseJ_demo_phases.csv"

    plot_inversej_demo_keyframes(source, keyframes_path, dpi=dpi)
    plot_inversej_demo_snapshots(source, snapshots_path, dpi=dpi)

    dmp = SimpleDMP_V0()
    dmp.imitate(demo_subset, source["phase_frames"]["approaching"], start_frame, end_frame)
    reproduced, _ = dmp.generate()
    tracking = compute_tracking_error(demo_subset, reproduced)

    import pandas as pd

    rows = []
    for name, frame in source["phase_frames"].items():
        pose = source["trajectory"][frame]
        tip = pose_to_points(pose, SEGMENT_LENGTHS, ROBOT_BASE)[-1]
        rows.append(
            {
                "phase": name,
                "frame": frame,
                "tip_x_mm": tip[0],
                "tip_y_mm": tip[1],
                "tip_z_mm": tip[2],
                "theta_1_deg": pose[3],
                "theta_2_deg": pose[4],
                "theta_3_deg": pose[5],
                "dmp_total_rmse_deg": tracking["total_rmse"],
            }
        )
    pd.DataFrame(rows).to_csv(phase_csv_path, index=False)

    return {
        "keyframes": keyframes_path,
        "snapshots": snapshots_path,
        "phase_csv": phase_csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        type=Path,
        default=_ROOT / SOURCE_FILENAME,
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    source = _load_source_demo(args.demo)

    _, phi_data, theta_data = load_inversej_robot_data(args.demo)
    traj_full = np.hstack([phi_data, theta_data])
    # 角度边界紧贴演示，禁止为泛化把臂拉直。
    extra_margin = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=float)
    angle_lower = np.maximum(traj_full.min(axis=0) - extra_margin, [0, 0, 0, 0, 0, 0])
    angle_upper = np.minimum(traj_full.max(axis=0) + extra_margin, [180, 180, 180, 170, 170, 170])
    angle_bounds = (angle_lower, angle_upper)

    print("保存 inverse-J 原始演示图（v3 风格关键帧 + 快照）...")
    original_outputs = save_v3_style_original_demo_outputs(source, args.output_dir, dpi=150)

    print("运行 inverse-J GP-IL 泛化（v4 风格，16 组，木头两侧各 8 组镜像）...")
    successes = _run_generalization(source, angle_bounds)
    n_side = sum(1 for s in successes if s.get("approach") == "side")
    n_top_down = sum(1 for s in successes if s.get("approach") == "top-down")
    n_below = sum(1 for s in successes if s.get("approach") == "below")
    mirror_axis_y = float(source["source_center"][1] + MIRROR_AXIS_Y_OFFSET)
    n_neg_side = sum(1 for s in successes if s["target_center"][1] < mirror_axis_y)
    n_pos_side = sum(1 for s in successes if s["target_center"][1] >= mirror_axis_y)
    print(
        f"成功: {len(successes)} / 目标 {N_SUCCESS}  "
        f"(从上往下 {n_top_down}, 侧面 {n_side}, 自下而上 {n_below}, 两侧 {n_neg_side}+{n_pos_side})"
    )

    if len(successes) < N_SUCCESS:
        raise SystemExit(
            f"成功泛化不足或 top-down 不足: 成功 {len(successes)}/{N_SUCCESS}, "
            f"top-down {n_top_down}/{N_TOP_DOWN_MIN}, 两侧 {n_neg_side}+{n_pos_side}。"
            "请检查候选范围或镜像约束。"
        )

    png_path = args.output_dir / "GP_IL_inverseJ.png"
    png_alpha = args.output_dir / "GP_IL_inverseJ_transparent.png"
    grid_path = args.output_dir / "GP_IL_inverseJ_posture_grid.png"
    overview_path = args.output_dir / "GP_IL_inverseJ_overview.png"

    title = "GP-IL inverse-J: mirrored top-down J-shape generalization (16 targets)"
    plot_gp_il_v4(source, successes, png_path, dpi=args.dpi, transparent=False, figure_title=title)
    plot_gp_il_v4(source, successes, png_alpha, dpi=args.dpi, transparent=True, figure_title=title)
    plot_posture_grid(successes, source["filename"], grid_path, dpi=150)
    plot_overview_only(successes, overview_path, dpi=args.dpi, transparent=True)

    import pandas as pd

    rows = []
    for i, r in enumerate(successes, 1):
        rows.append(
            {
                "id": i,
                "target_x_mm": r["target_center"][0],
                "target_y_mm": r["target_center"][1],
                "target_z_mm": r["target_center"][2],
                "center_error_mm": r["center_error"],
                "side_tip_error_mm": r.get("side_tip_error", np.nan),
                "wrap_gap_mm": r.get("wrap_gap", np.nan),
                "tip_surface_gap_mm": r.get("tip_surface_gap", np.nan),
                "top_down_score": r.get("top_down_score", np.nan),
                "max_angle_delta_deg": float(np.max(np.abs(r["pose_delta"]))),
                "tip_reach_mm": r.get("tip_reach", np.nan),
                "source_reach_mm": r.get("source_reach", np.nan),
                "reach_ratio": (
                    float(r["tip_reach"] / r["source_reach"])
                    if r.get("tip_reach") is not None and r.get("source_reach")
                    else np.nan
                ),
                "approach": r.get("approach", "side"),
                "mirror_side": (
                    "neg_y_side"
                    if r["target_center"][1] < mirror_axis_y
                    else "pos_y_side"
                ),
                "mirror_axis_y_mm": mirror_axis_y,
                "below_score": r.get("below_score", 0.0),
                "quadrant": _quadrant(r["target_center"][:2], ROBOT_BASE[:2]),
            }
        )
    success_csv = args.output_dir / "GP_IL_inverseJ_success.csv"
    pd.DataFrame(rows).to_csv(success_csv, index=False)

    print(f"已保存: {original_outputs['keyframes']}")
    print(f"已保存: {original_outputs['snapshots']}")
    print(f"已保存: {original_outputs['phase_csv']}")
    print(f"已保存: {png_path}")
    print(f"已保存: {png_alpha}")
    print(f"已保存: {grid_path}")
    print(f"已保存: {overview_path}")
    print(f"已保存: {success_csv}")


if __name__ == "__main__":
    main()
