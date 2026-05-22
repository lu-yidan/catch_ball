#!/usr/bin/env python
"""
GP_IL_v4 — 最后一帧卷曲姿态 + 论文风格泛化图（15 组成功）

约束：
  - 小木块相对 base 在 ±x、±y 方向分散分布
  - 优先侧面卷取（木块在地面）；至多 2 组允许自下而上卷取

输出：
  - GP_IL_v4.png / GP_IL_v4_transparent.png  双面板 A/B（最后一帧姿态 + 俯视运输）
  - GP_IL_v4_posture_grid.png                 4×4 分实验最后一帧 + 合览
  - GP_IL_v4_overview.png                     15 组合览

Usage:
    python GP_IL_v4.py
    python GP_IL_v4.py --demo Task2_robot_state_data/2020_03_13_chishuru0003_4_angles_data.csv
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
sys.path.insert(0, str(_ROOT))
from GP_IL import (  # noqa: E402
    GaussianProcessPolicyTransporter,
    build_object_keypoints,
    compute_tip_trajectory,
    create_cylinder,
    estimate_curl_object_center,
    generate_fixed_base_curl_trajectory,
    get_curl_center_from_points,
    load_robot_data,
    parse_data_info,
    pose_to_points,
)

OUT_DIR = _ROOT / "output_gp_il_v4"
SEGMENT_LENGTHS = np.array([150.0, 150.0, 150.0])
ROBOT_BASE = np.array([0.0, 0.0, 1000.0])
# 与 GP_IL 一致，略加长圆柱（木块/抓取目标）
CYL_R, CYL_H = 20.0, 58.0
SEG_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
N_SUCCESS = 15
N_FROM_BELOW_MAX = 2
# 相对 base 的水平散布（mm）
SPREAD_X = (-135.0, 135.0)
SPREAD_Y = (-125.0, 125.0)
MIN_XY_FROM_BASE = 42.0
MIN_TARGET_SEP_XY = 40.0
MAX_PER_QUADRANT = 5
RES_4K = (3840, 2160)

SIDE_TIP_RADIUS = CYL_R + 1.0
SIDE_TIP_Z_OFFSET = -0.12 * CYL_H
# 相对演示最后一帧的最大角度偏移（禁止为够到目标而把臂“拉直/拉长”）
MAX_POSE_DELTA = np.array([34.0, 34.0, 34.0, 40.0, 40.0, 36.0], dtype=float)
MAX_TIP_REACH_SCALE = 1.0


def _load_source_demo(demo_csv: Path, grasp_time_txt: Path):
    filename = demo_csv.name
    frames, phi, theta = load_robot_data(str(demo_csv))
    traj = np.hstack([phi, theta])

    phase_info = parse_data_info(str(grasp_time_txt)) if grasp_time_txt.is_file() else {}
    if filename in phase_info:
        info = phase_info[filename]
        if "ready_to_grasp" in info:
            grasp_start = info["grasp"][0] if "grasp" in info else info["ready_to_grasp"][1] + 1
            grasp_end = info["grasp"][1] if "grasp" in info else len(traj) - 1
        elif "grasp" in info:
            grasp_start, grasp_end = info["grasp"]
        else:
            grasp_start = len(traj) // 4
            grasp_end = len(traj) - 1
    else:
        grasp_start = len(traj) // 4
        grasp_end = len(traj) - 1

    grasp_end = min(grasp_end, len(traj) - 1)
    grasp_start = min(grasp_start, grasp_end)
    curl_slice = traj[: grasp_end + 1]
    source_center, _ = estimate_curl_object_center(traj, grasp_end, SEGMENT_LENGTHS, ROBOT_BASE)
    source_tip = compute_tip_trajectory(curl_slice, SEGMENT_LENGTHS, ROBOT_BASE)
    return {
        "filename": filename,
        "curl_slice": curl_slice,
        "source_center": source_center,
        "source_tip": source_tip,
        "final_pose": curl_slice[-1],
    }


def _quadrant(xy: np.ndarray, origin: np.ndarray) -> str:
    """相对 base 的象限标签 (+x+y, +x-y, -x+y, -x-y, axis)。"""
    d = xy - origin
    if abs(d[0]) < 18 and abs(d[1]) < 18:
        return "axis"
    qx = "+" if d[0] >= 0 else "-"
    qy = "+" if d[1] >= 0 else "-"
    return f"{qx}x{qy}y"


def _side_direction(target_center: np.ndarray) -> np.ndarray:
    """从 base 指向木块的水平单位向量，作为侧面接触方向。"""
    direction = target_center[:2] - ROBOT_BASE[:2]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return direction / norm


def _side_tip_target(target_center: np.ndarray, transported_tip: np.ndarray) -> np.ndarray:
    """末端贴在圆柱朝向 GP 运输末端方向的一侧外表面。"""
    direction = transported_tip[:2] - target_center[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 12.0:
        direction = ROBOT_BASE[:2] - target_center[:2]
        norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm
    return target_center + np.array(
        [
            direction[0] * SIDE_TIP_RADIUS,
            direction[1] * SIDE_TIP_RADIUS,
            SIDE_TIP_Z_OFFSET,
        ]
    )


def below_wrap_score(final_points: np.ndarray, target_center: np.ndarray) -> float:
    """0=侧面卷取，1=自下而上（末端/卷曲段在圆柱底部以下且水平接近）。"""
    bottom_z = target_center[2] - CYL_H / 2.0
    distal = final_points[34:]
    tip = final_points[-1]

    tip_h = float(np.linalg.norm(tip[:2] - target_center[:2]))
    tip_under = max(0.0, bottom_z - tip[2]) / 35.0
    tip_under *= max(0.0, 1.0 - tip_h / (CYL_R * 2.2))

    near = distal[np.linalg.norm(distal[:, :2] - target_center[:2], axis=1) < CYL_R * 2.5]
    if len(near) == 0:
        near_under = 0.0
    else:
        near_under = float(np.mean(np.clip((bottom_z - near[:, 2]) / 35.0, 0.0, 1.0)))

    return float(np.clip(0.65 * tip_under + 0.35 * near_under, 0.0, 1.0))


def classify_wrap_approach(final_points: np.ndarray, target_center: np.ndarray) -> str:
    """below 当末端/卷曲段确实在圆柱底部以下从下方接近。"""
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
    bottom_z = target_center[2] - CYL_H / 2.0

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

        tip_h = float(np.linalg.norm(tip[:2] - target_center[:2]))
        tip_surface_gap = abs(tip_h - CYL_R)
        side_radius_error = abs(tip_h - SIDE_TIP_RADIUS)
        side_height_penalty = 3.0 * max(0.0, bottom_z - 3.0 - tip[2])

        horiz_dist = np.linalg.norm(distal[:, :2] - target_center[:2], axis=1)
        surface_gap = np.abs(horiz_dist - CYL_R)
        wrap_gap = float(np.min(surface_gap))
        wrap_contact_penalty = float(6.0 * np.mean(np.clip(surface_gap - 3.0, 0.0, None)))

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


def _select_diverse_successes(candidates: list[dict], n: int = N_SUCCESS) -> list[dict]:
    """侧面优先；四象限均衡；目标 xy 间距≥MIN_TARGET_SEP_XY；至多 2 个自下而上。"""
    origin = ROBOT_BASE[:2]
    sides = sorted(
        [c for c in candidates if c["approach"] == "side"],
        key=lambda item: (
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
        chosen.append({**item, "approach": "below" if is_below else "side"})
        chosen_ids.add(iid)
        if q in quad_used:
            quad_used[q] += 1
        return True

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
            is_below = item["approach"] == "below" and n_below < N_FROM_BELOW_MAX
            if is_below:
                n_below += 1
            chosen.append({**item, "approach": "below" if is_below else "side"})
            chosen_ids.add(id(item))

    return chosen[:n]


def _run_generalization(source: dict, angle_bounds) -> list[dict]:
    """收集 can_wrap 成功结果，再按侧面优先 + 四象限分散选出 N_SUCCESS 组。"""
    pool: list[dict] = []
    transporter = GaussianProcessPolicyTransporter(
        length_scale=120.0, signal_variance=1.0, noise_variance=1e-6
    )

    for target_center in _candidate_targets(source["source_center"]):
        target_keypoints = build_object_keypoints(target_center, CYL_R, CYL_H)
        transporter.fit(build_object_keypoints(source["source_center"], CYL_R, CYL_H), target_keypoints)
        transported_tip = transporter.transform(source["source_tip"][-1])

        tuned_pose, info = tune_side_final_pose(
            source["final_pose"],
            target_center,
            transported_tip,
            SEGMENT_LENGTHS,
            ROBOT_BASE,
            angle_bounds,
        )
        if not info["can_wrap"]:
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
            }
        )

    return _select_diverse_successes(pool, N_SUCCESS)


def _draw_small_cylinder_2d(ax, xy, color="#8d6e63", alpha=0.85, scale: float = 1.0):
    """俯视图：竖直圆柱投影为正圆。"""
    cx, cy = xy
    circle = patches.Circle(
        (cx, cy),
        radius=CYL_R * scale,
        facecolor=color,
        edgecolor="#5d4037",
        lw=1.1,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(circle)


def _add_cylinder_3d(ax, center, facecolor="#8d6e63", alpha=0.65):
    _, faces = create_cylinder(center, CYL_R, CYL_H, axis="z")
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
    _set_3d_limits(ax, [pts, result["target_center"]])
    approach = result.get("approach", "side")
    tag = "SIDE wrap" if approach == "side" else "FROM-BELOW (allowed)"
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
        elev=19,
        azim=-58,
        set_view=True,
    )
    ax_a.set_xlabel("X (mm)", fontproperties=bold, labelpad=8)
    ax_a.set_ylabel("Y (mm)", fontproperties=bold, labelpad=8)
    ax_a.set_zlabel("Z (mm)", fontproperties=bold, labelpad=8)
    ax_a.tick_params(labelsize=10)
    ax_a.text2D(0.02, 0.98, "A  Final curl frame", transform=ax_a.transAxes, fontproperties=panel_fp, va="top")

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
        figure_title or "GP-IL: final-frame wrap (15 targets, side-dominant, tall cylinders)",
        ha="center",
        fontproperties=FontProperties(family="Arial", weight="bold", size=18),
        color="#111111",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, transparent=transparent, facecolor="none", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_posture_grid(results: list[dict], source_filename: str, output_png: Path, dpi: int = 150):
    """4×4 网格：15 组最后一帧 + 合览。"""
    results = results[:N_SUCCESS]
    n_below = sum(1 for r in results if r.get("approach") == "below")
    fig = plt.figure(figsize=(26, 22))
    fig.suptitle(
        f"Fixed-base GP-IL: final curl (15 blocks, {n_below} from-below allowed)\n"
        f"Source: {source_filename}",
        fontsize=14,
        fontweight="bold",
    )

    for idx in range(N_SUCCESS):
        ax = fig.add_subplot(4, 4, idx + 1, projection="3d")
        _plot_one_experiment(ax, results[idx], idx + 1, cmap_color=False)

    ax_ov = fig.add_subplot(4, 4, 16, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, N_SUCCESS))
    overview_pts = [ROBOT_BASE]
    for idx, (res, color) in enumerate(zip(results, colors)):
        pts = res["final_points"]
        ax_ov.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=2.4, label=f"Exp {idx + 1}")
        ax_ov.scatter(*pts[-1], color="red", s=28, marker="o", zorder=12)
        _add_cylinder_3d(ax_ov, res["target_center"], facecolor=color, alpha=0.22)
        overview_pts.extend([pts, res["target_center"]])
    ax_ov.scatter(*ROBOT_BASE, color="black", s=100, marker="^")
    _set_3d_limits(ax_ov, overview_pts, margin=80)
    ax_ov.set_title("Whole-arm final posture overview", fontsize=10, fontweight="bold")
    ax_ov.legend(loc="best", fontsize=7)

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
    _set_3d_limits(ax, overview_pts, margin=85)
    n_below = sum(1 for r in results if r.get("approach") == "below")
    ax.set_title(
        f"15 final curl postures ({n_below} from-below)",
        fontweight="bold",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(output_png, dpi=dpi, transparent=transparent, bbox_inches="tight", facecolor="none")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        type=Path,
        default=_ROOT / "data" / "demo" / "2020_03_13_chishuru0003_4_angles_data.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    grasp_txt = args.demo.parent / "grasp_time.txt"
    source = _load_source_demo(args.demo, grasp_txt)

    _, phi_data, theta_data = load_robot_data(str(args.demo))
    traj_full = np.hstack([phi_data, theta_data])
    # 角度边界紧贴演示，禁止为泛化把臂拉直。
    extra_margin = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=float)
    angle_lower = np.maximum(traj_full.min(axis=0) - extra_margin, [0, 0, 0, 0, 0, 0])
    angle_upper = np.minimum(traj_full.max(axis=0) + extra_margin, [180, 180, 180, 170, 170, 170])
    angle_bounds = (angle_lower, angle_upper)

    print("运行 GP-IL 泛化（侧面优先，±x/±y 分散，15 组）...")
    successes = _run_generalization(source, angle_bounds)
    n_side = sum(1 for s in successes if s.get("approach") == "side")
    n_below = sum(1 for s in successes if s.get("approach") == "below")
    print(f"成功: {len(successes)} / 目标 {N_SUCCESS}  (侧面 {n_side}, 自下而上 {n_below})")

    if len(successes) < N_SUCCESS:
        raise SystemExit(
            f"成功泛化不足 ({len(successes)}/{N_SUCCESS})。"
            "请检查演示数据或增大 GP_IL_v4._candidate_targets 搜索范围。"
        )

    png_path = args.output_dir / "GP_IL_v4.png"
    png_alpha = args.output_dir / "GP_IL_v4_transparent.png"
    grid_path = args.output_dir / "GP_IL_v4_posture_grid.png"
    overview_path = args.output_dir / "GP_IL_v4_overview.png"

    plot_gp_il_v4(source, successes, png_path, dpi=args.dpi, transparent=False)
    plot_gp_il_v4(source, successes, png_alpha, dpi=args.dpi, transparent=True)
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
                "max_angle_delta_deg": float(np.max(np.abs(r["pose_delta"]))),
                "approach": r.get("approach", "side"),
                "below_score": r.get("below_score", 0.0),
                "quadrant": _quadrant(r["target_center"][:2], ROBOT_BASE[:2]),
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "GP_IL_v4_success.csv", index=False)

    print(f"已保存: {png_path}")
    print(f"已保存: {png_alpha}")
    print(f"已保存: {grid_path}")
    print(f"已保存: {overview_path}")
    print(f"已保存: {args.output_dir / 'GP_IL_v4_success.csv'}")


if __name__ == "__main__":
    main()
