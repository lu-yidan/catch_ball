#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3 抓取仿真可视化：规划后绘制初始/完成姿态，评估能否够到球。

坐标系 = 相机 soft_arm_center（与 ball_target.json 一致）:
  +X 侧向, +Y 向下, -Z 朝相机
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grasp_excute import (
    DEFAULT_ARM_BASE_MM,
    fk_inverted_pcc_centerline,
    vision_center_to_arm_mm,
)
from grasp_planning_v2 import load_arm_axes_config

STRAIGHT_HOME_POSE_DEG = np.zeros(6, dtype=float)
from paths import OUTPUT_V3_SIM_DIR

OUT_DIR = OUTPUT_V3_SIM_DIR
# 末端进入此半径视为仿真“可抓住”（中心距 - 球半径）
DEFAULT_GRASP_MARGIN_MM = 25.0


@dataclass
class GraspSimReport:
    base_mm: np.ndarray
    ball_mm: np.ndarray
    ball_radius_mm: float
    tip_initial_mm: np.ndarray
    tip_final_mm: np.ndarray
    dist_initial_mm: float
    dist_final_mm: float
    gap_initial_mm: float
    gap_final_mm: float
    center_error_final_mm: np.ndarray
    likely_grasp: bool
    motor14_ok: bool
    notes: list[str]

    def summary_lines(self) -> list[str]:
        lines = [
            f"  base (arm)     = {self.base_mm.round(1).tolist()}",
            f"  ball center    = {self.ball_mm.round(1).tolist()}  R={self.ball_radius_mm:.0f} mm",
            f"  tip initial    = {self.tip_initial_mm.round(1).tolist()}  "
            f"dist={self.dist_initial_mm:.0f} mm  surface_gap={self.gap_initial_mm:.0f} mm",
            f"  tip final      = {self.tip_final_mm.round(1).tolist()}  "
            f"dist={self.dist_final_mm:.0f} mm  surface_gap={self.gap_final_mm:.0f} mm",
            f"  final error    = {self.center_error_final_mm.round(1).tolist()}",
            f"  sim likely_grasp = {self.likely_grasp}  (gap_final <= {DEFAULT_GRASP_MARGIN_MM:.0f} mm)",
            f"  motor14_ok     = {self.motor14_ok}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return lines


def _ball_arm(
    ball_center_mm: np.ndarray,
    base_mm: np.ndarray,
    arm_axes_config: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_arm_axes_config(arm_axes_config)
    scale = np.asarray(cfg.get("vision_to_arm_scale", [1.0, 1.0, 1.0]), dtype=float)
    if base_mm is None:
        base_mm = np.asarray(cfg.get("arm_base_mm", DEFAULT_ARM_BASE_MM.tolist()), dtype=float)
    ball_arm = vision_center_to_arm_mm(ball_center_mm, scale)
    return np.asarray(base_mm, dtype=float).reshape(3), ball_arm.reshape(3)


def evaluate_grasp_sim(
    result: Any,
    ball_center_mm: np.ndarray,
    radius_mm: float,
    base_mm: np.ndarray | None = None,
    arm_axes_config: Path | None = None,
    grasp_margin_mm: float = DEFAULT_GRASP_MARGIN_MM,
) -> GraspSimReport:
    """根据绳组 FK 评估初始/末端与球关系。"""
    base_mm, ball_mm = _ball_arm(ball_center_mm, base_mm, arm_axes_config)
    radius_mm = float(radius_mm)

    tip_i = fk_inverted_pcc_centerline(STRAIGHT_HOME_POSE_DEG, base_mm)[-1]
    tip_f = fk_inverted_pcc_centerline(result.final_pose, base_mm)[-1]

    d0 = float(np.linalg.norm(tip_i - ball_mm))
    df = float(np.linalg.norm(tip_f - ball_mm))
    g0 = d0 - radius_mm
    gf = df - radius_mm
    err = tip_f - ball_mm

    motor14_ok = bool(result.info.get("motor14_in_range", False))
    likely = bool(gf <= grasp_margin_mm and motor14_ok)

    notes: list[str] = []
    if ball_mm[0] <= base_mm[0] + 20:
        notes.append("球 X 不在 base 正侧，规划可能非典型 +X 场景")
    if g0 < gf:
        notes.append("末端比初始更远球 — 检查规划或 FK")
    if not motor14_ok:
        notes.append("motor1/4 不在可行域，实机不宜执行")
    if likely and ball_mm[1] - tip_f[1] > 80:
        notes.append("XZ 可触及但 Y 偏差大，实机可能仅侧面靠近")

    return GraspSimReport(
        base_mm=base_mm,
        ball_mm=ball_mm,
        ball_radius_mm=radius_mm,
        tip_initial_mm=tip_i,
        tip_final_mm=tip_f,
        dist_initial_mm=d0,
        dist_final_mm=df,
        gap_initial_mm=g0,
        gap_final_mm=gf,
        center_error_final_mm=err,
        likely_grasp=likely,
        motor14_ok=motor14_ok,
        notes=notes,
    )


def _draw_ball_xz(ax, ball: np.ndarray, r: float, **kwargs) -> None:
    th = np.linspace(0, 2 * np.pi, 64)
    ax.plot(ball[0] + r * np.cos(th), ball[2] + r * np.sin(th), **kwargs)


def _draw_ball_xy(ax, ball: np.ndarray, r: float, **kwargs) -> None:
    th = np.linspace(0, 2 * np.pi, 64)
    ax.plot(ball[0] + r * np.cos(th), ball[1] + r * np.sin(th), **kwargs)


def plot_grasp_preview(
    result: Any,
    ball_center_mm: np.ndarray,
    radius_mm: float,
    base_mm: np.ndarray | None = None,
    arm_axes_config: Path | None = None,
    *,
    show: bool = True,
    save_path: Path | None = None,
) -> GraspSimReport:
    """绘制初始(直立)与抓取完成姿态：base、臂、球。"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        raise SystemExit("需要 matplotlib: pip install matplotlib") from exc

    report = evaluate_grasp_sim(
        result, ball_center_mm, radius_mm, base_mm, arm_axes_config
    )
    base = report.base_mm
    ball = report.ball_mm
    r = report.ball_radius_mm

    pts_init = fk_inverted_pcc_centerline(STRAIGHT_HOME_POSE_DEG, base)
    pts_final = fk_inverted_pcc_centerline(result.final_pose, base)
    traj = result.trajectory
    tips_path = np.array([fk_inverted_pcc_centerline(p, base)[-1] for p in traj])

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(
        f"v3 grasp preview | likely_grasp={report.likely_grasp} | "
        f"gap_final={report.gap_final_mm:.0f}mm",
        fontsize=11,
    )

    # --- 3D ---
    ax3 = fig.add_subplot(131, projection="3d")
    ax3.plot(pts_init[:, 0], pts_init[:, 1], pts_init[:, 2], "o-", color="#888", lw=2, ms=2, label="initial")
    ax3.plot(pts_final[:, 0], pts_final[:, 1], pts_final[:, 2], "o-", color="#2ca02c", lw=2.5, ms=2, label="grasp final")
    ax3.plot(tips_path[:, 0], tips_path[:, 1], tips_path[:, 2], ":", color="#4C78A8", alpha=0.5, label="tip path")
    ax3.scatter(*base, c="k", s=60, marker="s", label="base")
    ax3.scatter(*ball, c="gold", s=100, depthshade=False, label="ball center")
    u, v = np.mgrid[0 : 2 * np.pi : 12j, 0 : np.pi : 8j]
    ax3.plot_surface(
        ball[0] + r * np.cos(u) * np.sin(v),
        ball[1] + r * np.sin(u) * np.sin(v),
        ball[2] + r * np.cos(v),
        color="gold",
        alpha=0.25,
        linewidth=0,
    )
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.legend(fontsize=7, loc="upper left")

    # --- XZ (主卷取平面) ---
    ax_xz = fig.add_subplot(132)
    ax_xz.plot(pts_init[:, 0], pts_init[:, 2], "o-", color="#888", lw=2, label="initial")
    ax_xz.plot(pts_final[:, 0], pts_final[:, 2], "o-", color="#2ca02c", lw=2.5, label="grasp final")
    ax_xz.plot(tips_path[:, 0], tips_path[:, 2], ":", color="#4C78A8", alpha=0.6)
    _draw_ball_xz(ax_xz, ball, r, color="gold", lw=1.5, ls="--")
    ax_xz.scatter(ball[0], ball[2], c="gold", s=80, zorder=5)
    ax_xz.scatter(base[0], base[2], c="k", s=50, marker="s", zorder=6)
    ax_xz.annotate("+X", xy=(base[0] + 80, base[2]), fontsize=9, color="gray")
    ax_xz.set_xlabel("X (lateral)"); ax_xz.set_ylabel("Z (camera)")
    ax_xz.set_title("XZ — curl toward +X ball")
    ax_xz.axis("equal"); ax_xz.grid(True, alpha=0.3); ax_xz.legend(fontsize=8)

    # --- XY ---
    ax_xy = fig.add_subplot(133)
    ax_xy.plot(pts_init[:, 0], pts_init[:, 1], "o-", color="#888", lw=2, label="initial")
    ax_xy.plot(pts_final[:, 0], pts_final[:, 1], "o-", color="#2ca02c", lw=2.5, label="grasp final")
    _draw_ball_xy(ax_xy, ball, r, color="gold", lw=1.5, ls="--")
    ax_xy.scatter(ball[0], ball[1], c="gold", s=80)
    ax_xy.scatter(base[0], base[1], c="k", s=50, marker="s")
    ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y (down)")
    ax_xy.set_title("XY side view")
    ax_xy.grid(True, alpha=0.3); ax_xy.legend(fontsize=8)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = OUT_DIR / "v3_grasp_preview.png"
    fig.savefig(save_path, dpi=150)
    print(f"[v3 sim] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return report


def print_sim_report(report: GraspSimReport) -> None:
    print("=" * 60)
    print("v3 simulation (camera / arm frame)")
    print("=" * 60)
    for line in report.summary_lines():
        print(line)
    print("=" * 60)
