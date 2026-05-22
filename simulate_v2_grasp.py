#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2 抓取仿真（仅本文件；规划/实机仍用 grasp_excute 绳组模型）。

正运动学:
  - 主路径: grasp_excute.fk_inverted_pcc_centerline（XZ 绳组 phi, +Y 主轴）
  - 对照: GP marker FK + rope_phi_to_marker_plane_deg（无经验角偏移）

motor 经验 (相对原始 XOY PCC):
  motor1+ -> -X (phi=180°), motor2+ -> -Z (phi=270°), motor6/5+ -> +X (phi=0°)

用法:
  python simulate_v2_grasp.py --coord-file ball_target.json
  python simulate_v2_grasp.py --ball 200 400 0
  python simulate_v2_grasp.py --fk rope|marker|both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import GP_tennis as gp  # noqa: E402
from grasp_excute import (  # noqa: E402
    DEFAULT_ARM_BASE_MM,
    SEGMENT_LENGTHS_MM,
    fk_inverted_pcc_centerline,
    marker_plane_to_rope_phi_deg,
    motor_pull_unit_xz,
    motor_steps_to_bend_delta_xz,
    rope_phi_to_marker_plane_deg,
)
from grasp_planning import load_external_ball_coordinate
from grasp_planning_v2 import DEFAULT_BALL_RADIUS_MM, plan_grasp_v2
from paths import DEFAULT_COORD_FILE as DEFAULT_COORD_FILE_PATH, OUTPUT_V2_SIM_DIR
STRAIGHT_HOME_POSE_DEG = np.zeros(6, dtype=float)
OUT_DIR = OUTPUT_V2_SIM_DIR


def pose_rope_to_marker(pose_rope_deg: np.ndarray) -> np.ndarray:
    p = np.asarray(pose_rope_deg, dtype=float).copy()
    p[0:3] = [rope_phi_to_marker_plane_deg(x) for x in p[0:3]]
    # GP 内部对 Theta 乘 INVERTED_PCC_THETA_SIGN=-1；绳组规划角需先取反
    p[3:6] = -p[3:6]
    return p


def gp_points_to_arm_frame(points_gp: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_gp, dtype=float)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = gp.ROBOT_BASE[2] - pts[:, 2]
    out[:, 2] = pts[:, 1]
    return out


def fk_marker_centerline(
    pose_rope_deg: np.ndarray,
    base_mm: np.ndarray | None = None,
    points_per_segment: int = 20,
) -> np.ndarray:
    """GP 倒立 PCC（CSV Plane 角）+ 臂系坐标变换。"""
    if base_mm is None:
        base_mm = DEFAULT_ARM_BASE_MM
    marker_pose = pose_rope_to_marker(pose_rope_deg)
    pts_gp = gp.pose_to_points(
        marker_pose,
        SEGMENT_LENGTHS_MM,
        gp.ROBOT_BASE,
        num_points_per_segment=points_per_segment,
    )
    return gp_points_to_arm_frame(pts_gp)


def tip_rope(pose: np.ndarray, base_mm: np.ndarray | None = None) -> np.ndarray:
    return fk_inverted_pcc_centerline(pose, base_mm=base_mm)[-1]


def tip_marker(pose: np.ndarray, base_mm: np.ndarray | None = None) -> np.ndarray:
    return fk_marker_centerline(pose, base_mm=base_mm)[-1]


def verify_motor_directions() -> None:
    """打印 +motor 步进对应的 XZ 弯曲方向（与实机经验对照）。"""
    print("--- motor +step -> bend_xz (deg along u(phi)) ---")
    for mid, phi in sorted({1: 180, 2: 270, 3: 270, 4: 270, 5: 0, 6: 0}.items()):
        d = motor_steps_to_bend_delta_xz({mid: 1000})
        u = motor_pull_unit_xz(phi)
        print(f"  motor{mid} phi={phi:3.0f}  u={u}  +1000step -> d={d}")


def toward_ball_metrics(
    trajectory: np.ndarray,
    ball_mm: np.ndarray,
    base_mm: np.ndarray,
    fk_mode: str,
) -> dict:
    tip_fn = tip_rope if fk_mode == "rope" else tip_marker
    tips = np.array([tip_fn(p, base_mm) for p in trajectory])
    to_ball = ball_mm - base_mm
    dists = np.linalg.norm(tips - ball_mm, axis=1)
    align = [
        float(np.dot(t - base_mm, to_ball) / (np.linalg.norm(t - base_mm) * np.linalg.norm(to_ball) + 1e-9))
        for t in tips
    ]
    to_xz = to_ball[[0, 2]]
    if np.linalg.norm(to_xz) > 1e-6:
        u = to_xz / np.linalg.norm(to_xz)
        xz_progress = float(np.dot(tips[-1][[0, 2]] - base_mm[[0, 2]], u)) - float(
            np.dot(tips[0][[0, 2]] - base_mm[[0, 2]], u)
        )
    else:
        xz_progress = 0.0
    return {
        "tips": tips,
        "final_dist_mm": float(dists[-1]),
        "home_dist_mm": float(dists[0]),
        "final_align": float(align[-1]),
        "home_align": float(align[0]),
        "xz_progress_mm": xz_progress,
        "dist_improved": bool(dists[-1] < dists[0] - 5.0),
        "align_improved": bool(align[-1] > align[0] + 0.02),
    }


def curl_success(m: dict, ball_mm: np.ndarray, base_mm: np.ndarray) -> bool:
    return bool(
        m["dist_improved"]
        and m["align_improved"]
        and m["final_align"] > 0.35
        and m["xz_progress_mm"] > 15.0
    )


def print_report(
    result,
    ball_mm: np.ndarray,
    base_mm: np.ndarray,
    fk_mode: str = "rope",
) -> bool:
    traj = result.trajectory
    info = result.info
    m = toward_ball_metrics(traj, ball_mm, base_mm, fk_mode)
    ok = curl_success(m, ball_mm, base_mm)

    print("=" * 72)
    print(f"v2 仿真 FK={fk_mode}  (motor1+@-X/180°, motor2+@-Z/270°, 绳组 u=[cos,sin] in XZ)")
    print("=" * 72)
    print(f"ball_mm           = {ball_mm.tolist()}")
    print(f"to_ball_xz plan   = {info.get('to_ball_xz_mm')}")
    print(f"phi12 IK (rope)   = {info.get('section12_phi_ik_deg'):.2f}°")
    print(f"motor1/4 cmd      = {info.get('motor1_steps_cmd')} / {info.get('motor4_steps_cmd')}")
    d14 = motor_steps_to_bend_delta_xz(
        {1: info.get("motor1_steps_cmd", 0), 4: info.get("motor4_steps_cmd", 0)}
    )
    print(f"motor14 bend_xz   = {np.round(d14, 2).tolist()}  (deg-equivalent)")
    print("-" * 72)
    print("straight home tip:", np.round(tip_rope(STRAIGHT_HOME_POSE_DEG, base_mm), 1).tolist())
    for label, idx in [
        ("25%", len(traj) // 4),
        ("50%", len(traj) // 2),
        ("75%", 3 * len(traj) // 4),
        ("final", -1),
    ]:
        p = traj[idx]
        tr = tip_rope(p, base_mm)
        tm = tip_marker(p, base_mm)
        print(
            f"  [{label:5}] rope_phi={np.round(p[0:3], 1)} "
            f"marker_plane={np.round([rope_phi_to_marker_plane_deg(x) for x in p[0:3]], 1)} "
            f"tip_rope={np.round(tr, 1)} tip_marker={np.round(tm, 1)}"
        )
    print("-" * 72)
    print(
        f"dist {m['home_dist_mm']:.0f} -> {m['final_dist_mm']:.0f} mm  "
        f"align {m['home_align']:.3f} -> {m['final_align']:.3f}  "
        f"xz_progress={m['xz_progress_mm']:.1f} mm  success={ok}"
    )
    print("=" * 72)
    return ok


def plot_result(result, ball_mm: np.ndarray, base_mm: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    traj = result.trajectory
    plot_poses = [
        ("straight", STRAIGHT_HOME_POSE_DEG),
        ("25%", traj[len(traj) // 4]),
        ("final", traj[-1]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for lab, pose in plot_poses:
        pr = fk_inverted_pcc_centerline(pose, base_mm)
        pm = fk_marker_centerline(pose, base_mm)
        axes[0].plot(pr[:, 0], pr[:, 2], lw=2, label=f"{lab} rope")
        axes[0].plot(pm[:, 0], pm[:, 2], lw=1.5, ls="--", label=f"{lab} marker")
    axes[0].scatter(ball_mm[0], ball_mm[2], c="gold", s=80, zorder=5)
    axes[0].set_xlabel("X"); axes[0].set_ylabel("Z"); axes[0].legend(fontsize=7)
    axes[0].set_title("XZ: rope FK (solid) vs GP marker (dashed)")
    axes[0].axis("equal"); axes[0].grid(True, alpha=0.3)

    tips_r = np.array([tip_rope(p, base_mm) for p in traj])
    axes[1].plot(tips_r[:, 0], tips_r[:, 2], "o-", ms=3, label="tip path (rope)")
    axes[1].scatter(ball_mm[0], ball_mm[2], c="gold", s=80)
    axes[1].set_xlabel("X"); axes[1].set_ylabel("Z"); axes[1].legend()
    axes[1].set_title("Tip trajectory XZ"); axes[1].grid(True, alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "v2_grasp_fk.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    print(f"saved {p}")
    plt.close(fig)


def resolve_ball_and_radius(
    coord_file: Path,
    ball_cli: list[float] | None,
    radius_cli: float | None,
    diameter_cli: float,
) -> tuple[np.ndarray, float]:
    if ball_cli is not None:
        ball_mm = np.asarray(ball_cli, dtype=float)
        radius = (
            float(radius_cli)
            if radius_cli is not None
            else float(diameter_cli) * 0.5
        )
        print(f"[sim] 使用命令行球心: {ball_mm.tolist()} mm")
        return ball_mm, radius

    coord_file = Path(coord_file)
    if not coord_file.is_file():
        raise SystemExit(
            f"未指定 --ball 且找不到 {coord_file}。"
            "请先运行: python run_d435_vision.py"
        )
    ball_mm, file_radius, meta = load_external_ball_coordinate(coord_file)
    radius = (
        float(radius_cli)
        if radius_cli is not None
        else float(file_radius)
        if file_radius is not None
        else float(diameter_cli) * 0.5
    )
    print(
        f"[sim] 从视觉坐标文件读取球心: {coord_file}\n"
        f"      center_mm={ball_mm.round(2).tolist()}  frame={meta.get('source_frame', '?')}"
    )
    return ball_mm, radius


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coord-file",
        type=Path,
        default=DEFAULT_COORD_FILE_PATH,
        help="视觉写的球位 JSON（与 grasp_plan_execute_v2 相同，默认 ball_target.json）",
    )
    ap.add_argument(
        "--ball",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="手动球心 mm，覆盖 --coord-file（仅调试）",
    )
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--diameter", type=float, default=70.0)
    ap.add_argument("--base", type=float, nargs=3, default=DEFAULT_ARM_BASE_MM.tolist())
    ap.add_argument("--fk", choices=("rope", "marker", "both"), default="both")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--verify-motors", action="store_true")
    args = ap.parse_args()

    ball_mm, radius_mm = resolve_ball_and_radius(
        args.coord_file, args.ball, args.radius, args.diameter
    )
    base_mm = np.asarray(args.base, dtype=float)

    if args.verify_motors:
        verify_motor_directions()

    result = plan_grasp_v2(ball_center_mm=ball_mm, radius=radius_mm)
    ok = True
    if args.fk in ("rope", "both"):
        ok = print_report(result, ball_mm, base_mm, "rope") and ok
    if args.fk in ("marker", "both"):
        ok_m = print_report(result, ball_mm, base_mm, "marker")
        if args.fk == "marker":
            ok = ok_m

    if not args.no_plot:
        plot_result(result, ball_mm, base_mm)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
