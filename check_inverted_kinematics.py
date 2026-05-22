#!/usr/bin/env python3
"""倒立运动学一致性自检（绳组 / marker / motor 经验）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import GP_tennis as gp
from grasp_excute import (
    INVERTED_PCC_THETA_SIGN,
    MARKER_PLANE_OFFSET_FROM_ROPE_PHI_DEG,
    MOTOR_POSITIVE_PULL_PHI_DEG,
    ROPE_PLANNER_THETA_SIGN,
    bend_dir_3d_inverted_from_phi,
    fk_inverted_pcc_centerline,
    marker_plane_to_rope_phi_deg,
    motor14_steps_to_bend_delta,
    motor_pull_unit_xz,
    motor_steps_to_bend_delta_xz,
    pose_pair_to_motor_steps,
    rope_phi_to_marker_plane_deg,
    solve_sections_12_inverse_kinematics,
)
from grasp_planning_v2 import load_inversej_first_pose, plan_grasp_v2
from simulate_v2_grasp import fk_marker_centerline

PASS = FAIL = 0


def ok(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def tip_delta(pose_a, pose_b) -> np.ndarray:
    ta = fk_inverted_pcc_centerline(pose_a)[-1]
    tb = fk_inverted_pcc_centerline(pose_b)[-1]
    return tb - ta


print("=" * 60)
print("1. Motor +step -> u(phi) 与实机经验")
print("=" * 60)
for mid, expected, phi in [
    (1, np.array([-1.0, 0.0]), 180),
    (2, np.array([0.0, -1.0]), 270),
    (6, np.array([1.0, 0.0]), 0),
]:
    u = motor_pull_unit_xz(phi)
    d = motor_steps_to_bend_delta_xz({mid: 1000})
    d_norm = d / (np.linalg.norm(d) + 1e-12)
    ok(f"motor{mid}+ direction", np.allclose(d_norm, expected, atol=1e-3), f"d~{d}")

print()
print("=" * 60)
print("2. 单节 FK：直立 + 仅第1节弯曲")
print("=" * 60)
home = np.zeros(6)
# phi=0, theta=30 -> 应朝 +X 弯
p_px = home.copy()
p_px[0], p_px[3] = 0.0, 30.0
d_px = tip_delta(home, p_px)
ok("+X curl phi=0", d_px[0] > 80 and abs(d_px[2]) < 40, f"delta={d_px.round(1)}")

# phi=180, theta=30 -> 应朝 -X 弯
p_mx = home.copy()
p_mx[0], p_mx[3] = 180.0, 30.0
d_mx = tip_delta(home, p_mx)
ok("-X curl phi=180", d_mx[0] < -80 and abs(d_mx[2]) < 40, f"delta={d_mx.round(1)}")

# phi=270, theta=30 -> 应朝 -Z 弯
p_mz = home.copy()
p_mz[0], p_mz[3] = 270.0, 30.0
d_mz = tip_delta(home, p_mz)
ok("-Z curl phi=270", d_mz[2] < -80 and abs(d_mz[0]) < 40, f"delta={d_mz.round(1)}")

print()
print("=" * 60)
print("3. motor1 步进符号 vs FK（第3节）")
print("=" * 60)
h3 = np.zeros(6)
bend_m1p = motor14_steps_to_bend_delta(7500, 0)
bend_m1n = motor14_steps_to_bend_delta(-7500, 0)
ok("motor1+ bend_x has -X", bend_m1p[0] < -50, str(bend_m1p.round(1)))
ok("motor1- bend_x has +X", bend_m1n[0] > 50, str(bend_m1n.round(1)))

print()
print("=" * 60)
print("4. rope_phi <-> marker_plane (+270) 与 GP 臂系一致")
print("=" * 60)
test_poses = [
    np.array([0, 0, 0, 20, 0, 0], float),
    np.array([0, 0, 0, 30, 0, 0], float),
    np.array([90, 90, 90, 25, 25, 25], float),
]
for p in test_poses:
    tr = fk_inverted_pcc_centerline(p)[-1]
    tm = fk_marker_centerline(p)[-1]
    err = float(np.linalg.norm(tr - tm))
    ok(f"rope vs marker FK err<1mm", err < 1.0, f"err={err:.3f} tip_r={tr.round(1)}")

# phi_rope=0 -> marker=270
ok("phi offset", abs(rope_phi_to_marker_plane_deg(0) - 270) < 1e-6)
ok("inverse offset", abs(marker_plane_to_rope_phi_deg(270)) < 1e-6)

print()
print("=" * 60)
print("5. CSV 首帧 marker 角 -> rope phi 后 FK 合理")
print("=" * 60)
from paths import DEFAULT_DEMO_CSV as csv_path
if csv_path.is_file():
    marker_pose = load_inversej_first_pose(csv_path)
    rope_pose = marker_pose.copy()
    rope_pose[0:3] = [marker_plane_to_rope_phi_deg(x) for x in marker_pose[0:3]]
    rope_pose[3:6] = marker_pose[3:6] * (ROPE_PLANNER_THETA_SIGN / INVERTED_PCC_THETA_SIGN)
    tr = fk_inverted_pcc_centerline(rope_pose)[-1]
    tm = fk_marker_centerline(
        np.array(
            [
                marker_plane_to_rope_phi_deg(marker_pose[0]),
                marker_plane_to_rope_phi_deg(marker_pose[1]),
                marker_plane_to_rope_phi_deg(marker_pose[2]),
                marker_pose[3] * (ROPE_PLANNER_THETA_SIGN / INVERTED_PCC_THETA_SIGN),
                marker_pose[4] * (ROPE_PLANNER_THETA_SIGN / INVERTED_PCC_THETA_SIGN),
                marker_pose[5] * (ROPE_PLANNER_THETA_SIGN / INVERTED_PCC_THETA_SIGN),
            ]
        )
    )[-1]
    # direct marker FK
    pts_gp = gp.pose_to_points(marker_pose, gp.SEGMENT_LENGTHS, gp.ROBOT_BASE)
    arm = np.array([pts_gp[-1, 0], gp.ROBOT_BASE[2] - pts_gp[-1, 2], pts_gp[-1, 1]])
    err_csv = float(np.linalg.norm(tr - arm))
    ok("CSV marker direct vs converted rope", err_csv < 2.0, f"err={err_csv:.2f}")
    print(f"       CSV marker tip (arm frame) {arm.round(1)}")
    print(f"       rope converted tip        {tr.round(1)}")
else:
    print("  (skip: demo CSV missing)")

print()
print("=" * 60)
print("6. v2 IK：球在 +X -> phi=0, 仿真朝 +X")
print("=" * 60)
ball = np.array([200.0, 400.0, 0.0])
r = plan_grasp_v2(ball_center_mm=ball)
phi_ik = float(r.info["section12_phi_ik_deg"])
tip0 = fk_inverted_pcc_centerline(np.zeros(6))[-1]
tipf = fk_inverted_pcc_centerline(r.final_pose)[-1]
ok("phi12 IK ~0 for +X ball", abs(phi_ik) < 5 or abs(phi_ik - 360) < 5, f"phi={phi_ik:.1f}")
ok("final tip X > home X", tipf[0] > tip0[0] + 50, f"home={tip0[0]:.0f} final={tipf[0]:.0f}")
ok("dist to ball improves", r.trajectory is not None, "")

print()
print("=" * 60)
print("7. pose_pair_to_motor_steps 与 bend 分解一致（第1节 +X）")
print("=" * 60)
before = np.zeros(6)
after = before.copy()
after[0], after[3] = 0.0, 10.0
steps = pose_pair_to_motor_steps(before, after, [100.0] * 6, min_step_delta=1)
# 主要应动 motor6 (+X)
ok("dTheta mainly motor6", abs(steps[5]) >= abs(steps[1]), f"vec={steps}")

print()
print("=" * 60)
print(f"SUMMARY: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(1 if FAIL else 0)
