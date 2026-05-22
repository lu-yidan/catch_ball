#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端2 v2：绳组运动学规划 + 执行（不用 GP）。

流程:
  1. 等 grasp_ready（球闪动<75mm 持续 2s）
  2. 绳组逆运动学规划 → grasp_plan_v2.json（第1–2节 Plane/Theta，第3节 motor1/4）
  3. 确认仍静止后，按规划执行:
       - 第1–2节: motor2/3/5/6（Plane/Theta）
       - 第3节: 规划得到的 motor1/4（6500<|m|<8000，|m1|+|m4|<15000）

Usage:
    python grasp_plan_execute_v2.py --coord-file ball_target.json --port COM5
    python grasp_plan_execute_v2.py --exec-mode simple   # 仅第三节，不跑接近
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grasp_excute import (
    DEFAULT_MIN_STEP_DELTA,
    DEFAULT_SPEED,
    FeetechBus,
    execute_kinematic_plan_v2,
    execute_motor14_grasp,
    parse_float_list,
    project_motor14_to_grasp_range,
)
from grasp_plan_execute import (
    DEFAULT_COORD_FILE,
    DEFAULT_WAIT_READY_TIMEOUT_S,
    resolve_radius_mm,
    wait_grasp_ready,
)
from grasp_planning_v2 import (
    DEFAULT_OUTPUT_V2,
    plan_grasp_v2,
    save_plan_v2,
)

try:
    import serial
except ImportError:
    serial = None

from paths import DEFAULT_DEMO_CSV as DEFAULT_DEMO

_ROOT = Path(__file__).resolve().parent
DEFAULT_START_DELAY_S = 5.0
DEFAULT_HOLD_S = 10.0
MIN_STABLE_BEFORE_EXEC_S = 2.0


def confirm_grasp_ready(path: Path, min_stable_s: float = MIN_STABLE_BEFORE_EXEC_S) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("valid") or not data.get("grasp_ready"):
        raise SystemExit("执行前 grasp_ready 失效，请重新等球静止")
    if float(data.get("stationary_elapsed_s", 0.0)) < min_stable_s:
        raise SystemExit(
            f"执行前静止仅 {data['stationary_elapsed_s']:.1f}s < {min_stable_s:.1f}s"
        )


def print_plan_summary_v2(plan_path: Path, result) -> None:
    info = result.info
    home = info.get("home_pose_deg", [0] * 6)
    final = info.get("final_pose_deg", [0] * 6)
    print(
        f"Saved plan: {plan_path}\n"
        f"  臂系球心={info.get('ball_arm_mm')}  to_ball_XZ={info.get('to_ball_xz_mm')}\n"
        f"  scale={info.get('vision_to_arm_scale')}  dist_XZ={info.get('approach_dist_xz_mm', '?'):.1f} mm\n"
        f"  home  Plane=[{home[0]:.1f},{home[1]:.1f}]°  Theta=[{home[3]:.1f},{home[4]:.1f}]°\n"
        f"  final Plane=[{final[0]:.1f},{final[1]:.1f}]°  Theta=[{final[3]:.1f},{final[4]:.1f}]°  "
        f"(倒立 XZ IK, phi=atan2(Z,X))\n"
        f"  第3节 motor1={info.get('motor1_steps_cmd')}  motor4={info.get('motor4_steps_cmd')}  "
        f"|m1|+|m4|={info.get('motor14_sum_abs')}  "
        f"auto={info.get('motor14_auto', True)}  motor14_ok={info.get('motor14_in_range')}"
    )


def execute_on_bus_v2(args: argparse.Namespace, plan: dict, motor1: int, motor4: int) -> None:
    if serial is None:
        raise SystemExit("pip install pyserial")

    verbose = not args.quiet
    steps_per_degree = parse_float_list(args.steps_per_degree, 6, "steps_per_degree")
    step_limits = [int(v) for v in parse_float_list(args.step_limits, 6, "step_limits")]

    try:
        bus = FeetechBus(args.port, args.baud, debug=args.debug)
        print(f"[Serial] 已打开 {args.port} @ {args.baud} baud")
    except (serial.SerialException, PermissionError, OSError) as exc:
        raise SystemExit(f"无法打开串口 {args.port}: {exc}") from exc

    try:
        for sid in range(1, 7):
            bus.enable_torque(sid, 1)
            time.sleep(0.03)
        time.sleep(0.08)

        if args.exec_mode == "simple":
            print("--- 执行模式 simple: 仅第三节 (跳过第1–2节轨迹) ---")
            execute_motor14_grasp(
                bus,
                motor1,
                motor4,
                args.speed,
                start_delay_s=args.start_delay,
                hold_s=args.hold,
                return_home=not args.no_return_home,
                verbose=verbose,
            )
        else:
            print("--- 执行模式 kinematic: 第1–2节接近 + 第3节抓取 ---")
            execute_kinematic_plan_v2(
                bus,
                plan,
                args.speed,
                steps_per_degree,
                waypoint_stride=args.waypoint_stride,
                min_step_delta=args.min_step_delta,
                max_abs_steps=step_limits,
                start_delay_s=args.start_delay,
                hold_grasp_s=args.hold,
                hold_approach_s=args.hold_approach,
                return_home=not args.no_return_home,
                verbose=verbose,
            )
        print("完成。")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        bus.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2: 静止2s -> 运动学规划 -> 分节执行")
    p.add_argument("--coord-file", type=Path, default=DEFAULT_COORD_FILE)
    p.add_argument("--wait-timeout", type=float, default=DEFAULT_WAIT_READY_TIMEOUT_S)
    p.add_argument("--radius", type=float)
    p.add_argument("--diameter", type=float, default=70.0)
    p.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    p.add_argument(
        "--motor1",
        type=int,
        default=None,
        help="手动覆盖第三节 motor1（默认按球位自动规划）",
    )
    p.add_argument(
        "--motor4",
        type=int,
        default=None,
        help="手动覆盖第三节 motor4（默认按球位自动规划）",
    )
    p.add_argument("--n-waypoints", type=int, default=40)
    p.add_argument(
        "--phi-offset",
        type=float,
        default=0.0,
        help="第1–2节 Plane 角额外偏移 (度)，默认 0",
    )
    p.add_argument(
        "--arm-axes-config",
        type=Path,
        default=None,
        help="臂系/视觉映射 JSON（默认 config/soft_arm_arm_axes.json）",
    )
    p.add_argument("--plan-output", type=Path, default=DEFAULT_OUTPUT_V2)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--exec-mode",
        choices=["kinematic", "simple"],
        default="kinematic",
        help="kinematic=第1–2节+第3节; simple=仅 motor1/4",
    )
    p.add_argument("--port", default="COM5")
    p.add_argument("--baud", type=int, default=1000000)
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    p.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_S)
    p.add_argument("--hold", type=float, default=DEFAULT_HOLD_S)
    p.add_argument("--hold-approach", type=float, default=2.0, help="第1–2节接近后保持 (s)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--steps-per-degree", default="100")
    p.add_argument("--waypoint-stride", type=int, default=3)
    p.add_argument("--min-step-delta", type=int, default=DEFAULT_MIN_STEP_DELTA)
    p.add_argument("--step-limits", default="8000,4000,4000,8000,4000,4000")
    p.add_argument("--no-return-home", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.demo.is_file():
        raise SystemExit(f"缺少演示 CSV: {args.demo}")

    print("[v2] 步骤1/3: 等待球静止 grasp_ready...")
    ball_center, file_radius, meta = wait_grasp_ready(
        args.coord_file, timeout_s=args.wait_timeout
    )
    radius = resolve_radius_mm(args, file_radius)

    print(
        f"[v2] 步骤2/3: 绳组运动学规划 @ {ball_center.round(2).tolist()} mm\n"
        f"       球心来源: {args.coord_file.resolve()} (D435/视觉 soft_arm_center，非写死)"
    )
    if args.motor1 is not None or args.motor4 is not None:
        if args.motor1 is None or args.motor4 is None:
            raise SystemExit("手动覆盖第三节需同时指定 --motor1 与 --motor4")
        m1, m4 = project_motor14_to_grasp_range(args.motor1, args.motor4)
        print(f"  使用手动 motor1/4: {m1}, {m4}")
    else:
        m1, m4 = None, None
        print("  第3节: 绳组逆运动学 (motor1/4 可行域内朝球)")

    result = plan_grasp_v2(
        ball_center_mm=ball_center,
        radius=radius,
        demo_csv=args.demo,
        n_waypoints=args.n_waypoints,
        motor1_steps=m1,
        motor4_steps=m4,
        phi_offset_deg=args.phi_offset,
        arm_axes_config=args.arm_axes_config,
    )
    m1 = int(result.info["motor1_steps_cmd"])
    m4 = int(result.info["motor4_steps_cmd"])
    save_plan_v2(args.plan_output, result, ball_center, radius)
    if meta:
        with args.plan_output.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["vision"] = meta
        payload["exec_mode"] = args.exec_mode
        with args.plan_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print_plan_summary_v2(args.plan_output, result)

    if not result.info["motor14_in_range"]:
        raise SystemExit(f"motor1/4 超范围: m1={m1}, m4={m4}")

    if args.plan_only:
        print("--plan-only: 已规划未执行")
        return

    if args.dry_run:
        print(f"--dry-run: exec_mode={args.exec_mode}")
        return

    print("[v2] 步骤3/3: 确认仍静止，按规划执行...")
    confirm_grasp_ready(args.coord_file)
    time.sleep(0.5)

    with args.plan_output.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    execute_on_bus_v2(args, plan, m1, m4)


if __name__ == "__main__":
    main()
