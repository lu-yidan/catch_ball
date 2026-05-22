#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端2 v3：规划 -> 仿真 plot -> 人工确认 -> 可选执行。

流程:
  1. 等 grasp_ready（或 --ball 手动指定，与相机坐标系一致）
  2. 绳组规划 v3 -> grasp_plan_v3.json
  3. 仿真绘制初始/完成姿态（base、臂、球），评估能否抓住
  4. 终端输入 yes 才上实机；其它任意键取消

Usage:
    python grasp_plan_execute_v3.py --coord-file ball_target.json --port COM5
    python grasp_plan_execute_v3.py --ball 200 0 0 --no-wait   # 测试 +X 球，不连相机
    python grasp_plan_execute_v3.py --plan-only
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
from grasp_plan_execute_v2 import (
    MIN_STABLE_BEFORE_EXEC_S,
    confirm_grasp_ready,
    execute_on_bus_v2,
    print_plan_summary_v2,
)
from grasp_planning_v3 import (
    DEFAULT_OUTPUT_V3,
    plan_grasp_v3,
    save_plan_v3,
)
from simulate_v3_grasp import plot_grasp_preview, print_sim_report

try:
    import serial
except ImportError:
    serial = None

from paths import DEFAULT_DEMO_CSV as DEFAULT_DEMO

_ROOT = Path(__file__).resolve().parent
DEFAULT_START_DELAY_S = 5.0
DEFAULT_HOLD_S = 10.0


def prompt_execute() -> bool:
    print()
    print("-" * 60)
    print("仿真已完成。若要在实机执行，请在终端输入:  yes")
    print("其它任意输入将取消执行（规划 JSON 已保存）。")
    print("-" * 60)
    try:
        ans = input("Execute on robot? [yes/N]: ").strip().lower()
    except EOFError:
        return False
    return ans == "yes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3: plan -> sim plot -> yes -> execute")
    p.add_argument("--coord-file", type=Path, default=DEFAULT_COORD_FILE)
    p.add_argument("--wait-timeout", type=float, default=DEFAULT_WAIT_READY_TIMEOUT_S)
    p.add_argument(
        "--ball",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="手动球心 mm（soft_arm_center），跳过 wait grasp_ready",
    )
    p.add_argument("--no-wait", action="store_true", help="与 --ball 合用：不读 JSON 等待")
    p.add_argument("--radius", type=float)
    p.add_argument("--diameter", type=float, default=70.0)
    p.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    p.add_argument("--motor1", type=int, default=None)
    p.add_argument("--motor4", type=int, default=None)
    p.add_argument("--n-waypoints", type=int, default=40)
    p.add_argument("--phi-offset", type=float, default=0.0)
    p.add_argument("--arm-axes-config", type=Path, default=None)
    p.add_argument("--plan-output", type=Path, default=DEFAULT_OUTPUT_V3)
    p.add_argument("--plan-only", action="store_true", help="只规划+仿真，不询问执行")
    p.add_argument("--no-plot", action="store_true", help="不弹窗，仅保存 png")
    p.add_argument("--yes", action="store_true", help="跳过确认直接执行（慎用）")
    p.add_argument(
        "--exec-mode",
        choices=["kinematic", "simple"],
        default="kinematic",
    )
    p.add_argument("--port", default="COM5")
    p.add_argument("--baud", type=int, default=1000000)
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    p.add_argument("--start-delay", type=float, default=DEFAULT_START_DELAY_S)
    p.add_argument("--hold", type=float, default=DEFAULT_HOLD_S)
    p.add_argument("--hold-approach", type=float, default=2.0)
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

    meta: dict | None = None
    if args.ball is not None:
        import numpy as np

        ball_center = np.asarray(args.ball, dtype=float)
        file_radius = None
        print(f"[v3] 使用手动球心 (camera frame): {ball_center.tolist()} mm")
    elif args.no_wait:
        raise SystemExit("未指定 --ball 时不能 --no-wait")
    else:
        print("[v3] 步骤1: 等待球静止 grasp_ready...")
        ball_center, file_radius, meta = wait_grasp_ready(
            args.coord_file, timeout_s=args.wait_timeout
        )

    radius = resolve_radius_mm(args, file_radius if args.ball is None else None)

    print(f"[v3] 步骤2: 绳组规划 @ {ball_center.round(2).tolist()} mm")
    if args.motor1 is not None or args.motor4 is not None:
        if args.motor1 is None or args.motor4 is None:
            raise SystemExit("手动 motor1/4 需同时指定")
        m1, m4 = project_motor14_to_grasp_range(args.motor1, args.motor4)
    else:
        m1, m4 = None, None

    result = plan_grasp_v3(
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
    save_plan_v3(args.plan_output, result, ball_center, radius)

    if meta:
        with args.plan_output.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["vision"] = meta
        payload["exec_mode"] = args.exec_mode
        with args.plan_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print_plan_summary_v2(args.plan_output, result)

    print("[v3] 步骤3: 仿真评估与绘图...")
    sim_report = plot_grasp_preview(
        result,
        ball_center,
        radius,
        arm_axes_config=args.arm_axes_config,
        show=not args.no_plot,
    )
    print_sim_report(sim_report)

    # 写入仿真结果到 plan json
    with args.plan_output.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["sim"] = {
        "likely_grasp": sim_report.likely_grasp,
        "gap_final_mm": sim_report.gap_final_mm,
        "gap_initial_mm": sim_report.gap_initial_mm,
        "tip_initial_mm": sim_report.tip_initial_mm.tolist(),
        "tip_final_mm": sim_report.tip_final_mm.tolist(),
        "center_error_final_mm": sim_report.center_error_final_mm.tolist(),
        "notes": sim_report.notes,
    }
    with args.plan_output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if not result.info["motor14_in_range"]:
        print("[v3] 警告: motor1/4 不在可行域，不建议执行")

    if args.plan_only:
        print("--plan-only: 已规划并仿真，未询问执行")
        return

    if args.dry_run:
        print("--dry-run: 不执行实机")
        return

    do_exec = args.yes
    if not do_exec:
        do_exec = prompt_execute()

    if not do_exec:
        print("[v3] 已取消实机执行。规划见:", args.plan_output)
        return

    if args.ball is None:
        print("[v3] 步骤4: 确认仍静止后执行...")
        confirm_grasp_ready(args.coord_file)
        time.sleep(0.5)

    with args.plan_output.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    execute_on_bus_v2(args, plan, m1, m4)


if __name__ == "__main__":
    main()
