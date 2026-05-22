#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端2：规划 + 舵机执行（不打开相机）。

请先另开终端运行:
    python run_d435_vision.py

本脚本读取 ball_target.json，等 grasp_ready=true 后规划并卷取（最多10s）再复位。

试验绳组规划（无 GP、固定第三节 -7500/+7000）请用:
    python grasp_plan_execute_v2.py --coord-file ball_target.json --port COM5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from grasp_excute import (
    DEFAULT_MIN_STEP_DELTA,
    DEFAULT_SPEED,
    DEFAULT_WAYPOINT_DELAY_S,
    FeetechBus,
    add_vec,
    clamp_vec_to_step_limits,
    execute_plan,
    parse_float_list,
    send_vec,
    trajectory_to_step_vectors,
)
import GP_tennis as gp

from grasp_planning import (
    DEFAULT_BALL_RADIUS_MM,
    DEFAULT_OUTPUT,
    load_center_to_robot,
    load_external_ball_coordinate,
    plan_grasp,
    resolve_ball_center_for_planning,
    save_plan,
)

try:
    import serial
except ImportError:
    serial = None

from paths import DEFAULT_DEMO_CSV as DEFAULT_DEMO

_ROOT = Path(__file__).resolve().parent
DEFAULT_COORD_FILE = _ROOT / "ball_target.json"
DEFAULT_MAX_GRASP_TIME_S = 10.0
DEFAULT_WAIT_READY_TIMEOUT_S = 300.0
MOTOR_ENABLE_DELAY_S = 1.0


def resolve_radius_mm(args: argparse.Namespace, fallback_mm: float | None = None) -> float:
    if args.radius is not None:
        return float(args.radius)
    if args.diameter is not None:
        return float(args.diameter) * 0.5
    if fallback_mm is not None:
        return float(fallback_mm)
    return DEFAULT_BALL_RADIUS_MM


def wait_grasp_ready(
    path: Path,
    timeout_s: float,
    poll_s: float = 0.2,
) -> tuple[np.ndarray, float, dict]:
    """Poll JSON until valid and grasp_ready (written by run_d435_vision.py)."""
    deadline = time.time() + timeout_s
    path = Path(path)
    print(f"[Execute] 等待 {path} 中 grasp_ready=true (超时 {timeout_s:.0f}s)...")

    while time.time() < deadline:
        if not path.is_file():
            time.sleep(poll_s)
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            time.sleep(poll_s)
            continue
        except PermissionError:
            time.sleep(poll_s)
            continue
        except OSError:
            time.sleep(poll_s)
            continue

        if data.get("valid") and data.get("grasp_ready"):
            center, radius, meta = load_external_ball_coordinate(path)
            meta["grasp_ready"] = True
            meta["stationary_elapsed_s"] = data.get("stationary_elapsed_s")
            meta["flicker_mm"] = data.get("flicker_mm", data.get("speed_mm_s"))
            print(
                f"[Execute] grasp_ready, center_mm={center.round(2).tolist()}, "
                f"stable={data.get('stationary_elapsed_s', '?')}s"
            )
            return center, radius, meta

        elapsed = data.get("stationary_elapsed_s", 0.0)
        flicker = data.get("flicker_mm", data.get("speed_mm_s", 0.0))
        valid = data.get("valid", False)
        print(
            f"\r[Execute] 等待静止... valid={valid} stable={elapsed:.1f}s "
            f"flicker={flicker:.0f}mm   ",
            end="",
            flush=True,
        )
        time.sleep(poll_s)

    raise TimeoutError(f"grasp_ready not set in {path} within {timeout_s:.0f}s")


def print_plan_summary(plan_path: Path, result, radius: float) -> None:
    info = result.info
    print(
        f"Saved plan: {plan_path}\n"
        f"  approach={result.approach}, success={info['success']}, "
        f"radius={radius:.1f} mm, "
        f"center_error={info.get('center_error', float('nan')):.1f} mm, "
        f"wrap_gap={info.get('wrap_gap', float('nan')):.1f} mm, "
        f"tip_surface_gap={info.get('tip_surface_gap', float('nan')):.1f} mm, "
        f"motor1={info.get('motor1_steps_from_home')}, "
        f"motor4={info.get('motor4_steps_from_home')}, "
        f"|m1|+|m4|={info.get('motor14_sum_abs', '?')}, "
        f"motor14_ok={info.get('motor14_in_range', '?')}"
    )
    if not info["success"]:
        print(
            "  规划未达标：通常是坐标系未标定或球位超出演示工作空间。"
            "检查 config/center_to_robot.json 或加 --execute-unsafe 试执行。"
        )


def print_dry_run_vectors(args: argparse.Namespace, plan: dict, step_limits: list[int]) -> None:
    steps_per_degree = parse_float_list(args.steps_per_degree, 6, "steps_per_degree")
    vectors = trajectory_to_step_vectors(
        plan["trajectory_deg"],
        steps_per_degree,
        waypoint_stride=args.waypoint_stride,
        min_step_delta=args.min_step_delta,
    )
    current = [0, 0, 0, 0, 0, 0]
    safe_vectors = []
    for vec in vectors:
        safe_vec = clamp_vec_to_step_limits(vec, current, step_limits)
        current = add_vec(current, safe_vec)
        safe_vectors.append(safe_vec)
    print("--dry-run: no serial.")
    print(f"vectors={vectors}")
    print(f"final_steps={current}")


def execute_plan_on_bus(args: argparse.Namespace, plan: dict) -> None:
    if serial is None:
        raise SystemExit("pip install pyserial")

    steps_per_degree = parse_float_list(args.steps_per_degree, 6, "steps_per_degree")
    step_limits = [int(v) for v in parse_float_list(args.step_limits, 6, "step_limits")]

    if args.dry_run:
        print_dry_run_vectors(args, plan, step_limits)
        return

    try:
        bus = FeetechBus(args.port, args.baud, debug=args.debug)
    except (serial.SerialException, PermissionError, OSError) as exc:
        raise SystemExit(f"无法打开串口 {args.port}: {exc}") from exc

    try:
        for sid in range(1, 7):
            bus.enable_torque(sid, 1)
            time.sleep(0.03)
        time.sleep(0.08)
        print(f"--- 上电延时 {MOTOR_ENABLE_DELAY_S:.1f}s ---")
        time.sleep(MOTOR_ENABLE_DELAY_S)

        print(f"--- 卷取开始 (最长 {args.max_grasp_time:.1f}s) ---")
        total_vec = execute_plan(
            bus,
            plan,
            args.speed,
            steps_per_degree,
            args.waypoint_stride,
            args.waypoint_delay,
            args.min_step_delta,
            return_home=False,
            max_abs_steps=step_limits,
            max_grasp_s=args.max_grasp_time,
        )

        if not args.no_return_home:
            home_vec = [-x for x in total_vec]
            print(f"--- 复位 home_vec={home_vec} ---")
            send_vec(bus, home_vec, args.speed)
        print("完成。")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        bus.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="读取 ball_target.json，规划并执行抓取（需先运行 run_d435_vision.py）"
    )
    p.add_argument(
        "--coord-file",
        type=Path,
        default=DEFAULT_COORD_FILE,
        help="run_d435_vision.py 写入的目标文件",
    )
    p.add_argument(
        "--wait-timeout",
        type=float,
        default=DEFAULT_WAIT_READY_TIMEOUT_S,
        help="等待 grasp_ready 的最长时间 (s)",
    )
    p.add_argument("--radius", type=float)
    p.add_argument("--diameter", type=float, default=70.0)
    p.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    p.add_argument(
        "--center-to-robot",
        type=Path,
        default=None,
        help="soft_arm_center -> robot 标定 JSON（默认 config/center_to_robot.json）",
    )
    p.add_argument("--approach", choices=["auto", "side", "top-down", "below"], default="auto")
    p.add_argument("--n-waypoints", type=int, default=80)
    p.add_argument("--plan-output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--execute-unsafe", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--port", default="COM5")
    p.add_argument("--baud", type=int, default=1000000)
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--steps-per-degree", default="100")
    p.add_argument("--waypoint-stride", type=int, default=3)
    p.add_argument("--waypoint-delay", type=float, default=DEFAULT_WAYPOINT_DELAY_S)
    p.add_argument("--min-step-delta", type=int, default=DEFAULT_MIN_STEP_DELTA)
    p.add_argument("--step-limits", default="8000,4000,4000,8000,4000,4000")
    p.add_argument("--no-return-home", action="store_true")
    p.add_argument("--max-grasp-time", type=float, default=DEFAULT_MAX_GRASP_TIME_S)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.demo.is_file():
        raise SystemExit(f"缺少演示 CSV: {args.demo}")

    ball_center, file_radius, meta = wait_grasp_ready(
        args.coord_file, timeout_s=args.wait_timeout
    )
    radius = resolve_radius_mm(args, file_radius)
    print(
        f"[Target] vision {ball_center.round(2).tolist()} mm "
        f"(frame={meta.get('source_frame', '?')}), r={radius:.1f} mm"
    )

    calib = load_center_to_robot(args.center_to_robot)
    demo_source = gp._load_source_demo(args.demo)["source_center"]
    try:
        plan_center, frame_meta = resolve_ball_center_for_planning(
            ball_center,
            meta.get("source_frame"),
            calib,
            demo_source,
            vision_meta=meta,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    meta.update(frame_meta)
    print(f"[Target] robot  {plan_center.round(2).tolist()} mm (for GP planner)")

    result = plan_grasp(
        ball_center=plan_center,
        radius=radius,
        approach=args.approach,
        demo_csv=args.demo,
        n_waypoints=args.n_waypoints,
    )
    save_plan(args.plan_output, result, ball_center, radius)
    if meta:
        with args.plan_output.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["vision"] = meta
        with args.plan_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    print_plan_summary(args.plan_output, result, radius)

    if args.strict and not result.info["success"]:
        raise SystemExit("规划未达 success 阈值")
    if args.plan_only:
        print("--plan-only: 未执行舵机")
        return
    if not result.info["success"] and not args.execute_unsafe and not args.dry_run:
        raise SystemExit("规划 success=False，加 --execute-unsafe 可强制执行")

    with args.plan_output.open("r", encoding="utf-8") as f:
        plan = json.load(f)
    execute_plan_on_bus(args, plan)


if __name__ == "__main__":
    main()
