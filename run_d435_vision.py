#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端1：D435 视觉 = 网球(绿) + 末端蓝盘(蓝) + AprilTag + centre 坐标系，写 ball_target.json。

终端2：grasp_plan_execute.py 读 JSON 规划+舵机。

Usage:
    python run_d435_vision.py
    python run_d435_vision.py --center-origin-in-tag-cm 50 -30 -27 --output-json ball_target.json
    python run_d435_vision.py --show-mask   # 右侧显示 HSV 掩膜
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from vision_d435_tracker import TennisBallTracker

_ROOT = Path(__file__).resolve().parent
DEFAULT_CENTER_ORIGIN_TAG_CM = (50.0, -30.0, -27.0)
DEFAULT_OUTPUT_JSON = _ROOT / "ball_target.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D435 网球视觉（独立进程，带窗口）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--tag-id", type=int, default=0)
    p.add_argument("--tag-size", type=float, default=0.25)
    p.add_argument("--tag-every", type=int, default=5)
    p.add_argument("--tag-max-age", type=int, default=15)
    p.add_argument(
        "--center-origin-in-tag-cm",
        type=float,
        nargs=3,
        default=DEFAULT_CENTER_ORIGIN_TAG_CM,
        metavar=("X", "Y", "Z"),
    )
    p.add_argument("--tennis-radius", type=float, default=0.035)
    p.add_argument("--blue-h-low", type=int, default=94)
    p.add_argument("--blue-h-high", type=int, default=104)
    p.add_argument("--blue-s-min", type=int, default=80)
    p.add_argument("--blue-v-min", type=int, default=35)
    p.add_argument("--blue-diameter", type=float, default=0.026, help="末端蓝盘直径 m")
    p.add_argument("--blue-circularity", type=float, default=0.30)
    p.add_argument("--center-axis-len", type=float, default=0.10, help="centre 坐标轴可视化长度 m")
    p.add_argument("--show-mask", action="store_true", help="显示网球/蓝盘 HSV 掩膜")
    p.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="供 grasp_plan_execute.py 读取",
    )
    p.add_argument(
        "--stationary-time",
        type=float,
        default=2.0,
        help="闪动范围内持续多久后 grasp_ready=true (s)",
    )
    p.add_argument(
        "--stationary-flicker",
        type=float,
        default=75.0,
        help="相对锚点的位置闪动上限 (mm)，超过则重新计时",
    )
    p.add_argument("--no-ema", action="store_true")
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="关闭窗口（一般保持默认开启）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.show_viz = not args.no_viz

    tracker = TennisBallTracker.from_namespace(args)
    print(
        "[D435] 视觉: 网球 + 末端蓝盘 + AprilTag\n"
        f"  窗口: {'ON' if args.show_viz else 'OFF'}  掩膜: {'ON' if args.show_mask else 'OFF'}\n"
        f"  Tag id={args.tag_id} 每 {args.tag_every} 帧检测  center_origin_cm={args.center_origin_in_tag_cm}\n"
        f"  输出: {args.output_json}\n"
        f"  闪动<{args.stationary_flicker:.0f}mm 持续 {args.stationary_time:.0f}s -> grasp_ready=true\n"
        "  画面: 绿圈=网球  蓝圈=末端蓝盘  彩色轴=centre系  需看到 TAG\n"
        "  终端2: python grasp_plan_execute.py  按 q 退出"
    )
    tracker.start()
    try:
        while tracker._thread is not None and tracker._thread.is_alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[D435] 用户中断")
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
