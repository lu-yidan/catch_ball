#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 AprilTag 标定 soft_arm_center -> GP robot 坐标（写入 config/center_to_robot.json）。

前提（与 run_d435_vision.py 相同）：
  - 已贴 AprilTag id=0，并设置 --center-origin-in-tag-cm
  - 视觉里 tag_status=TAG，ball_target.json 含 tennis_tag_mm

单点标定（默认）：
  把网球放在「演示 CSV 抓取时」的同一位置，等 grasp_ready 后运行本脚本。
  用演示轨迹终点的 robot 球心作为真值，拟合 tag 轴 scale/offset。

用法:
  python calibrate_center_to_robot.py --coord-file ball_target.json
  python calibrate_center_to_robot.py --coord-file ball_target.json \\
      --robot-ball-mm 122 -205 853
  python calibrate_center_to_robot.py --coord-file ball_target.json \\
      --center-origin-in-tag-cm 50 -30 -27
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import GP_tennis as gp  # noqa: E402
from grasp_planning import (  # noqa: E402
    DEFAULT_CENTER_TO_ROBOT,
    DEFAULT_TAG_AXIS_SCALES,
    center_mm_to_tag_mm,
    fit_tag_to_robot_axes,
    load_external_ball_coordinate,
    save_center_to_robot_calib,
    tag_mm_to_robot,
)

DEFAULT_DEMO = _ROOT / gp.SOURCE_FILENAME
DEFAULT_ORIGIN_CM = (50.0, -30.0, -27.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AprilTag 链式标定 center/tag -> robot")
    p.add_argument("--coord-file", type=Path, default=_ROOT / "ball_target.json")
    p.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    p.add_argument(
        "--center-origin-in-tag-cm",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="与 run_d435_vision.py 一致；缺省用 JSON 内或 50 -30 -27",
    )
    p.add_argument(
        "--robot-ball-mm",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="该抓取点的 GP robot 球心 (mm)；缺省用演示 CSV 终态",
    )
    p.add_argument(
        "--tag-scales",
        type=float,
        nargs=3,
        default=DEFAULT_TAG_AXIS_SCALES,
        metavar=("SX", "SY", "SZ"),
        help="固定 tag 轴缩放，默认 1 -1 1（Y 反向）",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_CENTER_TO_ROBOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.coord_file.is_file():
        raise SystemExit(f"缺少 {args.coord_file}，请先运行 run_d435_vision.py")

    center_mm, _, meta = load_external_ball_coordinate(args.coord_file)
    if meta.get("tag_status", "").startswith("NO_TAG"):
        raise SystemExit("当前无 AprilTag（tag_status=NO_TAG），无法做 tag 标定")

    if args.center_origin_in_tag_cm is not None:
        origin_mm = np.asarray(args.center_origin_in_tag_cm, dtype=float) * 10.0
    elif meta.get("center_origin_in_tag_mm") is not None:
        origin_mm = np.asarray(meta["center_origin_in_tag_mm"], dtype=float)
    else:
        origin_mm = np.asarray(DEFAULT_ORIGIN_CM, dtype=float) * 10.0
        print(f"[Calib] 使用默认 center_origin_in_tag_cm={DEFAULT_ORIGIN_CM}")

    calib_stub = {"center_origin_in_tag_mm": origin_mm.tolist()}
    if meta.get("tennis_tag_mm") is not None:
        tag_mm = np.asarray(meta["tennis_tag_mm"], dtype=float)
        print("[Calib] 使用 JSON 内 tennis_tag_mm")
    else:
        tag_mm = center_mm_to_tag_mm(center_mm, calib_stub)
        print("[Calib] 由 tennis_center_mm + origin 推算 tag_mm")

    if args.robot_ball_mm is not None:
        robot_mm = np.asarray(args.robot_ball_mm, dtype=float)
    elif args.demo.is_file():
        robot_mm = gp._load_source_demo(args.demo)["source_center"]
        print(f"[Calib] robot 真值 = 演示终态球心 {robot_mm.round(2).tolist()} mm")
    else:
        raise SystemExit("需要 --robot-ball-mm 或可用的 --demo CSV")

    axes = fit_tag_to_robot_axes(tag_mm, robot_mm, scales=list(args.tag_scales))
    payload = {
        "mode": "tag",
        "comment": (
            "AprilTag 链: p_tag_mm = p_center_mm + center_origin_in_tag_mm; "
            "p_robot = axis_map(p_tag). 与 run_d435_vision 的 center-origin 一致。"
        ),
        "center_origin_in_tag_cm": (origin_mm / 10.0).tolist(),
        "center_origin_in_tag_mm": origin_mm.tolist(),
        "demo_robot_center_mm": robot_mm.tolist(),
        "calib_ball_center_mm": center_mm.tolist(),
        "calib_tag_mm": tag_mm.tolist(),
        "tag_axis_scales": list(args.tag_scales),
        "axes": axes,
    }
    save_center_to_robot_calib(args.output, payload)

    check = tag_mm_to_robot(tag_mm, payload)
    err = float(np.linalg.norm(check - robot_mm))
    print(f"[Calib] 已写入 {args.output}")
    print(f"  center_mm = {center_mm.round(2).tolist()}")
    print(f"  tag_mm    = {tag_mm.round(2).tolist()}")
    print(f"  robot_ref = {robot_mm.round(2).tolist()}")
    print(f"  robot_fit = {check.round(2).tolist()}  (err={err:.2f} mm)")
    for axis in axes:
        print(
            f"  robot_{axis['robot']} = {axis['scale']:.0f}*tag_{axis['tag']} "
            f"+ {axis['offset']:.1f}"
        )
    if err > 5.0:
        print(
            "\n[警告] 残差 >5mm：请确认网球在演示抓取位置，且 center-origin 与视觉一致。"
        )


if __name__ == "__main__":
    main()
