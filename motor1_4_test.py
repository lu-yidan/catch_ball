#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三节绳组测试：motor1/4 在抓取范围内（各 6500~8000，|m1|+|m4|<15000），保持 10 s 后复位。

Usage:
    python motor1_4_test.py --port COM5
    python motor1_4_test.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time

from grasp_excute import (
    DEFAULT_SPEED,
    FeetechBus,
    motor14_in_grasp_range,
    project_motor14_to_grasp_range,
    send_vec,
)

try:
    import serial
except ImportError:
    serial = None

DEFAULT_MOTOR1_STEPS = -7500
DEFAULT_MOTOR4_STEPS = 7400
DEFAULT_HOLD_S = 10.0
START_DELAY_S = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Section-3 rope test: motor1 + motor4, then home.")
    p.add_argument("--port", default="COM5")
    p.add_argument("--baud", type=int, default=1000000)
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    p.add_argument("--motor1", type=int, default=DEFAULT_MOTOR1_STEPS, help="motor1 step delta")
    p.add_argument("--motor4", type=int, default=DEFAULT_MOTOR4_STEPS, help="motor4 step delta")
    p.add_argument("--hold", type=float, default=DEFAULT_HOLD_S, help="hold time after move (s)")
    p.add_argument("--start-delay", type=float, default=START_DELAY_S)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    m1, m4 = project_motor14_to_grasp_range(args.motor1, args.motor4)
    if (m1, m4) != (args.motor1, args.motor4):
        print(f"[Test] clamp motor steps -> motor1={m1}, motor4={m4}")
    if not motor14_in_grasp_range(m1, m4):
        print(
            f"[Test] WARNING: motor1/4 out of grasp range "
            f"(need 6500<|m|<8000 and |m1|+|m4|<15000): {m1}, {m4}"
        )
    move_vec = [m1, 0, 0, m4, 0, 0]
    home_vec = [-m1, 0, 0, -m4, 0, 0]

    print(f"[Test] move vec={move_vec}")
    print(f"[Test] hold {args.hold:.1f}s")
    print(f"[Test] home vec={home_vec}")

    if args.dry_run:
        print("--dry-run: no serial commands sent.")
        return

    if serial is None:
        print("Please install pyserial: pip install pyserial")
        sys.exit(1)

    try:
        bus = FeetechBus(args.port, args.baud, debug=args.debug)
    except (serial.SerialException, PermissionError, OSError) as exc:
        print(f"Failed to open {args.port}: {exc}")
        sys.exit(1)

    try:
        for sid in range(1, 7):
            bus.enable_torque(sid, 1)
            time.sleep(0.03)
        time.sleep(0.08)

        if args.start_delay > 0:
            print(f"--- Start delay {args.start_delay:.1f}s ---")
            time.sleep(args.start_delay)

        print(f"--- Move: motor1={m1}, motor4={m4} (|m1|+|m4|={abs(m1)+abs(m4)}) ---")
        send_vec(bus, move_vec, args.speed, simple=True, verbose=args.debug)

        print(f"--- Hold {args.hold:.1f}s ---")
        time.sleep(args.hold)

        print(f"--- Return home: {home_vec} ---")
        send_vec(bus, home_vec, args.speed, simple=True, verbose=args.debug)
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
