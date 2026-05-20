"""
camera_ball_dual_hsv.py

RealSense D455 dual HSV detector:
  - tennis ball: HSV circle detection + visual/sensor depth fusion
  - blue end disk: HSV circle detection + RealSense depth at detected centre

Outputs both positions in camera body frame (X-forward, Y-left, Z-up).

Usage:
    python camera_ball_dual_hsv.py
    python camera_ball_dual_hsv.py --no-viz
    python camera_ball_dual_hsv.py --show-mask
"""

import argparse
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from transform.camera_to_base import optical_to_body


# Tennis ball HSV defaults.
TENNIS_H_LOW = 25
TENNIS_H_HIGH = 80
TENNIS_S_MIN = 80
TENNIS_V_MIN = 80
TENNIS_RADIUS_M = 0.033

# Blue end disk HSV defaults estimated from tools/recording/1.png and 2.png.
BLUE_H_LOW = 94
BLUE_H_HIGH = 104
BLUE_S_MIN = 80
BLUE_V_MIN = 35
BLUE_DIAMETER_M = 0.026

DEPTH_SAMPLE_RADIUS = 5
DEPTH_MIN = 0.15
DEPTH_MAX = 8.0
VIS_WEIGHT = 0.5
EMA_ALPHA = 0.6
EMA_GATE = 0.6


class _FPS:
    def __init__(self, window=30):
        self._t = []
        self._w = window

    def tick(self):
        now = time.perf_counter()
        self._t.append(now)
        if len(self._t) > self._w:
            self._t.pop(0)

    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


def detect_colored_circle(
    frame_bgr,
    hsv_low,
    hsv_high,
    min_radius_px,
    max_radius_px,
    min_circularity,
):
    """Return (cx, cy, r_px, mask), or (None, None, None, mask)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_radius_px ** 2:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if not (min_radius_px <= r <= max_radius_px):
            continue

        score = area * circularity
        if best is None or score > best[0]:
            best = (score, int(cx), int(cy), float(r), float(area), float(circularity))

    if best is None:
        return None, None, None, mask

    _, cx, cy, r, _, _ = best
    return cx, cy, r, mask


def color_pixel_to_depth_pixel(cx, cy, depth_arr, depth_scale, color_intrin, depth_intrin, color_to_depth_extr):
    """Map a color pixel to a depth pixel using the same lightweight mapping as camera_ball_color.py."""
    dh, dw = depth_arr.shape
    ndcx = (cx - color_intrin.ppx) / color_intrin.fx
    ndcy = (cy - color_intrin.ppy) / color_intrin.fy

    dx0 = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
    dy0 = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
    dx0 = max(0, min(dw - 1, dx0))
    dy0 = max(0, min(dh - 1, dy0))
    raw0 = depth_arr[dy0, dx0]
    depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0

    tx = color_to_depth_extr.translation[0]
    ty = color_to_depth_extr.translation[1]
    dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx + tx / depth_coarse * depth_intrin.fx + 0.5)
    dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy + ty / depth_coarse * depth_intrin.fy + 0.5)
    dx = max(0, min(dw - 1, dx))
    dy = max(0, min(dh - 1, dy))
    return dx, dy


def median_depth_at(dx, dy, depth_arr, depth_scale, radius_px):
    dh, dw = depth_arr.shape
    patch = depth_arr[
        max(0, dy - radius_px):min(dh, dy + radius_px + 1),
        max(0, dx - radius_px):min(dw, dx + radius_px + 1),
    ].astype(np.float32) * depth_scale
    valid = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def deproject_body(cx, cy, depth_m, color_intrin):
    if not (DEPTH_MIN < depth_m < DEPTH_MAX):
        return None
    p_opt = rs.rs2_deproject_pixel_to_point(color_intrin, [cx, cy], depth_m)
    return optical_to_body(p_opt)


def update_ema(prev, value, disabled):
    if value is None:
        return prev
    if disabled or prev is None:
        return value.copy()
    gate_dist = np.linalg.norm(value - prev)
    if gate_dist > EMA_GATE:
        return value.copy()
    return EMA_ALPHA * value + (1 - EMA_ALPHA) * prev


def format_point(p):
    if p is None:
        return "—"
    return f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"


def main():
    parser = argparse.ArgumentParser(description="Dual HSV tennis ball + blue end disk detector")
    parser.add_argument("--no-viz", action="store_true", help="disable OpenCV window")
    parser.add_argument("--no-ema", action="store_true", help="disable EMA smoothing")
    parser.add_argument("--show-mask", action="store_true", help="show tennis/blue HSV masks")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")

    parser.add_argument("--tennis-h-low", type=int, default=TENNIS_H_LOW)
    parser.add_argument("--tennis-h-high", type=int, default=TENNIS_H_HIGH)
    parser.add_argument("--tennis-s-min", type=int, default=TENNIS_S_MIN)
    parser.add_argument("--tennis-v-min", type=int, default=TENNIS_V_MIN)
    parser.add_argument("--tennis-radius", type=float, default=TENNIS_RADIUS_M, help="tennis ball radius in meters")

    parser.add_argument("--blue-h-low", type=int, default=BLUE_H_LOW)
    parser.add_argument("--blue-h-high", type=int, default=BLUE_H_HIGH)
    parser.add_argument("--blue-s-min", type=int, default=BLUE_S_MIN)
    parser.add_argument("--blue-v-min", type=int, default=BLUE_V_MIN)
    parser.add_argument("--blue-diameter", type=float, default=BLUE_DIAMETER_M, help="blue disk diameter in meters")

    parser.add_argument("--min-radius-px", type=float, default=3.0, help="minimum blob radius in pixels")
    parser.add_argument("--max-radius-px", type=float, default=220.0, help="maximum blob radius in pixels")
    parser.add_argument("--tennis-circularity", type=float, default=0.60)
    parser.add_argument("--blue-circularity", type=float, default=0.30)
    parser.add_argument("--depth-radius", type=int, default=DEPTH_SAMPLE_RADIUS, help="depth median patch radius in pixels")
    args = parser.parse_args()

    viz = not args.no_viz
    tennis_low = np.array([args.tennis_h_low, args.tennis_s_min, args.tennis_v_min], dtype=np.uint8)
    tennis_high = np.array([args.tennis_h_high, 255, 255], dtype=np.uint8)
    blue_low = np.array([args.blue_h_low, args.blue_s_min, args.blue_v_min], dtype=np.uint8)
    blue_high = np.array([args.blue_h_high, 255, 255], dtype=np.uint8)

    print(f"[INFO] Tennis HSV: H=[{args.tennis_h_low},{args.tennis_h_high}] "
          f"S>={args.tennis_s_min} V>={args.tennis_v_min} radius={args.tennis_radius:.3f}m")
    print(f"[INFO] Blue HSV:   H=[{args.blue_h_low},{args.blue_h_high}] "
          f"S>={args.blue_s_min} V>={args.blue_v_min} diameter={args.blue_diameter:.3f}m")

    pipeline = rs.pipeline()
    fps_tries = [(60, 60), (30, 30), (15, 15)]

    def start_pipeline():
        last_err = None
        for c_fps, d_fps in fps_tries:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps)
            cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, d_fps)
            try:
                print(f"[INFO] Starting RealSense ({args.width}x{args.height} color {c_fps}Hz depth {d_fps}Hz)...")
                profile = pipeline.start(cfg)
                pipeline.wait_for_frames(timeout_ms=5000)
                print(f"[INFO] RealSense OK ({c_fps}/{d_fps} Hz)")
                return profile
            except RuntimeError as e:
                last_err = e
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
                print(f"[WARN] Profile failed: {e}")
        raise RuntimeError(f"RealSense failed. Tried {fps_tries}. Last: {last_err!r}")

    profile = start_pipeline()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrin = color_profile.get_intrinsics()
    depth_intrin = depth_profile.get_intrinsics()
    color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"[INFO] Color fx={color_intrin.fx:.1f} fy={color_intrin.fy:.1f}")
    print(f"[INFO] Depth fx={depth_intrin.fx:.1f} scale={depth_scale:.4f}")

    tennis_ema = None
    blue_ema = None
    fps = _FPS()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_arr = np.asanyarray(depth_frame.get_data())

            tennis_cx, tennis_cy, tennis_r, tennis_mask = detect_colored_circle(
                color, tennis_low, tennis_high, args.min_radius_px, args.max_radius_px, args.tennis_circularity)
            blue_cx, blue_cy, blue_r, blue_mask = detect_colored_circle(
                color, blue_low, blue_high, args.min_radius_px, args.max_radius_px, args.blue_circularity)

            tennis_raw = None
            tennis_depth = 0.0
            tennis_depth_src = "—"
            if tennis_cx is not None:
                visual_depth = color_intrin.fx * args.tennis_radius / tennis_r if tennis_r > 0 else 0.0
                dx, dy = color_pixel_to_depth_pixel(
                    tennis_cx, tennis_cy, depth_arr, depth_scale,
                    color_intrin, depth_intrin, color_to_depth_extr)
                surface_depth = median_depth_at(dx, dy, depth_arr, depth_scale, args.depth_radius)
                sensor_depth = surface_depth + args.tennis_radius if surface_depth > 0 else 0.0
                vis_ok = DEPTH_MIN < visual_depth < DEPTH_MAX
                sensor_ok = DEPTH_MIN < sensor_depth < DEPTH_MAX
                if vis_ok and sensor_ok:
                    ratio = visual_depth / sensor_depth
                    if 0.5 < ratio < 2.0:
                        tennis_depth = VIS_WEIGHT * visual_depth + (1 - VIS_WEIGHT) * sensor_depth
                        tennis_depth_src = "fused"
                    else:
                        tennis_depth = visual_depth
                        tennis_depth_src = "visual"
                elif vis_ok:
                    tennis_depth = visual_depth
                    tennis_depth_src = "visual"
                elif sensor_ok:
                    tennis_depth = sensor_depth
                    tennis_depth_src = "sensor"
                tennis_raw = deproject_body(tennis_cx, tennis_cy, tennis_depth, color_intrin)

            blue_raw = None
            blue_depth = 0.0
            if blue_cx is not None:
                dx, dy = color_pixel_to_depth_pixel(
                    blue_cx, blue_cy, depth_arr, depth_scale,
                    color_intrin, depth_intrin, color_to_depth_extr)
                # The blue end is a disk. Use sensor depth directly for the visible face center.
                blue_depth = median_depth_at(dx, dy, depth_arr, depth_scale, args.depth_radius)
                blue_raw = deproject_body(blue_cx, blue_cy, blue_depth, color_intrin)

            tennis_ema = update_ema(tennis_ema, tennis_raw, args.no_ema)
            blue_ema = update_ema(blue_ema, blue_raw, args.no_ema)
            fps.tick()

            tennis_r_str = f"{tennis_r:.1f}px" if tennis_r is not None else "—"
            blue_r_str = f"{blue_r:.1f}px" if blue_r is not None else "—"
            print(f"\r[TENNIS] {format_point(tennis_ema)} r={tennis_r_str} "
                  f"d={tennis_depth:.2f}m/{tennis_depth_src}  "
                  f"[BLUE] {format_point(blue_ema)} r={blue_r_str} d={blue_depth:.2f}m  "
                  f"fps={fps.fps:5.1f}", end="", flush=True)

            if viz or args.show_mask:
                vis = color.copy()
                if tennis_cx is not None:
                    cv2.circle(vis, (tennis_cx, tennis_cy), int(tennis_r), (0, 255, 0), 2)
                    cv2.circle(vis, (tennis_cx, tennis_cy), 3, (0, 0, 255), -1)
                    cv2.putText(vis, f"tennis {format_point(tennis_ema)}",
                                (tennis_cx + 8, max(18, tennis_cy - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                if blue_cx is not None:
                    cv2.circle(vis, (blue_cx, blue_cy), int(blue_r), (255, 0, 0), 2)
                    cv2.circle(vis, (blue_cx, blue_cy), 3, (0, 0, 255), -1)
                    cv2.putText(vis, f"blue {format_point(blue_ema)}",
                                (blue_cx + 8, max(36, blue_cy + 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                cv2.putText(vis, f"dual HSV {fps.fps:.1f} fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                if args.show_mask:
                    tennis_bgr = cv2.cvtColor(tennis_mask, cv2.COLOR_GRAY2BGR)
                    blue_bgr = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
                    out = np.hstack([vis, tennis_bgr, blue_bgr])
                else:
                    out = vis

                cv2.imshow("dual_hsv: image | tennis mask | blue mask", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        pipeline.stop()
        if viz or args.show_mask:
            cv2.destroyAllWindows()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
