"""
camera_ball_color.py

RealSense D455 + HSV color segmentation for tennis ball detection.
Outputs ball position in camera body frame (X-forward, Y-left, Z-up).

Depth is estimated two ways and fused:
  1. Visual (apparent size): depth = fx * BALL_RADIUS / r_pixels
     - Gives ball CENTRE depth directly (no +BALL_RADIUS offset needed)
     - Fast, motion-blur resistant, no depth sensor required
  2. RealSense depth sensor: patch median at colour→depth mapped pixel
     - More accurate at close range when surface is visible
     - Fails on shiny/fast objects (holes in depth map)

Fusion rule:
  - Both valid  → weighted average (visual weight = VIS_WEIGHT)
  - Sensor only → sensor + BALL_RADIUS (front surface → centre)
  - Visual only → visual
  - Neither     → coast / no measurement

Usage:
    python camera_ball_color.py                  # with OpenCV window
    python camera_ball_color.py --no-viz         # headless
    python camera_ball_color.py --show-mask      # show HSV binary mask (for tuning)
    python camera_ball_color.py --h-low 30 --h-high 75  # tune HSV hue range

LCM publishing (if lcm + lcm_types available):
    channel: camera_ball_lcmt, coords in camera body frame
"""

import argparse
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from transform.camera_to_base import optical_to_body

try:
    import lcm
    from lcm_types.lidar_lcmt import lidar_lcmt
    HAS_LCM = True
except ImportError:
    HAS_LCM = False

# ── Tennis ball HSV range ─────────────────────────────────────────────────────
# Fluorescent yellow-green: H≈35–75, high S and V.
# Tune with --show-mask if lighting differs.
HSV_H_LOW  = 25
HSV_H_HIGH = 80
HSV_S_MIN  = 80
HSV_V_MIN  = 80

# ── Detection params ──────────────────────────────────────────────────────────
BALL_RADIUS      = 0.033   # m — tennis ball physical radius
MIN_RADIUS_PX    = 5       # px — ignore blobs smaller than this
MAX_RADIUS_PX    = 200     # px — ignore blobs larger than this
MIN_CIRCULARITY  = 0.65    # 0–1, 1=perfect circle; filters non-circular blobs
DEPTH_MIN        = 0.15    # m
DEPTH_MAX        = 8.0     # m
DEPTH_SAMPLE_RADIUS = 5    # px — depth patch radius around detected centre
VIS_WEIGHT       = 0.5     # fusion weight for visual depth (0=sensor only, 1=visual only)
EMA_ALPHA        = 0.6
EMA_GATE         = 0.6     # m
COAST_FRAMES     = 10


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


def detect_tennis_ball(frame_bgr, hsv_low, hsv_high):
    """
    Detect tennis ball in BGR frame using HSV colour segmentation.

    Returns (cx, cy, r_px, mask) where (cx, cy) is the ball centre in pixels,
    r_px is the pixel radius of the best candidate circle, and mask is the
    binary HSV mask. Returns (None, None, None, mask) if no ball found.
    """
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # Morphological cleanup: close small holes, remove thin noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None   # (score, cx, cy, r)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * MIN_RADIUS_PX ** 2:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if not (MIN_RADIUS_PX <= r <= MAX_RADIUS_PX):
            continue

        # Score: prefer large, circular blobs
        score = area * circularity
        if best is None or score > best[0]:
            best = (score, int(cx), int(cy), r)

    if best is None:
        return None, None, None, mask

    _, cx, cy, r = best
    return cx, cy, r, mask


def main():
    parser = argparse.ArgumentParser(
        description="RealSense D455 HSV tennis ball → camera body frame"
    )
    parser.add_argument("--no-viz",    action="store_true", help="disable OpenCV window")
    parser.add_argument("--no-ema",    action="store_true", help="disable EMA smoothing")
    parser.add_argument("--show-mask", action="store_true", help="show HSV binary mask window")
    parser.add_argument("--width",     type=int, default=640)
    parser.add_argument("--height",    type=int, default=480)
    parser.add_argument("--h-low",     type=int, default=HSV_H_LOW,  help="HSV hue lower bound")
    parser.add_argument("--h-high",    type=int, default=HSV_H_HIGH, help="HSV hue upper bound")
    parser.add_argument("--s-min",     type=int, default=HSV_S_MIN,  help="HSV saturation min")
    parser.add_argument("--v-min",     type=int, default=HSV_V_MIN,  help="HSV value min")
    args = parser.parse_args()

    viz      = not args.no_viz
    hsv_low  = np.array([args.h_low,  args.s_min, args.v_min], dtype=np.uint8)
    hsv_high = np.array([args.h_high, 255,        255       ], dtype=np.uint8)

    print(f"[INFO] HSV range: H=[{args.h_low},{args.h_high}]  "
          f"S>={args.s_min}  V>={args.v_min}")
    print(f"[INFO] Visual depth: depth = fx × {BALL_RADIUS}m / r_px")
    print(f"[INFO] Fusion: visual_weight={VIS_WEIGHT}  "
          f"(0=sensor only, 1=visual only)")

    # ── LCM ──────────────────────────────────────────────────────────────────
    lc = None
    if HAS_LCM:
        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        print("[INFO] LCM ready — publishing on 'camera_ball_lcmt'")

    # ── RealSense pipeline ────────────────────────────────────────────────────
    pipeline   = rs.pipeline()
    _FPS_TRIES = [(60, 60), (30, 30), (15, 15)]

    def _start_pipeline():
        last_err = None
        for c_fps, d_fps in _FPS_TRIES:
            rs_cfg = rs.config()
            rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps)
            rs_cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,  d_fps)
            for attempt in range(2):
                try:
                    print(f"[INFO] Starting RealSense ({c_fps}/{d_fps} Hz attempt {attempt+1})...")
                    profile = pipeline.start(rs_cfg)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "resolve" in msg or "couldn't" in msg:
                        last_err = e
                        print(f"[WARN] Profile not supported: {e}")
                        break
                    raise
                try:
                    pipeline.wait_for_frames(timeout_ms=5000)
                    print(f"[INFO] RealSense OK ({c_fps}/{d_fps} Hz)")
                    return profile
                except RuntimeError:
                    print("[WARN] Frame timeout — hardware reset...")
                    pipeline.stop()
                    devs = rs.context().query_devices()
                    if len(devs) == 0:
                        raise RuntimeError("No RealSense device found.")
                    devs[0].hardware_reset()
                    time.sleep(5)
        raise RuntimeError(f"RealSense failed. Tried {_FPS_TRIES}. Last: {last_err!r}")

    profile = _start_pipeline()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrin  = color_profile.get_intrinsics()
    depth_intrin  = depth_profile.get_intrinsics()
    color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
    depth_scale   = profile.get_device().first_depth_sensor().get_depth_scale()

    fx = color_intrin.fx   # used for visual depth: depth = fx * R / r_px

    print(f"[INFO] Color  fx={color_intrin.fx:.1f} fy={color_intrin.fy:.1f} "
          f"ppx={color_intrin.ppx:.1f} ppy={color_intrin.ppy:.1f}")
    print(f"[INFO] Depth  fx={depth_intrin.fx:.1f}  scale={depth_scale:.4f}")
    t_cd = color_to_depth_extr.translation
    print(f"[INFO] Color→Depth  tx={t_cd[0]*1000:.1f}mm "
          f"ty={t_cd[1]*1000:.1f}mm tz={t_cd[2]*1000:.1f}mm")

    # ── Shared state: camera → detection thread ───────────────────────────────
    buf_lock    = threading.Lock()
    buf_frames  = None
    buf_updated = threading.Event()
    stop_flag   = threading.Event()

    # ── Shared state: detection thread → main (display) ──────────────────────
    disp_lock  = threading.Lock()
    disp_frame = [None]

    # ── Detection thread ──────────────────────────────────────────────────────
    def detection_worker():
        import os
        try:
            param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO) - 1)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
        except (PermissionError, OSError):
            try:
                os.nice(-10)
            except PermissionError:
                pass

        center_ema = None
        last_cx = last_cy = last_r = None
        miss_count = 0
        det_fps    = _FPS()

        while not stop_flag.is_set():
            if not buf_updated.wait(timeout=1.0):
                continue
            buf_updated.clear()

            with buf_lock:
                frames = buf_frames

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            color     = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data()).copy()
            dh, dw    = depth_arr.shape

            # ── HSV detection ─────────────────────────────────────────────────
            cx, cy, r_px, mask = detect_tennis_ball(color, hsv_low, hsv_high)

            if cx is not None:
                miss_count = 0
                last_cx, last_cy, last_r = cx, cy, r_px
            else:
                miss_count += 1

            p_cam_str    = "—"
            depth_vis    = 0.0
            depth_sensor = 0.0
            depth_fused  = 0.0
            detected     = False
            coasting     = False

            if last_cx is not None and miss_count <= COAST_FRAMES:
                detected = (cx is not None)
                coasting = not detected

                # ── Visual depth: depth = fx * R / r_px ──────────────────────
                # This gives distance to ball CENTRE directly (no +R offset).
                if last_r > 0:
                    depth_vis = fx * BALL_RADIUS / last_r

                # ── Sensor depth via 3-step Color→Depth mapping ───────────────
                ndcx = (last_cx - color_intrin.ppx) / color_intrin.fx
                ndcy = (last_cy - color_intrin.ppy) / color_intrin.fy

                dx0 = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
                dy0 = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
                dx0 = max(0, min(dw - 1, dx0))
                dy0 = max(0, min(dh - 1, dy0))
                raw0 = depth_arr[dy0, dx0]
                depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0

                tx = color_to_depth_extr.translation[0]
                ty = color_to_depth_extr.translation[1]
                dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx
                         + tx / depth_coarse * depth_intrin.fx + 0.5)
                dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy
                         + ty / depth_coarse * depth_intrin.fy + 0.5)
                dx = max(0, min(dw - 1, dx))
                dy = max(0, min(dh - 1, dy))

                r   = DEPTH_SAMPLE_RADIUS
                patch   = (depth_arr[max(0, dy - r):min(dh, dy + r + 1),
                                     max(0, dx - r):min(dw, dx + r + 1)]
                           .astype(np.float32) * depth_scale)
                valid_d = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
                depth_surface = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0
                # Sensor reads front surface; shift to ball centre
                depth_sensor = depth_surface + BALL_RADIUS if depth_surface > 0 else 0.0

                # ── Depth fusion ──────────────────────────────────────────────
                vis_ok    = DEPTH_MIN < depth_vis    < DEPTH_MAX
                sensor_ok = DEPTH_MIN < depth_sensor < DEPTH_MAX

                if vis_ok and sensor_ok:
                    # Sanity check: reject if the two estimates disagree >50%
                    ratio = depth_vis / depth_sensor
                    if 0.5 < ratio < 2.0:
                        depth_fused = VIS_WEIGHT * depth_vis + (1 - VIS_WEIGHT) * depth_sensor
                    else:
                        # Large disagreement: trust visual (sensor likely hit background)
                        depth_fused = depth_vis
                        print(f"\n[WARN] Depth disagreement: vis={depth_vis:.2f}m "
                              f"sensor={depth_sensor:.2f}m → using visual", flush=True)
                elif vis_ok:
                    depth_fused = depth_vis
                elif sensor_ok:
                    depth_fused = depth_sensor
                else:
                    depth_fused = 0.0

                if depth_fused > 0:
                    p_opt     = rs.rs2_deproject_pixel_to_point(
                        color_intrin, [last_cx, last_cy], depth_fused)
                    p_cam_arr = optical_to_body(p_opt)

                    if args.no_ema or center_ema is None:
                        center_ema = p_cam_arr.copy()
                    else:
                        gate_dist = np.linalg.norm(p_cam_arr - center_ema)
                        if gate_dist < EMA_GATE:
                            center_ema = EMA_ALPHA * p_cam_arr + (1 - EMA_ALPHA) * center_ema
                        else:
                            print(f"\n[WARN] EMA gate {gate_dist:.2f}m > {EMA_GATE}m, resetting",
                                  flush=True)
                            center_ema = p_cam_arr.copy()

                    x, y, z   = float(center_ema[0]), float(center_ema[1]), float(center_ema[2])
                    p_cam_str = f"({x:+.3f}, {y:+.3f}, {z:+.3f})"

                    if lc is not None:
                        lcm_msg = lidar_lcmt()
                        lcm_msg.offset_time = int(time.time() * 1e6)
                        lcm_msg.x, lcm_msg.y, lcm_msg.z = x, y, z
                        lc.publish("camera_ball_lcmt", lcm_msg.encode())

            det_fps.tick()

            # ── Terminal output ───────────────────────────────────────────────
            src = "—"
            if depth_vis > 0 and depth_sensor > 0:
                src = f"fused(vis={depth_vis:.2f} sen={depth_sensor:.2f})"
            elif depth_vis > 0:
                src = f"visual({depth_vis:.2f}m)"
            elif depth_sensor > 0:
                src = f"sensor({depth_sensor:.2f}m)"

            status = "BALL " if detected else ("COAST" if coasting else "     ")
            r_str  = f"{last_r:.1f}px" if last_r else "—"
            print(f"\r[{status}] cam={p_cam_str}  r={r_str}  "
                  f"depth={src}  fused={depth_fused:.2f}m  "
                  f"det={det_fps.fps:5.1f}fps",
                  end="", flush=True)

            # ── Annotated display frame ───────────────────────────────────────
            if viz or args.show_mask:
                vis = color.copy()

                if last_cx is not None and miss_count <= COAST_FRAMES:
                    circle_color = (0, 255, 0) if detected else (0, 165, 255)
                    # Draw detected circle
                    cv2.circle(vis, (last_cx, last_cy), int(last_r), circle_color, 2)
                    cv2.circle(vis, (last_cx, last_cy), 3, (0, 0, 255), -1)
                    tag = "ball" if detected else f"coast {miss_count}/{COAST_FRAMES}"
                    cv2.putText(vis, f"{tag}  r={last_r:.0f}px",
                                (last_cx - int(last_r), max(last_cy - int(last_r) - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 1)
                    if p_cam_str != "—":
                        cv2.putText(vis, f"cam {p_cam_str}",
                                    (last_cx - int(last_r), max(last_cy - int(last_r) - 24, 28)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, circle_color, 1)
                    # Show depth source
                    cv2.putText(vis, f"d={depth_fused:.2f}m [{src[:10]}]",
                                (10, vis.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(vis, "No ball", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.putText(vis, f"HSV det {det_fps.fps:.1f} fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                with disp_lock:
                    if args.show_mask:
                        # Side-by-side: colour frame + HSV mask
                        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        disp_frame[0] = np.hstack([vis, mask_bgr])
                    else:
                        disp_frame[0] = vis

    det_thread = threading.Thread(target=detection_worker, daemon=True)
    det_thread.start()

    # ── Main thread: camera capture + display ─────────────────────────────────
    win_title = "camera_ball_color (q=quit)"
    print(f"[INFO] Running. Press {'q' if viz else 'Ctrl+C'} to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            with buf_lock:
                buf_frames = frames
            buf_updated.set()

            if viz or args.show_mask:
                with disp_lock:
                    frame = disp_frame[0]
                if frame is not None:
                    cv2.imshow(win_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        det_thread.join(timeout=2)
        pipeline.stop()
        if viz or args.show_mask:
            cv2.destroyAllWindows()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
