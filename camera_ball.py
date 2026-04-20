"""
camera_ball.py

RealSense D455 + YOLOv8 tennis ball detection.
Outputs ball position in camera body frame (X-forward, Y-left, Z-up).

Usage:
    python camera_ball.py              # with OpenCV visualization
    python camera_ball.py --no-viz     # headless
    python camera_ball.py --no-ema     # disable EMA smoothing
    python camera_ball.py --model yolo11m.pt --imgsz 480

LCM publishing (if lcm + lcm_types available):
    channel: camera_ball_lcmt, coords in camera body frame
"""

import argparse
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from transform.camera_to_base import optical_to_body

try:
    import lcm
    from lcm_types.lidar_lcmt import lidar_lcmt
    HAS_LCM = True
except ImportError:
    HAS_LCM = False

# ── Detection params ──────────────────────────────────────────────────────────
SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD       = 0.3
DEPTH_SAMPLE_RADIUS  = 5
DEPTH_MIN            = 0.1    # m
DEPTH_MAX            = 10.0   # m
BALL_RADIUS          = 0.033  # m — tennis ball radius; depth sensor reads front surface
EMA_ALPHA            = 0.6
EMA_GATE             = 0.6    # m — jump larger than this resets EMA
COAST_FRAMES         = 10


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


def main():
    parser = argparse.ArgumentParser(description="RealSense D455 tennis ball → camera body frame")
    parser.add_argument("--no-viz",  action="store_true", help="disable OpenCV window")
    parser.add_argument("--no-ema",  action="store_true", help="disable EMA smoothing")
    parser.add_argument("--model",   default="models/yolov8n.pt", help="YOLO model path")
    parser.add_argument("--imgsz",   type=int, default=480, help="YOLO input size")
    parser.add_argument("--width",   type=int, default=640, help="camera capture width")
    parser.add_argument("--height",  type=int, default=480, help="camera capture height")
    args = parser.parse_args()

    viz = not args.no_viz

    # ── LCM ──────────────────────────────────────────────────────────────────
    lc = None
    if HAS_LCM:
        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        print("[INFO] LCM ready — publishing on 'camera_ball_lcmt'")

    # ── YOLO ─────────────────────────────────────────────────────────────────
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    half   = device != "cpu"
    print(f"[INFO] Loading YOLO: {args.model}  device={device}  half={half}")
    model = YOLO(args.model)

    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    print("[INFO] Warming up YOLO...")
    for _ in range(3):
        model(dummy, verbose=False, device=device, half=half)
    t0 = time.perf_counter()
    for _ in range(5):
        model(dummy, verbose=False, device=device, half=half)
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"[INFO] YOLO warmup: {ms:.1f} ms/frame  (≈ {1000/ms:.0f} FPS upper bound)")

    # ── RealSense pipeline ────────────────────────────────────────────────────
    pipeline  = rs.pipeline()
    _FPS_TRIES = [(60, 60), (30, 30), (15, 15)]

    def _start_pipeline():
        last_err = None
        for c_fps, d_fps in _FPS_TRIES:
            rs_cfg = rs.config()
            rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, c_fps)
            rs_cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16,  d_fps)
            for attempt in range(2):
                try:
                    print(f"[INFO] Starting RealSense (color {c_fps}Hz depth {d_fps}Hz attempt {attempt+1})...")
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

    # ── Intrinsics, extrinsics, depth scale ───────────────────────────────────
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrin  = color_profile.get_intrinsics()
    depth_intrin  = depth_profile.get_intrinsics()
    # Rotation + translation mapping a point from color frame → depth frame.
    # tx ≈ −14.5 mm for D455 (color sensor is to the right of depth).
    color_to_depth_extr = color_profile.get_extrinsics_to(depth_profile)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"[INFO] Color  fx={color_intrin.fx:.1f} fy={color_intrin.fy:.1f} "
          f"ppx={color_intrin.ppx:.1f} ppy={color_intrin.ppy:.1f}")
    print(f"[INFO] Depth  fx={depth_intrin.fx:.1f} fy={depth_intrin.fy:.1f} "
          f"ppx={depth_intrin.ppx:.1f} ppy={depth_intrin.ppy:.1f}  scale={depth_scale:.4f}")
    t_cd = color_to_depth_extr.translation
    print(f"[INFO] Color→Depth  tx={t_cd[0]*1000:.1f}mm "
          f"ty={t_cd[1]*1000:.1f}mm tz={t_cd[2]*1000:.1f}mm")

    # ── Shared state: camera thread → YOLO thread ─────────────────────────────
    buf_lock    = threading.Lock()
    buf_frames  = None   # raw rs2 composite_frame (just a reference swap)
    buf_updated = threading.Event()
    stop_flag   = threading.Event()

    # ── Shared state: YOLO thread → main thread (display) ─────────────────────
    disp_lock  = threading.Lock()
    disp_frame = [None]  # latest annotated BGR frame for cv2.imshow

    # ── YOLO worker thread ────────────────────────────────────────────────────
    def yolo_worker():
        import os
        try:
            param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO) - 1)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
            print("[INFO] YOLO thread: SCHED_FIFO set.")
        except (PermissionError, OSError):
            try:
                os.nice(-10)
            except PermissionError:
                pass

        center_ema = None
        last_bbox  = None
        miss_count = 0
        yolo_fps   = _FPS()

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

            # Copy color to Python heap (needed for YOLO numpy input).
            # Copy depth to Python heap for fast numpy patch slicing.
            color     = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data()).copy()  # uint16, mm units
            dh, dw    = depth_arr.shape

            # Resize color for YOLO, keep scale factors to map back to full res.
            orig_h, orig_w = color.shape[:2]
            color_small    = cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz

            results  = model.track(color_small, conf=CONF_THRESHOLD,
                                   persist=True, verbose=False,
                                   device=device, half=half)
            best_box  = None
            best_conf = 0.0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_box = c, box

            if best_box is not None:
                miss_count = 0
                x1s, y1s, x2s, y2s = best_box.xyxy[0]
                last_bbox = (int(x1s * sx), int(y1s * sy),
                             int(x2s * sx), int(y2s * sy))
            else:
                miss_count += 1

            p_cam_str     = "—"
            depth_surface = 0.0
            depth_m       = 0.0
            detected      = False
            coasting      = False

            if last_bbox is not None and miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = last_bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ── Color → Depth pixel mapping (3-step) ─────────────────────
                # D455 color and depth sensors have different FOV and a ~14.5mm
                # baseline. Using color pixel coords directly in depth_arr causes
                # up to ~14 cm lateral error at image edges.
                #
                # Step 1: normalise with color intrinsics → direction vector
                ndcx = (cx - color_intrin.ppx) / color_intrin.fx
                ndcy = (cy - color_intrin.ppy) / color_intrin.fy

                # Step 2a: FOV-only mapping to depth pixel (no parallax yet)
                dx0 = int(ndcx * depth_intrin.fx + depth_intrin.ppx + 0.5)
                dy0 = int(ndcy * depth_intrin.fy + depth_intrin.ppy + 0.5)
                dx0 = max(0, min(dw - 1, dx0))
                dy0 = max(0, min(dh - 1, dy0))
                raw0 = depth_arr[dy0, dx0]
                depth_coarse = raw0 * depth_scale if raw0 > 0 else 1.0  # fallback 1 m

                # Step 2b: add parallax correction using color→depth baseline
                tx = color_to_depth_extr.translation[0]
                ty = color_to_depth_extr.translation[1]
                dx = int(ndcx * depth_intrin.fx + depth_intrin.ppx
                         + tx / depth_coarse * depth_intrin.fx + 0.5)
                dy = int(ndcy * depth_intrin.fy + depth_intrin.ppy
                         + ty / depth_coarse * depth_intrin.fy + 0.5)
                dx = max(0, min(dw - 1, dx))
                dy = max(0, min(dh - 1, dy))

                # Step 3: sample median of patch at corrected depth pixel
                r   = DEPTH_SAMPLE_RADIUS
                patch   = (depth_arr[max(0, dy - r):min(dh, dy + r + 1),
                                     max(0, dx - r):min(dw, dx + r + 1)]
                           .astype(np.float32) * depth_scale)
                valid_d = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
                depth_surface = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0

                # Depth sensor reads the front surface of the ball.
                # Shift by BALL_RADIUS along the optical axis to reach ball centre.
                depth_m = depth_surface + BALL_RADIUS if depth_surface > 0 else 0.0

                if depth_m > 0:
                    # Deproject using color intrinsics + color pixel → optical frame
                    p_opt     = rs.rs2_deproject_pixel_to_point(color_intrin, [cx, cy], depth_m)
                    # optical (Z-fwd, X-right, Y-down) → body (X-fwd, Y-left, Z-up)
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
                    detected  = best_box is not None
                    coasting  = not detected

                    if lc is not None:
                        lcm_msg = lidar_lcmt()
                        lcm_msg.offset_time = int(time.time() * 1e6)
                        lcm_msg.x, lcm_msg.y, lcm_msg.z = x, y, z
                        lc.publish("camera_ball_lcmt", lcm_msg.encode())

            yolo_fps.tick()

            status = "BALL " if detected else ("COAST" if coasting else "     ")
            print(f"\r[{status}] cam={p_cam_str}  "
                  f"surf={depth_surface:.2f}m ctr={depth_m:.2f}m  "
                  f"YOLO={yolo_fps.fps:4.1f}fps",
                  end="", flush=True)

            # ── Build annotated display frame ─────────────────────────────────
            if viz:
                vis = color.copy()
                if last_bbox is not None and miss_count <= COAST_FRAMES:
                    x1, y1, x2, y2 = last_bbox
                    cx2, cy2       = (x1 + x2) // 2, (y1 + y2) // 2
                    box_color      = (0, 255, 0) if detected else (0, 165, 255)
                    tag            = "ball" if detected else f"coast {miss_count}/{COAST_FRAMES}"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                    cv2.circle(vis, (cx2, cy2), 4, box_color, -1)
                    cv2.putText(vis, f"{tag} {best_conf:.2f}",
                                (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                    if p_cam_str != "—":
                        cv2.putText(vis, f"cam {p_cam_str}",
                                    (x1, max(y1 - 28, 28)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
                else:
                    cv2.putText(vis, "No ball", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(vis, f"YOLO {yolo_fps.fps:.1f} fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                with disp_lock:
                    disp_frame[0] = vis

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
    yolo_thread.start()

    # ── Main thread: camera capture + display ─────────────────────────────────
    print(f"[INFO] Running. Press {'q' if viz else 'Ctrl+C'} to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            with buf_lock:
                buf_frames = frames   # reference swap, no copy
            buf_updated.set()

            if viz:
                with disp_lock:
                    frame = disp_frame[0]
                if frame is not None:
                    cv2.imshow("camera_ball", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        yolo_thread.join(timeout=2)
        pipeline.stop()
        if viz:
            cv2.destroyAllWindows()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
