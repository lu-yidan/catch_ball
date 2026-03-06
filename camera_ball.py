"""
camera_ball.py

RealSense D435 + YOLOv8 足球检测，坐标转换到 pelvis (base) 系，支持可视化开关。

用法:
    python camera_ball.py           # 带可视化
    python camera_ball.py --no-viz  # 纯终端，无 OpenCV 窗口

关节角来源（按优先级）:
    1. ROS2 /lowstate 话题（如果 rclpy 可用，后台线程订阅）
    2. 命令行参数 --q-wy / --q-wr / --q-wp / --q-head
    3. 默认值 0 / 0 / 0 / 0.593412

LCM 发布（如果 lcm + lcm_types 可用）:
    频道: camera_ball_lcmt，消息格式同 lidar_lcmt
"""

import argparse
import sys
import threading
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from transform.camera_to_base import transform_point_camera_to_base, optical_to_body

# ── 可选依赖 ───────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from unitree_hg.msg import LowState
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Node = object      # 占位，使 _JointListener 类定义不报错
    LowState = None    # 占位，使类型注解不报错

try:
    import lcm
    from lcm_types.lidar_lcmt import lidar_lcmt
    HAS_LCM = True
except ImportError:
    HAS_LCM = False

# ── 检测参数 ───────────────────────────────────────────────────────────────────
SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD       = 0.3   # 略低阈值，提升召回率
DEPTH_SAMPLE_RADIUS  = 5
DEPTH_MIN            = 0.1   # 米
DEPTH_MAX            = 10.0  # 米
EMA_ALPHA            = 0.6   # 指数滑动平均系数，越大越跟当前值
EMA_GATE             = 0.6   # 跳变门限（米），超过则不更新 EMA
COAST_FRAMES         = 10    # YOLO 漏检时保留最后位置的帧数


# ── 深度采样 ───────────────────────────────────────────────────────────────────

def get_depth_at_center(depth_frame, cx, cy, radius=DEPTH_SAMPLE_RADIUS):
    """在 (cx,cy) 周围 radius px 取深度中位数（米），无有效值返回 0.0。"""
    w, h = depth_frame.get_width(), depth_frame.get_height()
    x0, x1 = max(0, cx - radius), min(w - 1, cx + radius)
    y0, y1 = max(0, cy - radius), min(h - 1, cy + radius)

    samples = [
        depth_frame.get_distance(x, y)
        for y in range(y0, y1 + 1)
        for x in range(x0, x1 + 1)
        if DEPTH_MIN < depth_frame.get_distance(x, y) < DEPTH_MAX
    ]
    return float(np.median(samples)) if samples else 0.0


# ── ROS2 关节角订阅（后台线程） ────────────────────────────────────────────────

class _JointListener(Node):
    """后台 ROS2 节点，仅订阅 /lowstate 更新关节角。"""

    def __init__(self):
        super().__init__("camera_ball_joint_listener")
        self.q_wy   = 0.0
        self.q_wr   = 0.0
        self.q_wp   = 0.0
        self.q_head = 0.593412
        self.create_subscription(
            LowState, "/lowstate", self._cb, qos_profile_sensor_data
        )
        self.get_logger().info("camera_ball: subscribed to /lowstate")

    def _cb(self, msg: "LowState"):
        q = [m.q for m in msg.motor_state]
        self.q_wy   = q[12]
        self.q_wr   = q[13]
        self.q_wp   = q[14]


# ── FPS 计数器 ─────────────────────────────────────────────────────────────────

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


# ── 主程序 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RealSense 足球检测 → base 坐标系")
    parser.add_argument("--no-viz",  action="store_true",  help="关闭 OpenCV 可视化窗口")
    parser.add_argument("--model",   default="yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--imgsz",   type=int, default=480, help="YOLO 推理输入尺寸（越小越快）")
    parser.add_argument("--q-wy",    type=float, default=0.0,      metavar="RAD")
    parser.add_argument("--q-wr",    type=float, default=0.0,      metavar="RAD")
    parser.add_argument("--q-wp",    type=float, default=0.0,      metavar="RAD")
    parser.add_argument("--q-head",  type=float, default=0.593412, metavar="RAD")
    args = parser.parse_args()

    viz = not args.no_viz

    # ── 关节角来源 ────────────────────────────────────────────────────────────
    joint = argparse.Namespace(
        q_wy=args.q_wy, q_wr=args.q_wr, q_wp=args.q_wp, q_head=args.q_head
    )

    if HAS_ROS2:
        rclpy.init()
        ros_node = _JointListener()
        threading.Thread(
            target=rclpy.spin, args=(ros_node,), daemon=True
        ).start()
        joint = ros_node
        print("[INFO] ROS2 available — joint angles from /lowstate")
    else:
        print("[INFO] ROS2 not available — using fixed joint angles "
              f"(q_wy={joint.q_wy}, q_wr={joint.q_wr}, "
              f"q_wp={joint.q_wp}, q_head={joint.q_head})")

    # ── LCM ──────────────────────────────────────────────────────────────────
    lc = None
    if HAS_LCM:
        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        print("[INFO] LCM ready — publishing on 'camera_ball_lcmt'")

    # ── YOLO ─────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    # 打印模型参数量，并用随机图热身测速（3次取均值）
    info = model.info(verbose=False)
    dummy = np.zeros((480, 848, 3), dtype=np.uint8)
    model(dummy, verbose=False)  # 第一次含编译开销，不计入
    t0 = time.perf_counter()
    for _ in range(5):
        model(dummy, verbose=False)
    ms_per_frame = (time.perf_counter() - t0) / 5 * 1000
    print(f"[INFO] Model inference: {ms_per_frame:.1f} ms/frame  "
          f"(≈ {1000/ms_per_frame:.0f} FPS upper bound)")

    # ── RealSense ─────────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  60)

    def _start_pipeline():
        """启动 pipeline，若帧超时则自动硬件重置相机后重试（无需拔线）。"""
        for attempt in range(2):
            print(f"[INFO] Starting RealSense pipeline (attempt {attempt + 1})...")
            profile = pipeline.start(cfg)
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
                return profile
            except RuntimeError:
                print("[WARN] Frame timeout — performing hardware reset...")
                pipeline.stop()
                ctx = rs.context()
                devs = ctx.query_devices()
                if len(devs) == 0:
                    raise RuntimeError("No RealSense device found.")
                devs[0].hardware_reset()
                time.sleep(3)
        raise RuntimeError("RealSense failed to start after hardware reset.")

    profile = _start_pipeline()

    align      = rs.align(rs.stream.color)
    intrinsics = (
        profile.get_stream(rs.stream.color)
               .as_video_stream_profile()
               .get_intrinsics()
    )
    print(f"[INFO] Intrinsics — fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
          f"ppx={intrinsics.ppx:.1f}  ppy={intrinsics.ppy:.1f}")

    if viz:
        print("[INFO] Visualization ON  (press 'q' to quit)")
    else:
        print("[INFO] Visualization OFF  (press Ctrl+C to quit)")

    # ── 线程间共享状态 ────────────────────────────────────────────────────────
    # 相机线程写，YOLO 线程读
    buf_lock      = threading.Lock()
    buf_color     = None   # np.ndarray
    buf_depth     = None   # np.ndarray (uint16, mm)
    buf_updated   = threading.Event()

    # YOLO 线程写，主线程读
    res_lock      = threading.Lock()
    res_bbox      = None   # (x1,y1,x2,y2) 或 None
    res_conf      = 0.0
    res_cam_str   = "—"
    res_base_str  = "—"
    res_detected  = False
    res_coasting  = False
    res_yolo_fps  = 0.0
    # 速度外推：最近两次检测的 (时间戳, cx, cy)
    res_vel_t     = 0.0    # 最后一次 YOLO 结果时间戳
    res_vel_vx    = 0.0    # 像素/秒
    res_vel_vy    = 0.0

    stop_flag = threading.Event()

    # ── YOLO 线程 ─────────────────────────────────────────────────────────────
    def yolo_worker():
        nonlocal res_bbox, res_conf, res_cam_str, res_base_str
        nonlocal res_detected, res_coasting, res_yolo_fps
        nonlocal res_vel_t, res_vel_vx, res_vel_vy

        # ── 线程优先级（Linux）────────────────────────────────────────────────
        import os
        try:
            # 尝试实时调度（需要 sudo 或 CAP_SYS_NICE）
            param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO) - 1)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
            print("[INFO] YOLO thread: SCHED_FIFO real-time priority set.")
        except (PermissionError, OSError):
            try:
                os.nice(-10)  # 普通用户能降 nice 值
                print("[INFO] YOLO thread: nice=-10 set.")
            except PermissionError:
                print("[INFO] YOLO thread: running at default priority.")

        try:
            # 将 YOLO 线程固定到最后两个核心，避免和相机线程竞争
            n_cpu = os.cpu_count() or 2
            os.sched_setaffinity(0, set(range(max(0, n_cpu - 2), n_cpu)))
            print(f"[INFO] YOLO thread: pinned to CPU {max(0, n_cpu-2)}–{n_cpu-1}.")
        except (AttributeError, OSError):
            pass

        center_ema  = None
        last_bbox   = None
        miss_count  = 0
        yolo_fps    = _FPS()
        prev_cx     = None
        prev_cy     = None
        prev_t      = None

        while not stop_flag.is_set():
            if not buf_updated.wait(timeout=1.0):
                continue
            buf_updated.clear()

            with buf_lock:
                color = buf_color.copy()
                depth = buf_depth.copy()   # uint16 numpy array

            t_infer = time.perf_counter()

            # YOLO 追踪
            results   = model.track(color, conf=CONF_THRESHOLD,
                                    persist=True, verbose=False,
                                    imgsz=args.imgsz)
            # results   = model(color)
            best_box  = None
            best_conf = 0.0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_box = c, box

            yolo_fps.tick()

            if best_box is not None:
                miss_count = 0
                last_bbox  = tuple(map(int, best_box.xyxy[0]))
                # 速度估计（像素/秒）
                x1b, y1b, x2b, y2b = last_bbox
                cx_now, cy_now = (x1b + x2b) // 2, (y1b + y2b) // 2
                now = time.perf_counter()
                if prev_cx is not None and (now - prev_t) > 0:
                    dt = now - prev_t
                    vx = (cx_now - prev_cx) / dt
                    vy = (cy_now - prev_cy) / dt
                else:
                    vx, vy = 0.0, 0.0
                prev_cx, prev_cy, prev_t = cx_now, cy_now, now
            else:
                vx, vy = 0.0, 0.0

            p_cam_str  = "—"
            p_base_str = "—"
            detected   = False
            coasting   = False

            if last_bbox is not None and miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = last_bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                h, w   = depth.shape

                # 从 uint16 depth array 中采样（单位 mm → m）
                x0d = max(0, cx - DEPTH_SAMPLE_RADIUS)
                x1d = min(w - 1, cx + DEPTH_SAMPLE_RADIUS)
                y0d = max(0, cy - DEPTH_SAMPLE_RADIUS)
                y1d = min(h - 1, cy + DEPTH_SAMPLE_RADIUS)
                patch   = depth[y0d:y1d+1, x0d:x1d+1].astype(np.float32) * 0.001
                valid   = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
                depth_m = float(np.median(valid)) if len(valid) > 0 else 0.0

                if depth_m > 0:
                    p_opt     = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [cx, cy], depth_m)
                    p_cam_arr = optical_to_body(p_opt)

                    if center_ema is None:
                        center_ema = p_cam_arr.copy()
                    elif np.linalg.norm(p_cam_arr - center_ema) < EMA_GATE:
                        center_ema = (EMA_ALPHA * p_cam_arr
                                      + (1 - EMA_ALPHA) * center_ema)

                    p_base = transform_point_camera_to_base(
                        center_ema,
                        joint.q_wy, joint.q_wr, joint.q_wp, joint.q_head,
                    )

                    p_cam_str  = (f"(X={center_ema[0]:+.3f}, "
                                  f"Y={center_ema[1]:+.3f}, "
                                  f"Z={center_ema[2]:+.3f})")
                    p_base_str = (f"({p_base[0]:+.3f}, "
                                  f"{p_base[1]:+.3f}, "
                                  f"{p_base[2]:+.3f})")
                    detected   = best_box is not None
                    coasting   = best_box is None

                    if lc is not None:
                        lcm_msg = lidar_lcmt()
                        lcm_msg.offset_time = int(time.time() * 1e6)
                        lcm_msg.x, lcm_msg.y, lcm_msg.z = (
                            float(p_base[0]), float(p_base[1]), float(p_base[2]))
                        lc.publish("camera_ball_lcmt", lcm_msg.encode())

            if best_box is None:
                miss_count += 1

            with res_lock:
                res_bbox     = last_bbox if (miss_count <= COAST_FRAMES) else None
                res_conf     = best_conf
                res_cam_str  = p_cam_str
                res_base_str = p_base_str
                res_detected = detected
                res_coasting = coasting
                res_yolo_fps = yolo_fps.fps
                res_vel_t    = t_infer
                res_vel_vx   = vx
                res_vel_vy   = vy

            # 终端输出（含 YOLO 实际帧率）
            if detected:
                status = "BALL "
            elif coasting:
                status = "COAST"
            else:
                status = "     "
            print(
                f"\r[{status}] cam={p_cam_str}  base={p_base_str}  "
                f"YOLO={yolo_fps.fps:4.1f}fps",
                end="", flush=True,
            )

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
    yolo_thread.start()

    # ── 主线程：相机采集 + 显示（全速运行）─────────────────────────────────────
    cam_fps = _FPS()
    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf      = aligned.get_color_frame()
            df      = aligned.get_depth_frame()
            if not cf or not df:
                continue

            color_image  = np.asanyarray(cf.get_data())
            depth_array  = np.asanyarray(df.get_data())   # uint16

            with buf_lock:
                buf_color = color_image.copy()
                buf_depth = depth_array
            buf_updated.set()

            cam_fps.tick()

            # ── 叠加最新检测结果 ──────────────────────────────────────────────
            if viz:
                with res_lock:
                    bbox      = res_bbox
                    conf      = res_conf
                    cam_str   = res_cam_str
                    base_str  = res_base_str
                    detected  = res_detected
                    coasting  = res_coasting
                    yolo_fps_ = res_yolo_fps
                    vel_t     = res_vel_t
                    vx        = res_vel_vx
                    vy        = res_vel_vy

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    # 速度外推：将 bbox 向前平移 Δt × velocity
                    dt  = time.perf_counter() - vel_t
                    dx  = int(vx * dt)
                    dy  = int(vy * dt)
                    x1, y1, x2, y2 = x1+dx, y1+dy, x2+dx, y2+dy
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    color = (0, 255, 0) if detected else (0, 165, 255)
                    tag   = "Ball" if detected else "Coast"
                    top   = max(y1 - 30, 30)
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(color_image,
                                f"{tag} cam {cam_str}  {conf:.2f}",
                                (x1, top),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
                    cv2.putText(color_image,
                                f"     base{base_str}",
                                (x1, top + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    cv2.putText(color_image, "No ball", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                cv2.putText(color_image,
                            f"CAM {cam_fps.fps:.1f}  YOLO {yolo_fps_:.1f} fps",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow("camera_ball", color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        stop_flag.set()
        yolo_thread.join(timeout=2)
        pipeline.stop()
        cv2.destroyAllWindows()
        if HAS_ROS2:
            rclpy.shutdown()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
