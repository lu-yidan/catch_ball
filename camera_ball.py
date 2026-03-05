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

    # ── RealSense ─────────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16,  30)

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

    # ── 状态 ─────────────────────────────────────────────────────────────────
    center_ema  = None
    last_bbox   = None   # coasting 用：上一次检测到的 bbox
    miss_count  = 0      # 连续漏检帧数
    fps_counter = _FPS()

    try:
        while True:
            # ── 获取对齐帧 ────────────────────────────────────────────────────
            frames         = pipeline.wait_for_frames()
            aligned        = align.process(frames)
            color_frame    = aligned.get_color_frame()
            depth_frame    = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            fps_counter.tick()

            # ── YOLO 追踪（ByteTrack，跨帧保持 ID）────────────────────────────
            results   = model.track(color_image, conf=CONF_THRESHOLD,
                                    persist=True, verbose=False)
            best_box  = None
            best_conf = 0.0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_box = c, box

            # ── 处理检测结果 ──────────────────────────────────────────────────
            p_cam_str  = "—"
            p_base_str = "—"
            detected   = False
            coasting   = False

            if best_box is not None:
                miss_count = 0
                last_bbox  = tuple(map(int, best_box.xyxy[0]))

            if last_bbox is not None and miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = last_bbox
                cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
                depth_m = get_depth_at_center(depth_frame, cx, cy)

                if depth_m > 0:
                    p_opt     = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [cx, cy], depth_m
                    )  # 光学坐标系：Z前、X右、Y下
                    p_cam_arr = optical_to_body(p_opt)

                    if center_ema is None:
                        center_ema = p_cam_arr.copy()
                    elif np.linalg.norm(p_cam_arr - center_ema) < EMA_GATE:
                        center_ema = EMA_ALPHA * p_cam_arr + (1 - EMA_ALPHA) * center_ema

                    p_base = transform_point_camera_to_base(
                        center_ema,
                        joint.q_wy, joint.q_wr, joint.q_wp, joint.q_head,
                    )

                    p_cam_str  = f"(X={center_ema[0]:+.3f}, Y={center_ema[1]:+.3f}, Z={center_ema[2]:+.3f})"
                    p_base_str = f"({p_base[0]:+.3f}, {p_base[1]:+.3f}, {p_base[2]:+.3f})"
                    detected   = best_box is not None
                    coasting   = best_box is None

                    # LCM 发布（检测到或 coasting 均发布）
                    if lc is not None:
                        lcm_msg = lidar_lcmt()
                        lcm_msg.offset_time = int(time.time() * 1e6)
                        lcm_msg.x, lcm_msg.y, lcm_msg.z = (
                            float(p_base[0]), float(p_base[1]), float(p_base[2])
                        )
                        lc.publish("camera_ball_lcmt", lcm_msg.encode())

                    if viz:
                        # 绿色=检测到，橙色=coasting（保留上帧位置）
                        color = (0, 255, 0) if detected else (0, 165, 255)
                        label = (f"{'Ball' if detected else 'Coast'} "
                                 f"base={p_base_str}  {best_conf:.2f}")
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(color_image, label,
                                    (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

            if best_box is None:
                miss_count += 1

            # ── 终端输出 ──────────────────────────────────────────────────────
            if detected:
                status = "BALL "
            elif coasting:
                status = "COAST"
            else:
                status = "     "
            print(
                f"\r[{status}] cam={p_cam_str}  base={p_base_str}  "
                f"FPS={fps_counter.fps:5.1f}",
                end="", flush=True
            )

            # ── 可视化 ────────────────────────────────────────────────────────
            if viz:
                cv2.putText(color_image, f"FPS: {fps_counter.fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                if not detected and not coasting:
                    cv2.putText(color_image, "No ball", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("camera_ball", color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if HAS_ROS2:
            rclpy.shutdown()
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
