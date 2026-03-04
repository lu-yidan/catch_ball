"""
Soccer Ball 3D Position Detection
使用 RealSense D435 + YOLOv8 检测足球并获取其相对于相机的 3D 坐标
"""

import sys
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO

# COCO 数据集中 sports ball 的 class id
SPORTS_BALL_CLASS_ID = 32

# 深度有效范围（米）
DEPTH_MIN = 0.1
DEPTH_MAX = 10.0

# 取中心区域小块来平均深度，避免单点噪声（像素半径）
DEPTH_SAMPLE_RADIUS = 5

# 置信度阈值
CONF_THRESHOLD = 0.4


def get_depth_at_center(depth_frame, cx: int, cy: int, radius: int = DEPTH_SAMPLE_RADIUS) -> float:
    """
    在 (cx, cy) 周围 radius 像素范围内采样深度，返回有效值的中位数（米）。
    返回 0.0 表示无有效深度。
    """
    w = depth_frame.get_width()
    h = depth_frame.get_height()

    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)

    samples = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            d = depth_frame.get_distance(x, y)
            if DEPTH_MIN < d < DEPTH_MAX:
                samples.append(d)

    if not samples:
        return 0.0
    return float(np.median(samples))


def pixel_to_3d(intrinsics, cx: int, cy: int, depth_m: float):
    """
    利用相机内参将像素坐标 + 深度反投影为 3D 点（相机坐标系，单位：米）。
    返回 (X, Y, Z)。
    """
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
    return point  # [X, Y, Z]


def draw_result(image, bbox, label: str, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_pos = (x1, max(y1 - 10, 20))
    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    # ---------- 加载 YOLO 模型 ----------
    model_path = "yolov8n.pt"  # 首次运行会自动下载
    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    # ---------- 初始化 RealSense ----------
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    print("[INFO] Starting RealSense pipeline...")
    profile = pipeline.start(config)

    # 获取深度传感器并设置量程
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[INFO] Depth scale: {depth_scale:.6f} m/unit")

    # 将 Depth 对齐到 Color 坐标系
    align = rs.align(rs.stream.color)

    # 获取 color stream 内参（用于 3D 反投影）
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    print(f"[INFO] Color intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
          f"ppx={intrinsics.ppx:.2f}, ppy={intrinsics.ppy:.2f}")

    print("[INFO] Running... Press 'q' to quit.")

    try:
        while True:
            # ---------- 获取对齐帧 ----------
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # ---------- YOLO 推理 ----------
            results = model(color_image, conf=CONF_THRESHOLD, verbose=False)

            best_box = None
            best_conf = 0.0

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == SPORTS_BALL_CLASS_ID and conf > best_conf:
                        best_conf = conf
                        best_box = box

            display_image = color_image.copy()

            if best_box is not None:
                # 像素坐标
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 获取深度
                depth_m = get_depth_at_center(depth_frame, cx, cy)

                if depth_m > 0:
                    # 反投影到 3D
                    X, Y, Z = pixel_to_3d(intrinsics, cx, cy, depth_m)
                    label = f"Ball  X:{X:.3f}m  Y:{Y:.3f}m  Z:{Z:.3f}m  conf:{best_conf:.2f}"
                    print(f"[BALL] X={X:+.3f}m  Y={Y:+.3f}m  Z={Z:+.3f}m  conf={best_conf:.2f}")
                else:
                    label = f"Ball (no depth)  conf:{best_conf:.2f}"
                    print(f"[BALL] Detected but depth invalid at ({cx}, {cy})")

                draw_result(display_image, (x1, y1, x2, y2), label)

                # 标记中心点
                cv2.circle(display_image, (cx, cy), 5, (0, 0, 255), -1)
            else:
                cv2.putText(display_image, "No ball detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Soccer Ball Detection", display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Pipeline stopped.")


if __name__ == "__main__":
    main()
