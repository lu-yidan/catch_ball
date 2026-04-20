# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Uses a **conda** environment named `catchball` (Python 3.10).

```bash
conda activate catchball
```

`pyrealsense2` must be installed via conda-forge, not pip:
```bash
conda install -c conda-forge pyrealsense2 -y
pip install -r requirements.txt --ignore-installed pyrealsense2
```

First-time Linux setup: configure udev rules for RealSense USB access (one-time):
```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Running

### camera_ball.py (RealSense D455 + YOLO, tennis ball)

```bash
python camera_ball.py              # with visualization
python camera_ball.py --no-viz     # headless, no OpenCV window
python camera_ball.py --no-ema     # disable EMA position filtering
python camera_ball.py --model yolo11m.pt --imgsz 480
python camera_ball.py --width 640 --height 480  # camera resolution
```

### test-ball.py (Livox Mid-360 + ROS2)

```bash
python test-ball.py   # requires ROS2 + livox_ros_driver2 + unitree_hg msgs
```

First run auto-downloads `yolov8n.pt` (~6MB). Press `q` or `Ctrl+C` to quit.

## Architecture

Two main scripts sharing the `transform/` package:

- **`camera_ball.py`** — RealSense D435 + YOLOv8; optional ROS2 joint angles + LCM publishing
- **`test-ball.py`** — Livox Mid-360 LiDAR; requires ROS2

### Threading model (camera_ball.py)

Producer-consumer design with two threads:
- **Main thread**: `pipeline.wait_for_frames()` (releases GIL) → swaps `buf_frames` reference (no copy) → `cv2.imshow` from `disp_frame`
- **YOLO worker thread**: waits on `buf_updated`, copies color+depth arrays, runs YOLO, Color→Depth mapping, EMA, LCM publish, annotates `disp_frame`

YOLO worker attempts `SCHED_FIFO` real-time scheduling (needs `CAP_SYS_NICE`), falls back to `nice=-10`.

`COAST_FRAMES = 10`: on missed detections, the last bbox is held for up to 10 frames (shown in orange as "Coast").

### Data flow (camera_ball.py each frame)

1. RealSense pipeline → raw frameset (reference swap, no copy)
2. color.copy() + depth_arr.copy() in YOLO thread
3. `cv2.resize` → `model.track()` on color frame → finds `sports ball` (COCO class 32)
4. **Color→Depth 3-step mapping**: NDC normalise → FOV correction → parallax correction
5. 5px-radius median depth patch at corrected depth pixel → `depth_surface`
6. `depth_m = depth_surface + BALL_RADIUS` (0.033 m, tennis ball)
7. `rs2_deproject_pixel_to_point(color_intrin, [cx,cy], depth_m)` → optical frame
8. `optical_to_body()` → camera body frame (X-forward, REP-103)
9. EMA filter (gate = 0.6 m, resets on exceed) → output position
10. Optional LCM publish on channel `camera_ball_lcmt`

### LiDAR algorithm (test-ball.py)

1. Filter by reflectivity ≥ 149, distance 0.2–2.0 m, spatial ROI
2. Gauss-Newton least-squares sphere fit (`estimate_ball_center_ls`) with known radius 0.115 m
3. EMA temporal smoothing (α=0.6, gate=0.6 m)
4. `transform_point_mid360_to_base()` → pelvis frame
5. LCM publish on channel `lidar_lcmt`

## Optional Dependencies

- **LCM** (`lcm` + `lcm_types.lidar_lcmt`): if available, publishes ball position (camera body frame) on `camera_ball_lcmt` (multicast `udpm://239.255.76.67:7667?ttl=255`)

`test-ball.py` requires ROS2 (`rclpy` + `livox_ros_driver2` + `unitree_hg`) and LCM unconditionally.

## Transform Package

`transform/camera_to_base.py` — provides:
- `optical_to_body(p_optical)`: optical frame (Z-fwd, X-right, Y-down) → body frame (X-fwd, Y-left, Z-up). Used by `camera_ball.py`.
- `transform_point_camera_to_base(p_cam, q_wy, q_wr, q_wp, q_head)`: kinematic chain to pelvis frame (not used by `camera_ball.py` currently).

`transform/mid360_to_base.py` — used by `test-ball.py`.

Kinematic chain (for future use):
```
pelvis → waist_yaw(Rz) → waist_roll(Rx) → waist_pitch(Ry) → head(Ry) → sensor(fixed)
```

## Model Selection

| Model | Params | mAP | Notes |
|---|---|---|---|
| `yolov8n.pt` | 3M | 37.3 | default, fastest |
| `yolov8s.pt` | 11M | 44.9 | |
| `yolo11s.pt` | 9M | 47.0 | newer arch |
| `yolo11m.pt` | 20M | 51.5 | **recommended** |
| `yolo11x.pt` | 57M | 54.7 | highest accuracy |

`*.pt` files are gitignored; downloaded automatically on first use.

## Key Tuning Constants

All in `camera_ball.py` top-level:

| Constant | Default | Effect |
|---|---|---|
| `CONF_THRESHOLD` | 0.3 | lower → more recall, more false positives |
| `BALL_RADIUS` | 0.033 m | tennis ball radius; depth reads front surface |
| `DEPTH_MIN/MAX` | 0.1/10.0 m | filter invalid depth readings |
| `EMA_ALPHA` | 0.6 | position smoothing (higher = faster response) |
| `EMA_GATE` | 0.6 m | jump gate; exceeding resets EMA |
| `COAST_FRAMES` | 10 | frames to hold last detection after miss |

## Docs

- `doc/core_logic.md` — per-function walkthrough of `camera_ball.py`
- `doc/alignment.md` — full explanation of the RGB↔Depth coordinate alignment math
