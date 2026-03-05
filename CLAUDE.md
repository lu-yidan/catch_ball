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

## Running

```bash
python main.py
```

First run auto-downloads `yolov8n.pt` (~6MB). Press `q` or `Ctrl+C` to quit.

## Architecture

Two main scripts, sharing a common `transform/` package:

- **`camera_ball.py`** — RealSense D435 + YOLOv8 ball detection with base frame transform
- **`test-ball.py`** — Livox Mid-360 LiDAR + ROS2 ball detection with base frame transform

Data flow in `camera_ball.py` each frame:

1. **RealSense pipeline** → aligned color + depth frames
2. **`align` object** → warps depth into color camera's coordinate frame
3. **YOLOv8** inference on color frame → finds `sports ball` (COCO class 32)
4. **`get_depth_at_center()`** → 5px-radius median depth patch around BBox center
5. **`rs2_deproject_pixel_to_point()`** → pixel + depth → 3D in camera optical frame (Z-forward)
6. **`optical_to_body()`** → optical frame (Z-forward) → body frame (X-forward, REP-103)
7. **`transform_point_camera_to_base()`** → kinematic chain → pelvis/base frame

Key constants at the top of `camera_ball.py`: `CONF_THRESHOLD`, `DEPTH_SAMPLE_RADIUS`, `DEPTH_MIN/MAX`, `SPORTS_BALL_CLASS_ID`, `EMA_ALPHA/GATE`.

## Transform Package

`transform/camera_to_base.py` and `transform/mid360_to_base.py` implement the kinematic chain from URDF `g1_sysid_23dof.urdf`:

```
pelvis → waist_yaw(Rz) → waist_roll(Rx) → waist_pitch(Ry) → head(Ry) → sensor(fixed)
```

**Coordinate convention**: RealSense outputs optical frame (Z-forward). Must call `optical_to_body()` before passing to `transform_point_camera_to_base()`. URDF `head_camera_link` uses body convention (X-forward, REP-103).

## Docs

- `doc/core_logic.md` — per-function walkthrough of `camera_ball.py`
- `doc/alignment.md` — full explanation of the RGB↔Depth coordinate alignment math (intrinsics, extrinsics, reprojection)
