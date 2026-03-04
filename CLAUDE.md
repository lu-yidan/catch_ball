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

Single-file application (`main.py`). Data flows in one direction each frame:

1. **RealSense pipeline** → aligned color + depth frames
2. **`align` object** → warps depth into color camera's coordinate frame (so pixel (u,v) means the same point in both images)
3. **YOLOv8** inference on color frame → finds `sports ball` (COCO class 32)
4. **`get_depth_at_center()`** → samples a 5px-radius patch around the BBox center, returns median of valid depths (filters 0-values and out-of-range)
5. **`rs2_deproject_pixel_to_point()`** → uses color camera intrinsics (read from firmware) to back-project pixel + depth into 3D (X, Y, Z) in meters

Key constants at the top of `main.py`: `CONF_THRESHOLD`, `DEPTH_SAMPLE_RADIUS`, `DEPTH_MIN/MAX`, `SPORTS_BALL_CLASS_ID`.

## Docs

- `doc/core_logic.md` — per-function walkthrough of `main.py`
- `doc/alignment.md` — full explanation of the RGB↔Depth coordinate alignment math (intrinsics, extrinsics, reprojection)
