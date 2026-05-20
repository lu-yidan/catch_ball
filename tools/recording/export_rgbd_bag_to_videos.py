"""
export_rgbd_bag_to_videos.py

Convert a RealSense .bag recording to RGB/depth outputs.

Outputs per bag:
  color.mp4       - RGB video
  depth_vis.mp4   - colorized depth preview
  depth/*.png     - raw 16-bit depth frames in RealSense units
  timestamps.csv  - bag timestamps
  metadata.json   - export settings and stream metadata

Usage:
    python export_rgbd_bag_to_videos.py recordings/rgbd_bag_YYYYMMDD_HHMMSS.bag
    python export_rgbd_bag_to_videos.py recordings/test.bag --no-depth-png
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def intrinsics_to_dict(intrin):
    return {
        "width": intrin.width,
        "height": intrin.height,
        "ppx": intrin.ppx,
        "ppy": intrin.ppy,
        "fx": intrin.fx,
        "fy": intrin.fy,
        "model": str(intrin.model),
        "coeffs": list(intrin.coeffs),
    }


def depth_to_colormap(depth_raw: np.ndarray, depth_scale: float, depth_max_m: float):
    depth_m = depth_raw.astype(np.float32) * depth_scale
    depth_8 = np.clip(depth_m / depth_max_m * 255.0, 0, 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_8, cv2.COLORMAP_JET)
    depth_vis[depth_raw == 0] = (0, 0, 0)
    return depth_vis


def open_writer(path: Path, fps: float, frame_shape):
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def stream_fps(profile, fallback: float):
    try:
        return float(profile.fps())
    except Exception:
        return fallback


def main():
    parser = argparse.ArgumentParser(description="Export RGB-D data from a RealSense .bag")
    parser.add_argument("bag", help="input .bag file")
    parser.add_argument("--out-dir", default="", help="output directory; default is bag stem")
    parser.add_argument("--fps", type=float, default=0.0, help="output video FPS; 0 uses bag stream FPS")
    parser.add_argument("--depth-max", type=float, default=6.0, help="max depth in meters for visualization")
    parser.add_argument("--no-depth-png", action="store_true", help="do not export raw 16-bit depth PNG frames")
    parser.add_argument("--no-align", action="store_true", help="do not align depth to color pixels")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        raise FileNotFoundError(bag_path)

    out_dir = Path(args.out_dir) if args.out_dir else bag_path.with_suffix("")
    depth_dir = out_dir / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_depth_png:
        depth_dir.mkdir(exist_ok=True)

    color_path = out_dir / "color.mp4"
    depth_vis_path = out_dir / "depth_vis.mp4"
    timestamps_path = out_dir / "timestamps.csv"
    metadata_path = out_dir / "metadata.json"

    pipeline = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag_path), repeat_playback=False)

    align = None if args.no_align else rs.align(rs.stream.color)
    color_writer = None
    depth_writer = None
    frame_idx = 0
    pipeline_started = False
    metadata = {
        "source_bag": str(bag_path),
        "error": "export did not start",
    }

    print(f"[INFO] Reading bag: {bag_path}")
    print(f"[INFO] Exporting to: {out_dir}")

    try:
        profile = pipeline.start(cfg)
        pipeline_started = True
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        output_fps = args.fps if args.fps > 0 else stream_fps(color_profile, 30.0)

        metadata = {
            "source_bag": str(bag_path),
            "output_fps": output_fps,
            "depth_scale_m_per_unit": depth_scale,
            "depth_aligned_to_color": align is not None,
            "color_intrinsics": intrinsics_to_dict(color_profile.get_intrinsics()),
            "depth_intrinsics": intrinsics_to_dict(depth_profile.get_intrinsics()),
            "files": {
                "color_video": color_path.name,
                "depth_visualization_video": depth_vis_path.name,
                "raw_depth_dir": "depth" if not args.no_depth_png else None,
                "timestamps": timestamps_path.name,
            },
        }

        with timestamps_path.open("w", newline="", encoding="utf-8") as ts_file:
            ts_writer = csv.writer(ts_file)
            ts_writer.writerow(["frame", "bag_timestamp_ms", "color_frame_number", "depth_frame_number"])

            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    break

                if align is not None:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_vis = depth_to_colormap(depth_raw, depth_scale, args.depth_max)

                if color_writer is None:
                    color_writer = open_writer(color_path, output_fps, color.shape)
                    depth_writer = open_writer(depth_vis_path, output_fps, depth_vis.shape)

                color_writer.write(color)
                depth_writer.write(depth_vis)
                if not args.no_depth_png:
                    cv2.imwrite(str(depth_dir / f"{frame_idx:06d}.png"), depth_raw)

                ts_writer.writerow([
                    frame_idx,
                    f"{frames.get_timestamp():.3f}",
                    color_frame.get_frame_number(),
                    depth_frame.get_frame_number(),
                ])

                frame_idx += 1
                print(f"\r[EXPORT] frames={frame_idx}", end="", flush=True)

    finally:
        if color_writer is not None:
            color_writer.release()
        if depth_writer is not None:
            depth_writer.release()
        if pipeline_started:
            pipeline.stop()

        metadata["frame_count"] = frame_idx
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\n[INFO] Exported {frame_idx} frames to {out_dir}")


if __name__ == "__main__":
    main()
