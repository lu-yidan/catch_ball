"""
record_rgbd_to_videos.py

Record synchronized RealSense RGB and depth streams.

Outputs per session:
  color.mp4       - playable RGB video
  depth_vis.mp4   - playable colorized depth preview
  depth/*.png     - raw 16-bit depth frames in RealSense units
  timestamps.csv  - frame index and host timestamp
  metadata.json   - camera intrinsics, depth scale, and recording settings

Usage:
    python record_rgbd_to_videos.py
    python record_rgbd_to_videos.py --duration 10
    python record_rgbd_to_videos.py --width 848 --height 480 --fps 60
    python record_rgbd_to_videos.py --no-preview --no-depth-png
"""

import argparse
import csv
import json
import time
from datetime import datetime
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


def main():
    parser = argparse.ArgumentParser(description="Record RealSense RGB + depth")
    parser.add_argument("--out-dir", default="recordings", help="directory for recording sessions")
    parser.add_argument("--prefix", default="rgbd", help="session directory name prefix")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--fps", type=int, default=30, help="capture FPS")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to record; 0 means until q/Ctrl+C")
    parser.add_argument("--depth-max", type=float, default=6.0, help="max depth in meters for visualization")
    parser.add_argument("--no-preview", action="store_true", help="disable OpenCV preview window")
    parser.add_argument("--no-depth-png", action="store_true", help="do not save raw 16-bit depth PNG frames")
    parser.add_argument("--no-align", action="store_true", help="do not align depth to color pixels")
    args = parser.parse_args()

    session_name = f"{args.prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = Path(args.out_dir) / session_name
    depth_dir = session_dir / "depth"
    session_dir.mkdir(parents=True, exist_ok=False)
    if not args.no_depth_png:
        depth_dir.mkdir()

    color_path = session_dir / "color.mp4"
    depth_vis_path = session_dir / "depth_vis.mp4"
    timestamps_path = session_dir / "timestamps.csv"
    metadata_path = session_dir / "metadata.json"

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    align = None if args.no_align else rs.align(rs.stream.color)
    color_writer = None
    depth_writer = None
    frame_idx = 0
    started_at = time.time()
    pipeline_started = False

    print(f"[INFO] Output: {session_dir}")
    print(f"[INFO] Starting RealSense color/depth {args.width}x{args.height}@{args.fps}...")

    try:
        profile = pipeline.start(cfg)
        pipeline_started = True
        pipeline.wait_for_frames(timeout_ms=5000)

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        metadata = {
            "session": session_name,
            "started_at_unix": started_at,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
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
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        with timestamps_path.open("w", newline="", encoding="utf-8") as ts_file:
            ts_writer = csv.writer(ts_file)
            ts_writer.writerow(["frame", "host_time_unix", "elapsed_sec"])

            print("[INFO] Recording. Press q in preview window or Ctrl+C to stop.")
            while True:
                frames = pipeline.wait_for_frames()
                if align is not None:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                now = time.time()
                elapsed = now - started_at
                color = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_vis = depth_to_colormap(depth_raw, depth_scale, args.depth_max)

                if color_writer is None:
                    color_writer = open_writer(color_path, args.fps, color.shape)
                    depth_writer = open_writer(depth_vis_path, args.fps, depth_vis.shape)

                current_idx = frame_idx
                color_writer.write(color)
                depth_writer.write(depth_vis)
                if not args.no_depth_png:
                    cv2.imwrite(str(depth_dir / f"{current_idx:06d}.png"), depth_raw)
                ts_writer.writerow([current_idx, f"{now:.6f}", f"{elapsed:.6f}"])
                frame_idx += 1

                if not args.no_preview:
                    preview = np.hstack([color, depth_vis])
                    cv2.putText(preview, f"frame={current_idx}  t={elapsed:.1f}s",
                                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.imshow("record_rgbd_to_videos: color | depth (q=quit)", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                print(f"\r[REC] frames={frame_idx} elapsed={elapsed:.1f}s", end="", flush=True)

                if args.duration > 0 and elapsed >= args.duration:
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        if pipeline_started:
            pipeline.stop()
        if color_writer is not None:
            color_writer.release()
        if depth_writer is not None:
            depth_writer.release()
        cv2.destroyAllWindows()

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["ended_at_unix"] = time.time()
            metadata["frame_count"] = frame_idx
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"\n[INFO] Saved {frame_idx} frames to {session_dir}")


if __name__ == "__main__":
    main()
