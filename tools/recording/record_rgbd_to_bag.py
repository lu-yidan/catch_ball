"""
record_rgbd_to_bag.py

Record RealSense RGB-D streams to a native .bag file.

This is the preferred path for strict 1280x720@30fps capture because the
recording path stores the RealSense streams directly instead of encoding MP4 or
writing depth PNGs in the capture loop.

Usage:
    python record_rgbd_to_bag.py
    python record_rgbd_to_bag.py --duration 10
    python record_rgbd_to_bag.py --width 1280 --height 720 --fps 30
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Record RealSense RGB-D to .bag")
    parser.add_argument("--out-dir", default="recordings", help="directory for recording sessions")
    parser.add_argument("--prefix", default="rgbd_bag", help="session file name prefix")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument("--fps", type=int, default=30, help="capture FPS")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to record; 0 means until q/Ctrl+C")
    parser.add_argument("--preview", action="store_true", help="show lightweight RGB preview while recording")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_path = out_dir / f"{args.prefix}_{stamp}.bag"
    metadata_path = out_dir / f"{args.prefix}_{stamp}.json"

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_record_to_file(str(bag_path))

    print(f"[INFO] Recording bag: {bag_path}")
    print(f"[INFO] Streams: color/depth {args.width}x{args.height}@{args.fps}")
    print("[INFO] Press q in preview window or Ctrl+C to stop.")

    frame_count = 0
    first_time = None
    last_time = None
    pipeline_started = False
    started_at = time.time()

    try:
        profile = pipeline.start(cfg)
        pipeline_started = True

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

        metadata = {
            "bag_file": bag_path.name,
            "started_at_unix": started_at,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "depth_scale_m_per_unit": depth_scale,
            "color_intrinsics": intrinsics_to_dict(color_profile.get_intrinsics()),
            "depth_intrinsics": intrinsics_to_dict(depth_profile.get_intrinsics()),
            "note": "Use export_rgbd_bag_to_videos.py to convert this bag to mp4/png outputs.",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        while True:
            frames = pipeline.wait_for_frames()
            now = time.time()
            if first_time is None:
                first_time = now
            last_time = now
            frame_count += 1

            elapsed = now - started_at
            actual_fps = 0.0
            if first_time is not None and last_time > first_time:
                actual_fps = (frame_count - 1) / (last_time - first_time)
            print(f"\r[REC BAG] frames={frame_count} elapsed={elapsed:.1f}s "
                  f"loop_fps={actual_fps:.1f}", end="", flush=True)

            if args.preview:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color = np.asanyarray(color_frame.get_data())
                    cv2.imshow("record_rgbd_to_bag preview (q=quit)", color)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.duration > 0 and elapsed >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()

        ended_at = time.time()
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update({
                "ended_at_unix": ended_at,
                "frame_count": frame_count,
                "elapsed_sec": ended_at - started_at,
            })
            if first_time is not None and last_time is not None and last_time > first_time:
                metadata["capture_loop_fps"] = (frame_count - 1) / (last_time - first_time)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"\n[INFO] Saved bag: {bag_path}")
        print(f"[INFO] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
