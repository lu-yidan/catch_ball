#!/usr/bin/env bash
# 录制 RealSense RGB + depth 数据。
#
# 用法：
#   bash tools/recording/run_record_rgbd_to_videos.sh                         # 1280×720 @30fps，带预览
#   bash tools/recording/run_record_rgbd_to_videos.sh --duration 10           # 录制 10 秒
#   bash tools/recording/run_record_rgbd_to_videos.sh --width 848 --height 480 --fps 60
#   bash tools/recording/run_record_rgbd_to_videos.sh --no-preview            # 后台录制
#   bash tools/recording/run_record_rgbd_to_videos.sh --no-depth-png          # 只保存 color.mp4 + depth_vis.mp4

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

echo "[run_record_rgbd_to_videos.sh] Killing stale camera/record processes..."
STALE_PATTERN="detect_tennis_ball_yolo.py|detect_tennis_ball_hsv.py|detect_tennis_and_blue_disk_hsv.py|record_rgbd_to_videos.py"
STALE_PIDS=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run_record_rgbd_to_videos.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    STILL_ALIVE=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run_record_rgbd_to_videos.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run_record_rgbd_to_videos.sh] No stale processes found."
fi

if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run_record_rgbd_to_videos.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

echo "[run_record_rgbd_to_videos.sh] Starting record_rgbd_to_videos.py $*"
cd "$REPO_ROOT"
exec "$PYTHON" -u "$SCRIPT_DIR/record_rgbd_to_videos.py" "$@"
