#!/usr/bin/env bash
# 严格 RGB-D 录制：实时只写 RealSense 原生 .bag。
#
# 用法：
#   bash tools/recording/run_record_rgbd_to_bag.sh                         # 1280×720 @30fps
#   bash tools/recording/run_record_rgbd_to_bag.sh --duration 10           # 录制 10 秒
#   bash tools/recording/run_record_rgbd_to_bag.sh --preview               # 显示轻量 RGB 预览
#   bash tools/recording/run_record_rgbd_to_bag.sh --width 848 --height 480 --fps 60

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

echo "[run_record_rgbd_to_bag.sh] Killing stale camera/record processes..."
STALE_PATTERN="detect_tennis_ball_yolo.py|detect_tennis_ball_hsv.py|detect_tennis_and_blue_disk_hsv.py|record_rgbd_to_videos.py|record_rgbd_to_bag.py"
STALE_PIDS=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run_record_rgbd_to_bag.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    STILL_ALIVE=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run_record_rgbd_to_bag.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run_record_rgbd_to_bag.sh] No stale processes found."
fi

if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run_record_rgbd_to_bag.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

echo "[run_record_rgbd_to_bag.sh] Starting record_rgbd_to_bag.py $*"
cd "$REPO_ROOT"
exec "$PYTHON" -u "$SCRIPT_DIR/record_rgbd_to_bag.py" "$@"
