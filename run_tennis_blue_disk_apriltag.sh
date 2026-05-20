#!/usr/bin/env bash
# 启动 detect_tennis_blue_disk_apriltag.py
#
# 同时检测网球、蓝色末端圆片、AprilTag 36h11 tag0，并输出目标相对 tag/中心点坐标。
#
# 用法：
#   bash run_tennis_blue_disk_apriltag.sh
#   bash run_tennis_blue_disk_apriltag.sh --show-mask
#   bash run_tennis_blue_disk_apriltag.sh --no-viz
#   bash run_tennis_blue_disk_apriltag.sh --center-origin-in-tag-cm 50 -30 -27
#   bash run_tennis_blue_disk_apriltag.sh --tag-origin-in-center 0.2 0.0 0.5
#   bash run_tennis_blue_disk_apriltag.sh --tag-every 5 --width 848 --height 480

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

echo "[run_tennis_blue_disk_apriltag.sh] Killing stale camera processes..."
STALE_PATTERN="detect_tennis_ball_yolo.py|detect_tennis_ball_hsv.py|detect_tennis_and_blue_disk_hsv.py|detect_tennis_blue_disk_apriltag.py"
STALE_PIDS=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run_tennis_blue_disk_apriltag.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    STILL_ALIVE=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run_tennis_blue_disk_apriltag.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run_tennis_blue_disk_apriltag.sh] No stale processes found."
fi

if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run_tennis_blue_disk_apriltag.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

echo "[run_tennis_blue_disk_apriltag.sh] Starting detect_tennis_blue_disk_apriltag.py $*"
cd "$SCRIPT_DIR"
exec "$PYTHON" -u detect_tennis_blue_disk_apriltag.py "$@"
