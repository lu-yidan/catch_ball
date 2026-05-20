#!/usr/bin/env bash
# 启动 camera_ball_dual_hsv.py（HSV 同时识别网球和蓝色末端圆片）
#
# 用法：
#   bash run_dual_hsv.sh
#   bash run_dual_hsv.sh --no-viz
#   bash run_dual_hsv.sh --show-mask
#   bash run_dual_hsv.sh --blue-h-low 86 --blue-h-high 108 --blue-s-min 35 --blue-v-min 25

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

echo "[run_dual_hsv.sh] Killing stale camera processes..."
STALE_PIDS=$(pgrep -f "camera_ball|camera_ball_dual_hsv" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run_dual_hsv.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    STILL_ALIVE=$(pgrep -f "camera_ball|camera_ball_dual_hsv" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run_dual_hsv.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run_dual_hsv.sh] No stale processes found."
fi

if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run_dual_hsv.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

echo "[run_dual_hsv.sh] Starting camera_ball_dual_hsv.py $*"
cd "$SCRIPT_DIR"
exec "$PYTHON" -u camera_ball_dual_hsv.py "$@"
