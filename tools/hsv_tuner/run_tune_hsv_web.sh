#!/usr/bin/env bash
# 启动 Web HSV/MOG2 调参工具。
#
# 用法：
#   bash tools/hsv_tuner/run_tune_hsv_web.sh
#   bash tools/hsv_tuner/run_tune_hsv_web.sh --port 5001
#   bash tools/hsv_tuner/run_tune_hsv_web.sh --video recordings/rgbd_YYYYMMDD_HHMMSS/color.mp4

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

echo "[run_tune_hsv_web.sh] Killing stale camera/tuner processes..."
STALE_PIDS=$(pgrep -f "camera_ball|tune_hsv_web.py" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run_tune_hsv_web.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    STILL_ALIVE=$(pgrep -f "camera_ball|tune_hsv_web.py" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run_tune_hsv_web.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run_tune_hsv_web.sh] No stale processes found."
fi

if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run_tune_hsv_web.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

echo "[run_tune_hsv_web.sh] Starting tune_hsv_web.py $*"
cd "$REPO_ROOT"
exec "$PYTHON" -u "$SCRIPT_DIR/tune_hsv_web.py" "$@"
