#!/usr/bin/env bash
# 启动 camera_ball_color.py（RealSense D455 + HSV 网球检测）
#
# 用法：
#   bash run_color.sh                        # 带可视化
#   bash run_color.sh --no-viz               # 无窗口
#   bash run_color.sh --show-mask            # 显示 HSV 二值掩码（调参用）
#   bash run_color.sh --h-low 30 --h-high 75 # 手动指定色相范围
#
# 每次启动前自动杀掉残留进程，确保相机干净释放。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

# ── 1. 杀掉所有残留的 camera_ball 进程 ───────────────────────────────────────
echo "[run_color.sh] Killing stale camera_ball processes..."
STALE_PIDS=$(pgrep -f "camera_ball" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run_color.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    STILL_ALIVE=$(pgrep -f "camera_ball" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run_color.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run_color.sh] No stale processes found."
fi

# ── 2. 等待相机 /dev/video* 释放 ─────────────────────────────────────────────
if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run_color.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

# ── 3. 启动 ───────────────────────────────────────────────────────────────────
echo "[run_color.sh] Starting camera_ball_color.py $*"
cd "$SCRIPT_DIR"
exec "$PYTHON" -u camera_ball_color.py "$@"
