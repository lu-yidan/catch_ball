#!/usr/bin/env bash
# 启动 camera_ball.py（RealSense D455 + YOLO 网球检测）
#
# 用法：
#   bash run.sh                          # 带可视化
#   bash run.sh --no-viz                 # 无窗口
#   bash run.sh --model yolo11m.pt       # 切换模型
#   bash run.sh --no-viz --imgsz 320     # 低延迟模式
#
# 每次启动前自动杀掉残留进程，确保相机干净释放。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

# ── 1. 杀掉所有残留的 camera_ball 进程 ───────────────────────────────────────
echo "[run.sh] Killing stale camera_ball processes..."
STALE_PIDS=$(pgrep -f "camera_ball.py" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    # Force kill if still alive
    STILL_ALIVE=$(pgrep -f "camera_ball.py" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run.sh] No stale processes found."
fi

# ── 2. 等待相机 /dev/video* 释放 ─────────────────────────────────────────────
if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

# ── 3. 启动 ───────────────────────────────────────────────────────────────────
echo "[run.sh] Starting camera_ball.py $*"
cd "$SCRIPT_DIR"
exec "$PYTHON" -u camera_ball.py "$@"
