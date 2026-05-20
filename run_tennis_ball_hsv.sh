#!/usr/bin/env bash
# 启动 detect_tennis_ball_hsv.py（RealSense D455 + HSV 网球检测）
#
# 用法：
#   bash run_tennis_ball_hsv.sh                              # 1280×720 @30fps，带可视化（默认）
#   bash run_tennis_ball_hsv.sh --no-viz                     # 无窗口
#   bash run_tennis_ball_hsv.sh --show-mask                  # 显示 HSV 二值掩码（调参用）
#   bash run_tennis_ball_hsv.sh --width 848 --height 480     # 848×480 @60fps（帧率优先）
#   bash run_tennis_ball_hsv.sh --width 640 --height 480     # 640×480 @90fps（最快）
#   bash run_tennis_ball_hsv.sh --h-low 30 --h-high 75       # 手动指定色相范围
#   bash run_tennis_ball_hsv.sh --no-traj                    # 关闭轨迹预测叠加层
#   bash run_tennis_ball_hsv.sh --record                     # 录制（自动命名 ball_YYYYMMDD_HHMMSS.mp4）
#   bash run_tennis_ball_hsv.sh --record out.mp4             # 录制到指定文件
#
# 每次启动前自动杀掉残留进程，确保相机干净释放。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

# ── 1. 杀掉所有残留的相机检测进程 ─────────────────────────────────────────────
STALE_PATTERN="detect_tennis_ball_yolo.py|detect_tennis_ball_hsv.py|detect_tennis_and_blue_disk_hsv.py"
echo "[run_tennis_ball_hsv.sh] Killing stale camera detector processes..."
STALE_PIDS=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
if [ -n "$STALE_PIDS" ]; then
    echo "[run_tennis_ball_hsv.sh] Found PIDs: $STALE_PIDS — sending SIGTERM..."
    kill $STALE_PIDS 2>/dev/null || true
    sleep 2
    STILL_ALIVE=$(pgrep -f "$STALE_PATTERN" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        echo "[run_tennis_ball_hsv.sh] Force killing: $STILL_ALIVE"
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 1
    fi
else
    echo "[run_tennis_ball_hsv.sh] No stale processes found."
fi

# ── 2. 等待相机 /dev/video* 释放 ─────────────────────────────────────────────
if ls /dev/video* &>/dev/null; then
    BUSY=$(fuser /dev/video* 2>/dev/null || true)
    if [ -n "$BUSY" ]; then
        echo "[run_tennis_ball_hsv.sh] Camera devices still busy ($BUSY), waiting 3s..."
        sleep 3
    fi
fi

# ── 3. 启动 ───────────────────────────────────────────────────────────────────
echo "[run_tennis_ball_hsv.sh] Starting detect_tennis_ball_hsv.py $*"
cd "$SCRIPT_DIR"
exec "$PYTHON" -u detect_tennis_ball_hsv.py "$@"
