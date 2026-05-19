#!/usr/bin/env bash
# 将 RealSense .bag 离线导出为 color.mp4、depth_vis.mp4 和 depth PNG。
#
# 用法：
#   bash tools/recording/run_export_bag.sh recordings/rgbd_bag_YYYYMMDD_HHMMSS.bag
#   bash tools/recording/run_export_bag.sh recordings/test.bag --no-depth-png

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/home/ydlu/miniconda3/envs/catchball/bin/python"

if [ $# -lt 1 ]; then
    echo "Usage: bash tools/recording/run_export_bag.sh <file.bag> [export options]"
    exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON" -u "$SCRIPT_DIR/export_bag_rgbd.py" "$@"
