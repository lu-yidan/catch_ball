# Soccer Ball 3D Detection

使用 **Intel RealSense D435** + **YOLOv8** 实时检测足球，支持输出相机坐标系和机器人 base（pelvis）坐标系下的位置。

## 文件结构

```
catch_ball/
├── camera_ball.py           # 相机版主程序（RealSense + YOLO）
├── test-ball.py             # 雷达版主程序（Livox Mid-360 + ROS2）
├── transform/
│   ├── __init__.py
│   ├── camera_to_base.py    # head_camera_link → pelvis 变换
│   └── mid360_to_base.py    # mid360_link → pelvis 变换
├── doc/
│   ├── alignment.md         # RGB↔Depth 对齐原理
│   └── core_logic.md        # 核心逻辑说明
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n catchball python=3.10 -y
conda activate catchball
```

### 2. 安装 pyrealsense2（必须用 conda，不能用 pip）

`pyrealsense2` 通过 conda-forge 安装可避免 Linux 上的 libusb 冲突：

```bash
conda install -c conda-forge pyrealsense2 -y
```

### 3. 安装其余 Python 依赖

```bash
pip install -r requirements.txt --ignore-installed pyrealsense2
```

> `--ignore-installed pyrealsense2` 防止 pip 覆盖刚才 conda 安装的版本。

### 4. 配置 RealSense USB 权限（Linux，只需一次）

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# 重新插拔相机后验证：
rs-enumerate-devices
```

### 5. 可选：ROS2 + LCM（仅 camera_ball.py 需要）

ROS2 和 LCM 是**可选的**。没有安装时，`camera_ball.py` 仍可运行，关节角使用命令行参数或默认值。

- ROS2：按照官方文档安装 Humble/Iron，并确保 `unitree_hg` 消息包已编译
- LCM：`pip install lcm`，并确保 `lcm_types` 包在 Python 路径中

---

## 运行

### main.py（最小版）

```bash
conda activate catchball
python main.py
```

首次运行自动下载 `yolov8n.pt`（约 6MB）。按 `q` 或 `Ctrl+C` 退出。

### camera_ball.py（完整版）

```bash
conda activate catchball

# 带可视化（默认）
python camera_ball.py

# 无头模式（无 OpenCV 窗口，适合部署）
python camera_ball.py --no-viz

# 手动指定关节角（无 ROS2 时调试用，单位 rad）
python camera_ball.py --no-viz --q-head 0.3 --q-wp 0.1

# 换精度更高的模型
python camera_ball.py --model yolov8s.pt
```

有 ROS2 时，关节角自动从 `/lowstate` 话题读取，无需手动指定。

---

## 坐标系说明

**相机坐标系（camera frame）：**

| 轴 | 方向 |
|---|---|
| X | 向右 |
| Y | 向下 |
| Z | 远离相机（即深度方向） |

**base / pelvis 坐标系：**

通过以下运动学链变换（关节角实时读取）：

```
pelvis → waist_yaw(Rz) → waist_roll(Rx) → waist_pitch(Ry) → head(Ry) → head_camera(fixed)
```

固定关节参数来自 URDF（`head_camera_joint`）。

---

## 常见问题

**`NameError: name 'Node' is not defined`**
确认 `camera_ball.py` 版本是最新的（已在 import 失败时将 `Node = object` 作为占位）。

**相机无法打开**
确认 USB 3.0 连接，已配置 udev rules，用 `rs-enumerate-devices` 验证。

**深度始终为 0**
足球距离过近（< 0.1m）或过远（> 10m），或表面高光反射。可调整 `DEPTH_MIN` / `DEPTH_MAX`。

**检测不到足球**
降低 `CONF_THRESHOLD`（如 0.3），或换用 `yolov8s.pt` / `yolov8m.pt`。
