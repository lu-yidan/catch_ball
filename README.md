# Soccer Ball 3D Detection

使用 **Intel RealSense D435** + **YOLOv8** 实时检测足球，并输出足球相对于相机的 3D 坐标（米）。

## 依赖

- Intel RealSense D435 及 USB 3.0 连接
- Python 3.10+
- Conda（推荐）

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n catchball python=3.10 -y
conda activate catchball
```

### 2. 安装 pyrealsense2

`pyrealsense2` 通过 conda-forge 安装比 pip 更稳定，可避免 Linux 上的 libusb 冲突：

```bash
conda install -c conda-forge pyrealsense2 -y
```

### 3. 安装其余依赖

```bash
pip install -r requirements.txt --ignore-installed pyrealsense2
```

> `--ignore-installed pyrealsense2` 防止 pip 覆盖刚才 conda 安装的版本。

### 4. 配置 RealSense USB 权限（Linux）

```bash
# 下载官方 udev rules
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# 重新插拔相机
```

## 运行

```bash
conda activate catchball
python main.py
```

首次运行会自动下载 `yolov8n.pt`（约 6MB）。

按 `q` 退出，或 `Ctrl+C`。

## 输出说明

终端每帧打印检测到的足球 3D 坐标：

```
[BALL] X=+0.032m  Y=+0.015m  Z=+1.823m  conf=0.87
```

| 轴 | 方向 |
|---|---|
| X | 向右为正 |
| Y | 向下为正 |
| Z | 远离相机为正（即距离） |

窗口叠加显示 BBox、坐标文字和中心点红点。

## 文件结构

```
catch_ball/
├── main.py           # 主程序
├── requirements.txt  # Python 依赖
└── README.md
```

## 常见问题

**相机无法打开**
确认 USB 3.0 连接，并已配置 udev rules，尝试 `rs-enumerate-devices` 验证。

**深度为 0**
足球距离过近（< 0.1m）或过远（> 10m），或表面反光。可调整 `DEPTH_MIN` / `DEPTH_MAX`。

**检测不到足球**
降低 `CONF_THRESHOLD`（如 0.3），或换用精度更高的模型 `yolov8s.pt`。
