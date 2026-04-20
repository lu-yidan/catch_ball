# Tennis Ball 3D Detection

使用 **Intel RealSense D455** + **YOLOv8** 实时检测网球，输出球相对相机的三维位置（camera body frame：X前 Y左 Z上）。

---

## 环境配置

```bash
conda create -n catchball python=3.10 -y
conda activate catchball

# pyrealsense2 必须通过 conda 安装（pip 版在 Linux 上有 libusb 冲突）
conda install -c conda-forge pyrealsense2 -y
pip install -r requirements.txt --ignore-installed pyrealsense2
```

首次 Linux 设置：配置 RealSense USB 权限（只需一次）：

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 两种检测方案

| | YOLO 方案 (`camera_ball.py`) | HSV 方案 (`camera_ball_color.py`) |
|--|--|--|
| 检测原理 | 神经网络目标检测 | HSV 色块分割 + 圆形拟合 |
| 速度 | ~60 fps（GPU） | >200 fps（CPU） |
| 运动模糊 | 差（bbox 变形） | 好（颜色比边缘更耐模糊） |
| 光照依赖 | 低 | 高（需调 HSV 阈值） |
| 遮挡处理 | 好 | 差（遮挡会低估圆半径） |
| 深度来源 | 深度传感器（front surface + R） | 视觉测距 + 传感器融合 |
| 适用场景 | 通用 | 高速球、颜色稳定环境 |

---

## 运行

### YOLO 方案

```bash
bash run.sh                              # 带可视化
bash run.sh --no-viz                     # 无窗口
bash run.sh --model models/yolo11m.pt    # 精度更高的模型
```

首次运行自动下载 `models/yolov8n.pt`（约 6MB）。

### HSV 方案

```bash
bash run_color.sh                              # 1280×720 @30fps（默认，最远 ~6.8m）
bash run_color.sh --width 848 --height 480     # 848×480  @60fps（最远 ~5.6m）
bash run_color.sh --width 640 --height 480     # 640×480  @90fps（最远 ~4.3m）
bash run_color.sh --show-mask                  # 显示 HSV 二值掩码（调参用）
bash run_color.sh --h-low 30 --h-high 75       # 手动指定 HSV 色相范围
```

按 `q` 或 `Ctrl+C` 退出。

### HSV 检测距离上限

检测距离由几何约束决定：

```
D_max = fx × BALL_RADIUS / MIN_RADIUS_PX
```

| 分辨率 | fx | MIN_RADIUS_PX=3 | FPS |
|--------|-----|-----------------|-----|
| 1280×720 | ~616 | **6.8 m** | 30 |
| 848×480  | ~424 | **4.7 m** | 60 |
| 640×480  | ~388 | **4.3 m** | 90 |

> 超出范围后球的像素半径小于 3px，圆形拟合不可靠。
> 调小 `MIN_RADIUS_PX`（如 2）可进一步延伸，但误检率会上升。

---

## YOLO 模型选择

| 模型 | 参数量 | mAP | 说明 |
|------|--------|-----|------|
| `yolov8n.pt` | 3M | 37.3 | 默认，最快 |
| `yolov8s.pt` | 11M | 44.9 | |
| `yolo11s.pt` | 9M | 47.0 | 新架构 |
| `yolo11m.pt` | 20M | 51.5 | **推荐** |
| `yolo11x.pt` | 57M | 54.7 | 精度最高 |

模型文件存放于 `models/`（gitignore，首次使用自动下载）。

---

## 坐标系

```
optical frame (RealSense 输出)     camera body frame (输出)
  Z 朝前                       →     X 朝前
  X 朝右                       →     Y 朝左
  Y 朝下                       →     Z 朝上
```

输出字段 `(x, y, z)` 含义：x = 球在相机前方距离，y = 左方距离，z = 上方距离（均为米）。

---

## 架构说明

双线程设计：

- **主线程**：`pipeline.wait_for_frames()`（释放 GIL）+ 引用交换 + `cv2.imshow`
- **YOLO 线程**：帧拷贝 → resize → GPU 推理 → Color→Depth 三步像素映射 → 深度采样 → EMA 平滑 → LCM 发布

`camera_ball.py` 主要改进点（相较旧版）：

- **去掉 `align.process()`**：全图深度对齐耗时 ~50ms 且持有 GIL，改为对球心单点做精确 Color→Depth 映射（<0.1ms）
- **Color→Depth 三步映射**：修正 FOV 差异 + 基线视差，消除旧版最大 ~14cm 横向误差
- **BALL_RADIUS = 0.033m**：网球半径，深度传感器测前表面，加此偏移得球心
- **EMA gate 超限时重置**：旧版静默跳过导致球飞远后 EMA 永久冻结

详见 `doc/camera_ball.md`。

---

## LiDAR 版本

`test-ball.py` — Livox Mid-360 + ROS2，需要 `livox_ros_driver2` 和 `unitree_hg`。

---

## 常见问题

**`RuntimeError: Couldn't resolve requests`**
相机帧率组合不支持，代码自动尝试 60/60→30/30→15/15 Hz 降级。检查 USB 3.0 连接。

**深度始终为 0**
球距离过近（< 0.1m）或过远（> 10m），或高光反射。调整 `DEPTH_MIN/MAX` 或 `DEPTH_SAMPLE_RADIUS`。

**检测不到网球**
降低 `CONF_THRESHOLD`（如 0.2），或换用 `yolo11m.pt`。
