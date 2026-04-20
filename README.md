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

## 运行

```bash
conda activate catchball
python camera_ball.py              # 带 OpenCV 可视化
python camera_ball.py --no-viz     # 无窗口（headless）
python camera_ball.py --no-ema     # 关闭 EMA 平滑
python camera_ball.py --model yolo11m.pt --imgsz 480
```

首次运行自动下载 `yolov8n.pt`（约 6MB）。按 `q` 或 `Ctrl+C` 退出。

正常输出示例：

```
[BALL ] cam=(+0.821, +0.012, -0.003)  surf=0.79m ctr=0.82m  YOLO=35.2fps
[COAST] cam=(+0.820, +0.011, -0.003)  surf=0.79m ctr=0.82m  YOLO=35.2fps
[     ] no ball  YOLO=35.2fps
```

| 状态 | 含义 |
|------|------|
| `BALL` | 本帧 YOLO 检测到球 |
| `COAST` | 漏检，沿用上帧位置（最多 10 帧） |
| `(空)` | 超过 10 帧未检测到 |

---

## 模型选择

| 模型 | 参数量 | mAP | 说明 |
|------|--------|-----|------|
| `yolov8n.pt` | 3M | 37.3 | 默认，最快 |
| `yolov8s.pt` | 11M | 44.9 | |
| `yolo11s.pt` | 9M | 47.0 | 新架构 |
| `yolo11m.pt` | 20M | 51.5 | **推荐** |
| `yolo11x.pt` | 57M | 54.7 | 精度最高 |

`*.pt` 文件已加入 `.gitignore`，首次使用自动下载。

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
