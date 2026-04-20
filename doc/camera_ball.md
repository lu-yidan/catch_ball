# camera_ball.py 详细说明

RealSense D455 + YOLOv8 网球检测，输出球相对相机的三维位置（camera body frame）。

---

## 目录

1. [整体架构](#1-整体架构)
2. [双线程模型](#2-双线程模型)
3. [Color→Depth 像素映射](#3-colordepth-像素映射)
4. [YOLO 检测与追踪](#4-yolo-检测与追踪)
5. [Coasting（惯性保持）](#5-coasting惯性保持)
6. [深度采样与球心偏移](#6-深度采样与球心偏移)
7. [坐标变换](#7-坐标变换)
8. [EMA 位置滤波](#8-ema-位置滤波)
9. [实时性保障](#9-实时性保障)
10. [相机启动与硬件重置](#10-相机启动与硬件重置)
11. [LCM 发布](#11-lcm-发布)
12. [可视化](#12-可视化)
13. [关键参数速查](#13-关键参数速查)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│ 主线程                                                        │
│  pipeline.wait_for_frames()  ← 释放 GIL，阻塞等新帧          │
│  → buf_frames = frames        (引用交换，无拷贝)              │
│  → buf_updated.set()                                         │
│  → cv2.imshow(disp_frame)    ← 显示 YOLO 线程标注好的帧     │
└──────────────────────┬──────────────────────────────────────┘
                       │ buf_updated Event
┌──────────────────────▼──────────────────────────────────────┐
│ YOLO 线程（daemon）                                           │
│  buf_updated.wait()                                          │
│  → color.copy() / depth_arr.copy()                          │
│  → cv2.resize → model.track()    ← GPU 推理                  │
│  → Color→Depth 三步像素映射                                   │
│  → patch 中位数深度 + BALL_RADIUS → depth_m                  │
│  → rs2_deproject → optical_to_body → EMA                    │
│  → LCM 发布 (可选)                                           │
│  → 标注 vis 帧 → disp_frame[0] = vis                        │
└─────────────────────────────────────────────────────────────┘
```

主线程只做 `pipeline.wait_for_frames()`（C 扩展，释放 GIL）+ 引用交换 + `cv2.imshow`，YOLO 线程在 GIL 空隙中并行运行。

---

## 2. 双线程模型

### 帧缓冲（主线程写）

```python
frames = pipeline.wait_for_frames()   # 释放 GIL，~16ms
with buf_lock:
    buf_frames = frames                # 仅交换引用，无拷贝
buf_updated.set()
```

`pipeline.wait_for_frames()` 在 C 层阻塞并释放 GIL，YOLO 线程可以在此期间并行运行。

### YOLO 线程消费

```python
buf_updated.wait(timeout=1.0)
buf_updated.clear()
with buf_lock:
    frames = buf_frames

cf = frames.get_color_frame()
df = frames.get_depth_frame()
color     = np.asanyarray(cf.get_data()).copy()   # 必须拷贝，YOLO 需要 numpy 数组
depth_arr = np.asanyarray(df.get_data()).copy()   # 拷贝到堆，避免 DMA 读取慢
```

`buf_updated` 是 `threading.Event`，主线程在 YOLO 处理期间产生的多帧会被自动丢弃，只保留最新帧。

### 显示帧传递（YOLO 线程→主线程）

```python
# YOLO 线程写
with disp_lock:
    disp_frame[0] = vis

# 主线程读
with disp_lock:
    frame = disp_frame[0]
if frame is not None:
    cv2.imshow("camera_ball", frame)
```

---

## 3. Color→Depth 像素映射

D455 的 color 和 depth 传感器**不共光心**，有不同 FOV 和约 −14.5mm 基线。直接用 color 像素坐标索引 `depth_arr` 会导致最大约 14–18 cm 横向误差（图像边缘处深度采样到背景）。

### 三步映射

```
Color 像素 (cx, cy)
        │
        │ Step 1 — 用 color 内参归一化，得到方向向量
        │   ndcx = (cx − ppx_color) / fx_color
        │   ndcy = (cy − ppy_color) / fy_color
        ▼
归一化方向 (ndcx, ndcy)
        │
        │ Step 2a — 加入 depth 内参，修正 FOV 差异
        │   dx0 = ndcx × fx_depth + ppx_depth
        │   dy0 = ndcy × fy_depth + ppy_depth
        │
        │ Step 2b — 读粗略深度，修正基线视差
        │   Z_coarse = depth_arr[dy0, dx0] × depth_scale  (或默认 1.0m)
        │   dx = dx0 + tx / Z_coarse × fx_depth   (tx ≈ −0.0145m)
        │   dy = dy0 + ty / Z_coarse × fy_depth
        ▼
深度图像素 (dx, dy)
        │
        │ Step 3 — numpy patch 采样（11×11，中位数）
        ▼
depth_surface（球前表面深度）
```

反投影时仍用 **color 内参 + color 像素**（`cx, cy`），保证 3D 点在 color 相机坐标系下。

---

## 4. YOLO 检测与追踪

```python
color_small = cv2.resize(color, (args.imgsz, args.imgsz))
results = model.track(color_small, conf=CONF_THRESHOLD,
                      persist=True, verbose=False,
                      device=device, half=half)
```

- 先缩放到 `imgsz×imgsz` 再送 YOLO，检测框坐标按比例映射回原图
- `model.track(persist=True)` 启用 ByteTrack 跨帧追踪，减少运动模糊下的漏检
- 检测 COCO class 32（sports ball，含网球）
- 取置信度最高的一个检测结果

---

## 5. Coasting（惯性保持）

YOLO 漏检时（运动模糊、遮挡、光照），沿用上一帧 bbox 继续采深度和计算坐标：

```
检测到 → [BALL]   绿色框，miss_count=0
漏检   → [COAST]  橙色框，沿用 last_bbox，miss_count++
漏检 >COAST_FRAMES → [    ]  清空，显示 "No ball"
```

Coasting 期间从 `last_bbox` 位置重新采样深度，EMA 跳变门限会过滤掉深度明显跑偏的情况。

---

## 6. 深度采样与球心偏移

```python
patch   = depth_arr[dy-r:dy+r+1, dx-r:dx+r+1].astype(np.float32) * depth_scale
valid_d = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
depth_surface = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0

# 深度传感器测到的是球的前表面，加半径得到球心深度
depth_m = depth_surface + BALL_RADIUS if depth_surface > 0 else 0.0
```

`BALL_RADIUS = 0.033m`（网球直径 6.54–6.86cm，半径约 3.3cm）。

中位数比均值对边缘飞点和高光反射无效点更鲁棒。

---

## 7. 坐标变换

```
rs2_deproject_pixel_to_point(color_intrin, [cx, cy], depth_m)
    → 光学坐标系 (Z前, X右, Y下)
        ↓ optical_to_body()
    → camera body 坐标系 (X前, Y左, Z上)  ← 当前输出
```

`optical_to_body`：
```
X_body =  Z_optical
Y_body = -X_optical
Z_body = -Y_optical
```

**当前只输出 camera body frame，不做运动学链变换到 pelvis 系。**

---

## 8. EMA 位置滤波

在 camera body 坐标系对位置做指数滑动平均：

```python
gate_dist = np.linalg.norm(p_cam_arr - center_ema)
if gate_dist < EMA_GATE:
    center_ema = EMA_ALPHA * p_cam_arr + (1 - EMA_ALPHA) * center_ema
else:
    # 跳变太大 → 重置 EMA（不再冻结在旧位置）
    print(f"[WARN] EMA gate {gate_dist:.2f}m > {EMA_GATE}m, resetting")
    center_ema = p_cam_arr.copy()
```

`EMA_ALPHA = 0.6`：新值占 60%，历史值占 40%。

gate 超限时**重置**而非跳过，避免球大幅移动后 EMA 永久冻结。

---

## 9. 实时性保障

YOLO 线程启动时尝试：

1. `SCHED_FIFO` 实时调度（需要 `CAP_SYS_NICE` 或 root）
2. `nice(-10)` 降低调度优先级

主线程 `pipeline.wait_for_frames()` 在 C 层释放 GIL，两线程真正并行运行。

---

## 10. 相机启动与硬件重置

依次尝试 (60/60, 30/30, 15/15) Hz 组合，避免 `RuntimeError: Couldn't resolve requests`：

```python
_FPS_TRIES = [(60, 60), (30, 30), (15, 15)]
```

每个组合最多尝试两次，超时则触发 `hardware_reset()`（等效拔插相机），3 秒后重试。

---

## 11. LCM 发布

可选依赖，无 `lcm` / `lcm_types` 时自动跳过。

```
频道: camera_ball_lcmt
字段: offset_time (μs), x, y, z  ← camera body frame (X前, Y左, Z上)
```

---

## 12. 可视化

`--no-viz` 关闭 OpenCV 窗口（适合无显示器部署）：

```
[BALL ] cam=(+0.821, +0.012, -0.003)  surf=0.79m ctr=0.82m  YOLO=35.2fps
```

| 状态 | 含义 |
|------|------|
| `BALL` | 本帧 YOLO 检测到球 |
| `COAST` | 漏检，沿用上帧（橙色框） |
| `(空)` | 超过 COAST_FRAMES 帧未检测到 |

---

## 13. 关键参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONF_THRESHOLD` | 0.3 | YOLO 置信度阈值 |
| `DEPTH_SAMPLE_RADIUS` | 5 px | 深度采样 patch 半径（11×11 点） |
| `DEPTH_MIN / MAX` | 0.1 / 10.0 m | 有效深度范围 |
| `BALL_RADIUS` | 0.033 m | 网球半径，前表面深度 + 此值 = 球心深度 |
| `EMA_ALPHA` | 0.6 | 平滑系数（越大跟踪越快） |
| `EMA_GATE` | 0.6 m | 超出则重置 EMA |
| `COAST_FRAMES` | 10 帧 | 漏检保持帧数 |
| `--model` | yolov8n.pt | YOLO 模型路径 |
| `--imgsz` | 480 | YOLO 推理输入边长 |
| `--width/height` | 640/480 | 相机采集分辨率 |
| `--no-viz` | False | 关闭 OpenCV 窗口 |
| `--no-ema` | False | 关闭 EMA，直接用原始测量值 |
