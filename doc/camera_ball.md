# camera_ball.py 详细说明

本文档覆盖 `camera_ball.py` 的全部机制，包括整体架构、每个模块的设计决策和参数含义。

---

## 目录

1. [整体架构](#1-整体架构)
2. [启动流程](#2-启动流程)
3. [双线程模型](#3-双线程模型)
4. [YOLO 检测与追踪](#4-yolo-检测与追踪)
5. [Coasting（惯性保持）](#5-coasting惯性保持)
6. [深度采样](#6-深度采样)
7. [坐标变换链](#7-坐标变换链)
8. [EMA 位置滤波](#8-ema-位置滤波)
9. [速度估计与显示外推](#9-速度估计与显示外推)
10. [实时性保障](#10-实时性保障)
11. [相机硬件重置](#11-相机硬件重置)
12. [关节角来源](#12-关节角来源)
13. [LCM 发布](#13-lcm-发布)
14. [可视化](#14-可视化)
15. [关键参数速查](#15-关键参数速查)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│ 主线程                                                        │
│  RealSense pipeline.wait_for_frames()  ← 阻塞，最多 33ms    │
│  align.process()                                             │
│  → 写 buf_color / buf_depth                                  │
│  → buf_updated.set()                                         │
│  → 读 res_* 叠加到画面                                       │
│  → cv2.imshow()                         ← 始终 30fps        │
└──────────────────────┬──────────────────────────────────────┘
                       │ buf_updated Event
┌──────────────────────▼──────────────────────────────────────┐
│ YOLO 线程（daemon）                                           │
│  buf_updated.wait()                                          │
│  → 拷贝最新帧                                                │
│  → model.track()                        ← 独立速度           │
│  → 深度采样 → 坐标变换 → EMA → 速度估计                      │
│  → 写 res_bbox / res_cam_str / res_base_str / res_vel_*      │
│  → LCM 发布                                                  │
└─────────────────────────────────────────────────────────────┘
```

两线程通过两把 `threading.Lock` 和一个 `threading.Event` 交互：

| 对象 | 方向 | 保护内容 |
|---|---|---|
| `buf_lock` | 主→YOLO | `buf_color`, `buf_depth` |
| `buf_updated` | 主→YOLO | 新帧到达信号 |
| `res_lock` | YOLO→主 | 所有 `res_*` 结果变量 |
| `stop_flag` | 主→YOLO | 退出信号 |

**为什么要双线程？**

RealSense 采集（`wait_for_frames`）和 YOLO 推理串行时，显示帧率 = min(相机帧率, YOLO帧率)。以 YOLO 18ms/帧计算，显示最多 ~55fps，但实际还要加上 align、深度采样等开销，通常降至 20fps 以下。双线程后主线程只做采集和显示，始终跑满相机帧率（30/60fps），YOLO 独立运行不影响画面流畅度。

---

## 2. 启动流程

```
argparse 解析参数
    ↓
关节角来源确认（ROS2 / 命令行 / 默认值）
    ↓
LCM 初始化（可选）
    ↓
YOLO 加载 → 热身推理 → 打印 ms/frame
    ↓
RealSense pipeline 启动（含超时自动重置）
    ↓
获取相机内参
    ↓
启动 YOLO 后台线程
    ↓
主循环（采集 + 显示）
```

### 热身推理

```python
model(dummy, verbose=False)          # 第一次含 JIT 编译，不计入
t0 = time.perf_counter()
for _ in range(5):
    model(dummy, verbose=False)
ms_per_frame = (time.perf_counter() - t0) / 5 * 1000
```

用全零图像跑 5 次取平均，反映当前机器上该模型的真实推理延迟（含预处理）。这个数值决定了 YOLO 线程的实际帧率上限。

---

## 3. 双线程模型

### 帧缓冲（主线程写）

```python
with buf_lock:
    buf_color = color_image.copy()   # 深拷贝，避免数据竞争
    buf_depth = depth_array          # uint16 numpy array，直接赋值
buf_updated.set()                    # 通知 YOLO 线程有新帧
```

注意：`depth_array` 是 `np.asanyarray(df.get_data())` 的结果，是对 RealSense 帧内存的 **零拷贝视图**。YOLO 线程会在 `with buf_lock` 内部再做一次 `.copy()`，所以主线程的 `depth_array` 赋值本身不需要拷贝。

### YOLO 线程消费

```python
buf_updated.wait(timeout=1.0)   # 阻塞等待，1s 超时防止死锁
buf_updated.clear()             # 重置 Event，确保每帧只处理一次
with buf_lock:
    color = buf_color.copy()
    depth = buf_depth.copy()
```

`buf_updated` 是 `threading.Event`（非 Semaphore），因此如果主线程在 YOLO 处理期间产生了多帧，YOLO 只会取最新的一帧处理，旧帧自动丢弃——这正是我们想要的"最新帧优先"行为。

### 结果写回

```python
with res_lock:
    res_bbox     = ...
    res_cam_str  = ...
    res_base_str = ...
    res_detected = ...
    res_coasting = ...
    res_yolo_fps = ...
    res_vel_t    = t_infer    # YOLO 开始推理的时间戳
    res_vel_vx   = vx         # 像素/秒
    res_vel_vy   = vy
```

所有结果在一个 `with res_lock` 块内原子写入，主线程读取时同样加锁，避免读到半更新状态。

---

## 4. YOLO 检测与追踪

### model.track() vs model()

| | `model()` | `model.track()` |
|---|---|---|
| 帧间关联 | 无 | ByteTrack 追踪器 |
| 遮挡处理 | 每帧独立 | 卡尔曼滤波预测 |
| 运动模糊 | 容易丢失 | ID 保持，降低漏检 |
| 计算开销 | 低 | 略高（追踪状态维护）|

```python
results = model.track(
    color,
    conf=CONF_THRESHOLD,   # 置信度阈值
    persist=True,          # 跨帧保持追踪状态（必须）
    verbose=False,
    imgsz=args.imgsz,      # 推理输入尺寸
)
```

`persist=True` 告诉 YOLO 在连续调用之间保留追踪器内部状态。如果不设置，每帧都会重置追踪器，等同于没有跨帧关联。

### imgsz 参数

YOLO 在推理前会将输入图像 resize 到 `imgsz × imgsz`（或最长边为 imgsz 的等比例尺寸）。相机输出是 848×480，实际关系：

| imgsz | 推理分辨率 | 速度 | 小目标精度 |
|---|---|---|---|
| 640 | 640×360 | 慢 | 最好 |
| 480 | 480×272 | 中（默认） | 好 |
| 320 | 320×192 | 最快 | 一般 |

对于足球检测（目标通常不极小），480 是精度与速度的合理平衡点。

### 球的筛选

COCO 数据集 class_id 32 = `sports ball`，包含足球、篮球等圆形球类。取所有检测结果中置信度最高的一个：

```python
for box in result.boxes:
    if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
        c = float(box.conf[0])
        if c > best_conf:
            best_conf, best_box = c, box
```

---

## 5. Coasting（惯性保持）

### 问题

YOLO 在以下情况会漏检（`best_box = None`）：
- 运动模糊（快速移动）
- 部分遮挡
- 镭射/反光表面导致检测置信度低于阈值
- 光照突变

若漏检时立即清空位置，机器人控制器会收到位置跳变或缺失，影响运动规划。

### 实现

```
COAST_FRAMES = 10    # 允许连续漏检的最大帧数
```

```python
if best_box is not None:
    miss_count = 0
    last_bbox  = tuple(map(int, best_box.xyxy[0]))
else:
    miss_count += 1

# 只要 miss_count 未超限，继续用 last_bbox 取深度和坐标
if last_bbox is not None and miss_count <= COAST_FRAMES:
    # ... 正常处理 ...
    coasting = (best_box is None)
```

状态机：

```
检测到 → [BALL]   miss_count=0，绿色框
漏检1帧 → [COAST] miss_count=1，橙色框，位置沿用上帧
漏检N帧 → [COAST] miss_count=N，橙色框
漏检>10帧 → [    ] 清空，显示 "No ball"
```

Coasting 期间深度仍然从 `last_bbox` 的像素位置重新采样，所以如果球停在原地只是 YOLO 漏检，坐标仍然准确。如果球已经移动，深度会错误（取到背景），EMA 的跳变门限会拒绝这种异常值。

---

## 6. 深度采样

### 为什么不直接取单点

D435 深度图存在以下噪声：
- **飞点（Flying Pixels）**：物体边缘深度值不稳定，介于前景和背景之间
- **高光反射**：镭射足球反光区域深度无效（返回 0）
- **随机噪声**：单点深度抖动约 ±3mm（1m 处）

### 实现

从 uint16 depth numpy array 中取 BBox 中心附近的 patch：

```python
patch   = depth[y0d:y1d+1, x0d:x1d+1].astype(np.float32) * 0.001  # mm→m
valid   = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
depth_m = float(np.median(valid)) if len(valid) > 0 else 0.0
```

中位数比均值对异常值更鲁棒：即使 patch 里有 30% 的无效点（0 或反光导致的异常值），中位数仍然正确。

`DEPTH_SAMPLE_RADIUS = 5` 意味着采样区域是 11×11 = 121 个像素点。

### 为什么在 YOLO 线程中直接操作 numpy array

原来的实现用 `depth_frame.get_distance(x, y)` 逐点访问 RealSense 帧对象。双线程后 RealSense 帧对象不能安全地跨线程访问（帧对象有内部引用计数，可能被相机线程回收）。改为在主线程拷贝 uint16 numpy array，YOLO 线程只操作 numpy array，避免了跨线程访问 SDK 对象的风险。

---

## 7. 坐标变换链

详见 [alignment.md](./alignment.md)。简述如下：

```
rs2_deproject_pixel_to_point()
    → 光学坐标系 (Z前, X右, Y下)
        ↓ optical_to_body()
    → 相机 body 坐标系 (X前, Y左, Z上)
        ↓ transform_point_camera_to_base()
    → pelvis/base 坐标系 (X前, Y左, Z上)
```

`transform_point_camera_to_base` 的运动学链：

```
pelvis
  └─ waist_yaw_joint   Rz(q_wy),   t=[0, 0, 0]
      └─ waist_roll_joint  Rx(q_wr),  t=[-0.0039635, 0, 0.044]
          └─ waist_pitch_joint  Ry(q_wp),  t=[0, 0, 0]
              └─ head_joint   Ry(q_head),  t=[0.0039635, 0, 0.3159]
                  └─ head_camera_joint (固定)
                         xyz=[0.0448, 0.01, 0.1219]
                         rpy=[0.0119, 0.8377, 0.0053]
```

关节角在每次调用时实时传入，支持机器人运动过程中的动态更新。

---

## 8. EMA 位置滤波

指数滑动平均（Exponential Moving Average）在相机 body 坐标系下对位置做平滑：

```python
if center_ema is None:
    center_ema = p_cam_arr.copy()           # 冷启动
elif np.linalg.norm(p_cam_arr - center_ema) < EMA_GATE:
    center_ema = EMA_ALPHA * p_cam_arr + (1 - EMA_ALPHA) * center_ema
```

### EMA_ALPHA = 0.6

- 越接近 1：跟踪越快，但噪声抑制越弱
- 越接近 0：越平滑，但响应越慢（滞后越大）
- 0.6 表示新值占 60%，历史值占 40%

### EMA_GATE = 0.6 m

跳变门限：如果新测量值与当前 EMA 相差超过 0.6m，则**拒绝**该测量值，不更新 EMA。

设计目的：防止以下情况污染平滑结果：
1. 深度突然无效导致反投影错误
2. YOLO 误检到远处另一个球
3. Coasting 期间 bbox 位置已不准确导致深度错误

如果球做正常运动，0.6m/帧 的跳变阈值通常足够（球速 > 18m/s 时才会触发，大部分场景不会发生）。

**为什么在相机系做 EMA 而不是 base 系？**

相机系是传感器直接测量的坐标系，噪声特性稳定（深度噪声是 Z 向的，各向异性）。base 系经过运动学链变换后，噪声特性更复杂，不适合直接滤波。

---

## 9. 速度估计与显示外推

### 问题：显示滞后

双线程架构中，主线程显示的 bbox 来自 YOLO 上一次推理的结果。设 YOLO 推理耗时 T_yolo，相机帧间隔 T_cam，则显示时 bbox 已经"过时"了 T_yolo + T_cam 左右。

对于 18ms 推理 + 33ms 帧间隔，bbox 最多落后约 50ms。球速 3m/s 时，像素位移约 15px，主观上框明显跟不上球。

### 速度估计

在 YOLO 线程中，记录相邻两次成功检测的中心点和时间戳：

```python
cx_now, cy_now = bbox中心
now = time.perf_counter()

if prev_cx is not None and (now - prev_t) > 0:
    dt = now - prev_t
    vx = (cx_now - prev_cx) / dt    # 像素/秒
    vy = (cy_now - prev_cy) / dt
prev_cx, prev_cy, prev_t = cx_now, cy_now, now
```

漏检时 `vx, vy = 0.0, 0.0`（停止外推，保持最后位置）。

### 显示外推

主线程每帧从 `res_vel_t`（YOLO 推理时刻）和 `res_vel_vx/vy` 读取速度，计算当前时刻应在的位置：

```python
dt = time.perf_counter() - vel_t    # 距上次 YOLO 结果的时间
dx = int(vx * dt)
dy = int(vy * dt)
x1, y1, x2, y2 = x1+dx, y1+dy, x2+dx, y2+dy
```

**局限性**：这是线性外推，不考虑加速度。球做弧线运动或突然变向时外推会有误差，但对于短时间窗口（<50ms）仍比不外推好得多。

---

## 10. 实时性保障

### 三层策略（优先级从高到低）

**第一层：SCHED_FIFO 实时调度**

```python
param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO) - 1)
os.sched_setscheduler(0, os.SCHED_FIFO, param)
```

Linux 实时调度策略，YOLO 线程不会被普通进程抢占。需要 root 权限或 `CAP_SYS_NICE` capability：

```bash
sudo python camera_ball.py
# 或授予 capability（不需要完整 sudo）：
sudo setcap cap_sys_nice+ep $(which python)
```

**第二层：nice 值降低**（普通用户可用）

```python
os.nice(-10)
```

Linux nice 值范围 -20（最高优先级）到 +19（最低）。普通用户可以将自己的进程降低到 -20，但无法超过 0 提升到负值（需要 root）。实际上 `nice(-10)` 在普通用户下通常会失败，建议用：

```bash
nice -n -10 python camera_ball.py   # shell 层面设置
```

**第三层：CPU 亲和性绑定**

```python
n_cpu = os.cpu_count() or 2
os.sched_setaffinity(0, set(range(max(0, n_cpu - 2), n_cpu)))
```

将 YOLO 线程绑定到最后 2 个 CPU 核心（如 4 核机器绑定到 core 2, 3）。主线程默认不绑定（使用全部核心），YOLO 线程有专属核心不与相机线程竞争调度。

### imgsz 对延迟的影响

| imgsz | 典型推理时间 | YOLO 线程帧率 | 显示外推误差 |
|---|---|---|---|
| 640 | ~25ms | ~40fps | ~25ms |
| 480 | ~15ms | ~65fps | ~15ms |
| 320 | ~8ms | ~120fps | ~8ms |

```bash
python camera_ball.py --imgsz 320   # 最低延迟，适合高速球
python camera_ball.py --imgsz 480   # 默认，平衡
python camera_ball.py --imgsz 640   # 最高精度
```

---

## 11. 相机硬件重置

Ctrl+C 强制终止时，RealSense pipeline 可能未正常关闭，相机硬件仍处于"流传输中"状态，下次启动时 `wait_for_frames` 超时。

```python
def _start_pipeline():
    for attempt in range(2):
        profile = pipeline.start(cfg)
        try:
            pipeline.wait_for_frames(timeout_ms=5000)
            return profile                          # 成功
        except RuntimeError:
            pipeline.stop()
            ctx = rs.context()
            ctx.query_devices()[0].hardware_reset() # 硬件复位
            time.sleep(3)                           # 等待相机重新枚举
    raise RuntimeError("RealSense failed to start after hardware reset.")
```

`hardware_reset()` 等效于物理上拔插相机（USB 重新枚举），3秒后相机重新可用。最多重试 2 次，避免无限重试。

---

## 12. 关节角来源

按优先级：

| 优先级 | 来源 | 条件 |
|---|---|---|
| 1 | ROS2 `/lowstate` 话题 | `rclpy` 和 `unitree_hg` 均可 import |
| 2 | 命令行参数 | `--q-wy`, `--q-wr`, `--q-wp`, `--q-head` |
| 3 | 硬编码默认值 | `q_wy=0, q_wr=0, q_wp=0, q_head=0.593412` |

ROS2 可用时，`_JointListener` 节点在独立线程中运行 `rclpy.spin()`，回调更新关节角。YOLO 线程每帧直接读 `joint.q_wy` 等属性（Python GIL 保护，float 赋值是原子操作，无需额外加锁）。

`q_head=0.593412 rad ≈ 34°` 是头部关节的默认俯仰角，对应 Unitree G1 标准站立姿态。

---

## 13. LCM 发布

```python
lc.publish("camera_ball_lcmt", lcm_msg.encode())
```

消息格式复用 `lidar_lcmt`（与雷达版 `test-ball.py` 兼容）：

| 字段 | 含义 |
|---|---|
| `offset_time` | Unix 时间戳（微秒） |
| `x` | base 系 X 坐标（米，向前） |
| `y` | base 系 Y 坐标（米，向左） |
| `z` | base 系 Z 坐标（米，向上） |

仅在深度有效时发布（`depth_m > 0`）。Coasting 期间继续发布最后的 EMA 位置。

---

## 14. 可视化

### 画面元素

```
┌─────────────────────────────────────────────┐
│ CAM 30.0  YOLO 18.5 fps          (左上，青色) │
│                                              │
│  Ball cam (X=+0.821, Y=+0.012, Z=+0.003)    │ ← 第一行，绿色
│       base(+1.023, -0.045, +1.234) 0.87     │ ← 第二行，绿色
│  ┌──────────────┐                            │
│  │              │  ← 绿框（检测到）           │
│  │      ●       │  ← 红点（中心）             │
│  └──────────────┘                            │
└─────────────────────────────────────────────┘
```

### 颜色含义

| 颜色 | 状态 |
|---|---|
| 绿色框 | `[BALL]` YOLO 当前帧检测到 |
| 橙色框 | `[COAST]` YOLO 漏检，沿用上帧位置 |
| 无框 + "No ball" | 连续漏检超过 `COAST_FRAMES` 帧 |

### bbox 位置说明

显示的框已经过**速度外推**（见第 9 节），位置比 YOLO 原始结果更接近球的当前实际位置。显示的坐标文字（cam/base）是 EMA 滤波后的 3D 坐标，未做外推（外推只在像素空间做）。

---

## 15. 关键参数速查

| 参数 | 位置 | 默认值 | 说明 |
|---|---|---|---|
| `CONF_THRESHOLD` | 顶部常量 | 0.3 | YOLO 置信度阈值，越低召回越高但误检越多 |
| `DEPTH_SAMPLE_RADIUS` | 顶部常量 | 5 px | 深度采样半径 |
| `DEPTH_MIN` | 顶部常量 | 0.1 m | 深度有效下限 |
| `DEPTH_MAX` | 顶部常量 | 10.0 m | 深度有效上限 |
| `EMA_ALPHA` | 顶部常量 | 0.6 | EMA 平滑系数，越大越跟当前值 |
| `EMA_GATE` | 顶部常量 | 0.6 m | 跳变门限，超过则拒绝新测量 |
| `COAST_FRAMES` | 顶部常量 | 10 帧 | 漏检后保持位置的最大帧数 |
| `--model` | 命令行 | yolov8n.pt | YOLO 模型文件 |
| `--imgsz` | 命令行 | 480 | YOLO 推理输入尺寸 |
| `--no-viz` | 命令行 | False | 关闭 OpenCV 窗口 |
| `--q-head` | 命令行 | 0.593412 | 头关节角（rad），无 ROS2 时使用 |
