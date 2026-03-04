# 核心逻辑说明

## 整体流程

```
RealSense D435
  ├── Color Frame ──→ YOLO 检测足球 ──→ 获取 BBox 中心像素 (cx, cy)
  └── Depth Frame ──→ align 对齐到 Color ──→ 读取 depth(cx, cy)
                                                      ↓
                                  rs2_deproject_pixel_to_point
                                                      ↓
                                          3D 坐标 (X, Y, Z)  单位：米
```

---

## 各模块说明

### 1. RealSense 初始化

```python
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
```

- 同时开启 Color 和 Depth 两路流
- `z16` 格式：每像素 16-bit 无符号整数，存原始深度计数值
- 实际深度（米）= 计数值 × depth_scale（通常 0.001，即 1mm/unit）

### 2. Depth 对齐到 Color

```python
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
```

对齐后 `aligned_depth_frame[v][u]` 与 `color_frame[v][u]` 对应同一个空间点。
详细原理见 [alignment.md](./alignment.md)。

### 3. YOLO 检测足球

```python
results = model(color_image, conf=CONF_THRESHOLD, verbose=False)
# COCO class_id 32 = "sports ball"
if cls_id == SPORTS_BALL_CLASS_ID and conf > best_conf:
    best_box = box
```

- 使用 YOLOv8 预训练权重，COCO 数据集 class 32 即 `sports ball`
- 多个检测结果时取置信度最高的一个

### 4. 中心点深度采样

```python
def get_depth_at_center(depth_frame, cx, cy, radius=5):
    # 在 (cx, cy) 周围 radius 像素内采集所有有效深度值
    # 返回中位数（米）
```

**为什么不直接取单点？**

足球表面可能有高光反射或遮挡，单点深度容易为 0 或异常值。
取 5px 半径小块的中位数可显著提升稳定性。

有效范围过滤：`DEPTH_MIN=0.1m` ～ `DEPTH_MAX=10.0m`，超出范围的点丢弃。

### 5. 2D 像素 → 3D 坐标（反投影）

```python
point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
# 返回 [X, Y, Z]，单位：米，相机坐标系
```

内部执行针孔相机反投影公式：

```
X = (cx - ppx) * Z / fx
Y = (cy - ppy) * Z / fy
Z = depth_m
```

其中 `fx, fy, ppx, ppy` 是 Color 相机的内参，从固件中读取。

**坐标轴方向（相机坐标系）：**

```
        Z（远离相机）
       /
      /
     O ────── X（向右）
     |
     Y（向下）
```

---

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `CONF_THRESHOLD` | 0.4 | YOLO 置信度阈值，降低可提升召回率 |
| `DEPTH_SAMPLE_RADIUS` | 5 | 深度采样半径（像素） |
| `DEPTH_MIN` | 0.1m | 深度有效下限 |
| `DEPTH_MAX` | 10.0m | 深度有效上限 |
| `model_path` | `yolov8n.pt` | 更换为 `yolov8s/m.pt` 可提升精度 |
