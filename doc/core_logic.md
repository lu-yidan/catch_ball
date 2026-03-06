# 核心逻辑说明

> 本文件是旧版简述，完整的逐模块说明请见 [camera_ball.md](./camera_ball.md)。

## 整体流程（简版）

```
RealSense D435
  ├── Color Frame ──→ YOLO tracking ──→ BBox 中心 (cx, cy)
  └── Depth Frame ──→ align 对齐 ──→ 深度 patch 中位数 depth_m
                                              ↓
                          rs2_deproject_pixel_to_point()
                                              ↓
                            光学系 (Z前) → optical_to_body() → body 系 (X前)
                                              ↓
                          transform_point_camera_to_base()
                                              ↓
                                  base/pelvis 坐标 (X,Y,Z) 米
```

## 各模块速查

| 模块 | 文档位置 |
|---|---|
| RGB↔Depth 对齐原理 | [alignment.md](./alignment.md) |
| 双线程架构 | [camera_ball.md § 3](./camera_ball.md#3-双线程模型) |
| YOLO 追踪 + coasting | [camera_ball.md § 4–5](./camera_ball.md#4-yolo-检测与追踪) |
| 深度采样 | [camera_ball.md § 6](./camera_ball.md#6-深度采样) |
| 坐标变换链 | [camera_ball.md § 7](./camera_ball.md#7-坐标变换链) |
| EMA 滤波 | [camera_ball.md § 8](./camera_ball.md#8-ema-位置滤波) |
| 速度外推 | [camera_ball.md § 9](./camera_ball.md#9-速度估计与显示外推) |
| 实时性保障 | [camera_ball.md § 10](./camera_ball.md#10-实时性保障) |
| 相机硬件重置 | [camera_ball.md § 11](./camera_ball.md#11-相机硬件重置) |
| 关键参数速查 | [camera_ball.md § 15](./camera_ball.md#15-关键参数速查) |
