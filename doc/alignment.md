# RGB 像素 → Depth 像素 对应原理

## 问题根源：两个独立传感器

D435 上有两个物理上分开的传感器：

```
   [RGB 摄像头]          [深度传感器（双目红外）]
        |                        |
   自己的光心                自己的光心
   自己的焦距                自己的焦距
   自己的畸变                自己的畸变
        |_______  物理间距  _____|
                 （baseline）
```

两者看的是同一个世界，但**视角不同、分辨率不同、内参不同**，
所以同一个物体在两张图上的像素坐标不一样。
直接用 RGB 检测到的像素坐标去读 Depth 图，拿到的是错误位置的深度。

---

## 坐标系转换链

完整的对应关系需要经过三步：

```
深度像素 (u', v')  +  深度值 d
        ↓  1. 反投影（Depth 内参）
   3D 点（Depth 相机坐标系）
        ↓  2. 外参变换 R, T（Depth → RGB）
   3D 点（RGB 相机坐标系）
        ↓  3. 投影（RGB 内参）
   RGB 像素 (u, v)
```

### 第 1 步：Depth 像素 → 3D 点

针孔相机反投影公式：

```
X_d = (u' - ppx_d) × d / fx_d
Y_d = (v' - ppy_d) × d / fy_d
Z_d = d
```

`fx_d, fy_d, ppx_d, ppy_d` 是 Depth 传感器的内参（焦距、主点）。

### 第 2 步：Depth 坐标系 → RGB 坐标系

刚体变换（旋转 + 平移）：

```
P_rgb = R × P_depth + T
```

- `R`：3×3 旋转矩阵（两传感器间的朝向差）
- `T`：3×1 平移向量（两传感器的物理间距，单位米）

### 第 3 步：3D 点 → RGB 像素

投影公式：

```
u = fx_rgb × (X_rgb / Z_rgb) + ppx_rgb
v = fy_rgb × (Y_rgb / Z_rgb) + ppy_rgb
```

---

## RealSense SDK 的实现方式

`rs.align(rs.stream.color)` 将上面三步全部封装，对原始 Depth 图做**图像重采样**：

```
原始 Depth 图（Depth 传感器视角）
        ↓
对每一个 Depth 像素，走完三步转换，算出对应的 RGB 像素位置
        ↓
将深度值"搬"到 RGB 像素坐标上
        ↓
对齐后的 Depth 图（与 RGB 图尺寸、视角完全一致）
```

对齐后：

```python
aligned_depth_frame[v][u]  ←→  color_frame[v][u]
        ↑                              ↑
    同一个空间点，同一个像素坐标
```

---

## 内参 / 外参从哪里来

D435 **出厂标定**，参数存储在相机固件中，SDK 直接读取，无需用户手动标定：

```python
# Color 内参
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()
# intrinsics.fx, fy    焦距（像素）
# intrinsics.ppx, ppy  主点（像素）
# intrinsics.coeffs    畸变系数

# Depth → Color 外参（R, T）
depth_profile = profile.get_stream(rs.stream.depth)
extrinsics = depth_profile.get_extrinsics_to(color_profile)
# extrinsics.rotation      → [float × 9]，3×3 行优先
# extrinsics.translation   → [float × 3]，单位：米
```

`align` 和 `rs2_deproject_pixel_to_point` 内部都使用了这些参数。

---

## 为什么对齐后边缘会有黑边（深度为 0）

RGB 和 Depth 传感器的视野角（FOV）不同：

```
RGB  水平 FOV ≈ 69°   ████████████████
Depth 水平 FOV ≈ 87°  ██████████████████████
```

对齐到 RGB 视角后，Depth 原本覆盖但 RGB 覆盖不到的区域（边缘）无法反向映射，
填充为 0（无效深度）。反过来，RGB 视角内 Depth 盲区也会出现 0。

---

## 反投影：对齐深度图 + RGB 像素 → 3D 坐标

对齐完成后，用 **RGB 内参**（不是 Depth 内参）做反投影：

```python
# 使用对齐后的深度 + RGB 内参
point = rs.rs2_deproject_pixel_to_point(
    intrinsics,   # Color 相机内参
    [cx, cy],     # RGB 图上的像素坐标
    depth_m       # 对齐后该像素的深度（米）
)
# 返回 [X, Y, Z]，在 RGB 相机坐标系下，单位：米
```

此时 3D 坐标已经是 RGB 相机坐标系，X 向右、Y 向下、Z 朝前。
