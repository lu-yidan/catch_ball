# 软体臂抓取流水线 (v2 / v3)

坐标系与相机 `soft_arm_center` 一致：**+X 侧向，+Y 向下（主轴），-Z 朝相机**。

## 终端分工

| 终端 | 脚本 | 作用 |
|------|------|------|
| 1 | `run_d435_vision.py` | RealSense + AprilTag，写 `ball_target.json` |
| 2 | `grasp_plan_execute_v3.py` | 规划 → 仿真图 → 输入 `yes` 后执行 |

v2 执行（无仿真确认）：`grasp_plan_execute_v2.py`

## 推荐命令 (v3)

```bash
conda activate catchball

# 终端1
python -u run_d435_vision.py

# 终端2（球位从 ball_target.json 读取，非写死）
python -u grasp_plan_execute_v3.py --coord-file ball_target.json --port COM5
```

仿真通过后终端输入 `yes` 才会动实机。

## 仅规划 / 仿真

```bash
python -u grasp_plan_execute_v3.py --coord-file ball_target.json --plan-only
python -u simulate_v2_grasp.py --coord-file ball_target.json
python -u check_inverted_kinematics.py
```

## 第三节绳组测试

```bash
python motor1_4_test.py --port COM5
```

## 配置文件

- `config/soft_arm_arm_axes.json` — 视觉→臂系 scale、弯曲平面说明
- `config/center_to_robot.json` — 标定（如有）

## 输出目录（gitignore）

- `ball_target.json` — 实时球心
- `grasp_plan_v3.json` — 规划轨迹
- `output_v3_sim/` — 仿真预览图
