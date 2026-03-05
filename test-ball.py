import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data

from livox_ros_driver2.msg import CustomMsg
from unitree_hg.msg import LowState
from transform.mid360_to_base import transform_point_mid360_to_base
from lcm_types.lidar_lcmt import lidar_lcmt
class BallCenterEstimator(Node):
    def __init__(self):
        super().__init__("ball_center_estimator")

        # ---- params (你之后可以改成 declare_parameter) ----
        self.r = 0.115  # 球半径
        self.reflect_thr = 149 # 反射率阈值：先从 60/80/100 试
        self.min_points = 3   # 候选点数下限
        self.inlier_eps = 0.02 # |dist-r| 允许误差（2cm 起步）
        self.max_range = 2.0  # ROI 距离
        self.min_range = 0.2

        # 高度门限（按你雷达安装高度调）
        self.z_low = -1.5
        self.z_high = 1.5

        self.x_low = -0.0
        self.x_high = 5.0

        # 简单时间滤波（指数滑动平均）
        self.alpha = 0.6
        self.center_ema = None
        self.last_t = None

        # 关节角（默认值，后续可以从 ROS2 话题获取）
        self.q_wy = 0.0   # waist_yaw_joint
        self.q_wr = 0.0   # waist_roll_joint
        self.q_wp = 0.0   # waist_pitch_joint
        self.q_head = 0.593412  # head_joint
        self.q_mid = 0.0  # mid360_joint

        # qos = QoSProfile(
        #     reliability=ReliabilityPolicy.BEST_EFFORT,
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=5
        # )

        self.create_subscription(CustomMsg, "/livox/lidar", self.cb_lidar, 5)
        self.get_logger().info("BallCenterEstimator subscribed to /livox/lidar")

        self.create_subscription(LowState, "/lowstate", self.cb_g1, qos_profile_sensor_data)
        self.get_logger().info("BallCenterEstimator subscribed to /lowstate")


        import lcm
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    
    def cb_g1(self, msg: LowState):
        motor_states = msg.motor_state
        q_list = [m.q for m in motor_states]
        self.q_wy = q_list[12]
        self.q_wr = q_list[13]
        self.q_wp = q_list[14]
        self.get_logger().info(
            f"Waist YRP: ({self.q_wy:.3f}, {self.q_wr:.3f}, {self.q_wp:.3f}) | "
        )



    def cb_lidar(self, msg: CustomMsg):
        t0 = time.time()

        # -------- unpack points --------
        pts = msg.points
        print("Through points:",len(pts))
        if len(pts) == 0:
            return

        xyz = np.empty((len(pts), 3), dtype=np.float32)
        refl = np.empty((len(pts),), dtype=np.int16)

        for i, p in enumerate(pts):
            xyz[i, 0] = p.x
            xyz[i, 1] = p.y
            xyz[i, 2] = p.z
            refl[i] = p.reflectivity

        # -------- ROI + reflectivity filter --------
        d = np.linalg.norm(xyz, axis=1)
        mask = (
            (refl >= self.reflect_thr) &
            (d >= self.min_range) & (d <= self.max_range) &
            (xyz[:, 2] >= self.z_low) & (xyz[:, 2] <= self.z_high)&
            (xyz[:, 0] >= self.x_low) & (xyz[:, 0] <= self.x_high)
        )
        cand = xyz[mask]

        print("point_nums:", cand.shape[0])
        if cand.shape[0] < self.min_points:
            # self.get_logger().info(f"Too few candidates: {cand.shape[0]}")
            return
        

        center = estimate_ball_center_ls(cand, self.r, max_iter=10)

        # -------- temporal smoothing / gating --------
        now = time.time()
        if self.center_ema is None:
            self.center_ema = center
            self.last_t = now
        else:
            # gating: 跳变太大就不更新（你可以改成用KF预测）
            if np.linalg.norm(center - self.center_ema) < 0.6:
                self.center_ema = self.alpha * center + (1.0 - self.alpha) * self.center_ema

        # -------- 坐标转换：mid360 (EMA) -> base --------
        center_base_ema = transform_point_mid360_to_base(
            self.center_ema,
            self.q_wy, self.q_wr, self.q_wp, self.q_head, self.q_mid
        )

        dt_ms = (time.time() - t0) * 1000.0
        cx, cy, cz = self.center_ema.tolist()
        cx_base, cy_base, cz_base = center_base_ema.tolist()
        msg = lidar_lcmt()
        msg.offset_time = int(time.time() * 1e6)
        msg.x = cx_base
        msg.y = cy_base
        msg.z = cz_base
        # publish
        self.lc.publish("lidar_lcmt", msg.encode())
        self.get_logger().info(
            f"Ball center (mid360 frame, EMA): ({cx:.3f}, {cy:.3f}, {cz:.3f}) | "
            f"(base frame, EMA): ({cx_base:.3f}, {cy_base:.3f}, {cz_base:.3f}) "
            f"cand={cand.shape[0]} cost={dt_ms:.1f}ms"
        )

        

def estimate_ball_center_ls(points, r=0.115, max_iter=10):
    """
    points: (N,3) numpy array
    r: known radius
    return: center (3,)
    """

    # 初始值：点云均值
    c = points.mean(axis=0).astype(np.float64)

    for _ in range(max_iter):

        v = points - c[None, :]
        dist = np.linalg.norm(v, axis=1) + 1e-9
        resid = dist - r

        # Jacobian: d(dist)/dc = -(p-c)/dist
        J = -(v / dist[:, None])   # Nx3

        # Solve normal equation
        A = J.T @ J
        b = -J.T @ resid

        # 正则化防止奇异
        A += 1e-6 * np.eye(3)

        dc = np.linalg.solve(A, b)
        c = c + dc

        if np.linalg.norm(dc) < 1e-5:
            break

    return c.astype(np.float32)



def main():
    rclpy.init()
    node = BallCenterEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
