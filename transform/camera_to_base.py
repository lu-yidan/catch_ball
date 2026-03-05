"""
camera_to_base.py

将 head_camera_link（RealSense D435，body 坐标系：X前、Y左、Z上）下的三维点
变换到 pelvis (base) 坐标系。

运动学链（来自 URDF g1_sysid_23dof.urdf）：
    pelvis
      └─ waist_yaw_joint   (Rz, origin=[0, 0, 0])
          └─ waist_roll_joint  (Rx, origin=[-0.0039635, 0, 0.044])
              └─ waist_pitch_joint (Ry, origin=[0, 0, 0])
                  └─ head_joint   (Ry, origin=[0.0039635, 0, 0.3159])
                      └─ head_camera_joint (fixed,
                             xyz=[0.0448353662, 0.01, 0.1219029938]
                             rpy=[0.0119142, 0.8377475, 0.0053045])

注意：RealSense rs2_deproject_pixel_to_point 返回光学坐标系（Z前、X右、Y下），
      调用本函数前需先转换为 body 坐标系：
          X_body =  Z_optical
          Y_body = -X_optical
          Z_body = -Y_optical
"""

import numpy as np


def _Rx(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def _Ry(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def _Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def _rpy_to_R(roll, pitch, yaw):
    """URDF 约定: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)

def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# head_camera_joint 固定参数（来自 URDF）
_T_HEAD_CAMERA = _T(
    _rpy_to_R(roll=0.0119142, pitch=0.8377475, yaw=0.0053045),
    [0.0448353662, 0.01, 0.1219029938],
)


def transform_point_camera_to_base(p_cam, q_wy, q_wr, q_wp, q_head):
    """
    将 head_camera_link (body 系) 下的三维点变换到 pelvis (base) 坐标系。

    Args:
        p_cam  : array-like (3,)，相机 body 坐标系下的点（米）
        q_wy   : waist_yaw_joint 角度（rad）
        q_wr   : waist_roll_joint 角度（rad）
        q_wp   : waist_pitch_joint 角度（rad）
        q_head : head_joint 角度（rad）

    Returns:
        np.ndarray (3,)，pelvis 坐标系下的点（米）
    """
    T = (
        _T(_Rz(q_wy), [0.0, 0.0, 0.0])
        @ _T(_Rx(q_wr), [-0.0039635, 0.0, 0.044])
        @ _T(_Ry(q_wp), [0.0, 0.0, 0.0])
        @ _T(_Ry(q_head), [0.0039635, 0.0, 0.3159])
        @ _T_HEAD_CAMERA
    )
    return (T @ np.array([*p_cam, 1.0]))[:3]


def optical_to_body(p_optical):
    """
    RealSense 光学坐标系 → body 坐标系（REP-103）。

    光学系：Z前、X右、Y下
    body 系：X前、Y左、Z上

    Args:
        p_optical : array-like (3,)，光学坐标系下的点

    Returns:
        np.ndarray (3,)，body 坐标系下的点
    """
    x, y, z = p_optical
    return np.array([z, -x, -y])
