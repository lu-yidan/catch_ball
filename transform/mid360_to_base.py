"""
mid360_to_base.py

将 mid360_link（Livox Mid-360 激光雷达）下的三维点变换到 pelvis (base) 坐标系。

运动学链（来自 URDF g1_sysid_23dof.urdf）：
    pelvis
      └─ waist_yaw_joint   (Rz, origin=[0, 0, 0])
          └─ waist_roll_joint  (Rx, origin=[-0.0039635, 0, 0.044])
              └─ waist_pitch_joint (Ry, origin=[0, 0, 0])
                  └─ head_joint   (Ry, origin=[0.0039635, 0, 0.3159])
                      └─ mid360_joint (Ry + fixed rpy,
                             xyz=[0, 0.00003, 0.10028]
                             rpy=[0, 3.101, 3.1415])
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
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)

def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# mid360_joint 固定偏置（来自 URDF）
_R_MID360_FIXED = _rpy_to_R(roll=0.0, pitch=3.101, yaw=3.1415)
_T_HEAD_MID360_ORIGIN = np.array([0.0, 0.00003, 0.10028])


def transform_point_mid360_to_base(p_mid360, q_wy, q_wr, q_wp, q_head, q_mid=0.0):
    """
    将 mid360_link 坐标系下的三维点变换到 pelvis (base) 坐标系。

    Args:
        p_mid360 : array-like (3,)，mid360 坐标系下的点（米）
        q_wy     : waist_yaw_joint 角度（rad）
        q_wr     : waist_roll_joint 角度（rad）
        q_wp     : waist_pitch_joint 角度（rad）
        q_head   : head_joint 角度（rad）
        q_mid    : mid360_joint 角度（rad，默认 0）

    Returns:
        np.ndarray (3,)，pelvis 坐标系下的点（米）
    """
    T_head_mid360 = _T(_R_MID360_FIXED @ _Ry(q_mid), _T_HEAD_MID360_ORIGIN)

    T = (
        _T(_Rz(q_wy), [0.0, 0.0, 0.0])
        @ _T(_Rx(q_wr), [-0.0039635, 0.0, 0.044])
        @ _T(_Ry(q_wp), [0.0, 0.0, 0.0])
        @ _T(_Ry(q_head), [0.0039635, 0.0, 0.3159])
        @ T_head_mid360
    )
    return (T @ np.array([*p_mid360, 1.0]))[:3]
