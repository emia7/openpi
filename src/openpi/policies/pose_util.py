import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

# 3d pos + 3d rot <-> 4*4 matrix
def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot


# 3d pos + 3d rot <-> 6d pose
def pos_rot_to_pose6(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose6_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot


# 6d pose <-> 4*4 matrix
def pose6_to_mat(pose):
    return pos_rot_to_mat(*pose6_to_pos_rot(pose))

def mat_to_pose6(mat):
    return pos_rot_to_pose6(*mat_to_pos_rot(mat))   


# 对一个6d pose应用一个变换矩阵tx, 通常用于坐标系转换
def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose6_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose6(tf_pose_mat)
    return tf_pose

# 对3D点应用刚体变换（旋转+平移）
def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

# 相机投影,将3D点投影到2D像素平面
def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out

# 将 7d pose(3d pos + 4d quant)转换为3d pos + 3d rot
def pose7_to_pos_rotvec(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = pose7[:3].astype(np.float32)
    quat = pose7[3:7].astype(np.float32)  # (qx,qy,qz,qw)
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)  # axis-angle vec
    return pos, rotvec

def pose7_to_mat(pose7: np.ndarray) -> np.ndarray:
    """
    将 7维位姿 [x, y, z, qx, qy, qz, qw] 转换为 4x4 变换矩阵。
    支持 Batch/Sequence 维度。
    """
    pos = pose7[..., :3]
    quat = pose7[..., 3:] # [qx, qy, qz, qw]
    
    # 创建 Rotation 对象
    rot = R.from_quat(quat)
    
    # 调用 pose_util 中的基础函数合成矩阵
    return pos_rot_to_mat(pos, rot)
