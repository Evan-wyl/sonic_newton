"""WXYZ math for SONIC's existing motion library and checkpoint contract.

Isaac Lab 3.0 math uses XYZW. Keep conversion explicit here and at asset
read/write boundaries; never change the format of stored motion data implicitly.
"""

import torch
from isaaclab.utils import math as _lab


@torch.jit.script
def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:], q[..., :1]), dim=-1)


@torch.jit.script
def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., -1:], q[..., :-1]), dim=-1)


def quat_apply(q, v):
    return _lab.quat_apply(wxyz_to_xyzw(q), v)


def quat_apply_inverse(q, v):
    return _lab.quat_apply_inverse(wxyz_to_xyzw(q), v)


def quat_apply_yaw(q, v):
    return _lab.quat_apply_yaw(wxyz_to_xyzw(q), v)


def quat_conjugate(q):
    return xyzw_to_wxyz(_lab.quat_conjugate(wxyz_to_xyzw(q)))


def quat_inv(q, eps=1e-9):
    return xyzw_to_wxyz(_lab.quat_inv(wxyz_to_xyzw(q), eps))


def quat_mul(q1, q2):
    return xyzw_to_wxyz(_lab.quat_mul(wxyz_to_xyzw(q1), wxyz_to_xyzw(q2)))


def quat_from_euler_xyz(roll, pitch, yaw):
    return xyzw_to_wxyz(_lab.quat_from_euler_xyz(roll, pitch, yaw))


def matrix_from_quat(q):
    return _lab.matrix_from_quat(wxyz_to_xyzw(q))


def axis_angle_from_quat(q, eps=1e-6):
    return _lab.axis_angle_from_quat(wxyz_to_xyzw(q), eps)


def quat_error_magnitude(q1, q2):
    return _lab.quat_error_magnitude(wxyz_to_xyzw(q1), wxyz_to_xyzw(q2))


def subtract_frame_transforms(t01, q01, t02=None, q02=None):
    position, rotation = _lab.subtract_frame_transforms(
        t01, wxyz_to_xyzw(q01), t02,
        None if q02 is None else wxyz_to_xyzw(q02),
    )
    return position, xyzw_to_wxyz(rotation)


sample_uniform = _lab.sample_uniform
quat_rotate_inverse = quat_apply_inverse
