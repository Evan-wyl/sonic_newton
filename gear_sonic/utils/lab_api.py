"""Explicit Isaac Lab 3.0 data boundaries for SONIC's WXYZ task tensors.

These helpers use public asset APIs, independent of the physics backend.
They do not implement Newton model editing or solver configuration.
"""

from .lab_math import wxyz_to_xyzw, xyzw_to_wxyz


def config_quat_xyzw(q):
    """Convert an existing WXYZ config/metadata tuple to Lab's XYZW tuple."""
    return tuple(q[1:]) + tuple(q[:1])


def as_torch(value):
    """Unwrap a Lab ProxyArray; retain tensors (including custom task buffers)."""
    return value.torch if hasattr(value, "torch") else value


def asset_tensor(asset, field):
    """Read public Lab data, exposing quaternions to SONIC in WXYZ order.

    Non-quaternion tensors retain the public buffer view. Quaternion conversion
    creates a task-side value; write rotations through the explicit pose helpers.
    """
    value = as_torch(getattr(asset.data, field))
    if value is not None and "quat" in field:
        return xyzw_to_wxyz(value)
    return value


def write_root_pose(asset, pose, *, env_ids=None):
    """Write a partial (N, 7) SONIC WXYZ root pose using Lab's index API."""
    native_pose = pose.clone()
    native_pose[..., 3:7] = wxyz_to_xyzw(pose[..., 3:7])
    asset.write_root_pose_to_sim_index(root_pose=native_pose, env_ids=env_ids)


def write_root_state(asset, state, *, env_ids=None):
    """Write root link pose and CoM velocity, matching the old root-state API."""
    write_root_pose(asset, state[..., :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=state[..., 7:13], env_ids=env_ids)


def write_joint_state(asset, position, velocity, *, joint_ids=None, env_ids=None):
    """Write partial joint state with explicit index selection."""
    asset.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)
