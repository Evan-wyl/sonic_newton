"""Event functions for domain randomization and environment resets in RL training."""

from __future__ import annotations

from gear_sonic.utils.lab_api import asset_tensor

from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = None
    add_joint_default_pos = None
    add_hand_joint_default_pos = None
    base_com = None

    # interval - balance training
    push_robot = None

    randomize_rigid_body_mass = None


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize joint default positions to simulate calibration errors.

    Applies random offsets to the default joint positions of the robot, modeling
    real-world joint encoder calibration inaccuracies. Also updates the action
    manager offset to keep action space aligned with the new defaults.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        asset_cfg: Scene entity config with joint IDs to randomize.
        pos_distribution_params: Min/max range for the position offset distribution.
        operation: How to combine the random value with the original ("add", "scale", "abs").
        distribution: Sampling distribution type.
    """
    asset = env.scene[asset_cfg.name]
    defaults = asset_tensor(asset, "default_joint_pos")
    if not hasattr(asset, "_sonic_nominal_joint_pos"):
        asset._sonic_nominal_joint_pos = defaults.clone()
        asset.data.default_joint_pos_nominal = defaults[0].clone()
    rows = torch.arange(asset.num_instances, device=asset.device)
    if env_ids is not None:
        rows = rows[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    joints = torch.arange(asset.num_joints, device=asset.device)[asset_cfg.joint_ids]
    if pos_distribution_params is None or rows.numel() == 0 or joints.numel() == 0:
        return
    candidate = _randomize_prop_by_op(
        asset._sonic_nominal_joint_pos.clone(), pos_distribution_params, rows, joints,
        operation=operation, distribution=distribution,
    )
    defaults[rows[:, None], joints] = candidate[rows[:, None], joints]
    # Keep action ordering explicit; paired advanced indexing would only
    # update a diagonal when both axes are selected.
    term = env.action_manager.get_term("joint_pos")
    names = term._joint_names
    asset_ids = [asset.joint_names.index(name) for name in names]
    term._offset[rows] = defaults[rows][:, asset_ids]


def randomize_rigid_body_com(env, env_ids, com_range, asset_cfg):
    """Startup local CoM offsets from nominal values (m), without accumulation.

    Equivalent to mjlab BeyondMimic's additive body_ipos randomization:
    mass and inertia about the CoM are not modified by this event.
    """
    from gear_sonic.envs.physics_adapter import set_body_com_startup

    asset = env.scene[asset_cfg.name]
    if not hasattr(asset, "_sonic_nominal_body_com"):
        asset._sonic_nominal_body_com = asset_tensor(asset, "body_com_pos_b").clone()
    rows = torch.arange(asset.num_instances, device=asset.device)
    if env_ids is not None:
        rows = rows[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    bodies = torch.arange(asset.num_bodies, device=asset.device)[asset_cfg.body_ids]
    if rows.numel() == 0 or bodies.numel() == 0:
        return
    positions = asset._sonic_nominal_body_com[rows[:, None], bodies].clone()
    for axis, name in enumerate(("x", "y", "z")):
        low, high = com_range.get(name, (0., 0.))
        if low > high or not torch.isfinite(torch.tensor([low, high])).all():
            raise ValueError(f"Invalid CoM range for {name}: {(low, high)}")
        positions[..., axis] += torch.empty_like(positions[..., axis]).uniform_(low, high)
    set_body_com_startup(env, asset, positions, body_ids=bodies, env_ids=rows)
