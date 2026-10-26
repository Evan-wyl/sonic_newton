"""Small project interfaces on top of Lab 3.0's public Newton asset APIs."""

import torch

from gear_sonic_newton.utils.lab_api import as_torch, asset_tensor
from gear_sonic_newton.utils.lab_math import quat_apply


def get_linear_jacobian(asset, body_name, joint_names, point_offset=None):
    """World linear Jacobian of a link point, columns in joint_names order.

    point_offset is in the link frame (m); None means the link origin.
    Floating-base columns are excluded. The fixed root has zero joint Jacobian.
    """
    body = asset.body_names.index(body_name)
    columns = [asset.joint_names.index(name) + asset.num_base_dofs for name in joint_names]
    full = as_torch(asset.data.body_link_jacobian_w)
    if asset.is_fixed_base and body == 0:
        return torch.zeros(full.shape[0], 3, len(columns), device=full.device, dtype=full.dtype)
    row = body - int(asset.is_fixed_base)
    linear = full[:, row, :3, columns].clone()
    if point_offset is not None:
        local = torch.as_tensor(point_offset, device=full.device, dtype=full.dtype).expand(full.shape[0], 3)
        offset = quat_apply(asset_tensor(asset, "body_quat_w")[:, body], local)
        angular = full[:, row, 3:6, columns]
        linear += torch.cross(angular.transpose(1, 2), offset[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
    return linear


def set_world_forces(asset, forces, *, body_ids=None, env_ids=None):
    """Persistent world forces at body CoMs; caller must explicitly clear them."""
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, body_ids=body_ids, env_ids=env_ids, is_global=True,
    )


def set_body_com_startup(env, asset, positions, *, body_ids, env_ids):
    """Set local CoMs and fully refresh MJWarp constants, outside CUDA capture.

    Like mjlab's body_ipos randomizer, mass and inertia about the CoM are
    preserved. This bridge is specific to the installed Newton SolverMuJoCo;
    keep private solver access here rather than in task event functions.
    """
    import mujoco_warp
    import warp as wp
    from newton import ModelFlags

    solver = env.sim.physics_manager._solver
    if getattr(solver, "use_mujoco_cpu", True) or not hasattr(solver, "mjw_model"):
        raise RuntimeError("CoM randomization requires the Newton MJWarp solver.")
    with wp.ScopedDevice(solver.model.device):
        if wp.get_device().is_capturing:
            raise RuntimeError("CoM randomization must run outside CUDA graph capture.")
        asset.set_coms_index(coms=positions, body_ids=body_ids, env_ids=env_ids)
        # Copies Newton body_com into MJWarp body_ipos, updates inertia frame
        # and invalidates Newton's cached contact conversion parameters.
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
        # Explicit full recomputation, including restoring the current-pose
        # mass matrix/factorization. Do not assume an array write is sufficient.
        # Newton excludes artificial kinematic locking armature from this
        # statistic; preserve its corrected value during the full refresh.
        mean_inertia = wp.clone(solver.mjw_model.stat.meaninertia)
        mujoco_warp.set_const(solver.mjw_model, solver.mjw_data)
        wp.copy(solver.mjw_model.stat.meaninertia, mean_inertia)
