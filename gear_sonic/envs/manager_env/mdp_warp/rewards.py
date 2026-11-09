"""Warp-first tracking reward terms (Stage 3c), mirror of ``../mdp/rewards.py``.

The experimental reward manager calls every term warp-style
(``func(env, out, **params)`` with a persistent ``(num_envs,)`` Warp ``out``),
so these are plain functions launching Warp kernels.

Parity note (matches the torch env exactly): reward terms run AFTER the
decimation/physics of the current step but BEFORE ``command.compute``, so the
reference side reads the command's persistent buffers (current ``time_steps``,
filled by the last command update) while the robot side reads LIVE sim state
-- fresh Warp views of the full robot body arrays are taken per call and
indexed in-kernel (``_wp_body_rcols`` / ``_wp_reward_rcols``), exactly like
the torch terms index ``robot.data.body_*`` properties live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .commands import TrackingCommandWarp
from .warp_utils import quat_error_magnitude_xyzw, quat_rotate_inv_xyzw, quat_rotate_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@wp.kernel
def _anchor_pos_error_kernel(
    robot_body_pos: wp.array(dtype=wp.float32, ndim=3),
    anchor_rcol: int,
    ref_pos: wp.array(dtype=wp.float32, ndim=2),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dx = robot_body_pos[i, anchor_rcol, 0] - ref_pos[i, 0]
    dy = robot_body_pos[i, anchor_rcol, 1] - ref_pos[i, 1]
    dz = robot_body_pos[i, anchor_rcol, 2] - ref_pos[i, 2]
    out[i] = wp.exp(-(dx * dx + dy * dy + dz * dz) / (std * std))


@wp.kernel
def _anchor_ori_error_kernel(
    robot_body_quat: wp.array(dtype=wp.float32, ndim=3),
    anchor_rcol: int,
    ref_quat: wp.array(dtype=wp.float32, ndim=2),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    qr = wp.vec4f(
        robot_body_quat[i, anchor_rcol, 0],
        robot_body_quat[i, anchor_rcol, 1],
        robot_body_quat[i, anchor_rcol, 2],
        robot_body_quat[i, anchor_rcol, 3],
    )
    qm = wp.vec4f(ref_quat[i, 0], ref_quat[i, 1], ref_quat[i, 2], ref_quat[i, 3])
    error = quat_error_magnitude_xyzw(qm, qr)
    out[i] = wp.exp(-(error * error) / (std * std))


@wp.kernel
def _body_pos_error_kernel(
    robot_body_pos: wp.array(dtype=wp.float32, ndim=3),
    ref_pos: wp.array(dtype=wp.float32, ndim=3),
    sub: wp.array(dtype=wp.int32),
    rcols: wp.array(dtype=wp.int32),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    # ref side: persistent buffer rows at the tracked-axis subset
    # robot side: LIVE full robot array at rcols[sub[.]]
    i = wp.tid()
    num_bodies = sub.shape[0]
    error_sum = float(0.0)
    for b in range(num_bodies):
        t = sub[b]
        col = rcols[t]
        dx = robot_body_pos[i, col, 0] - ref_pos[i, t, 0]
        dy = robot_body_pos[i, col, 1] - ref_pos[i, t, 1]
        dz = robot_body_pos[i, col, 2] - ref_pos[i, t, 2]
        error_sum += dx * dx + dy * dy + dz * dz
    out[i] = wp.exp(-(error_sum / float(num_bodies)) / (std * std))


@wp.kernel
def _body_ori_error_kernel(
    robot_body_quat: wp.array(dtype=wp.float32, ndim=3),
    ref_quat: wp.array(dtype=wp.float32, ndim=3),
    sub: wp.array(dtype=wp.int32),
    rcols: wp.array(dtype=wp.int32),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    num_bodies = sub.shape[0]
    error_sum = float(0.0)
    for b in range(num_bodies):
        t = sub[b]
        col = rcols[t]
        qr = wp.vec4f(
            robot_body_quat[i, col, 0],
            robot_body_quat[i, col, 1],
            robot_body_quat[i, col, 2],
            robot_body_quat[i, col, 3],
        )
        qm = wp.vec4f(ref_quat[i, t, 0], ref_quat[i, t, 1], ref_quat[i, t, 2], ref_quat[i, t, 3])
        error = quat_error_magnitude_xyzw(qm, qr)
        error_sum += error * error
    out[i] = wp.exp(-(error_sum / float(num_bodies)) / (std * std))


@wp.kernel
def _body_vel_error_kernel(
    robot_vel: wp.array(dtype=wp.float32, ndim=3),
    ref_vel: wp.array(dtype=wp.float32, ndim=3),
    sub: wp.array(dtype=wp.int32),
    rcols: wp.array(dtype=wp.int32),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    num_bodies = sub.shape[0]
    error_sum = float(0.0)
    for b in range(num_bodies):
        t = sub[b]
        col = rcols[t]
        dx = robot_vel[i, col, 0] - ref_vel[i, t, 0]
        dy = robot_vel[i, col, 1] - ref_vel[i, t, 1]
        dz = robot_vel[i, col, 2] - ref_vel[i, t, 2]
        error_sum += dx * dx + dy * dy + dz * dz
    out[i] = wp.exp(-(error_sum / float(num_bodies)) / (std * std))


@wp.kernel
def _local_5point_error_kernel(
    robot_body_pos: wp.array(dtype=wp.float32, ndim=3),
    robot_body_quat: wp.array(dtype=wp.float32, ndim=3),
    anchor_rcol: int,
    rcols: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.float32, ndim=2),
    ref_anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    ref_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    ref_points: wp.array(dtype=wp.float32, ndim=3),
    std: float,
    out: wp.array(dtype=wp.float32),
):
    # ref point:   quat_rotate_inv(ref anchor quat, ref_point - ref anchor pos)
    # robot point: body_pos + quat_rotate(body_quat, offset), anchor-localized
    #              with the LIVE robot anchor
    i = wp.tid()
    ra_q = wp.vec4f(
        robot_body_quat[i, anchor_rcol, 0],
        robot_body_quat[i, anchor_rcol, 1],
        robot_body_quat[i, anchor_rcol, 2],
        robot_body_quat[i, anchor_rcol, 3],
    )
    ra_p = wp.vec3f(
        robot_body_pos[i, anchor_rcol, 0],
        robot_body_pos[i, anchor_rcol, 1],
        robot_body_pos[i, anchor_rcol, 2],
    )
    rf_q = wp.vec4f(ref_anchor_quat[i, 0], ref_anchor_quat[i, 1], ref_anchor_quat[i, 2], ref_anchor_quat[i, 3])
    rf_p = wp.vec3f(ref_anchor_pos[i, 0], ref_anchor_pos[i, 1], ref_anchor_pos[i, 2])
    num_pts = rcols.shape[0]
    error_sum = float(0.0)
    for k in range(num_pts):
        col = rcols[k]
        bq = wp.vec4f(
            robot_body_quat[i, col, 0],
            robot_body_quat[i, col, 1],
            robot_body_quat[i, col, 2],
            robot_body_quat[i, col, 3],
        )
        off = quat_rotate_xyzw(
            bq, wp.vec3f(offsets[k, 0], offsets[k, 1], offsets[k, 2])
        )
        p_rob = quat_rotate_inv_xyzw(
            ra_q,
            wp.vec3f(
                robot_body_pos[i, col, 0] + off[0] - ra_p[0],
                robot_body_pos[i, col, 1] + off[1] - ra_p[1],
                robot_body_pos[i, col, 2] + off[2] - ra_p[2],
            ),
        )
        p_ref = quat_rotate_inv_xyzw(
            rf_q,
            wp.vec3f(
                ref_points[i, k, 0] - rf_p[0],
                ref_points[i, k, 1] - rf_p[1],
                ref_points[i, k, 2] - rf_p[2],
            ),
        )
        dx = p_rob[0] - p_ref[0]
        dy = p_rob[1] - p_ref[1]
        dz = p_rob[2] - p_ref[2]
        error_sum += dx * dx + dy * dy + dz * dz
    out[i] = wp.exp(-(error_sum / float(num_pts)) / (std * std))


@wp.kernel
def _anti_shake_ang_vel_kernel(
    robot_body_ang_vel: wp.array(dtype=wp.float32, ndim=3),
    sub: wp.array(dtype=wp.int32),
    rcols: wp.array(dtype=wp.int32),
    threshold: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    num_bodies = sub.shape[0]
    penalty_sum = float(0.0)
    for b in range(num_bodies):
        col = rcols[sub[b]]
        wx = robot_body_ang_vel[i, col, 0]
        wy = robot_body_ang_vel[i, col, 1]
        wz = robot_body_ang_vel[i, col, 2]
        speed = wp.sqrt(wx * wx + wy * wy + wz * wz)
        excess = speed - threshold
        if excess < 0.0:
            excess = 0.0
        penalty_sum += excess * excess
    out[i] = penalty_sum / float(num_bodies)


# --------------------------------------------------------------------------
# subset cache (indices into cfg.body_names + their robot-body columns)
# --------------------------------------------------------------------------


def _subset_views(command: TrackingCommandWarp, body_names):
    key = tuple(body_names) if body_names is not None else None
    # A captured reward graph keeps device pointers to every subset used by
    # its terms. Keep all subsets alive for the command's lifetime: replacing
    # a single cached subset lets earlier graph nodes read freed memory.
    cache = getattr(command, "_reward_subset_views", None)
    if cache is None:
        cache = {}
        command._reward_subset_views = cache
    hit = cache.get(key)
    if hit is None:
        tracked = [
            i
            for i, name in enumerate(command.cfg.body_names)
            if body_names is None or name in body_names
        ]
        sub = wp.array([int(v) for v in tracked], dtype=wp.int32, device=command.device)
        # rcols: tracked-axis index -> robot body column (from the command's
        # immutable map; identity sub is used when no subset is requested)
        rcols = command._wp_body_rcols
        hit = (sub, rcols)
        cache[key] = hit
    return hit


# --------------------------------------------------------------------------
# terms (signatures mirror ../mdp/rewards.py plus the warp ``out``)
# --------------------------------------------------------------------------


def tracking_anchor_pos_error(
    env: ManagerBasedRLEnv, out, command_name: str, std: float
) -> None:
    """Anchor position tracking reward. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    r_pos, _, _, _, _, _ = command._robot_wp_views()
    wp.launch(
        kernel=_anchor_pos_error_kernel,
        dim=env.num_envs,
        inputs=[r_pos, command._anchor_rcol, command.wp_anchor_pos_w, float(std), out],
        device=env.device,
    )


def tracking_anchor_ori_error(
    env: ManagerBasedRLEnv, out, command_name: str, std: float
) -> None:
    """Anchor orientation tracking reward. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    _, r_quat, _, _, _, _ = command._robot_wp_views()
    wp.launch(
        kernel=_anchor_ori_error_kernel,
        dim=env.num_envs,
        inputs=[r_quat, command._anchor_rcol, command.wp_anchor_quat_w, float(std), out],
        device=env.device,
    )


def tracking_relative_body_pos_error(
    env: ManagerBasedRLEnv, out, command_name: str, std: float, body_names: list | None = None
) -> None:
    """Anchor-relative tracked body position tracking reward. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    r_pos, _, _, _, _, _ = command._robot_wp_views()
    sub, rcols = _subset_views(command, body_names)
    wp.launch(
        kernel=_body_pos_error_kernel,
        dim=env.num_envs,
        inputs=[r_pos, command.wp_body_pos_rel_w, sub, rcols, float(std), out],
        device=env.device,
    )


def tracking_relative_body_ori_error(
    env: ManagerBasedRLEnv, out, command_name: str, std: float, body_names: list | None = None
) -> None:
    """Anchor-relative tracked body orientation tracking reward. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    _, r_quat, _, _, _, _ = command._robot_wp_views()
    sub, rcols = _subset_views(command, body_names)
    wp.launch(
        kernel=_body_ori_error_kernel,
        dim=env.num_envs,
        inputs=[r_quat, command.wp_body_quat_rel_w, sub, rcols, float(std), out],
        device=env.device,
    )


def tracking_body_linvel_error(
    env: ManagerBasedRLEnv, out, command_name: str, std: float, body_names: list | None = None
) -> None:
    """Tracked body linear velocity tracking reward. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    _, _, r_lin, _, _, _ = command._robot_wp_views()
    sub, rcols = _subset_views(command, body_names)
    wp.launch(
        kernel=_body_vel_error_kernel,
        dim=env.num_envs,
        inputs=[r_lin, command.wp_body_lin_vel_w, sub, rcols, float(std), out],
        device=env.device,
    )


def tracking_body_angvel_error(
    env: ManagerBasedRLEnv, out, command_name: str, std: float, body_names: list | None = None
) -> None:
    """Tracked body angular velocity tracking reward. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    _, _, _, r_ang, _, _ = command._robot_wp_views()
    sub, rcols = _subset_views(command, body_names)
    wp.launch(
        kernel=_body_vel_error_kernel,
        dim=env.num_envs,
        inputs=[r_ang, command.wp_body_ang_vel_w, sub, rcols, float(std), out],
        device=env.device,
    )


def tracking_local_vr_5point_error(
    env: ManagerBasedRLEnv, out, command_name: str, std: float
) -> None:
    """VR 5-point tracking reward in the anchor local frame. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    r_pos, r_quat, _, _, _, _ = command._robot_wp_views()
    wp.launch(
        kernel=_local_5point_error_kernel,
        dim=env.num_envs,
        inputs=[
            r_pos,
            r_quat,
            command._anchor_rcol,
            command._wp_reward_rcols,
            command._wp_reward_offsets,
            command.wp_anchor_pos_w,
            command.wp_anchor_quat_w,
            command.wp_reward_point_body_pos_w,
            float(std),
            out,
        ],
        device=env.device,
    )


def anti_shake_ang_vel_l2(
    env: ManagerBasedRLEnv,
    out,
    command_name: str,
    threshold: float = 1.5,
    body_names: list | None = None,
) -> None:
    """Angular-velocity jitter penalty with a deadzone. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    _, _, _, r_ang, _, _ = command._robot_wp_views()
    sub, rcols = _subset_views(command, body_names)
    wp.launch(
        kernel=_anti_shake_ang_vel_kernel,
        dim=env.num_envs,
        inputs=[r_ang, sub, rcols, float(threshold), out],
        device=env.device,
    )
