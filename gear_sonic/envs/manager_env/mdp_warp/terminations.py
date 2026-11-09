"""Warp-first tracking termination terms (Stage 3c), mirror of
``../mdp/terminations.py``.

The experimental termination manager calls every term warp-style
(``func(env, out, **params)`` with a persistent ``(num_envs,)`` Warp bool
``out``).  As with the rewards, the reference side reads the command's
persistent buffers and the robot side reads LIVE sim state (fresh views of the
full robot body arrays, indexed in-kernel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .commands import TrackingCommandWarp
from .warp_utils import quat_error_magnitude_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@wp.kernel
def _anchor_height_kernel(
    robot_body_pos: wp.array(dtype=wp.float32, ndim=3),
    anchor_rcol: int,
    ref_pos: wp.array(dtype=wp.float32, ndim=2),
    running_height: wp.array(dtype=wp.float32),
    threshold: float,
    adaptive: int,
    down_threshold: float,
    root_height_threshold: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    diff = ref_pos[i, 2] - robot_body_pos[i, anchor_rcol, 2]
    if diff < 0.0:
        diff = -diff
    thresh = threshold
    if adaptive != 0 and running_height[i] < root_height_threshold:
        thresh = down_threshold
    out[i] = diff > thresh


@wp.kernel
def _anchor_ori_kernel(
    robot_body_quat: wp.array(dtype=wp.float32, ndim=3),
    anchor_rcol: int,
    ref_quat: wp.array(dtype=wp.float32, ndim=2),
    threshold: float,
    out: wp.array(dtype=wp.bool),
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
    out[i] = error * error > threshold


@wp.kernel
def _body_pos_kernel(
    robot_body_pos: wp.array(dtype=wp.float32, ndim=3),
    ref_rel_pos: wp.array(dtype=wp.float32, ndim=3),
    sub: wp.array(dtype=wp.int32),
    rcols: wp.array(dtype=wp.int32),
    threshold: float,
    out: wp.array(dtype=wp.bool),
):
    # any tracked body (subset) whose anchor-relative reference position
    # deviates from the LIVE robot position by more than ``threshold``
    i = wp.tid()
    max_sq = float(0.0)
    for b in range(sub.shape[0]):
        t = sub[b]
        col = rcols[t]
        dx = robot_body_pos[i, col, 0] - ref_rel_pos[i, t, 0]
        dy = robot_body_pos[i, col, 1] - ref_rel_pos[i, t, 1]
        dz = robot_body_pos[i, col, 2] - ref_rel_pos[i, t, 2]
        max_sq = wp.max(max_sq, dx * dx + dy * dy + dz * dz)
    out[i] = max_sq > threshold * threshold


@wp.kernel
def _body_height_kernel(
    robot_body_pos: wp.array(dtype=wp.float32, ndim=3),
    ref_rel_pos: wp.array(dtype=wp.float32, ndim=3),
    sub: wp.array(dtype=wp.int32),
    rcols: wp.array(dtype=wp.int32),
    running_height: wp.array(dtype=wp.float32),
    threshold: float,
    adaptive: int,
    down_threshold: float,
    root_height_threshold: float,
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    thresh = threshold
    if adaptive != 0 and running_height[i] < root_height_threshold:
        thresh = down_threshold
    max_err = float(0.0)
    for b in range(sub.shape[0]):
        t = sub[b]
        col = rcols[t]
        diff = ref_rel_pos[i, t, 2] - robot_body_pos[i, col, 2]
        if diff < 0.0:
            diff = -diff
        max_err = wp.max(max_err, diff)
    out[i] = max_err > thresh


@wp.kernel
def _time_out_kernel(
    ts: wp.array(dtype=wp.int64),
    start: wp.array(dtype=wp.int64),
    total: wp.array(dtype=wp.int64),
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    out[i] = (ts[i] + start[i] + wp.int64(1)) >= total[i]


# --------------------------------------------------------------------------
# terms (signatures mirror ../mdp/terminations.py plus the warp ``out``)
# --------------------------------------------------------------------------


def _subset_views(command: TrackingCommandWarp, body_names):
    # multi-key cache: one PERSISTENT wp.array per (command, body_names).
    # A single-slot-per-command cache ping-pongs between distinct keys when
    # two subset terms are live: each call mints a fresh wp.array, so the
    # array baked into a captured graph loses its last reference after
    # capture and is freed -- replay then reads allocator-recycled storage
    # through garbage indices (CUDA error 700 at the first replay after the
    # learning phase reuses the block).
    key = tuple(body_names) if body_names is not None else None
    cache = _subset_views._cache.setdefault(id(command), {})
    hit = cache.get(key)
    if hit is None:
        tracked = [
            i
            for i, name in enumerate(command.cfg.body_names)
            if body_names is None or name in body_names
        ]
        sub = wp.array([int(v) for v in tracked], dtype=wp.int32, device=command.device)
        hit = (sub, command._wp_body_rcols)
        cache[key] = hit
    return hit[0], hit[1]


_subset_views._cache: dict[int, dict[tuple | None, tuple]] = {}


def exceeded_anchor_height(
    env: ManagerBasedEnv,
    out,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    root_height_threshold: float = 1.0,
) -> None:
    """Terminate on anchor Z-height error. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    r_pos, _, _, _, _, _ = command._robot_wp_views()
    wp.launch(
        kernel=_anchor_height_kernel,
        dim=env.num_envs,
        inputs=[
            r_pos,
            command._anchor_rcol,
            command.wp_anchor_pos_w,
            command.wp_running_ref_root_height,
            float(threshold),
            1 if threshold_adaptive else 0,
            float(down_threshold),
            float(root_height_threshold),
            out,
        ],
        device=env.device,
    )


def exceeded_anchor_ori(
    env: ManagerBasedEnv, out, asset_cfg, command_name: str, threshold: float
) -> None:
    """Terminate on squared anchor orientation error. Writes into ``out``.

    ``asset_cfg`` is unused (kept for API parity with the torch term).
    """
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    _, r_quat, _, _, _, _ = command._robot_wp_views()
    wp.launch(
        kernel=_anchor_ori_kernel,
        dim=env.num_envs,
        inputs=[
            r_quat,
            command._anchor_rcol,
            command.wp_anchor_quat_w,
            float(threshold),
            out,
        ],
        device=env.device,
    )


def exceeded_body_pos(
    env: ManagerBasedEnv, out, command_name: str, threshold: float, body_names: list | None = None
) -> None:
    """Terminate when any tracked body position error is too large. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    r_pos, _, _, _, _, _ = command._robot_wp_views()
    sub, rcols = _subset_views(command, body_names)
    wp.launch(
        kernel=_body_pos_kernel,
        dim=env.num_envs,
        inputs=[r_pos, command.wp_body_pos_rel_w, sub, rcols, float(threshold), out],
        device=env.device,
    )


def exceeded_body_height(
    env: ManagerBasedEnv,
    out,
    command_name: str,
    threshold: float,
    threshold_adaptive: bool = False,
    down_threshold: float = 0.5,
    body_names: list | None = None,
    root_height_threshold: float = 0.5,
) -> None:
    """Terminate when any tracked body Z-height error is too large. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    r_pos, _, _, _, _, _ = command._robot_wp_views()
    sub, rcols = _subset_views(command, body_names)
    wp.launch(
        kernel=_body_height_kernel,
        dim=env.num_envs,
        inputs=[
            r_pos,
            command.wp_body_pos_rel_w,
            sub,
            rcols,
            command.wp_running_ref_root_height,
            float(threshold),
            1 if threshold_adaptive else 0,
            float(down_threshold),
            float(root_height_threshold),
            out,
        ],
        device=env.device,
    )


def tracking_time_out(env: ManagerBasedEnv, out, command_name: str) -> None:
    """Terminate when the motion clip has been fully played. Writes into ``out``."""
    command: TrackingCommandWarp = env.command_manager.get_term(command_name)
    wp.launch(
        kernel=_time_out_kernel,
        dim=env.num_envs,
        inputs=[command._wp_ts, command._wp_motion_start, command._wp_total, out],
        device=env.device,
    )
