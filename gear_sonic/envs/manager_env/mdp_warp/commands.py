"""Warp-consumable variant of :class:`~gear_sonic.envs.manager_env.mdp.commands.TrackingCommand`.

Stage 3b of the Newton/Warp migration (the Newton/Warp migration).
The classic command does all per-step work in torch: ``_update_command``
advances ``time_steps``, calls the (usually empty) ``_resample_command`` whose
tail rebuilds the relative body targets with a fresh repeat/quat-ops chain, and
every obs/reward/termination reader then re-gathers the reference frame
through the ~200 lazy properties -- dozens of kernel launches and allocations
per step.  Only the reset/overflow resample (motion sampling, encoder
multinomial, sim writes, object placement) genuinely needs torch.

This subclass follows unitree's ``MotionCommandWarp`` pattern, adapted to
sonic's per-motion motion-lib layout (``data[steps + length_starts[ids]]``,
arrays pre-sliced to the tracked bodies) and term inventory:

* ``_metrics_kernel`` -- evaluates the 10 error metrics on the PRE-advance
  frame (exactly where ``CommandTerm.compute`` calls ``_update_metrics``)
  against the LIVE robot state;
* ``_reference_kernel`` -- gathers the reference frame at the post-advance
  cursor, snapshots the robot state, computes the anchor-relative body
  targets (yaw-only XYZW delta, see :func:`_delta_ori_xyzw`) and the
  vr/reward-point offset positions into persistent buffers consumed by the
  Stage-3c obs/reward/termination kernels;
* the multi-future / SMPL tokenizer sources are too entangled with the
  scrambled cat/reshape layouts the trained model expects (e.g.
  ``command_multi_future = cat([jpos, jvel]).view(N, F, -1)``) to port
  bit-exactly as kernels -- they stay torch but run ONCE per step here,
  copied into persistent buffers that the obs terms read with pure copy
  kernels (capturable; no per-term property chains).

Motion-end overflow (``time_steps + start >= total``) is detected from a
host-side numpy mirror -- the device tensors only change in the in-place
``+= 1`` (mirrored) and the torch resample path (which re-syncs the mirror) --
so the steady state performs no device readback.

Parity notes (verified against the torch command path):

* the base ``_update_command`` calls ``_resample_command`` every step with a
  usually-empty ``env_ids`` and its UNGUARDED tail recomputes the relative
  targets from the post-advance state each step -- so the reference kernel
  rebuilds them per step (they are NOT frozen between resamples);
* ``future_motion_ids`` / ``smpl_future_motion_ids`` are REASSIGNED by the
  base ``_resample_command`` at every resample (freshly allocated), so any
  cached int64 device copies of them must refresh unconditionally -- see
  :meth:`_sync_host_mirror`; the multi-future fills go through the base
  properties (which clamp time steps) verbatim;
* this copy keeps the existing WXYZ motion-library and Torch command contract.
  Reference quaternions are converted into XYZW buffers for Warp kernels;
  inherited Torch properties read WXYZ mirrors of the kernel outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from isaaclab.utils.configclass import configclass
import numpy as np
import torch
import warp as wp

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand, TrackingCommandCfg
from gear_sonic.trl.utils import common
from gear_sonic.utils.lab_api import as_torch
from gear_sonic.utils.lab_math import quat_apply, quat_inv

from .obs_history import register_env_hint
from .warp_utils import (
    quat_conj_xyzw,
    quat_error_magnitude_xyzw,
    quat_mul_xyzw,
    quat_rotate_xyzw,
    yaw_quat_xyzw,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _live_wp(x):
    """Zero-copy Warp view of a live torch tensor / Warp array (unitree pattern)."""
    if isinstance(x, wp.array):
        return x
    x = as_torch(x)
    if torch.is_tensor(x):
        if not x.is_contiguous():
            x = x.contiguous()
        return wp.from_torch(x)
    # lazy warp-backed buffers: resolve through torch
    return wp.from_torch(wp.to_torch(x))


@wp.func
def _delta_ori_xyzw(q_r: wp.vec4f, q_a: wp.vec4f) -> wp.vec4f:
    """Relative-anchor orientation used by sonic's relative body targets.

    Match the torch tail's ``yaw_quat(quat_mul(robot, quat_inv(ref)))``.
    """
    return yaw_quat_xyzw(quat_mul_xyzw(q_r, quat_conj_xyzw(q_a)))


@wp.kernel
def _gather_contig2d(src: wp.array(dtype=wp.float32, ndim=2), dst: wp.array(dtype=wp.float32, ndim=2)):
    # strided-source gather into the contiguous robot-state shadows
    i, j = wp.tid()
    dst[i, j] = src[i, j]


@wp.kernel
def _gather_contig3d(src: wp.array(dtype=wp.float32, ndim=3), dst: wp.array(dtype=wp.float32, ndim=3)):
    i, j, k = wp.tid()
    dst[i, j, k] = src[i, j, k]


@wp.kernel
def _xyzw_to_wxyz3d(src: wp.array(dtype=wp.float32, ndim=3), dst: wp.array(dtype=wp.float32, ndim=3)):
    i, j = wp.tid()
    dst[i, j, 0] = src[i, j, 3]
    dst[i, j, 1] = src[i, j, 0]
    dst[i, j, 2] = src[i, j, 1]
    dst[i, j, 3] = src[i, j, 2]


@wp.kernel
def _xyzw_to_wxyz2d(src: wp.array(dtype=wp.float32, ndim=2), dst: wp.array(dtype=wp.float32, ndim=2)):
    i = wp.tid()
    dst[i, 0] = src[i, 3]
    dst[i, 1] = src[i, 0]
    dst[i, 2] = src[i, 1]
    dst[i, 3] = src[i, 2]


@wp.func
def _motion_quat_xyzw(m_quat: wp.array(dtype=wp.float32, ndim=3), row: int, body: int) -> wp.vec4f:
    """Read this repository's WXYZ motion library in Lab/Warp XYZW order."""
    return wp.vec4f(m_quat[row, body, 1], m_quat[row, body, 2], m_quat[row, body, 3], m_quat[row, body, 0])


@wp.kernel
def _metrics_kernel(
    ts: wp.array(dtype=wp.int64),
    motion_ids: wp.array(dtype=wp.int64),
    motion_start: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    mj_pos: wp.array(dtype=wp.float32, ndim=2),
    mj_vel: wp.array(dtype=wp.float32, ndim=2),
    m_pos: wp.array(dtype=wp.float32, ndim=3),
    m_quat: wp.array(dtype=wp.float32, ndim=3),
    m_lin: wp.array(dtype=wp.float32, ndim=3),
    m_ang: wp.array(dtype=wp.float32, ndim=3),
    dof_mismatch: int,
    body_joint_idx: wp.array(dtype=wp.int32),
    anchor_mcol: int,
    anchor_rcol: int,
    body_rcols: wp.array(dtype=wp.int32),
    env_origins: wp.array(dtype=wp.float32, ndim=2),
    r_pos: wp.array(dtype=wp.float32, ndim=3),
    r_quat: wp.array(dtype=wp.float32, ndim=3),
    r_lin: wp.array(dtype=wp.float32, ndim=3),
    r_ang: wp.array(dtype=wp.float32, ndim=3),
    r_jpos: wp.array(dtype=wp.float32, ndim=2),
    r_jvel: wp.array(dtype=wp.float32, ndim=2),
    rel_pos: wp.array(dtype=wp.float32, ndim=3),
    rel_quat: wp.array(dtype=wp.float32, ndim=3),
    out_anchor_pos: wp.array(dtype=wp.float32),
    out_anchor_rot: wp.array(dtype=wp.float32),
    out_anchor_lin: wp.array(dtype=wp.float32),
    out_anchor_ang: wp.array(dtype=wp.float32),
    out_body_pos: wp.array(dtype=wp.float32),
    out_body_rot: wp.array(dtype=wp.float32),
    out_body_lin: wp.array(dtype=wp.float32),
    out_body_ang: wp.array(dtype=wp.float32),
    out_joint_pos: wp.array(dtype=wp.float32),
    out_joint_vel: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    row = int(ts[i] + motion_start[i] + length_starts[int(motion_ids[i])])

    # -- anchor errors (reference gets env_origins, robot state is global)
    dx = m_pos[row, anchor_mcol, 0] + env_origins[i, 0] - r_pos[i, anchor_rcol, 0]
    dy = m_pos[row, anchor_mcol, 1] + env_origins[i, 1] - r_pos[i, anchor_rcol, 1]
    dz = m_pos[row, anchor_mcol, 2] + env_origins[i, 2] - r_pos[i, anchor_rcol, 2]
    out_anchor_pos[i] = wp.sqrt(dx * dx + dy * dy + dz * dz)

    q_m = _motion_quat_xyzw(m_quat, row, anchor_mcol)
    q_r = wp.vec4f(
        r_quat[i, anchor_rcol, 0],
        r_quat[i, anchor_rcol, 1],
        r_quat[i, anchor_rcol, 2],
        r_quat[i, anchor_rcol, 3],
    )
    out_anchor_rot[i] = quat_error_magnitude_xyzw(q_m, q_r)

    lx = m_lin[row, anchor_mcol, 0] - r_lin[i, anchor_rcol, 0]
    ly = m_lin[row, anchor_mcol, 1] - r_lin[i, anchor_rcol, 1]
    lz = m_lin[row, anchor_mcol, 2] - r_lin[i, anchor_rcol, 2]
    out_anchor_lin[i] = wp.sqrt(lx * lx + ly * ly + lz * lz)

    ax = m_ang[row, anchor_mcol, 0] - r_ang[i, anchor_rcol, 0]
    ay = m_ang[row, anchor_mcol, 1] - r_ang[i, anchor_rcol, 1]
    az = m_ang[row, anchor_mcol, 2] - r_ang[i, anchor_rcol, 2]
    out_anchor_ang[i] = wp.sqrt(ax * ax + ay * ay + az * az)

    # -- body errors: per-body norm / quat error, then mean over tracked bodies
    num_bodies = rel_pos.shape[1]
    e_pos = float(0.0)
    e_rot = float(0.0)
    e_lin = float(0.0)
    e_ang = float(0.0)
    for b in range(num_bodies):
        col = body_rcols[b]
        px = rel_pos[i, b, 0] - r_pos[i, col, 0]
        py = rel_pos[i, b, 1] - r_pos[i, col, 1]
        pz = rel_pos[i, b, 2] - r_pos[i, col, 2]
        e_pos += wp.sqrt(px * px + py * py + pz * pz)

        q1 = wp.vec4f(rel_quat[i, b, 0], rel_quat[i, b, 1], rel_quat[i, b, 2], rel_quat[i, b, 3])
        q2 = wp.vec4f(r_quat[i, col, 0], r_quat[i, col, 1], r_quat[i, col, 2], r_quat[i, col, 3])
        e_rot += quat_error_magnitude_xyzw(q1, q2)

        lx = m_lin[row, b, 0] - r_lin[i, col, 0]
        ly = m_lin[row, b, 1] - r_lin[i, col, 1]
        lz = m_lin[row, b, 2] - r_lin[i, col, 2]
        e_lin += wp.sqrt(lx * lx + ly * ly + lz * lz)

        ax = m_ang[row, b, 0] - r_ang[i, col, 0]
        ay = m_ang[row, b, 1] - r_ang[i, col, 1]
        az = m_ang[row, b, 2] - r_ang[i, col, 2]
        e_ang += wp.sqrt(ax * ax + ay * ay + az * az)

    inv_bodies = 1.0 / float(num_bodies)
    out_body_pos[i] = e_pos * inv_bodies
    out_body_rot[i] = e_rot * inv_bodies
    out_body_lin[i] = e_lin * inv_bodies
    out_body_ang[i] = e_ang * inv_bodies

    # -- joint errors (abs-diff mean; dof-mismatch maps motion dofs into robot dofs)
    num_mj = mj_pos.shape[1]
    e_jpos = float(0.0)
    e_jvel = float(0.0)
    inv_mj = 1.0 / float(num_mj)
    for j in range(num_mj):
        if dof_mismatch == 1:
            rj = body_joint_idx[j]
        else:
            rj = j
        d = mj_pos[row, j] - r_jpos[i, rj]
        e_jpos += wp.abs(d)
        d = mj_vel[row, j] - r_jvel[i, rj]
        e_jvel += wp.abs(d)
    out_joint_pos[i] = e_jpos * inv_mj
    out_joint_vel[i] = e_jvel * inv_mj


@wp.kernel
def _reference_kernel(
    ts: wp.array(dtype=wp.int64),
    motion_ids: wp.array(dtype=wp.int64),
    motion_start: wp.array(dtype=wp.int64),
    length_starts: wp.array(dtype=wp.int64),
    mj_pos: wp.array(dtype=wp.float32, ndim=2),
    mj_vel: wp.array(dtype=wp.float32, ndim=2),
    m_pos: wp.array(dtype=wp.float32, ndim=3),
    m_quat: wp.array(dtype=wp.float32, ndim=3),
    m_lin: wp.array(dtype=wp.float32, ndim=3),
    m_ang: wp.array(dtype=wp.float32, ndim=3),
    anchor_mcol: int,
    vr_mcols: wp.array(dtype=wp.int32),
    reward_mcols: wp.array(dtype=wp.int32),
    vr_offsets: wp.array(dtype=wp.float32, ndim=2),
    reward_offsets: wp.array(dtype=wp.float32, ndim=2),
    env_origins: wp.array(dtype=wp.float32, ndim=2),
    r_pos: wp.array(dtype=wp.float32, ndim=3),
    r_quat: wp.array(dtype=wp.float32, ndim=3),
    r_lin: wp.array(dtype=wp.float32, ndim=3),
    r_ang: wp.array(dtype=wp.float32, ndim=3),
    anchor_rcol: int,
    body_rcols: wp.array(dtype=wp.int32),
    vr_rcols: wp.array(dtype=wp.int32),
    reward_rcols: wp.array(dtype=wp.int32),
    cmd_buf: wp.array(dtype=wp.float32, ndim=2),
    anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    anchor_lin: wp.array(dtype=wp.float32, ndim=2),
    anchor_ang: wp.array(dtype=wp.float32, ndim=2),
    body_pos: wp.array(dtype=wp.float32, ndim=3),
    body_quat: wp.array(dtype=wp.float32, ndim=3),
    body_lin: wp.array(dtype=wp.float32, ndim=3),
    body_ang: wp.array(dtype=wp.float32, ndim=3),
    vr_pos: wp.array(dtype=wp.float32, ndim=3),
    vr_quat: wp.array(dtype=wp.float32, ndim=3),
    reward_pos: wp.array(dtype=wp.float32, ndim=3),
    r_anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    r_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    r_body_pos: wp.array(dtype=wp.float32, ndim=3),
    r_body_quat: wp.array(dtype=wp.float32, ndim=3),
    r_body_lin: wp.array(dtype=wp.float32, ndim=3),
    r_body_ang: wp.array(dtype=wp.float32, ndim=3),
    r_vr_pos: wp.array(dtype=wp.float32, ndim=3),
    r_reward_pos: wp.array(dtype=wp.float32, ndim=3),
    rel_pos: wp.array(dtype=wp.float32, ndim=3),
    rel_quat: wp.array(dtype=wp.float32, ndim=3),
):
    i = wp.tid()
    row = int(ts[i] + motion_start[i] + length_starts[int(motion_ids[i])])
    num_bodies = body_pos.shape[1]
    num_vr = vr_mcols.shape[0]
    num_rp = reward_mcols.shape[0]

    # -- command buffer: [reference joint_pos | reference joint_vel] (current frame)
    num_j = mj_pos.shape[1]
    for j in range(num_j):
        cmd_buf[i, j] = mj_pos[row, j]
        cmd_buf[i, num_j + j] = mj_vel[row, j]

    # -- reference anchor (motion frame + env origins)
    ox = env_origins[i, 0]
    oy = env_origins[i, 1]
    oz = env_origins[i, 2]
    a_px = m_pos[row, anchor_mcol, 0] + ox
    a_py = m_pos[row, anchor_mcol, 1] + oy
    a_pz = m_pos[row, anchor_mcol, 2] + oz
    anchor_pos[i, 0] = a_px
    anchor_pos[i, 1] = a_py
    anchor_pos[i, 2] = a_pz
    anchor_quat[i, 0] = m_quat[row, anchor_mcol, 1]
    anchor_quat[i, 1] = m_quat[row, anchor_mcol, 2]
    anchor_quat[i, 2] = m_quat[row, anchor_mcol, 3]
    anchor_quat[i, 3] = m_quat[row, anchor_mcol, 0]
    q_a = wp.vec4f(anchor_quat[i, 0], anchor_quat[i, 1], anchor_quat[i, 2], anchor_quat[i, 3])
    for k in range(3):
        anchor_lin[i, k] = m_lin[row, anchor_mcol, k]
        anchor_ang[i, k] = m_ang[row, anchor_mcol, k]

    # -- robot anchor snapshot
    r_anchor_pos[i, 0] = r_pos[i, anchor_rcol, 0]
    r_anchor_pos[i, 1] = r_pos[i, anchor_rcol, 1]
    r_anchor_pos[i, 2] = r_pos[i, anchor_rcol, 2]
    r_anchor_quat[i, 0] = r_quat[i, anchor_rcol, 0]
    r_anchor_quat[i, 1] = r_quat[i, anchor_rcol, 1]
    r_anchor_quat[i, 2] = r_quat[i, anchor_rcol, 2]
    r_anchor_quat[i, 3] = r_quat[i, anchor_rcol, 3]
    q_r = wp.vec4f(
        r_anchor_quat[i, 0], r_anchor_quat[i, 1], r_anchor_quat[i, 2], r_anchor_quat[i, 3]
    )

    # -- delta transform (sonic tail semantics: xy from robot anchor, z from
    #    reference anchor; orientation delta is yaw-only in XYZW)
    delta_ori = _delta_ori_xyzw(q_r, q_a)
    delta_pos = wp.vec3f(r_anchor_pos[i, 0], r_anchor_pos[i, 1], a_pz)

    # -- per-body reference state, robot snapshots and relative targets
    for b in range(num_bodies):
        col = body_rcols[b]

        body_pos[i, b, 0] = m_pos[row, b, 0] + ox
        body_pos[i, b, 1] = m_pos[row, b, 1] + oy
        body_pos[i, b, 2] = m_pos[row, b, 2] + oz
        body_quat[i, b, 0] = m_quat[row, b, 1]
        body_quat[i, b, 1] = m_quat[row, b, 2]
        body_quat[i, b, 2] = m_quat[row, b, 3]
        body_quat[i, b, 3] = m_quat[row, b, 0]
        for k in range(3):
            body_lin[i, b, k] = m_lin[row, b, k]
            body_ang[i, b, k] = m_ang[row, b, k]

        r_body_pos[i, b, 0] = r_pos[i, col, 0]
        r_body_pos[i, b, 1] = r_pos[i, col, 1]
        r_body_pos[i, b, 2] = r_pos[i, col, 2]
        r_body_quat[i, b, 0] = r_quat[i, col, 0]
        r_body_quat[i, b, 1] = r_quat[i, col, 1]
        r_body_quat[i, b, 2] = r_quat[i, col, 2]
        r_body_quat[i, b, 3] = r_quat[i, col, 3]
        for k in range(3):
            r_body_lin[i, b, k] = r_lin[i, col, k]
            r_body_ang[i, b, k] = r_ang[i, col, k]

        q_b = _motion_quat_xyzw(m_quat, row, b)
        q_rel = quat_mul_xyzw(delta_ori, q_b)
        rel_quat[i, b, 0] = q_rel[0]
        rel_quat[i, b, 1] = q_rel[1]
        rel_quat[i, b, 2] = q_rel[2]
        rel_quat[i, b, 3] = q_rel[3]

        v = wp.vec3f(
            m_pos[row, b, 0] + ox - a_px,
            m_pos[row, b, 1] + oy - a_py,
            m_pos[row, b, 2] + oz - a_pz,
        )
        rv = quat_rotate_xyzw(delta_ori, v)
        rel_pos[i, b, 0] = delta_pos[0] + rv[0]
        rel_pos[i, b, 1] = delta_pos[1] + rv[1]
        rel_pos[i, b, 2] = delta_pos[2] + rv[2]

    # -- VR 3-point reference (motion pos + quat-rotated offsets + origins) and
    #    robot VR point snapshots
    for k in range(num_vr):
        mc = vr_mcols[k]
        rc = vr_rcols[k]
        off = quat_rotate_xyzw(
            _motion_quat_xyzw(m_quat, row, mc),
            wp.vec3f(vr_offsets[k, 0], vr_offsets[k, 1], vr_offsets[k, 2]),
        )
        vr_pos[i, k, 0] = m_pos[row, mc, 0] + ox + off[0]
        vr_pos[i, k, 1] = m_pos[row, mc, 1] + oy + off[1]
        vr_pos[i, k, 2] = m_pos[row, mc, 2] + oz + off[2]
        vr_quat[i, k, 0] = m_quat[row, mc, 1]
        vr_quat[i, k, 1] = m_quat[row, mc, 2]
        vr_quat[i, k, 2] = m_quat[row, mc, 3]
        vr_quat[i, k, 3] = m_quat[row, mc, 0]

        q_rb = wp.vec4f(r_quat[i, rc, 0], r_quat[i, rc, 1], r_quat[i, rc, 2], r_quat[i, rc, 3])
        roff = quat_rotate_xyzw(
            q_rb, wp.vec3f(vr_offsets[k, 0], vr_offsets[k, 1], vr_offsets[k, 2])
        )
        r_vr_pos[i, k, 0] = r_pos[i, rc, 0] + roff[0]
        r_vr_pos[i, k, 1] = r_pos[i, rc, 1] + roff[1]
        r_vr_pos[i, k, 2] = r_pos[i, rc, 2] + roff[2]

    # -- reward-point reference (with offsets) and robot reward-point snapshots
    for k in range(num_rp):
        mc = reward_mcols[k]
        rc = reward_rcols[k]
        off = quat_rotate_xyzw(
            _motion_quat_xyzw(m_quat, row, mc),
            wp.vec3f(reward_offsets[k, 0], reward_offsets[k, 1], reward_offsets[k, 2]),
        )
        reward_pos[i, k, 0] = m_pos[row, mc, 0] + ox + off[0]
        reward_pos[i, k, 1] = m_pos[row, mc, 1] + oy + off[1]
        reward_pos[i, k, 2] = m_pos[row, mc, 2] + oz + off[2]

        q_rb = wp.vec4f(r_quat[i, rc, 0], r_quat[i, rc, 1], r_quat[i, rc, 2], r_quat[i, rc, 3])
        roff = quat_rotate_xyzw(
            q_rb, wp.vec3f(reward_offsets[k, 0], reward_offsets[k, 1], reward_offsets[k, 2])
        )
        r_reward_pos[i, k, 0] = r_pos[i, rc, 0] + roff[0]
        r_reward_pos[i, k, 1] = r_pos[i, rc, 1] + roff[1]
        r_reward_pos[i, k, 2] = r_pos[i, rc, 2] + roff[2]


class TrackingCommandWarp(TrackingCommand):
    """Tracking command whose steady-state update runs as Warp kernels."""

    def __init__(self, cfg: TrackingCommandWarpCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # ManagerBasedRLEnvWarp creates the command manager before the
        # observation manager. Register this env so deferred history
        # descriptors can resolve dimensions during ObservationManager init.
        register_env_hint(env)

        # -- unsupported-in-Warp configs: fail loudly instead of silently diverging
        if self.cfg.use_height_map:
            raise NotImplementedError(
                "[TrackingCommandWarp] cfg.use_height_map is not ported."
            )
        if self.use_ref_motion_root_quat_w_as_anchor:
            raise NotImplementedError(
                "[TrackingCommandWarp] use_ref_motion_root_quat_w_as_anchor=True is not ported "
                "Use the Torch environment for this configuration."
            )

        num_envs = self.num_envs
        num_bodies = len(self.cfg.body_names)
        num_joints_motion = int(self.motion_lib.dof_pos.shape[1])
        num_vr = len(self.cfg.vr_3point_body)
        num_rp = len(self.cfg.reward_point_body)

        def persistent(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.float32, device=self.device)

        # -- persistent snapshots consumed by the 3c obs/reward/termination kernels
        self._command_buf = persistent(num_envs, 2 * num_joints_motion)
        self._anchor_pos_w = persistent(num_envs, 3)
        self._anchor_quat_w = persistent(num_envs, 4)
        self._anchor_lin_vel_w = persistent(num_envs, 3)
        self._anchor_ang_vel_w = persistent(num_envs, 3)
        self._body_pos_w = persistent(num_envs, num_bodies, 3)
        self._body_quat_w = persistent(num_envs, num_bodies, 4)
        self._body_lin_vel_w = persistent(num_envs, num_bodies, 3)
        self._body_ang_vel_w = persistent(num_envs, num_bodies, 3)
        self._vr_3point_body_pos_w = persistent(num_envs, num_vr, 3)
        self._vr_3point_body_quat_w = persistent(num_envs, num_vr, 4)
        self._reward_point_body_pos_w = persistent(num_envs, num_rp, 3)
        self._robot_anchor_pos_w = persistent(num_envs, 3)
        self._robot_anchor_quat_w = persistent(num_envs, 4)
        self._robot_body_pos_w = persistent(num_envs, num_bodies, 3)
        self._robot_body_quat_w = persistent(num_envs, num_bodies, 4)
        self._robot_body_lin_vel_w = persistent(num_envs, num_bodies, 3)
        self._robot_body_ang_vel_w = persistent(num_envs, num_bodies, 3)
        self._robot_vr_3point_pos_w = persistent(num_envs, num_vr, 3)
        self._robot_reward_point_body_pos_w = persistent(num_envs, num_rp, 3)
        self._running_ref_root_height = persistent(num_envs)
        # anchor-relative reference targets (canonical storage: the base
        # attributes are rebound to these so torch readers and kernels agree)
        self._body_pos_relative_w_buf = persistent(num_envs, num_bodies, 3)
        self._body_quat_relative_w_buf = persistent(num_envs, num_bodies, 4)

        self.wp_command = wp.from_torch(self._command_buf)
        self.wp_anchor_pos_w = wp.from_torch(self._anchor_pos_w)
        self.wp_anchor_quat_w = wp.from_torch(self._anchor_quat_w)
        self.wp_anchor_lin_vel_w = wp.from_torch(self._anchor_lin_vel_w)
        self.wp_anchor_ang_vel_w = wp.from_torch(self._anchor_ang_vel_w)
        self.wp_body_pos_w = wp.from_torch(self._body_pos_w)
        self.wp_body_quat_w = wp.from_torch(self._body_quat_w)
        self.wp_body_lin_vel_w = wp.from_torch(self._body_lin_vel_w)
        self.wp_body_ang_vel_w = wp.from_torch(self._body_ang_vel_w)
        self.wp_vr_3point_body_pos_w = wp.from_torch(self._vr_3point_body_pos_w)
        self.wp_vr_3point_body_quat_w = wp.from_torch(self._vr_3point_body_quat_w)
        self.wp_reward_point_body_pos_w = wp.from_torch(self._reward_point_body_pos_w)
        self.wp_robot_anchor_pos_w = wp.from_torch(self._robot_anchor_pos_w)
        self.wp_robot_anchor_quat_w = wp.from_torch(self._robot_anchor_quat_w)
        self.wp_robot_body_pos_w = wp.from_torch(self._robot_body_pos_w)
        self.wp_robot_body_quat_w = wp.from_torch(self._robot_body_quat_w)
        self.wp_robot_body_lin_vel_w = wp.from_torch(self._robot_body_lin_vel_w)
        self.wp_robot_body_ang_vel_w = wp.from_torch(self._robot_body_ang_vel_w)
        self.wp_robot_vr_3point_pos_w = wp.from_torch(self._robot_vr_3point_pos_w)
        self.wp_robot_reward_point_body_pos_w = wp.from_torch(self._robot_reward_point_body_pos_w)
        self.wp_running_ref_root_height = wp.from_torch(self._running_ref_root_height)
        self.wp_body_pos_rel_w = wp.from_torch(self._body_pos_relative_w_buf)
        self.wp_body_quat_rel_w = wp.from_torch(self._body_quat_relative_w_buf)

        # Warp computes in Lab's XYZW order; inherited Torch command properties
        # must continue exposing SONIC's WXYZ order to bridged task terms.
        self._legacy_quat_views = []
        for name in (
            "_anchor_quat_w", "_body_quat_w", "_vr_3point_body_quat_w",
            "_robot_anchor_quat_w", "_robot_body_quat_w", "_body_quat_relative_w_buf",
        ):
            source = getattr(self, name)
            legacy = torch.zeros_like(source)
            setattr(self, f"_legacy_{name.lstrip('_')}", legacy)
            self._legacy_quat_views.append((wp.from_torch(source), wp.from_torch(legacy), source.shape))

        # -- robot-state shadows (Stage-3d CUDA-graph discipline): the Newton
        # articulation exposes body/joint state as STRIDED views into packed
        # buffers; warp cannot view those zero-copy (contiguity required), so
        # _live_wp's .contiguous() mints a fresh temp per call.  A captured
        # manager graph bakes that temp's pointer and replay later reads
        # allocator-recycled storage (CUDA error 700).  Persistent contiguous
        # shadows plus a warp gather keep the pointers fixed; the gather is
        # launched from _robot_wp_views(), so eager callers sync per step and
        # captured stages record it into the graph (replaying every step).
        # Sources are storage-stable (verified: data_ptr constant across
        # steps/resets), so the baked source pointers stay valid.
        _d = self.robot.data
        self._robot_sync = []
        for _name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
            _src = as_torch(getattr(_d, _name))
            if not torch.is_tensor(_src):
                _src = wp.to_torch(_src)
            _shadow = torch.empty(tuple(_src.shape), dtype=_src.dtype, device=self.device)
            _src_wp = wp.array(
                ptr=_src.data_ptr(),
                shape=tuple(_src.shape),
                strides=tuple(s * _src.element_size() for s in _src.stride()),
                dtype=wp.float32,
                device=self.device,
            )
            # keep the resolved source tensor referenced: without an owner the
            # caching allocator may hand its block back to CUDA under OOM-retry
            # churn (learning phase), leaving the baked gather pointers unmapped
            self._robot_sync.append((_src, _src_wp, wp.from_torch(_shadow), tuple(_shadow.shape)))
        self._robot_views = tuple(e[2] for e in self._robot_sync)
        self._robot_sync_step = -1

        # -- immutable kernel inputs: column ids and per-point offsets
        self._anchor_mcol = int(self.motion_anchor_body_index)
        self._anchor_rcol = int(self.robot_anchor_body_index)
        self._wp_body_rcols = wp.array(
            [int(v) for v in self.body_indexes.tolist()], dtype=wp.int32, device=self.device
        )
        self._wp_vr_rcols = wp.array(
            [int(v) for v in self.vr_3point_body_indices], dtype=wp.int32, device=self.device
        )
        self._wp_reward_rcols = wp.array(
            [int(v) for v in self.reward_point_body_indices], dtype=wp.int32, device=self.device
        )
        self._wp_vr_mcols = wp.array(
            [int(v) for v in self.vr_3point_body_indices_motion],
            dtype=wp.int32,
            device=self.device,
        )
        self._wp_reward_mcols = wp.array(
            [int(v) for v in self.reward_point_body_indices_motion],
            dtype=wp.int32,
            device=self.device,
        )
        self._wp_vr_offsets = _live_wp(self.vr_3point_body_offsets.reshape(-1, 3))
        self._wp_reward_offsets = _live_wp(self.reward_point_body_offsets.reshape(-1, 3))
        self._wp_env_origins = _live_wp(self._env.scene.env_origins)
        self._wp_ts = wp.from_torch(self.time_steps)
        # per-env clip lengths (device copy, refreshed with the host mirror at
        # resamples) -- consumed by the tracking_time_out termination kernel
        self._dev_total = torch.empty(num_envs, dtype=torch.int64, device=self.device)
        self._wp_total = wp.from_torch(self._dev_total)
        self._dof_mismatch = 1 if self.has_dof_mismatch else 0
        body_joint_indices = getattr(self, "body_joint_indices", None)
        self._wp_body_joint_idx = wp.array(
            [int(v) for v in body_joint_indices.tolist()] if body_joint_indices is not None else [],
            dtype=wp.int32,
            device=self.device,
        )

        # -- metric outputs materialized up front (persistent storage for kernels)
        metric_keys = (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
        )
        for key in metric_keys:
            self.metrics.setdefault(
                key, torch.zeros(num_envs, dtype=torch.float32, device=self.device)
            )
        self._wp_metrics = tuple(wp.from_torch(self.metrics[key]) for key in metric_keys)

        # -- torch-computed copy sources (tokenizer/critic), filled once per step.
        #    Allocated from a one-time base-property call so shapes always match
        #    the shipped layouts (incl. the scrambled command_multi_future cat).
        self._copy_buffers: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for attr, value in (
                ("copy_command_z", self.command_z),
                ("copy_command_z_multi_future", self.command_z_multi_future),
                ("copy_command_multi_future", self.command_multi_future),
                ("copy_root_rot_dif_l_multi_future", self.root_rot_dif_l_multi_future),
                ("copy_command_multi_future_lower_body", self.command_multi_future_lower_body),
                ("copy_joint_pos_multi_future_for_smpl", self.joint_pos_multi_future_for_smpl),
                (
                    "copy_smpl_root_quat_w_dif_l_multi_future",
                    self.smpl_root_quat_w_dif_l_multi_future,
                ),
            ):
                flat = value.reshape(num_envs, -1).to(torch.float32)
                buf = torch.empty_like(flat)
                buf.copy_(flat)
                setattr(self, f"_{attr}", buf)
                self._copy_buffers[attr] = buf
            # smpl joints local: per-frame own-root canonicalization (the
            # smpl_joints_multi_future_local obs transform, applied once here)
            ref_joints = self.smpl_joints_multi_future
            ref_roots = self.smpl_root_quat_w_multi_future.unsqueeze(-2).repeat(
                1, 1, ref_joints.shape[-2], 1
            )
            local = quat_apply(quat_inv(ref_roots), ref_joints).reshape(num_envs, -1)
            buf = torch.empty_like(local)
            buf.copy_(local)
            self._copy_smpl_joints_multi_future_local = buf
            self._copy_buffers["copy_smpl_joints_multi_future_local"] = buf

        # per-buffer future-frame counts (torch non_flatten semantics: the
        # terms reshape the flat row to (N, frames, width)) for the obs
        # copy-term dim resolver
        nf = int(self.num_future_frames)
        sf = int(self.smpl_num_future_frames)
        self._copy_frames: dict[str, int] = {
            "copy_command_z": 1,
            "copy_command_z_multi_future": nf,
            "copy_command_multi_future": nf,
            "copy_root_rot_dif_l_multi_future": nf,
            "copy_command_multi_future_lower_body": nf,
            "copy_joint_pos_multi_future_for_smpl": sf,
            "copy_smpl_root_quat_w_dif_l_multi_future": sf,
            "copy_smpl_joints_multi_future_local": sf,
        }

        # -- motion-lib warp views (re-taken after resamples: wrist/upper-body
        #    augmentations may rebind the storage tensors), then the host
        #    mirror of the expiry state (sync-free overflow detection)
        self._refresh_motion_views()
        self._sync_host_mirror()

        # -- rel-target canonical storage + readiness flags
        self.body_pos_relative_w = self._body_pos_relative_w_buf
        self.body_quat_relative_w = self._legacy_body_quat_relative_w_buf
        self._persistent_ready = True
        self._warp_ready = True

    # ------------------------------------------------------------------
    # property overrides: persistent buffers once warm, base otherwise
    # (the base resample path reads these properties expecting fresh values
    # while the buffers still hold the pre-resample state -> fall-through)
    # ------------------------------------------------------------------

    def _buf(self, name: str):
        if getattr(self, "_persistent_ready", False):
            return getattr(self, name)
        return None

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        buf = self._buf("_anchor_pos_w")
        return buf if buf is not None else TrackingCommand.anchor_pos_w.fget(self)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        buf = self._buf("_legacy_anchor_quat_w")
        return buf if buf is not None else TrackingCommand.anchor_quat_w.fget(self)

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        buf = self._buf("_anchor_lin_vel_w")
        return buf if buf is not None else TrackingCommand.anchor_lin_vel_w.fget(self)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        buf = self._buf("_anchor_ang_vel_w")
        return buf if buf is not None else TrackingCommand.anchor_ang_vel_w.fget(self)

    @property
    def body_pos_w(self) -> torch.Tensor:
        buf = self._buf("_body_pos_w")
        return buf if buf is not None else TrackingCommand.body_pos_w.fget(self)

    @property
    def body_quat_w(self) -> torch.Tensor:
        buf = self._buf("_legacy_body_quat_w")
        return buf if buf is not None else TrackingCommand.body_quat_w.fget(self)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        buf = self._buf("_body_lin_vel_w")
        return buf if buf is not None else TrackingCommand.body_lin_vel_w.fget(self)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        buf = self._buf("_body_ang_vel_w")
        return buf if buf is not None else TrackingCommand.body_ang_vel_w.fget(self)

    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        buf = self._buf("_vr_3point_body_pos_w")
        return buf if buf is not None else TrackingCommand.vr_3point_body_pos_w.fget(self)

    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        buf = self._buf("_legacy_vr_3point_body_quat_w")
        return buf if buf is not None else TrackingCommand.vr_3point_body_quat_w.fget(self)

    @property
    def reward_point_body_pos_w(self) -> torch.Tensor:
        buf = self._buf("_reward_point_body_pos_w")
        return buf if buf is not None else TrackingCommand.reward_point_body_pos_w.fget(self)

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return TrackingCommand.robot_anchor_pos_w.fget(self)

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return TrackingCommand.robot_anchor_quat_w.fget(self)

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return TrackingCommand.robot_body_pos_w.fget(self)

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return TrackingCommand.robot_body_quat_w.fget(self)

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return TrackingCommand.robot_body_lin_vel_w.fget(self)

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return TrackingCommand.robot_body_ang_vel_w.fget(self)

    @property
    def robot_vr_3point_pos_w(self) -> torch.Tensor:
        return TrackingCommand.robot_vr_3point_pos_w.fget(self)

    @property
    def robot_reward_point_body_pos_w(self) -> torch.Tensor:
        return TrackingCommand.robot_reward_point_body_pos_w.fget(self)

    @property
    def running_ref_root_height(self) -> torch.Tensor:
        # plain attribute in the base (no property to defer to): pre-init
        # values live in a shadow slot until the persistent buffer takes over
        if getattr(self, "_persistent_ready", False):
            return self._running_ref_root_height
        return self._base_running_ref_root_height

    @running_ref_root_height.setter
    def running_ref_root_height(self, value) -> None:
        if getattr(self, "_persistent_ready", False):
            with torch.no_grad():
                self._running_ref_root_height.copy_(torch.as_tensor(value).reshape(-1))
        else:
            self._base_running_ref_root_height = value

    @property
    def command(self) -> torch.Tensor:
        # persistent (kernel-filled) instead of a fresh torch.cat, so the
        # official Warp generated_commands term keeps a valid view
        buf = self._buf("_command_buf")
        return buf if buf is not None else TrackingCommand.command.fget(self)

    # ------------------------------------------------------------------
    # warp helpers
    # ------------------------------------------------------------------

    def _robot_wp_views(self):
        """Warp views of the live robot state over persistent shadow buffers.

        Refreshes the shadows by launching the strided-source gather kernels.
        Eager callers sync once per env step (state only changes in
        scene.update / resets between steps); during CUDA-graph capture the
        launch is always performed so it gets RECORDED -- replays then
        re-gather the live state every step.  Never mints per-call temps:
        captured graphs must not bake pointers into allocator-recycled
        storage.
        """
        capturing = wp.get_device().is_capturing
        marker = getattr(self._env, "common_step_counter", None)
        if capturing or marker is None or marker != self._robot_sync_step:
            for _keepalive, src, dst, shape in self._robot_sync:
                if len(shape) == 3:
                    wp.launch(kernel=_gather_contig3d, dim=shape, inputs=[src, dst], device=self.device)
                else:
                    wp.launch(
                        kernel=_gather_contig2d,
                        dim=(shape[0], shape[1]),
                        inputs=[src, dst],
                        device=self.device,
                    )
            if marker is not None:
                self._robot_sync_step = marker
        return self._robot_views

    def _refresh_motion_views(self):
        """Refresh Warp views; kernels reorder WXYZ reference quaternions on read."""
        ml = self.motion_lib
        if "mujoco_to_isaaclab_body" not in ml.m_cfg:
            raise ValueError("Warp tracking requires the WXYZ motion-library body mapping.")
        self._wp_motion = (
            _live_wp(ml.dof_pos),
            _live_wp(ml.dof_vel),
            _live_wp(ml.body_pos_w),
            _live_wp(ml.body_quat_w),
            _live_wp(ml.body_lin_vel_w),
            _live_wp(ml.body_ang_vel_w),
        )
        if ml.length_starts.dtype == torch.int64:
            self._wp_length_starts = wp.from_torch(ml.length_starts)
        else:
            self._wp_length_starts = wp.from_torch(ml.length_starts.to(torch.int64))
        # motion_ids / motion_start_time_steps are written in-place at resample;
        # keep int64 device copies so kernel signatures stay fixed (refreshed
        # together with the host mirror in _sync_host_mirror).  Allocated ONCE
        # and never rebound: captured manager graphs bake these pointers, and
        # a rebind would drop the old tensors whose storage the caching
        # allocator can hand back to CUDA -- graph replay then reads unmapped
        # memory (CUDA error 700).  Content refresh happens exclusively via
        # _sync_host_mirror's in-place copy_.
        if not hasattr(self, "_wp_motion_ids"):
            self._dev_motion_ids = torch.empty(self.num_envs, dtype=torch.int64, device=self.device)
            self._dev_motion_start = torch.empty(self.num_envs, dtype=torch.int64, device=self.device)
            self._wp_motion_ids = wp.from_torch(self._dev_motion_ids)
            self._wp_motion_start = wp.from_torch(self._dev_motion_start)

    def _sync_host_mirror(self):
        """Refresh the host-side expiry mirror (and the int64 id/start copies)."""
        # NOTE: the device copies MUST refresh unconditionally -- they were
        # created via a dtype conversion (separate storage), so the base class
        # writes to motion_ids/motion_start_time_steps in place at every
        # resample without ever changing their data_ptr.
        self._dev_motion_ids.copy_(self.motion_ids)
        self._dev_motion_start.copy_(self.motion_start_time_steps)
        start = self.motion_start_time_steps.detach().cpu().numpy().astype(np.int64)
        total_dev = self.motion_lib.get_time_step_total(self.motion_ids).to(torch.int64)
        self._dev_total.copy_(total_dev)
        total = total_dev.detach().cpu().numpy().astype(np.int64)
        self._host_time_steps = self.time_steps.detach().cpu().numpy().astype(np.int64)
        self._host_start = start
        self._host_total = total

    # ------------------------------------------------------------------
    # per-step overrides
    # ------------------------------------------------------------------

    def _update_metrics(self):
        if not getattr(self, "_warp_ready", False):
            super()._update_metrics()
            return
        r_pos, r_quat, r_lin, r_ang, r_jpos, r_jvel = self._robot_wp_views()
        mj_pos, mj_vel, m_pos, m_quat, m_lin, m_ang = self._wp_motion
        wp.launch(
            kernel=_metrics_kernel,
            dim=self.num_envs,
            inputs=[
                self._wp_ts,
                self._wp_motion_ids,
                self._wp_motion_start,
                self._wp_length_starts,
                mj_pos,
                mj_vel,
                m_pos,
                m_quat,
                m_lin,
                m_ang,
                self._dof_mismatch,
                self._wp_body_joint_idx,
                self._anchor_mcol,
                self._anchor_rcol,
                self._wp_body_rcols,
                self._wp_env_origins,
                r_pos,
                r_quat,
                r_lin,
                r_ang,
                r_jpos,
                r_jvel,
                self.wp_body_pos_rel_w,
                self.wp_body_quat_rel_w,
                *self._wp_metrics,
            ],
            device=self.device,
        )

    def _launch_reference_update(self):
        r_pos, r_quat, r_lin, r_ang, _, _ = self._robot_wp_views()
        mj_pos, mj_vel, m_pos, m_quat, m_lin, m_ang = self._wp_motion
        wp.launch(
            kernel=_reference_kernel,
            dim=self.num_envs,
            inputs=[
                self._wp_ts,
                self._wp_motion_ids,
                self._wp_motion_start,
                self._wp_length_starts,
                mj_pos,
                mj_vel,
                m_pos,
                m_quat,
                m_lin,
                m_ang,
                self._anchor_mcol,
                self._wp_vr_mcols,
                self._wp_reward_mcols,
                self._wp_vr_offsets,
                self._wp_reward_offsets,
                self._wp_env_origins,
                r_pos,
                r_quat,
                r_lin,
                r_ang,
                self._anchor_rcol,
                self._wp_body_rcols,
                self._wp_vr_rcols,
                self._wp_reward_rcols,
                self.wp_command,
                self.wp_anchor_pos_w,
                self.wp_anchor_quat_w,
                self.wp_anchor_lin_vel_w,
                self.wp_anchor_ang_vel_w,
                self.wp_body_pos_w,
                self.wp_body_quat_w,
                self.wp_body_lin_vel_w,
                self.wp_body_ang_vel_w,
                self.wp_vr_3point_body_pos_w,
                self.wp_vr_3point_body_quat_w,
                self.wp_reward_point_body_pos_w,
                self.wp_robot_anchor_pos_w,
                self.wp_robot_anchor_quat_w,
                self.wp_robot_body_pos_w,
                self.wp_robot_body_quat_w,
                self.wp_robot_body_lin_vel_w,
                self.wp_robot_body_ang_vel_w,
                self.wp_robot_vr_3point_pos_w,
                self.wp_robot_reward_point_body_pos_w,
                self.wp_body_pos_rel_w,
                self.wp_body_quat_rel_w,
            ],
            device=self.device,
        )
        for source, legacy, shape in self._legacy_quat_views:
            if len(shape) == 3:
                wp.launch(kernel=_xyzw_to_wxyz3d, dim=shape[:2], inputs=[source, legacy], device=self.device)
            else:
                wp.launch(kernel=_xyzw_to_wxyz2d, dim=shape[0], inputs=[source, legacy], device=self.device)

    def _fill_persistent(self):
        """Reference kernel + EMA + the once-per-step torch copy-source fills."""
        self._launch_reference_update()
        # base EMA: 0.1 * anchor_z + 0.9 * prev (anchor read post-advance, i.e.
        # the buffer just filled by the kernel)
        self._running_ref_root_height.lerp_(self._anchor_pos_w[:, 2], 0.1)
        with torch.no_grad():
            N = self.num_envs
            self._copy_command_z.copy_(self.command_z.reshape(N, -1))
            self._copy_command_z_multi_future.copy_(
                self.command_z_multi_future.reshape(N, -1)
            )
            self._copy_command_multi_future.copy_(
                self.command_multi_future.reshape(N, -1)
            )
            self._copy_root_rot_dif_l_multi_future.copy_(
                self.root_rot_dif_l_multi_future.reshape(N, -1)
            )
            self._copy_command_multi_future_lower_body.copy_(
                self.command_multi_future_lower_body.reshape(N, -1)
            )
            self._copy_joint_pos_multi_future_for_smpl.copy_(
                self.joint_pos_multi_future_for_smpl.reshape(N, -1)
            )
            self._copy_smpl_root_quat_w_dif_l_multi_future.copy_(
                self.smpl_root_quat_w_dif_l_multi_future.reshape(N, -1)
            )
            ref_joints = self.smpl_joints_multi_future
            ref_roots = self.smpl_root_quat_w_multi_future.unsqueeze(-2).repeat(
                1, 1, ref_joints.shape[-2], 1
            )
            local = quat_apply(quat_inv(ref_roots), ref_joints).reshape(N, -1)
            self._copy_smpl_joints_multi_future_local.copy_(local)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # robot state mutates during env resets (write_data_to_sim + root
        # state resets) after the last per-step gather, so post-reset obs
        # consumers must re-gather instead of reading the step-cached shadows
        self._robot_sync_step = -1
        return super().reset(env_ids)

    def _update_command(self):
        if not getattr(self, "_warp_ready", False):
            super()._update_command()
            return
        if self.use_adaptive_sampling:
            with common.Timer("update_adaptive_sampling"):
                if self._adp_snapshot_valid:
                    self.motion_lib.update_adaptive_sampling(
                        self._env.reset_terminated,
                        self._adp_prev_motion_ids,
                        self._adp_prev_cursor,
                    )
                    # A snapshot is valid for exactly one physics step.
                    self._adp_snapshot_valid = False
        # advance the cursor (in-place, device-side) and mirror host-side
        self.time_steps += 1
        self._host_time_steps += 1
        overflow = np.nonzero(
            self._host_time_steps + self._host_start >= self._host_total
        )[0]
        if overflow.size > 0:
            # rare path: classic torch resample (motion sampling + sim writes);
            # the override re-syncs the mirror and refills the buffers
            env_ids = torch.as_tensor(overflow, dtype=torch.long, device=self.device)
            self._resample_command(env_ids)
        else:
            self._fill_persistent()

    def _resample_command(self, env_ids: Sequence[int]):
        if not getattr(self, "_warp_ready", False):
            super()._resample_command(env_ids)
            return
        # base resample reads the lazy properties expecting fresh values while
        # the persistent buffers still hold the pre-resample state -> fall through
        self._persistent_ready = False
        try:
            super()._resample_command(env_ids)
        finally:
            self._persistent_ready = True
        # the base tail rebinds the rel-target attributes to fresh tensors;
        # point them back at the canonical buffers
        self.body_pos_relative_w = self._body_pos_relative_w_buf
        self.body_quat_relative_w = self._legacy_body_quat_relative_w_buf
        if len(env_ids) > 0:
            self._refresh_motion_views()
            self._sync_host_mirror()
            self._fill_persistent()
            # base parity (line 3002): the resample hard-resets the EMA height
            # of the resampled envs to the NEW anchor z (during the super()
            # call the getter fall-through routed that in-place write into the
            # shadow slot instead of the persistent buffer)
            env_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            with torch.no_grad():
                self._running_ref_root_height[env_t] = self._anchor_pos_w[env_t, 2]


@configclass
class TrackingCommandWarpCfg(TrackingCommandCfg):
    """Configuration for the Warp-consumable tracking command."""

    class_type: type = TrackingCommandWarp
