"""Warp-first observation terms for the tracking command (Stage 3c).

Two families, both reading :class:`mdp_warp.commands.TrackingCommandWarp`'s
persistent buffers (filled once per step by the command's kernels / torch
fills), so the observation manager stage becomes pure Warp launches and is
CUDA-graph capturable:

* real kernels -- ``gravity_dir``, ``motion_anchor_pos_b``, ``motion_anchor_ori_b``,
  ``robot_body_pos_b``, ``robot_body_ori_b``, ``vr_3point_local_target``,
  ``vr_3point_local_orn_target`` mirror the torch term math on the reference
  and robot-anchor snapshots;
* copy terms (:func:`make_command_copy_obs`) -- the tokenizer/critic
  multi-future and SMPL sources whose scrambled cat/reshape layouts the
  trained model depends on are computed ONCE per step inside the command and
  flat-copied (optionally joint-selected, optionally noised) into ``out``;
  the source trailing shape is published for the env wrapper's 3D restore
  exactly like the Stage-3a bridge did.

Output dims that depend on the command config (body/point counts, future
frame counts) resolve through a deferred io-descriptor at ObservationManager
init (the command manager is already alive by then), mirroring
``torch_bridge._ProbeDim``.
"""

from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any, Callable

from isaaclab_experimental.envs.utils.io_descriptors import generic_io_descriptor_warp
import torch
import warp as wp

from . import obs_history as _obs_history
from .obs_history import _find_term_params, _parse_noise
from .warp_utils import quat_conj_xyzw, quat_mul_xyzw, quat_rotate_inv_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@wp.kernel
def _gravity_dir_kernel(
    r_quat: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    q = wp.vec4f(r_quat[i, 0], r_quat[i, 1], r_quat[i, 2], r_quat[i, 3])
    v = quat_rotate_inv_xyzw(q, wp.vec3f(0.0, 0.0, -1.0))
    out[i, 0] = v[0]
    out[i, 1] = v[1]
    out[i, 2] = v[2]


@wp.kernel
def _anchor_pos_b_kernel(
    r_pos: wp.array(dtype=wp.float32, ndim=2),
    r_quat: wp.array(dtype=wp.float32, ndim=2),
    m_pos: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    # subtract_frame_transforms(robot_anchor, ref_anchor) -> ref anchor in the
    # robot-anchor local frame
    i = wp.tid()
    q = wp.vec4f(r_quat[i, 0], r_quat[i, 1], r_quat[i, 2], r_quat[i, 3])
    v = quat_rotate_inv_xyzw(
        q,
        wp.vec3f(m_pos[i, 0] - r_pos[i, 0], m_pos[i, 1] - r_pos[i, 1], m_pos[i, 2] - r_pos[i, 2]),
    )
    out[i, 0] = v[0]
    out[i, 1] = v[1]
    out[i, 2] = v[2]


@wp.kernel
def _anchor_ori_b_kernel(
    r_quat: wp.array(dtype=wp.float32, ndim=2),
    m_quat: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    # 6D of quat_mul(quat_inv(robot_anchor_quat), ref_anchor_quat)
    i = wp.tid()
    qr = wp.vec4f(r_quat[i, 0], r_quat[i, 1], r_quat[i, 2], r_quat[i, 3])
    qm = wp.vec4f(m_quat[i, 0], m_quat[i, 1], m_quat[i, 2], m_quat[i, 3])
    q = quat_mul_xyzw(quat_conj_xyzw(qr), qm)
    xx = q[0] * q[0]
    yy = q[1] * q[1]
    zz = q[2] * q[2]
    xy = q[0] * q[1]
    xz = q[0] * q[2]
    yz = q[1] * q[2]
    wz = q[3] * q[2]
    wy = q[3] * q[1]
    wx = q[3] * q[0]
    # row-interleaved first two matrix columns, matching torch
    # matrix_from_quat(ori_b)[..., :2].reshape(N, -1) -> [R00, R01, R10, R11, R20, R21]
    out[i, 0] = 1.0 - 2.0 * (yy + zz)
    out[i, 1] = 2.0 * (xy - wz)
    out[i, 2] = 2.0 * (xy + wz)
    out[i, 3] = 1.0 - 2.0 * (xx + zz)
    out[i, 4] = 2.0 * (xz - wy)
    out[i, 5] = 2.0 * (yz + wx)


@wp.kernel
def _anchor_ori_b_repeat_kernel(
    r_quat: wp.array(dtype=wp.float32, ndim=2),
    m_quat: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    # non_flatten variant: the same current-frame 6D repeated over F slots
    i = wp.tid()
    qr = wp.vec4f(r_quat[i, 0], r_quat[i, 1], r_quat[i, 2], r_quat[i, 3])
    qm = wp.vec4f(m_quat[i, 0], m_quat[i, 1], m_quat[i, 2], m_quat[i, 3])
    q = quat_mul_xyzw(quat_conj_xyzw(qr), qm)
    xx = q[0] * q[0]
    yy = q[1] * q[1]
    zz = q[2] * q[2]
    xy = q[0] * q[1]
    xz = q[0] * q[2]
    yz = q[1] * q[2]
    wz = q[3] * q[2]
    wy = q[3] * q[1]
    wx = q[3] * q[0]
    num_frames = out.shape[1] // 6
    for f in range(num_frames):
        base = f * 6
        # row-interleaved first two matrix columns, matching torch
        # matrix_from_quat(ori_b)[..., :2].reshape(N, -1) -> [R00, R01, R10, R11, R20, R21]
        out[i, base + 0] = 1.0 - 2.0 * (yy + zz)
        out[i, base + 1] = 2.0 * (xy - wz)
        out[i, base + 2] = 2.0 * (xy + wz)
        out[i, base + 3] = 1.0 - 2.0 * (xx + zz)
        out[i, base + 4] = 2.0 * (xz - wy)
        out[i, base + 5] = 2.0 * (yz + wx)


@wp.kernel
def _robot_body_pos_b_kernel(
    r_anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    r_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    r_body_pos: wp.array(dtype=wp.float32, ndim=3),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    q = wp.vec4f(r_anchor_quat[i, 0], r_anchor_quat[i, 1], r_anchor_quat[i, 2], r_anchor_quat[i, 3])
    num_bodies = r_body_pos.shape[1]
    for b in range(num_bodies):
        v = quat_rotate_inv_xyzw(
            q,
            wp.vec3f(
                r_body_pos[i, b, 0] - r_anchor_pos[i, 0],
                r_body_pos[i, b, 1] - r_anchor_pos[i, 1],
                r_body_pos[i, b, 2] - r_anchor_pos[i, 2],
            ),
        )
        out[i, b * 3 + 0] = v[0]
        out[i, b * 3 + 1] = v[1]
        out[i, b * 3 + 2] = v[2]


@wp.kernel
def _robot_body_ori_b_kernel(
    r_anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    r_body_quat: wp.array(dtype=wp.float32, ndim=3),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    i = wp.tid()
    qa = wp.vec4f(r_anchor_quat[i, 0], r_anchor_quat[i, 1], r_anchor_quat[i, 2], r_anchor_quat[i, 3])
    num_bodies = r_body_quat.shape[1]
    for b in range(num_bodies):
        qb = wp.vec4f(
            r_body_quat[i, b, 0], r_body_quat[i, b, 1], r_body_quat[i, b, 2], r_body_quat[i, b, 3]
        )
        q = quat_mul_xyzw(quat_conj_xyzw(qa), qb)
        xx = q[0] * q[0]
        yy = q[1] * q[1]
        zz = q[2] * q[2]
        xy = q[0] * q[1]
        xz = q[0] * q[2]
        yz = q[1] * q[2]
        wz = q[3] * q[2]
        wy = q[3] * q[1]
        wx = q[3] * q[0]
        base = b * 6
        # row-interleaved first two matrix columns, matching torch
        # matrix_from_quat(ori_b)[..., :2].reshape(N, -1) -> [R00, R01, R10, R11, R20, R21]
        out[i, base + 0] = 1.0 - 2.0 * (yy + zz)
        out[i, base + 1] = 2.0 * (xy - wz)
        out[i, base + 2] = 2.0 * (xy + wz)
        out[i, base + 3] = 1.0 - 2.0 * (xx + zz)
        out[i, base + 4] = 2.0 * (xz - wy)
        out[i, base + 5] = 2.0 * (yz + wx)


@wp.kernel
def _vr_local_pos_kernel(
    anchor_pos: wp.array(dtype=wp.float32, ndim=2),
    anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    vr_pos: wp.array(dtype=wp.float32, ndim=3),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    # quat_apply(quat_inv(ref anchor quat), vr_pos - ref anchor pos)
    i = wp.tid()
    q = wp.vec4f(anchor_quat[i, 0], anchor_quat[i, 1], anchor_quat[i, 2], anchor_quat[i, 3])
    num_pts = vr_pos.shape[1]
    for k in range(num_pts):
        v = quat_rotate_inv_xyzw(
            q,
            wp.vec3f(
                vr_pos[i, k, 0] - anchor_pos[i, 0],
                vr_pos[i, k, 1] - anchor_pos[i, 1],
                vr_pos[i, k, 2] - anchor_pos[i, 2],
            ),
        )
        out[i, k * 3 + 0] = v[0]
        out[i, k * 3 + 1] = v[1]
        out[i, k * 3 + 2] = v[2]


@wp.kernel
def _vr_local_quat_kernel(
    anchor_quat: wp.array(dtype=wp.float32, ndim=2),
    vr_quat: wp.array(dtype=wp.float32, ndim=3),
    out: wp.array(dtype=wp.float32, ndim=2),
):
    # quat_mul(quat_inv(ref anchor quat), vr quat)
    i = wp.tid()
    qa = wp.vec4f(anchor_quat[i, 0], anchor_quat[i, 1], anchor_quat[i, 2], anchor_quat[i, 3])
    num_pts = vr_quat.shape[1]
    for k in range(num_pts):
        qk = wp.vec4f(vr_quat[i, k, 0], vr_quat[i, k, 1], vr_quat[i, k, 2], vr_quat[i, k, 3])
        q = quat_mul_xyzw(quat_conj_xyzw(qa), qk)
        # Preserve the existing SONIC observation contract (WXYZ).
        out[i, k * 4 + 0] = q[3]
        out[i, k * 4 + 1] = q[0]
        out[i, k * 4 + 2] = q[1]
        out[i, k * 4 + 3] = q[2]


@wp.kernel
def _copy_f32_kernel(
    src: wp.array(dtype=wp.float32, ndim=2), out: wp.array(dtype=wp.float32, ndim=2)
):
    i, j = wp.tid()
    out[i, j] = src[i, j]


@wp.kernel
def _copy_i64_f32_kernel(
    src: wp.array(dtype=wp.int64, ndim=2), out: wp.array(dtype=wp.float32, ndim=2)
):
    i, j = wp.tid()
    out[i, j] = float(src[i, j])


@wp.kernel
def _select_f32_kernel(
    src: wp.array(dtype=wp.float32, ndim=2),
    sel: wp.array(dtype=wp.int32),
    row_width: int,
    out: wp.array(dtype=wp.float32, ndim=2),
):
    # out[i, f * len(sel) + k] = src[i, f * row_width + sel[k]]
    i, j = wp.tid()
    num_sel = sel.shape[0]
    f = j // num_sel
    k = j - f * num_sel
    out[i, j] = src[i, f * row_width + sel[k]]


@wp.kernel
def _uniform_noise_kernel(
    out: wp.array(dtype=wp.float32, ndim=2),
    lo: float,
    hi: float,
    rng_state: wp.array(dtype=wp.int32, ndim=2),
):
    i, j = wp.tid()
    state = wp.uint32(rng_state[i, j])
    out[i, j] = out[i, j] + wp.randf(state, lo, hi)
    rng_state[i, j] = wp.int32(state + wp.uint32(1))


# --------------------------------------------------------------------------
# term plumbing (deferred dims resolved against the live command term)
# --------------------------------------------------------------------------


def _live_wp(x):
    if isinstance(x, wp.array):
        return x
    x = getattr(x, "torch", x)
    if torch.is_tensor(x):
        if not x.is_contiguous():
            x = x.contiguous()
        return wp.from_torch(x)
    return wp.from_torch(wp.to_torch(x))


class _CmdTermDim:
    """Deferred io-descriptor resolving ``out_dim`` from the live command term.

    The command manager inits before the observation manager, so at the
    ObservationManager's own ``__init__`` the TrackingCommandWarp buffers
    already exist and dims/trailing shapes can be read without calling any
    source term.  Mirrors ``torch_bridge._ProbeDim`` (cfg-time readers get the
    ``"torch_live"`` deferred sentinel).
    """

    __slots__ = ("_inner", "_wrapper", "_cached")

    def __init__(self, inner, wrapper):
        self._inner = inner
        self._wrapper = wrapper
        self._cached: dict[int, int] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def out_dim(self):
        hint = getattr(_obs_history, "_env_hint", None)
        env = hint() if hint is not None else None
        if env is None:
            return "torch_live"
        key = id(env)
        if key not in self._cached:
            raw = dict(getattr(self._wrapper, "_term_params", None) or {})
            raw.update(_find_term_params(env, self._wrapper))
            command = env.command_manager.get_term(raw.get("command_name", "motion"))
            dim, trailing = self._wrapper._resolver(command, raw)
            self._wrapper._torch_src_shapes[key] = trailing
            self._cached[key] = dim
        return self._cached[key]


def _noise_rng(cache: dict, key: int, n: int, d: int, device) -> wp.array:
    """Per-env int32 RNG state for ``_uniform_noise_kernel``, kept alive.

    Two lifetime rules matter here:

    * the wp view must own a LIVE torch backing -- ``wp.from_torch`` on a temp
      tensor dangles (the block returns to torch's caching allocator and can
      be handed to any later tensor; the kernel's in-place ``state + 1``
      advance would then stamp that tensor every step), so the torch tensor
      is cached alongside the view;
    * the cache key includes the width: several noisy terms of DIFFERENT
      widths run per step, and a single id(env) slot would be reallocated on
      every term alternation -- a stale narrower state made the kernel run
      past its end (the classic ``future_motion_ids`` corruption), and the
      reallocation itself landed inside the manager CUDA-graph capture
      (torch.randint during capture = cudaErrorStreamCaptureImplicit).  With
      the width in the key every state allocates exactly once, on the eager
      warmup pass that precedes capture.
    """
    ck = (key, n, d)
    entry = cache.get(ck)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "warp obs noise: RNG allocation attempted during CUDA-graph capture; "
                "the first call must run on an eager (warmup) step."
            )
        buf = torch.randint(0, 2**31 - 1, (n, d), dtype=torch.int32, device=device)
        entry = (buf, wp.from_torch(buf))
        cache[ck] = entry
    return entry[1]


def _apply_uniform_noise(env, out, lo: float, hi: float) -> None:
    """In-place uniform noise on a flat (N, D) warp out (obs_history RNG pattern).

    The per-env RNG state allocates lazily on the first (eager warmup) call;
    a first call during CUDA-graph capture is refused loudly.
    """
    n, d = out.shape[0], out.shape[1]
    rng = _noise_rng(_apply_uniform_noise._rng, id(env), n, d, env.device)
    wp.launch(
        kernel=_uniform_noise_kernel,
        dim=(n, d),
        inputs=[out, float(lo), float(hi), rng],
        device=env.device,
    )


_apply_uniform_noise._rng: dict[int, tuple] = {}


def native_with_noise(native: Callable, noise: Any) -> Callable:
    """Wrap an official experimental obs native with baked-in uniform noise.

    The experimental manager silently skips classic (stable) noise cfgs, so a
    noisy critic term routed to a native would lose its corruption; the wrapper
    keeps the native's io-descriptor and presented signature and adds the noise
    pass on the flat out (Stage-3a bridge convention: profile nulls the cfg
    noise and the term applies it itself).
    """
    lo, hi = _parse_noise(noise)

    def noisy_native(env: ManagerBasedEnv, out, **params):
        native(env, out, **params)
        _apply_uniform_noise(env, out, lo, hi)

    noisy_native._warp_protocol = True
    for attr in ("_descriptor", "__doc__"):
        if hasattr(native, attr):
            setattr(noisy_native, attr, getattr(native, attr))
    try:
        noisy_native.__signature__ = inspect.signature(native)
    except (TypeError, ValueError):
        pass
    noisy_native.__name__ = getattr(native, "__name__", str(native)) + "(noisy)"
    return noisy_native


def _make_obs_term(
    runner: Callable,
    resolver: Callable,
    param_names: tuple[str, ...],
    name: str,
    doc: str,
    term_params: dict | None = None,
    noise: Any = None,
) -> Callable:
    """Build a Warp-protocol obs term with command-resolved dims.

    Args:
        runner: ``(env, out, command, params) -> None`` launching the kernels.
        resolver: ``(command, params) -> (dim, trailing_shape)``.
        param_names: presented signature params after ``(env, out)``.
        term_params: the ObsTerm cfg params (for the deferred dim resolution).
        noise: optional classic noise cfg (uniform), applied on the flat out
            after the kernel (the experimental manager skips classic cfgs).
    """
    noise_lo, noise_hi = _parse_noise(noise)

    @generic_io_descriptor_warp(out_dim=None, dtype=torch.float32)
    def warp_obs_term(env: ManagerBasedEnv, out, **params):
        merged = {**warp_obs_term._term_params, **params}
        command = env.command_manager.get_term(merged.get("command_name", "motion"))
        runner(env, out, command, merged)
        if noise is not None:
            _apply_uniform_noise(env, out, noise_lo, noise_hi)

    warp_obs_term._warp_protocol = True
    warp_obs_term._torch_src_shapes: dict[int, tuple] = {}
    warp_obs_term._term_params = dict(term_params or {})
    warp_obs_term._resolver = resolver
    warp_obs_term._descriptor = _CmdTermDim(warp_obs_term._descriptor, warp_obs_term)
    sig_params = [
        inspect.Parameter("env", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("out", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *(
            inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
            for p in param_names
        ),
    ]
    warp_obs_term.__signature__ = inspect.Signature(parameters=sig_params)
    warp_obs_term.__name__ = name
    warp_obs_term.__doc__ = doc
    return warp_obs_term


# --------------------------------------------------------------------------
# kernel-term factories (routed by env_profile from the torch term funcs)
# --------------------------------------------------------------------------


def make_gravity_dir(term_params: dict | None = None, noise: Any = None) -> Callable:
    """``gravity_dir``: down vector in the robot-anchor frame (robot snapshot)."""

    def runner(env, out, command, params):
        wp.launch(
            kernel=_gravity_dir_kernel,
            dim=env.num_envs,
            inputs=[command.wp_robot_anchor_quat_w, out],
            device=env.device,
        )

    def resolver(command, params):
        return 3, (3,)

    return _make_obs_term(
        runner, resolver, ("command_name",), "gravity_dir(warp)", __doc__, term_params, noise=noise
    )


def make_motion_anchor_pos_b(term_params: dict | None = None, noise: Any = None) -> Callable:
    """``motion_anchor_pos_b``: ref anchor position in the robot-anchor frame."""

    def runner(env, out, command, params):
        wp.launch(
            kernel=_anchor_pos_b_kernel,
            dim=env.num_envs,
            inputs=[
                command.wp_robot_anchor_pos_w,
                command.wp_robot_anchor_quat_w,
                command.wp_anchor_pos_w,
                out,
            ],
            device=env.device,
        )

    def resolver(command, params):
        return 3, (3,)

    return _make_obs_term(
        runner, resolver, ("command_name",), "motion_anchor_pos_b(warp)", __doc__, term_params, noise=noise
    )


def make_motion_anchor_ori_b(term_params: dict | None = None, noise: Any = None) -> Callable:
    """``motion_anchor_ori_b``: 6D ref-anchor orientation in the robot-anchor frame.

    ``non_flatten=True`` repeats the current-frame 6D across the future-frame
    slots (torch term semantics).
    """

    def runner(env, out, command, params):
        kernel = (
            _anchor_ori_b_repeat_kernel
            if params.get("non_flatten", False)
            else _anchor_ori_b_kernel
        )
        wp.launch(
            kernel=kernel,
            dim=env.num_envs,
            inputs=[
                command.wp_robot_anchor_quat_w,
                command.wp_anchor_quat_w,
                out,
            ],
            device=env.device,
        )

    def resolver(command, params):
        if params.get("non_flatten", False):
            f = int(command.num_future_frames)
            return 6 * f, (f, 6)
        return 6, (6,)

    return _make_obs_term(
        runner,
        resolver,
        ("command_name", "non_flatten"),
        "motion_anchor_ori_b(warp)",
        __doc__,
        term_params,
        noise=noise,
    )


def make_robot_body_pos_b(term_params: dict | None = None, noise: Any = None) -> Callable:
    """``robot_body_pos_b``: tracked body positions in the robot-anchor frame."""

    def runner(env, out, command, params):
        wp.launch(
            kernel=_robot_body_pos_b_kernel,
            dim=env.num_envs,
            inputs=[
                command.wp_robot_anchor_pos_w,
                command.wp_robot_anchor_quat_w,
                command.wp_robot_body_pos_w,
                out,
            ],
            device=env.device,
        )

    def resolver(command, params):
        d = 3 * len(command.cfg.body_names)
        return d, (d,)

    return _make_obs_term(
        runner, resolver, ("command_name",), "robot_body_pos_b(warp)", __doc__, term_params, noise=noise
    )


def make_robot_body_ori_b(term_params: dict | None = None, noise: Any = None) -> Callable:
    """``robot_body_ori_b``: 6D tracked body orientations in the robot-anchor frame."""

    def runner(env, out, command, params):
        wp.launch(
            kernel=_robot_body_ori_b_kernel,
            dim=env.num_envs,
            inputs=[
                command.wp_robot_anchor_quat_w,
                command.wp_robot_body_quat_w,
                out,
            ],
            device=env.device,
        )

    def resolver(command, params):
        d = 6 * len(command.cfg.body_names)
        return d, (d,)

    return _make_obs_term(
        runner, resolver, ("command_name",), "robot_body_ori_b(warp)", __doc__, term_params, noise=noise
    )


def make_vr_3point_local_target(term_params: dict | None = None, noise: Any = None) -> Callable:
    """``vr_3point_local_target``: VR points in the REFERENCE anchor frame."""

    def runner(env, out, command, params):
        wp.launch(
            kernel=_vr_local_pos_kernel,
            dim=env.num_envs,
            inputs=[
                command.wp_anchor_pos_w,
                command.wp_anchor_quat_w,
                command.wp_vr_3point_body_pos_w,
                out,
            ],
            device=env.device,
        )

    def resolver(command, params):
        d = 3 * len(command.cfg.vr_3point_body)
        return d, (d,)

    return _make_obs_term(
        runner,
        resolver,
        ("command_name",),
        "vr_3point_local_target(warp)",
        __doc__,
        term_params,
        noise=noise,
    )


def make_vr_3point_local_orn_target(term_params: dict | None = None, noise: Any = None) -> Callable:
    """``vr_3point_local_orn_target``: VR quats in the REFERENCE anchor frame."""

    def runner(env, out, command, params):
        wp.launch(
            kernel=_vr_local_quat_kernel,
            dim=env.num_envs,
            inputs=[
                command.wp_anchor_quat_w,
                command.wp_vr_3point_body_quat_w,
                out,
            ],
            device=env.device,
        )

    def resolver(command, params):
        d = 4 * len(command.cfg.vr_3point_body)
        return d, (d,)

    return _make_obs_term(
        runner,
        resolver,
        ("command_name",),
        "vr_3point_local_orn_target(warp)",
        __doc__,
        term_params,
        noise=noise,
    )


# --------------------------------------------------------------------------
# copy-term factory (persistent command buffer -> out)
# --------------------------------------------------------------------------


def make_command_copy_obs(
    attr: str,
    trailing_shape: tuple[int, ...] | None = None,
    select_idx: list[int] | None = None,
    noise: Any = None,
    src_int: bool | None = None,
    term_params: dict | None = None,
) -> Callable:
    """Flat-copy a persistent TrackingCommandWarp buffer into the term ``out``.

    Args:
        attr: buffer attribute on the command (e.g. ``_copy_command_multi_future``).
        trailing_shape: per-env trailing shape for the wrapper's 3D restore;
            ``None`` reads it from the buffer at dim-resolution time.
        select_idx: optional joint selection -- the buffer holds the full
            ``(N, F, J)`` frames and the term emits ``(N, F, len(idx))``
            (``joint_pos_multi_future_select_joints_for_smpl`` semantics).
        noise: stable noise cfg (uniform only, like :mod:`mdp_warp.obs_history`)
            applied after the copy on the flat term value.
        src_int: source buffer is int64 (encoder index one-hot) -> cast to f32.
    """
    lo, hi = _parse_noise(noise)

    def _buf_of(command, params):
        # get_command_obs-style terms address the buffer through ``obs_name``
        # (e.g. obs_name="episode_encoder_index"); the tokenizer/critic terms
        # read the command's persistent ``_copy_*`` buffers
        obs_name = params.get("obs_name")
        return getattr(command, obs_name if obs_name else attr)

    def _sel_idx(params):
        return params.get("joints_idx", select_idx)

    def runner(env, out, command, params):
        fn = make_command_copy_obs
        key = id(env)
        buf = _buf_of(command, params)
        views = fn._views
        view = views.get(key)
        if view is None or view.ptr != buf.data_ptr():
            view = _live_wp(buf.reshape(buf.shape[0], -1))
            views[key] = view
        n, d = out.shape[0], out.shape[1]
        joints = _sel_idx(params)
        if joints is not None:
            sel = fn._sel.get(key)
            if sel is None:
                sel = wp.array([int(v) for v in joints], dtype=wp.int32, device=env.device)
                fn._sel[key] = sel
            # per-frame stride J of the (N, FS, J) source row, recovered from
            # the output width: d = FS * len(sel), src width = FS * J
            num_sel = len(joints)
            num_frames = d // num_sel
            row_width = int(view.shape[1]) // num_frames
            wp.launch(
                kernel=_select_f32_kernel,
                dim=(n, d),
                inputs=[view, sel, row_width, out],
                device=env.device,
            )
        else:
            int_src = src_int
            if int_src is None:
                int_src = buf.dtype in (torch.int64, torch.int32)
            kernel = _copy_i64_f32_kernel if int_src else _copy_f32_kernel
            wp.launch(kernel=kernel, dim=(n, d), inputs=[view, out], device=env.device)
        if noise is not None:
            # shared per-env RNG (shape-validated, torch backing kept alive)
            _apply_uniform_noise(env, out, lo, hi)

    def resolver(command, params):
        flat = _buf_of(command, params).reshape(command.num_envs, -1)
        joints = _sel_idx(params)
        if joints is not None:
            # buffer holds the full (N, FS, J) frames; the term emits
            # (N, FS, len(idx)) -- joint_pos_multi_future_select_joints
            num_frames = int(command.smpl_num_future_frames)
            trailing = (num_frames, len(joints))
            return int(math.prod(trailing)), trailing
        if params.get("non_flatten", False):
            # torch non_flatten semantics: the term reshapes the flat row to
            # (N, frames, width) -- frames recorded on the command at buffer
            # allocation (registry key = attr minus the _copy_ prefix)
            key = attr[1:] if attr.startswith("_copy_") else attr
            frames = command._copy_frames.get(key)
            if frames is None:
                raise AttributeError(
                    f"copy term '{attr}' has no recorded frame count for the "
                    "non_flatten layout"
                )
            width = int(flat.shape[1]) // frames
            trailing = (frames, width)
            return int(math.prod(trailing)), trailing
        if trailing_shape is not None:
            trailing = tuple(trailing_shape)
        else:
            trailing = (int(flat.shape[1]),)
        return int(math.prod(trailing)), trailing

    # one uniform presented signature covers every copy-term cfg (all params
    # defaulted: the manager's static set-check unions cfg params with params
    # that carry signature defaults)
    param_names = ("command_name", "non_flatten", "joints_idx", "obs_name")
    name = f"command_copy({attr})"
    return _make_obs_term(
        runner,
        resolver,
        param_names,
        name,
        __doc__,
        term_params,
    )


# per-env caches shared by every copy term (module-level: stable across terms)
make_command_copy_obs._views: dict[int, wp.array] = {}
make_command_copy_obs._sel: dict[int, wp.array] = {}
