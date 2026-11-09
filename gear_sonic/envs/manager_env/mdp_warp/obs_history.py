"""Device-side per-term observation history for the Warp manager-based env.

Ported from unitree_rl_lab's ``tasks/mimic/mdp_warp/obs_history.py`` (Isaac
Lab 3.0 Newton/Warp migration), with
two sonic-side adaptations:

* ``source_func`` may be a ``"module:attr"`` string (resolved with
  ``isaaclab.utils.string_to_callable``) so the wrapper factory can be wired
  straight from sonic's Hydra YAML term configs -- ``custom_instantiate``
  recursively instantiates the ``func:`` node below (see Usage);
* the env hint for deferred dim resolution will be registered by
  ``TrackingCommandWarp.__init__`` (sonic's motion command term) instead of
  unitree's ``MotionCommandWarp``.

Background
----------
The experimental Warp ``ObservationManager`` raises ``NotImplementedError``
for any ``history_length > 0`` ("History reshaping is not implemented yet
for warp").  The torch manager implements history with the host-indexed
``CircularBuffer`` + ``reshape``, neither of which fits the WARP_CAPTURED
(CUDA-graph) path: after capture no host Python runs per step, so the ring
bookkeeping must live in device memory and advance inside kernels.

sonic's policy group (e.g. ``observations/policy/local_dir_hist.yaml``) uses
PER-TERM ``history_length: ${actor_prop_history_length}`` (=10) on the stable
torch manager.  This module implements history **per observation term**,
entirely in task-side code (no ``isaaclab_experimental`` changes):

* :func:`make_history_obs` builds a drop-in Warp obs term (``func(env, out,
  **params)`` + ``generic_io_descriptor_warp``) with ``out_dim = frame_dim *
  history_length``; the ObsTerm's own ``params=`` flow through the wrapper
  to the source term unchanged.  The layout is the chronological old->new
  flatten per term (``[t-H+1, ..., t]``); group concatenation then yields
  ``[term0 block | term1 block | ...]``.  NOTE (parity TODO): this matches
  the torch manager's GROUP-level ``history_length=H`` layout one-to-one;
  sonic's stable per-term history flatten must be confirmed identical with a
  parity check before checkpoints are considered interchangeable.
* Per step (all device-side, CUDA-graph safe):

  1. ``source_func`` writes the clean current frame into a persistent
     scratch buffer (its own kernel(s), exactly as in the non-history cfg);
  2. ``_history_push_flatten_kernel`` adds uniform noise ONCE to that frame
     (torch semantics: corruption is applied before the value enters the
     history buffer, so every stored frame keeps its own step's noise
     draw), pushes it into ring slot ``t % H`` and writes the flattened
     history into ``out`` (one thread per element, H-loop in-kernel);
  3. ``_advance_counts_kernel`` advances the per-env append counters.

* Reset semantics mirror the torch CircularBuffer exactly: obs compute runs
  after ``_reset_idx`` (which calls ``CircularBuffer.reset`` -> num_pushes=0
  on the torch side), so an env that terminates at step T performs a
  FIRST push this step -- and CircularBuffer.append's ``is_first_push``
  branch fills the ENTIRE history with the fresh post-reset frame.  The
  policy therefore sees ``[fresh, ..., fresh]`` at the reset step and
  ``[fresh, ..., fresh, newer]`` one step later (NOT zero padding).  The
  per-env counts + the aggregated termination-manager dones reproduce this
  on device; when the dones buffer is unavailable a warning is printed
  once and reset-restart is disabled (stale frames then roll out naturally
  within H-1 steps).

CUDA-graph safety rules followed here (same discipline as unitree's
``MotionCommandWarp`` / the planned ``TrackingCommandWarp``):

* all buffers are allocated once, on the first (eager warmup) call:
  persistent torch storage + one-time ``wp.from_torch`` views, so pointers
  are stable across replays.  Init raises loudly if the stream is already
  capturing -- a one-time ``torch.zeros`` memset or RNG fill recorded into
  the graph would replay (and wipe the ring) every step;
* no host-side per-step bookkeeping: the ring position comes from a device
  counter and the noise RNG state is a device array advanced inside the
  kernel (``wp.randf(state, lo, hi)`` + manual ``+= 1``);
* the noise-free path is bit-identical to the source term
  (``v + randf(0.0, 0.0) == v``), so parity checks run corruption-free.

Known, benign deltas vs the torch manager (documented, not bugs):

* noise draws come from the Warp RNG instead of the torch generator -- same
  uniform distribution, different realizations;
* torch skips corruption on the very first step (``common_step_counter``
  gate); the wrapper noises from frame 0;
* noise is controlled by the cfg-time ``noise=`` argument (the same
  ``UniformNoiseCfg`` object the non-history cfgs put on the ObsTerm),
  not by the group's runtime ``enable_corruption`` flag.  To stay
  consistent, the wrapper resolves which cfg group owns it on first call
  and disables noise if that group has ``enable_corruption=False``.

Usage (sonic Hydra YAML -- the ``func:`` node is a factory call that
``trl.utils.common.custom_instantiate`` resolves recursively)::

    joint_pos:
      _target_: isaaclab_experimental.managers.ObservationTermCfg
      func:
        _target_: gear_sonic.envs.manager_env.mdp_warp:make_history_obs
        source_func: gear_sonic.envs.manager_env.mdp_warp.observations:joint_pos
        history_length: ${actor_prop_history_length}
        noise:
          _target_: isaaclab.utils.noise.UniformNoiseCfg
          n_min: -0.01
          n_max: 0.01
      # params= stays on the ObsTerm (the manager forwards them to the source
      # through the wrapper); noise= moves INSIDE the wrapper (same UniformNoiseCfg
      # object as in the non-history cfg): ObsTerm-level noise would be applied by
      # the manager to the whole H x d output every step (re-noising stored frames,
      # unlike torch), and the classic isaaclab Unoise is silently skipped by the
      # experimental manager anyway (not an isaaclab_experimental NoiseCfg instance).

``frame_dim`` is optional and normally omitted.  Concrete ``out_dim`` ints
and ``axes`` metadata infer at cfg time; sources whose dim only exists at
runtime (``"command"`` / ``"joint"`` / ``"action"`` sentinels, or the
no-metadata joint-state terms) get a deferred io-descriptor whose ``out_dim``
property resolves against the LIVE env during ``ObservationManager`` init --
which requires a playing sim, so every sentinel is resolvable there, exactly
like the framework resolves it for unwrapped terms -- and returns the value
multiplied by ``history_length`` (which the framework resolvers cannot
express).  The env is reachable then because the command manager (building
``TrackingCommandWarp``) inits before the observation manager and registers
an env hint (:func:`register_env_hint`).
"""

from __future__ import annotations

import copy
import inspect
import weakref

from isaaclab.utils import string_to_callable
from isaaclab_experimental.envs.utils.io_descriptors import generic_io_descriptor_warp
import torch
import warp as wp

__all__ = ["make_history_obs", "register_env_hint", "resolve_dones"]

# Weak reference to the env most recently under construction, registered by
# TrackingCommandWarp.__init__.  ManagerBasedRLEnvWarp builds the command
# manager BEFORE the observation manager (with the sim already playing), so
# when a deferred descriptor resolves the wrapper's out_dim during
# ObservationManager init, the live env is reachable from here.
_env_hint: weakref.ReferenceType | None = None


def register_env_hint(env) -> None:
    """Record the env being constructed for deferred obs-dim resolution.

    Weakref on purpose: the hint must not keep a destroyed env alive, and
    each new env construction (train / play / parity) simply refreshes it.
    """
    global _env_hint
    _env_hint = weakref.ref(env)


@wp.kernel
def _advance_counts_kernel(
    counts: wp.array(dtype=wp.int32),  # (N,) appends performed since last reset
    dones: wp.array(dtype=wp.bool),  # (N,) env terminated THIS step (reset already applied)
    episode_len: wp.array(dtype=wp.int64),  # (N,) env episode counter (0 right after any reset)
):
    i = wp.tid()
    if counts[i] == 0 or dones[i] or episode_len[i] == wp.int64(0):
        # first append after a reset happened this step (CircularBuffer:
        # reset zeroes num_pushes, the post-reset append is the new "first").
        # episode_len == 0 also covers explicit env.reset() calls, where the
        # termination dones buffer is stale and would miss the reset.
        counts[i] = 1
    else:
        counts[i] = counts[i] + 1


@wp.kernel
def _history_push_flatten_kernel(
    scratch: wp.array(dtype=wp.float32, ndim=2),  # (N, D) clean current frame
    ring: wp.array(dtype=wp.float32, ndim=3),  # (N, H, D)
    counts: wp.array(dtype=wp.int32),  # (N,) appends since last reset (pre-increment)
    rng_state: wp.array(dtype=wp.int32, ndim=2),  # (N, D) per-element RNG streams
    dones: wp.array(dtype=wp.bool),  # (N,) env terminated THIS step (reset already applied)
    episode_len: wp.array(dtype=wp.int64),  # (N,) env episode counter (0 right after any reset)
    noise_lo: float,
    noise_hi: float,
    history_length: int,
    frame_dim: int,
    out: wp.array(dtype=wp.float32, ndim=2),  # (N, H*D) chronological old -> new
):
    i, j = wp.tid()
    # noisy fresh frame; noise ONCE per element per step (torch applies
    # corruption before the value enters the history buffer)
    state = wp.uint32(rng_state[i, j])
    v = scratch[i, j] + wp.randf(state, noise_lo, noise_hi)
    rng_state[i, j] = wp.int32(state + wp.uint32(1))
    if counts[i] == 0 or dones[i] or episode_len[i] == wp.int64(0):
        # first append after a reset (obs compute runs after _reset_idx, so
        # dones marks exactly those envs): CircularBuffer.append's
        # is_first_push branch fills the ENTIRE history with the fresh frame
        # -- the torch policy sees [fresh, ..., fresh] at the reset step, not
        # zero padding
        for h in range(history_length):
            ring[i, h, j] = v
            out[i, h * frame_dim + j] = v
    else:
        t = counts[i]
        ring[i, t % history_length, j] = v
        out[i, (history_length - 1) * frame_dim + j] = v
        for k in range(history_length - 1):
            age = history_length - 1 - k  # 1 .. H-1 (k = 0 is the oldest frame)
            src = (t - age) % history_length
            if src < 0:
                # frame does not exist yet (first H-1 steps after a fill):
                # the slot still holds the first-push fill value, exactly the
                # copy the raw CircularBuffer buffer returns for those ages.
                # (warp % follows C semantics: sign of the dividend)
                src += history_length
            out[i, k * frame_dim + j] = ring[i, src, j]


def _as_wp_bool(view, num_envs: int):
    """Coerce a dones-like object to a contiguous (num_envs,) wp.bool array."""
    try:
        if isinstance(view, wp.array):
            arr = view
        elif torch.is_tensor(view):
            if view.dtype != torch.bool or not view.is_contiguous():
                return None
            arr = wp.from_torch(view)
        else:
            # lazy warp-backed buffer (Newton): round-trip like _live_wp
            t = wp.to_torch(view)
            if t.dtype != torch.bool or not t.is_contiguous():
                return None
            arr = wp.from_torch(t)
        if arr.dtype != wp.bool or tuple(arr.shape) != (num_envs,):
            return None
        return arr
    except Exception:
        return None


def resolve_dones(env):
    """Device-readable per-env "terminated this step" mask, or ``None``.

    Resolved per call (not cached): in eager mode the manager may rebind the
    buffer, and under graph capture the Python body only runs at capture time
    anyway, so the pointer seen there is what gets baked in.  The
    experimental termination terms write ``wp.bool`` outs, so the aggregated
    dones is expected to be a stable ``(num_envs,)`` wp.bool buffer written
    earlier in the same captured step.
    """
    term_manager = getattr(env, "termination_manager", None)
    return _as_wp_bool(getattr(term_manager, "dones", None), env.num_envs)


def _iter_obs_terms(env):
    """Yield ``(group_cfg, term_cfg)`` for every obs term of ``env.cfg``.

    These are the SAME objects the managers hold and mutate
    (``ManagerBasedEnvWarp`` stores ``self.cfg = cfg`` without copying), so
    by dim-inference time a term's materialized signature defaults and
    resolved SceneEntityCfgs are visible here.  Defensive: any
    introspection failure yields nothing.
    """
    try:
        for group in vars(env.cfg.observations).values():
            terms = None
            get_terms = getattr(group, "get_terms", None)
            if callable(get_terms):
                try:
                    terms = dict(get_terms())
                except Exception:
                    terms = None
            if terms is None:
                terms = {k: v for k, v in vars(group).items() if hasattr(v, "func")}
            for term in terms.values():
                yield group, term
    except Exception:
        pass


def _detect_group_corruption(env, term_func, default: bool = True) -> bool:
    """Find the obs group owning ``term_func`` and return its enable_corruption.

    Defensive: any introspection failure falls back to ``default``.  Runs
    once at init (host-side) and the answer is baked into the captured
    kernels -- corruption state does not change within an env instance.
    """
    for group, term in _iter_obs_terms(env):
        if getattr(term, "func", None) is term_func:
            return bool(getattr(group, "enable_corruption", default))
    return default


def _find_term_params(env, term_func) -> dict:
    """The ``params`` of the ObsTerm owning ``term_func`` ({} if not found)."""
    for _, term in _iter_obs_terms(env):
        if getattr(term, "func", None) is term_func:
            return dict(getattr(term, "params", None) or {})
    return {}


class _HistoryState:
    """Per-env persistent buffers of one wrapped term (allocated once, eager)."""

    __slots__ = (
        "scratch", "ring", "counter", "rng_state", "dones_fallback",
        "num_envs", "frame_dim", "warned", "lo", "hi", "params",
        "episode_len_ones", "episode_len_ones_t",
        # torch backings of the wp views above -- a from_torch view does not
        # own its storage, so these must live as long as the views or torch's
        # caching allocator can hand the blocks to other tensors (the history
        # kernels then stomp / get stomped -- same class of bug as the obs
        # noise RNG)
        "scratch_t", "ring_t", "counter_t", "rng_state_t", "dones_t",
    )


def _init_state(env, frame_dim: int, history_length: int) -> _HistoryState:
    is_capturing = getattr(wp, "is_capturing", lambda: False)
    if is_capturing():
        raise RuntimeError(
            "make_history_obs: first invocation happened during CUDA-graph capture. "
            "The persistent history buffers must be allocated on an eager (warmup) step; "
            "a one-time init recorded into the graph would replay every step."
        )
    n, d, h = env.num_envs, frame_dim, history_length
    dev = env.device
    st = _HistoryState()
    st.num_envs = n
    st.frame_dim = d
    st.warned = False
    st.episode_len_ones = None
    st.episode_len_ones_t = None
    # persistent torch storage + one-time zero-copy warp views (stable pointers)
    st.scratch_t = torch.zeros(n, d, dtype=torch.float32, device=dev)
    st.ring_t = torch.zeros(n, h, d, dtype=torch.float32, device=dev)
    # per-env append counter (CircularBuffer._num_pushes analog)
    st.counter_t = torch.zeros(n, dtype=torch.int32, device=dev)
    st.scratch = wp.from_torch(st.scratch_t)
    st.ring = wp.from_torch(st.ring_t)
    st.counter = wp.from_torch(st.counter_t)
    # per-element RNG streams: every wrapped term draws its own at init, so
    # streams are decorrelated across terms (and reproducible under a seeded
    # run -- the same global-torch-RNG dependency Unoise sampling has)
    st.rng_state_t = torch.randint(
        0, 2**31 - 1, (n, d), dtype=torch.int64, device=dev
    ).to(torch.int32)
    st.rng_state = wp.from_torch(st.rng_state_t)
    # used when the termination manager exposes no readable dones buffer
    st.dones_t = torch.zeros(n, dtype=torch.bool, device=dev)
    st.dones_fallback = wp.from_torch(st.dones_t)
    return st


def _resolve_source_params(env, source_func, source_params: dict) -> dict:
    """Materialize + resolve the source term's signature defaults.

    Mirrors what the manager does for the terms it sees:
    ``ManagerBase._resolve_common_term_cfg`` copies dataclass defaults from
    the signature into params, and ``_process_term_cfg_at_play`` calls
    ``SceneEntityCfg.resolve(scene)`` on them (filling ``joint_ids`` /
    ``joint_ids_wp``).  The manager only inspects the wrapper -- whose
    signature has no such defaults -- so the wrapped source term (e.g. the
    official ``joint_pos_rel``, which raises without a resolved
    ``asset_cfg.joint_ids_wp``) must be resolved here.  Runs once on the
    first eager call, when the simulation is playing and scene entities are
    resolvable; the resolved id buffers are pointer-stable for graph replay.
    """
    resolved = dict(source_params)
    try:
        signature = inspect.signature(source_func)
    except (TypeError, ValueError):
        return resolved
    positional = list(signature.parameters.values())
    protocol_names = {"out", "env_mask", "env_mask_wp", "env_ids"}
    skip = 2 if len(positional) > 1 and positional[1].name in protocol_names else 1
    for param in positional[skip:]:
        if param.default is inspect.Parameter.empty:
            continue
        if param.name in resolved or not hasattr(param.default, "__dataclass_fields__"):
            continue
        default = param.default
        resolved[param.name] = default.copy() if hasattr(default, "copy") else copy.deepcopy(default)
    for value in resolved.values():
        # SceneEntityCfg-style: resolve against the live scene (duck-typed to
        # stay import-agnostic; same shape the manager isinstance-checks)
        if hasattr(value, "resolve") and hasattr(value, "joint_names") and hasattr(value, "name"):
            value.resolve(env.scene)
    return resolved


def _get_descriptor(source_func):
    """The source term's io-descriptor (through functools.wraps chains), or None."""
    descriptor = getattr(source_func, "_descriptor", None)
    if descriptor is None:
        descriptor = getattr(getattr(source_func, "__wrapped__", None), "_descriptor", None)
    return descriptor


def _infer_frame_dim(source_func) -> int | None:
    """Cfg-time frame dim from the source term's io-descriptor, or ``None``.

    Mirrors the first two steps of the manager's ``_infer_term_dim_scalar``:
    a concrete ``out_dim`` int, else ``len(axes)``.  ``None`` means the dim
    only exists at runtime (sentinels ``"command"`` / ``"joint"`` /
    ``"action"`` / ``"body:N"``, or no descriptor metadata at all like the
    joint-state terms) -- ``make_history_obs`` then DEFERS the resolution to
    ``ObservationManager`` init via :class:`_DeferredDim`.
    """
    descriptor = _get_descriptor(source_func)
    if descriptor is None:
        return None
    out_dim = getattr(descriptor, "out_dim", None)
    if isinstance(out_dim, int):
        return int(out_dim)
    extras = getattr(descriptor, "extras", None)
    axes = extras.get("axes") if extras else None
    if axes is not None:
        return len(axes)
    return None


def _resolve_frame_dim_live(env, source_func, params: dict) -> int:
    """Per-frame dim of ``source_func`` resolved against a LIVE env.

    Task-side mirror of the manager's ``_infer_term_dim_scalar`` /
    ``_resolve_out_dim`` for the sources we wrap: descriptor int, then the
    sentinels (``"command"`` / ``"action"`` / ``"body:N"`` / ``"joint"``),
    then ``axes``, then the ``asset_cfg`` joint-count fallback.  ``params``
    must be the term's own params (materialized signature defaults +
    resolved SceneEntityCfgs) -- exactly what the manager has when it
    infers dims (sim is playing at ObservationManager init and
    ``_resolve_common_term_cfg`` ran for this term), and what
    ``_resolve_source_params`` produces for direct calls.
    """
    descriptor = _get_descriptor(source_func)
    out_dim = getattr(descriptor, "out_dim", None) if descriptor is not None else None
    if isinstance(out_dim, int):
        return int(out_dim)
    if out_dim == "command":
        cmd = env.command_manager.get_command(params.get("command_name"))
        return int(cmd.shape[-1])
    if out_dim == "action":
        return int(env.action_manager.action.shape[-1])
    if isinstance(out_dim, str) and out_dim.startswith("body:"):
        per_body = int(out_dim.split(":")[1])
        asset_cfg = params.get("asset_cfg")
        asset = env.scene[asset_cfg.name]
        body_ids = getattr(asset_cfg, "body_ids", None)
        if body_ids is None or isinstance(body_ids, slice):
            return per_body * len(asset.body_names)
        return per_body * len(body_ids)
    if out_dim == "joint" or out_dim is None:
        asset_cfg = params.get("asset_cfg")
        if asset_cfg is not None:
            joint_ids_wp = getattr(asset_cfg, "joint_ids_wp", None)
            if joint_ids_wp is not None:
                return int(joint_ids_wp.shape[0])
            joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
            if isinstance(joint_ids, slice):
                return int(len(env.scene[asset_cfg.name].joint_names))
            return int(len(joint_ids))
    extras = getattr(descriptor, "extras", None) if descriptor is not None else None
    axes = extras.get("axes") if extras else None
    if axes is not None:
        return len(axes)
    raise ValueError(
        f"make_history_obs: cannot resolve the frame dim of "
        f"{getattr(source_func, '__name__', source_func)!r} against the live env "
        f"(descriptor out_dim={out_dim!r}, no axes, no asset_cfg in params) -- "
        "pass frame_dim= explicitly."
    )


class _DeferredDim:
    """Descriptor proxy whose ``out_dim`` resolves to ``frame_dim * H`` lazily.

    Installed as ``wrapper._descriptor`` when the frame dim is not knowable
    at cfg time.  The manager reads ``descriptor.out_dim`` during
    ``ObservationManager`` init (``_infer_term_dim_scalar`` step 1, a plain
    ``getattr`` -- no isinstance check on the descriptor), at which point
    the sim is playing and the command/action managers exist, so the
    source's sentinel dim resolves exactly like the framework resolves it
    for unwrapped terms -- then gets multiplied by the history length,
    which the framework resolvers cannot express.  All other descriptor
    attributes pass through to the real descriptor.
    """

    __slots__ = ("_inner", "_wrapper", "_source_func", "_history_length", "_cached")

    def __init__(self, inner, wrapper, source_func, history_length: int):
        self._inner = inner
        self._wrapper = wrapper
        self._source_func = source_func
        self._history_length = history_length
        self._cached = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def out_dim(self):
        if self._cached is None:
            env = _env_hint() if _env_hint is not None else None
            if env is None:
                raise ValueError(
                    f"make_history_obs: no live env registered to resolve the deferred frame dim of "
                    f"{getattr(self._source_func, '__name__', self._source_func)!r} (the hint is set by "
                    "TrackingCommandWarp.__init__, which runs before the ObservationManager inits). "
                    "Pass frame_dim= explicitly for tasks without a TrackingCommandWarp command term."
                )
            # cfg-side params are the RAW ones: the manager materializes
            # signature defaults (asset_cfg) into its own term-cfg copies,
            # so run the same materialize+resolve mirror the wrapper body
            # uses for direct calls.  The ObsTerm's own params ride on the
            # wrapper (attached at construction): the env-side term lookup
            # is empty here -- this runs during the ObservationManager's own
            # __init__, before it is reachable through the env.
            # (deepcopy: resolve() mutates in place and must not touch
            # cfg-side objects)
            raw = dict(self._wrapper._term_params)
            raw.update(_find_term_params(env, self._wrapper))
            params = _resolve_source_params(env, self._source_func, copy.deepcopy(raw))
            frame_dim = _resolve_frame_dim_live(env, self._source_func, params)
            # publish the resolved value on the wrapper (tooling/debug aid)
            self._wrapper.frame_dim = frame_dim
            self._cached = frame_dim * self._history_length
        return self._cached


def _parse_noise(noise) -> tuple[float, float]:
    """``(lo, hi)`` of the uniform noise from a UniformNoiseCfg or a 2-tuple."""
    if noise is None:
        return 0.0, 0.0
    n_min, n_max = getattr(noise, "n_min", None), getattr(noise, "n_max", None)
    try:
        if n_min is not None and n_max is not None:
            # classic or experimental UniformNoiseCfg -- the same object the
            # non-history cfgs put on the ObsTerm's noise=
            return float(n_min), float(n_max)
        if isinstance(noise, (tuple, list)) and len(noise) == 2:
            return float(noise[0]), float(noise[1])
    except TypeError:
        pass
    raise ValueError(
        f"make_history_obs: unsupported noise cfg {noise!r} -- only uniform noise is "
        "supported (a UniformNoiseCfg with scalar n_min/n_max -- the same object the "
        "ObsTerm used to carry -- or an (n_min, n_max) tuple); the wrapper samples "
        "it in-kernel, once per frame, before the value enters the history ring."
    )


def make_history_obs(source_func, history_length: int = 4, frame_dim: int | None = None, noise=None, params=None):
    """Wrap a Warp obs term into a chronological ``history_length``-frame term.

    Args:
        source_func: the underlying per-step Warp term (e.g. any
            ``isaaclab_experimental.envs.mdp`` term or a ``mdp_warp``
            observation), or a ``"module:attr"`` string naming one (sonic's
            Hydra YAML wiring -- resolved here with
            ``isaaclab.utils.string_to_callable``, the same helper the
            managers use for string ``func=`` term cfgs).  It must write the
            clean current frame into the provided ``(num_envs, frame_dim)``
            buffer.  The ObsTerm's own ``params=`` reach it unchanged: the
            manager calls the wrapper with them as kwargs and the wrapper
            forwards them (merged over the source's signature defaults,
            which are materialized and resolved against the live scene on
            first call -- exactly what the manager does for the terms it
            sees directly).
        history_length: number of frames H kept and emitted (old -> new).
            Exposed as an attribute on the returned function for tooling
            (deploy/export paths that need to rebuild the frame layout).
        frame_dim: per-frame dimension of the source term.  Optional and
            normally omitted: concrete ``out_dim`` ints / ``axes`` infer at
            cfg time, and runtime sentinels (``"command"`` / ``"joint"`` /
            ``"action"``) or no-metadata joint terms resolve against the
            live env during ``ObservationManager init`` (deferred
            descriptor, see :class:`_DeferredDim`).  Pass explicitly only
            to bypass the deferred path (tasks with no ``TrackingCommandWarp``
            term to register the env hint).
        noise: uniform noise applied ONCE per frame before it enters the
            ring (mirrors the torch manager's noise->buffer ordering): the
            same ``UniformNoiseCfg`` object the non-history cfgs put on the
            ObsTerm (``Unoise(n_min=..., n_max=...)``), or an
            ``(n_min, n_max)`` tuple.  ``None`` = corruption-free term.
            Keep ``noise=None`` on the ObsTerm itself: manager-level noise
            would hit the whole H x d output every step (re-noising stored
            frames), and the classic Unoise is silently skipped by the
            experimental manager anyway (isinstance mismatch).
        params: the ObsTerm's own ``params=`` dict.  Consumed by the deferred
            dim descriptor: it resolves during the ObservationManager's own
            ``__init__`` (before the manager is reachable through the env),
            so the term's params cannot be looked up env-side and must ride
            on the wrapper.  Runtime calls receive them as kwargs from the
            manager regardless.

    Returns:
        A ``generic_io_descriptor_warp``-decorated term function with
        ``out_dim = frame_dim * history_length``.
    """
    if isinstance(source_func, str):
        # sonic YAML wiring: "module:attr" strings resolve exactly like the
        # managers resolve string func= cfgs
        source_func = string_to_callable(source_func)
    if frame_dim is None:
        frame_dim = _infer_frame_dim(source_func)  # None -> deferred to env init
    noise_lo, noise_hi = _parse_noise(noise)

    @generic_io_descriptor_warp(
        out_dim=None if frame_dim is None else frame_dim * history_length, dtype=torch.float32
    )
    def history_obs(env, out, **kwargs):
        states = history_obs._states
        state = states.get(id(env))
        if state is None or state.num_envs != env.num_envs:
            # resolve the owning group's corruption flag once (baked into the
            # captured kernels); noise=None terms stay clean regardless
            lo, hi = noise_lo, noise_hi
            if (lo != 0.0 or hi != 0.0) and not _detect_group_corruption(env, history_obs):
                lo, hi = 0.0, 0.0
            state_params = _resolve_source_params(env, source_func, {})
            d = frame_dim
            if d is None:
                # deferred dim, direct-call path (under the manager the
                # descriptor property already resolved it at init)
                d = _resolve_frame_dim_live(env, source_func, {**state_params, **kwargs})
                if history_obs.frame_dim is None:
                    history_obs.frame_dim = d
            state = _init_state(env, d, history_length)
            state.lo, state.hi = lo, hi
            state.params = state_params
            states[id(env)] = state
        # 1) clean current frame from the source term (its own kernel(s)):
        #    signature-default SceneEntityCfgs resolved once at init, plus
        #    the ObsTerm params= the manager forwards as kwargs every call
        source_func(env, state.scratch, **{**state.params, **kwargs})
        # 2) push + chronological flatten (device-side ring, race-free slots)
        dones = resolve_dones(env)
        if dones is None:
            dones = state.dones_fallback
            if not state.warned:
                print(
                    f"[WARN] obs_history/{history_obs.__name__}: termination_manager.dones not "
                    "readable as a (num_envs,) wp.bool buffer -- history reset-clearing disabled "
                    "(stale frames roll out within H-1 steps after a reset)."
                )
                state.warned = True
        # episode_length_buf == 0 marks envs reset since the last obs compute
        # on EVERY path (step-time _reset_idx and explicit env.reset()); the
        # env owns this persistent wp.int64 buffer.  Fallback: ones (never
        # triggers; dones alone then covers only step-time resets).
        episode_len = getattr(env, "_episode_length_buf_wp", None)
        if episode_len is None:
            if state.episode_len_ones is None:
                state.episode_len_ones_t = torch.ones(env.num_envs, dtype=torch.int64, device=env.device)
                state.episode_len_ones = wp.from_torch(state.episode_len_ones_t)
            episode_len = state.episode_len_ones
        wp.launch(
            kernel=_history_push_flatten_kernel,
            dim=(env.num_envs, state.frame_dim),
            inputs=[
                state.scratch,
                state.ring,
                state.counter,
                state.rng_state,
                dones,
                episode_len,
                state.lo,
                state.hi,
                history_length,
                state.frame_dim,
                out,
            ],
            device=env.device,
        )
        # 3) advance the per-env append counts (after all reads of them this
        # step); a reset this step restarts the count at the post-reset append
        wp.launch(
            kernel=_advance_counts_kernel,
            dim=env.num_envs,
            inputs=[state.counter, dones, episode_len],
            device=env.device,
        )

    if frame_dim is None:
        # sentinel/no-metadata source: hand the manager a descriptor whose
        # out_dim resolves (frame_dim * H) against the live env at
        # ObservationManager init -- see _DeferredDim
        history_obs._descriptor = _DeferredDim(
            history_obs._descriptor, history_obs, source_func, history_length
        )
    history_obs._states = {}
    history_obs.frame_dim = frame_dim
    history_obs.history_length = history_length
    history_obs.source_func = source_func
    history_obs._term_params = dict(params or {})
    history_obs.__name__ = f"{getattr(source_func, '__name__', 'obs')}_h{history_length}"
    # explicit Warp-protocol marker: the __signature__ passthrough below would
    # otherwise make torch_bridge.is_warp_term misread this wrapper as torch
    history_obs._warp_protocol = True
    # Present the SOURCE term's signature to the manager.  Its static check
    # (manager_base._resolve_common_term_cfg) rejects **kwargs passthrough --
    # VAR_KEYWORD has no default, so 'kwargs' lands in the mandatory list
    # ("expects mandatory parameters: ['kwargs']"; the framework flags this
    # case as unhandled right there).  With the source's signature exposed
    # via __signature__, the wrapper validates exactly like the unwrapped
    # term: the ObsTerm's params= pass the set check, dataclass defaults
    # (asset_cfg) are materialized into params and resolved at play time by
    # the manager itself, and the runtime call func(env, out_wp, **params)
    # is what the **kwargs body accepts and forwards.
    try:
        history_obs.__signature__ = inspect.signature(source_func)
    except (TypeError, ValueError):
        pass  # uninspectable source: keep the real (env, out, **kwargs)
    return history_obs
