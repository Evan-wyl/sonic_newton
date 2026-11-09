"""Torch→Warp MDP term adapters (Stage 3a wiring bridge).

The experimental Warp managers call every term through the Warp-first
protocols:

* observations / rewards / terminations: ``func(env, out, **params) -> None``
  (the term writes into the manager-preallocated ``out`` buffer),
* events: ``func(env, env_mask, **params) -> None`` (boolean warp mask).

sonic's tracking terms are torch functions (``func(env, **params) -> Tensor``).
Until each term gets its own Warp kernel port (REF plan 3b/3c), this module
wraps any torch term into the Warp protocol with a zero-copy
``torch_view.copy_(value)`` handoff:

* ``make_torch_obs_warp`` -- also carries the source's trailing shape
  (e.g. the tokenizer terms' ``(F, d)`` multi-future layout) on the wrapper
  (``_torch_src_shapes``) so :meth:`ManagerEnvWrapper.process_raw_obs` can
  restore the per-term 3D layout the model's ``parse_tokenizer_obs`` expects
  (the experimental manager only allocates flat 2D ``(N, D)`` outputs);
* ``make_torch_reward_warp`` / ``make_torch_termination_warp`` -- per-env
  scalar outputs ``(N,)`` float / bool;
* ``make_torch_event_warp`` -- converts the warp env mask to an index tensor
  and calls the torch event with ``env_ids=`` (unitree's ``_mask_to_env_ids``
  pattern; startup/reset/interval modes all work through it).

Dim resolution for observations: the experimental ObservationManager infers
each term's output dim from io-descriptor metadata, which torch terms do not
have.  The wrapper installs a deferred probe descriptor: at manager init
(sim playing, command manager alive -- the env hint registered by
``TrackingCommand.__init__``) the source term is called ONCE and its
flattened trailing dim is cached.  At cfg time the descriptor returns the
``"torch_live"`` sentinel, which every cfg-time reader
(:func:`mdp_warp.obs_history._infer_frame_dim`) correctly treats as
"deferred".

Parity deltas vs the stable torch manager (documented, both benign):

* corruption draws come from the torch global RNG (same impl object as the
  torch manager uses), but the first-step ``common_step_counter`` gate is not
  replicated -- bridged terms noise from step 0 (same delta as
  :mod:`mdp_warp.obs_history`);
* every adapter is ``@warp_capturable(False)`` (torch compute allocates), so
  managers holding bridged terms are auto-capped to WARP_NOT_CAPTURED and
  keep running eagerly.
"""

from __future__ import annotations

import copy
import inspect
import math
from typing import TYPE_CHECKING, Any, Callable

from isaaclab_experimental.envs.utils.io_descriptors import generic_io_descriptor_warp
from isaaclab_experimental.utils.warp import warp_capturable
import torch
import warp as wp

from . import obs_history as _obs_history
from .obs_history import _find_term_params, _resolve_source_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "is_warp_term",
    "make_torch_obs_warp",
    "make_torch_reward_warp",
    "make_torch_termination_warp",
    "make_torch_event_warp",
    "mask_to_env_ids",
]


# --------------------------------------------------------------------------
# protocol detection
# --------------------------------------------------------------------------


def _warp_style_params(func) -> dict[str, inspect.Parameter] | None:
    """Positional (env, second, ...) parameter table of ``func``, or None."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return {p.name: p for p in positional} if positional else None


def is_warp_term(func) -> bool:
    """True if ``func`` already follows an experimental Warp-first protocol.

    Discriminator is the SIGNATURE (second positional named ``out`` for
    obs/reward/termination terms, ``env_mask`` for events), not the
    ``_descriptor`` marker: stable official terms are also io-descriptor
    decorated (``isaaclab.envs.utils.io_descriptors.generic_io_descriptor``
    sets ``_descriptor`` too), so the marker alone misclassifies them.
    Wrappers produced by this package / ``make_history_obs`` set the
    explicit ``_warp_protocol`` marker, immune to the ``__signature__``
    passthrough they install for the managers' static param check.

    String ``func`` values (sonic's Hydra YAML wiring) resolve first: cfg-side
    they arrive as configclass ``ResolvableString`` (a str subclass whose
    ``__getattr__`` forwards to the RESOLVED callable), so the descriptor
    fallback below would see the resolved stable official descriptor and
    misclassify re-exported terms (``joint_pos_rel`` & co) as Warp terms.
    """
    if isinstance(func, str):
        from isaaclab.utils import string_to_callable

        try:
            func = string_to_callable(str(func))
        except Exception:
            return False
    if getattr(func, "_warp_protocol", False):
        return True
    params = _warp_style_params(func)
    if params is None:
        # uninspectable non-string callable: not a known Warp term
        return False
    names = list(params)
    return len(names) > 1 and names[1] in ("out", "env_mask", "env_mask_wp")


def mask_to_env_ids(env_mask: wp.array) -> torch.Tensor:
    """Boolean warp env mask -> torch index tensor."""
    return wp.to_torch(env_mask).nonzero(as_tuple=True)[0]


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


class _ProbeDim:
    """Deferred io-descriptor whose ``out_dim`` probes the torch source live.

    Installed as ``wrapper._descriptor`` (wrapping the decorator's original
    descriptor, whose other attributes pass through).  At
    ObservationManager init the env hint is registered
    (``TrackingCommand.__init__``) and the source can be called once to read
    its flattened trailing dim; the value is cached per (wrapper, env).
    Without a live env the ``"torch_live"`` sentinel is returned, which
    cfg-time readers treat as "deferred" (never an int, so e.g.
    :func:`mdp_warp.obs_history._infer_frame_dim` returns None).
    """

    # __slots__: keeps this object walk-opaque for configclass validation --
    # the cfg tree references the wrapper func, whose __dict__ holds this
    # descriptor, which would otherwise cycle back through _wrapper (the
    # validator walks anything with a __dict__, 996-deep RecursionError).
    __slots__ = ("_inner", "_wrapper", "_source_func", "_cached")

    def __init__(self, inner, wrapper, source_func):
        self._inner = inner
        self._wrapper = wrapper
        self._source_func = source_func
        self._cached: dict[int, int] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def out_dim(self):
        # read obs_history's hint dynamically: register_env_hint rebinds the
        # module global (``global _env_hint``), so a from-import here would
        # stay bound to the import-time None forever and never probe
        hint = getattr(_obs_history, "_env_hint", None)
        env = hint() if hint is not None else None
        if env is None:
            return "torch_live"  # sentinel: deferred (cfg-time readers ignore)
        key = id(env)
        if key not in self._cached:
            # the ObsTerm's params= (command_name etc.) -- attached at wrapper
            # construction (profile time); the env-side lookup cannot be used
            # here, it runs during the ObservationManager's own __init__
            # (env.observation_manager does not exist yet).  Fallback lookup
            # covers direct-call uses where a manager is already alive.
            raw = dict(getattr(self._wrapper, "_term_params", None) or {})
            raw.update(_find_term_params(env, self._wrapper))
            params = _resolve_source_params(env, self._source_func, copy.deepcopy(raw))
            value = self._source_func(env, **params)
            self._wrapper._torch_src_shapes[key] = tuple(value.shape[1:])
            self._cached[key] = math.prod(value.shape[1:]) if len(value.shape) > 1 else 1
        return self._cached[key]


def _present_source_signature(wrapper: Callable, source_func: Callable, second: str = "out") -> None:
    """Expose a WARP-style view of the source's signature on the wrapper.

    The managers' static param check
    (``manager_base._resolve_common_term_cfg``) rejects ``**kwargs``
    passthrough wrappers (VAR_KEYWORD has no default, so ``kwargs`` lands in
    the mandatory list), and it drops the first ``min_argc`` (=2) positional
    params -- ``env`` and the protocol buffer (``out`` / ``env_mask``) --
    before matching the term's ``params=`` against the remaining names.

    Two source shapes exist:

    * torch obs/reward/termination terms ``(env, <param>, ...)`` -- no
      protocol buffer, so a verbatim passthrough would shift the accounting
      by one (the source's first param after ``env`` gets eaten by the
      drop).  The presented signature inserts the buffer at position 1:
      ``(env, out, <source params after env>)``.
    * torch event terms ``(env, env_ids, <params> ...)`` -- ``env_ids``
      already occupies the drop slot, so the verbatim source signature is
      the correct presentation (the runtime body translates the warp mask
      to ``env_ids=`` and never receives it as a param).

    With that, the wrapper validates exactly like a native Warp term: the
    source's params pass the set check, its dataclass defaults
    (``asset_cfg``) are materialized into the term params, and the runtime
    call ``func(env, out, **params)`` is what the ``**params`` body accepts
    and forwards.
    """
    try:
        sig = inspect.signature(source_func)
    except (TypeError, ValueError):
        return  # uninspectable source: keep the real (env, out/env_mask, **params)
    params = list(sig.parameters.values())
    if not params:
        return
    if len(params) > 1 and params[1].name in ("out", "env_mask", "env_ids"):
        wrapper.__signature__ = sig  # already protocol-shaped (env, X, ...)
        return
    injected = inspect.Parameter(second, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    wrapper.__signature__ = sig.replace(parameters=[params[0], injected, *params[1:]])


def _as_torch(value):
    """Torch view of a term value (torch / proxy array / ``wp.array``).

    In the warp env the assets' data buffers are proxy arrays; official torch
    terms that return a buffer slice unindexed hand the raw proxy back, and
    its ``.reshape`` etc. are warp-flavored (``array.reshape() takes 2
    positional arguments...``).  ``.torch`` is the zero-copy torch view.
    """
    value = getattr(value, "torch", value)
    if isinstance(value, torch.Tensor):
        return value
    return wp.to_torch(value)


def _apply_stable_noise(value: torch.Tensor, noise_cfg) -> torch.Tensor:
    """Apply a stable ``isaaclab.utils.noise`` cfg with its own impl (or not)."""
    if noise_cfg is None:
        return value
    fn = getattr(noise_cfg, "func", None)
    if fn is None or not callable(fn):
        return value
    return fn(value, noise_cfg)


def make_torch_obs_warp(
    source_func: Callable,
    noise: Any = None,
    frame_dim: int | None = None,
    params: dict | None = None,
) -> Callable:
    """Wrap a torch obs term ``f(env, **params) -> (N, ...)`` into Warp protocol.

    The output is flattened into the manager's 2D ``(N, D)`` buffer; the
    source's trailing shape is published on the wrapper as
    ``_torch_src_shapes: {id(env): tuple}`` for the env wrapper's 3D restore.

    Args:
        source_func: torch observation term (a ``"module:attr"`` string is
            resolved like the managers resolve string ``func=`` cfgs).
        noise: a stable noise cfg (e.g. ``UniformNoiseCfg``) applied to the
            term value every step -- pass it here, NOT on the ObsTerm: the
            experimental manager silently skips classic noise cfgs, and for
            history-wrapped terms the noise must enter
            :func:`mdp_warp.obs_history.make_history_obs` instead.
        frame_dim: flattened per-frame dim.  Optional; when omitted it is
            probed against the live env during ObservationManager init.
        params: the ObsTerm's own ``params=`` dict (``command_name`` & co).
            Needed for the deferred dim probe: it runs during the
            ObservationManager's own ``__init__`` (before the manager is
            reachable through the env), so the probe cannot look the term up
            and must carry the params on the wrapper.  Runtime calls get the
            params as kwargs from the manager anyway.
    """
    if isinstance(source_func, str):
        from isaaclab.utils import string_to_callable

        source_func = string_to_callable(source_func)

    @generic_io_descriptor_warp(out_dim=None if frame_dim is None else int(frame_dim), dtype=torch.float32)
    @warp_capturable(False)
    def torch_obs_warp(env: ManagerBasedEnv, out, **params):
        resolved = torch_obs_warp._resolved.get(id(env))
        if resolved is None:
            resolved = _resolve_source_params(env, source_func, {})
            torch_obs_warp._resolved[id(env)] = resolved
        value = _as_torch(source_func(env, **{**resolved, **params}))
        torch_obs_warp._torch_src_shapes[id(env)] = tuple(value.shape[1:])
        value = _apply_stable_noise(value, noise)
        out_view = wp.to_torch(out)
        out_view.view(out_view.shape[0], -1).copy_(value.reshape(out_view.shape[0], -1))

    torch_obs_warp._torch_src_shapes: dict[int, tuple] = {}
    torch_obs_warp._resolved: dict[int, dict] = {}
    torch_obs_warp._torch_src = source_func
    torch_obs_warp._term_params = dict(params or {})
    torch_obs_warp._warp_protocol = True
    if frame_dim is None:
        torch_obs_warp._descriptor = _ProbeDim(
            torch_obs_warp._descriptor, torch_obs_warp, source_func
        )
    torch_obs_warp.__name__ = f"torch_obs_warp({getattr(source_func, '__name__', source_func)})"
    torch_obs_warp.__doc__ = (
        f"Warp-protocol bridge around the torch observation "
        f"``{getattr(source_func, '__name__', source_func)!r}`` (zero-copy torch handoff; "
        f"not CUDA-graph capturable)."
    )
    _present_source_signature(torch_obs_warp, source_func)
    return torch_obs_warp


# --------------------------------------------------------------------------
# rewards / terminations
# --------------------------------------------------------------------------


def make_torch_reward_warp(source_func: Callable) -> Callable:
    """Wrap a torch reward term ``f(env, **params) -> (N,)`` into Warp protocol."""
    if isinstance(source_func, str):
        from isaaclab.utils import string_to_callable

        source_func = string_to_callable(source_func)

    @warp_capturable(False)
    def torch_reward_warp(env: ManagerBasedEnv, out, **params):
        resolved = torch_reward_warp._resolved.get(id(env))
        if resolved is None:
            resolved = _resolve_source_params(env, source_func, {})
            torch_reward_warp._resolved[id(env)] = resolved
        wp.to_torch(out).copy_(_as_torch(source_func(env, **{**resolved, **params})).reshape(-1))

    torch_reward_warp._resolved: dict[int, dict] = {}
    torch_reward_warp._torch_src = source_func
    torch_reward_warp._warp_protocol = True
    torch_reward_warp.__name__ = f"torch_reward_warp({getattr(source_func, '__name__', source_func)})"
    torch_reward_warp.__doc__ = (
        f"Warp-protocol bridge around the torch reward "
        f"``{getattr(source_func, '__name__', source_func)!r}`` (zero-copy torch handoff)."
    )
    _present_source_signature(torch_reward_warp, source_func)
    return torch_reward_warp


def make_torch_termination_warp(source_func: Callable) -> Callable:
    """Wrap a torch termination term ``f(env, **params) -> (N,) bool`` into Warp protocol."""
    if isinstance(source_func, str):
        from isaaclab.utils import string_to_callable

        source_func = string_to_callable(source_func)

    @warp_capturable(False)
    def torch_termination_warp(env: ManagerBasedEnv, out, **params):
        resolved = torch_termination_warp._resolved.get(id(env))
        if resolved is None:
            resolved = _resolve_source_params(env, source_func, {})
            torch_termination_warp._resolved[id(env)] = resolved
        wp.to_torch(out).copy_(_as_torch(source_func(env, **{**resolved, **params})).reshape(-1))

    torch_termination_warp._resolved: dict[int, dict] = {}
    torch_termination_warp._torch_src = source_func
    torch_termination_warp._warp_protocol = True
    torch_termination_warp.__name__ = f"torch_termination_warp({getattr(source_func, '__name__', source_func)})"
    torch_termination_warp.__doc__ = (
        f"Warp-protocol bridge around the torch termination "
        f"``{getattr(source_func, '__name__', source_func)!r}`` (zero-copy torch handoff)."
    )
    _present_source_signature(torch_termination_warp, source_func)
    return torch_termination_warp


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


def make_torch_event_warp(source_func: Callable) -> Callable:
    """Wrap a torch event term ``f(env, env_ids, **params)`` into Warp protocol.

    The warp env mask is converted to an index tensor (unitree's
    ``_mask_to_env_ids`` pattern); startup/reset/interval modes all flow
    through it unchanged.
    """
    if isinstance(source_func, str):
        from isaaclab.utils import string_to_callable

        source_func = string_to_callable(source_func)

    @warp_capturable(False)
    def torch_event_warp(env: ManagerBasedEnv, env_mask, **params):
        resolved = torch_event_warp._resolved.get(id(env))
        if resolved is None:
            resolved = _resolve_source_params(env, source_func, {})
            torch_event_warp._resolved[id(env)] = resolved
        source_func(env, env_ids=mask_to_env_ids(env_mask), **{**resolved, **params})

    torch_event_warp._resolved: dict[int, dict] = {}
    torch_event_warp._torch_src = source_func
    torch_event_warp._warp_protocol = True
    torch_event_warp.__name__ = f"torch_event_warp({getattr(source_func, '__name__', source_func)})"
    torch_event_warp.__doc__ = (
        f"Warp-protocol bridge around the torch event "
        f"``{getattr(source_func, '__name__', source_func)!r}`` (env_mask -> env_ids)."
    )
    _present_source_signature(torch_event_warp, source_func, second="env_mask")
    return torch_event_warp
