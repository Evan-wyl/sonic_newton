"""Warp-convention events for the sonic tracking envs (Isaac Lab 3.0).

The experimental (warp) ``EventManager`` calls terms as
``func(env, env_mask, **params)`` with a boolean ``wp.array`` mask over envs,
which the stable torch event terms (``ManagerTermBase`` classes, or plain
functions taking ``env_ids: torch.Tensor | None``) do not accept.  This module
provides the task-side adapters; the math stays in the official/sonic torch
implementations (startup events run outside CUDA-graph capture, so torch is
fine here -- only interval events would need true kernels).

Contents:

* :func:`_stable_class_adapter` -- wraps the official backend-agnostic
  ``ManagerTermBase`` event classes as warp-convention functions.  The
  experimental manager instantiates stable class terms only for the
  ``prestartup`` mode, so the adapter instantiates them itself with a
  lightweight runtime cfg and caches one instance per
  (env, YAML term).  Exported adapters: ``randomize_rigid_body_material``,
  ``randomize_rigid_body_mass``, ``randomize_actuator_gains``.
* :func:`randomize_joint_default_pos` -- warp-convention port of sonic's
  torch event (:mod:`gear_sonic.envs.manager_env.mdp.events`), following the
  unitree_rl_lab ``mimic/mdp_warp/events.py`` precedent.  Same math for the
  uniform ``add``/``abs`` cases the tracking YAMLs use.

  The experimental ``JointPositionAction`` stores a shared 1-D offset, while
  the randomized default pose is per environment.  This event installs a
  Warp action-processing kernel that reads the per-env defaults, matching the
  stable action term's behavior.  The 1-D offset is also refreshed from env 0
  for IO descriptors and export.

Terms that need NO adapter (stage-3 YAML wiring points the ``func:`` string
directly at the official experimental warp terms):

* ``base_com`` -> ``isaaclab_experimental.envs.mdp:randomize_rigid_body_com``
* ``push_robot`` (interval) ->
  ``isaaclab_experimental.envs.mdp:push_by_setting_velocity``
"""

from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Callable, Literal

# Submodule imports on purpose: the Lab 3.0 lazy-loader breaks re-exports
# through package __init__ in some contexts (see REF plan Phase 1 notes).
from isaaclab.envs.mdp import events as _official_events
import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _process_actions_with_env_defaults(
    actions: wp.array2d(dtype=wp.float32),
    action_offset: int,
    scale: wp.array(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    joint_ids: wp.array(dtype=wp.int32),
    clip: wp.array2d(dtype=wp.float32),
    raw: wp.array2d(dtype=wp.float32),
    processed: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    value = actions[i, action_offset + j]
    raw[i, j] = value
    processed[i, j] = wp.clamp(value * scale[j] + default_joint_pos[i, joint_ids[j]], clip[j, 0], clip[j, 1])


def _process_per_env_default_actions(term, actions: wp.array, action_offset: int = 0):
    """Apply position actions relative to each env's randomized default pose."""
    wp.launch(
        _process_actions_with_env_defaults,
        dim=(term.num_envs, term.action_dim),
        inputs=[
            actions,
            int(action_offset),
            term._scale,
            term._event_default_joint_pos,
            term._default_joint_ids,
            term._clip,
            term._raw_actions,
            term._processed_actions,
        ],
        device=term.device,
    )


def _to_torch(x) -> torch.Tensor:
    """Zero-copy torch view of a Lab 3.0 array (ProxyArray or raw wp.array)."""
    torch_view = getattr(x, "torch", None)
    if torch_view is not None:  # isaaclab.utils.warp.proxy_array.ProxyArray
        return torch_view
    return wp.to_torch(x)


def _mask_to_env_ids(env_mask: wp.array) -> torch.Tensor:
    """Boolean warp env mask -> torch index tensor (unitree convention)."""
    return wp.to_torch(env_mask).nonzero(as_tuple=True)[0]


def _stable_class_adapter(cls) -> Callable:
    """Adapt an official ``ManagerTermBase`` event class to the warp protocol.

    The experimental manager calls plain-function terms as
    ``func(env, env_mask, **params)`` but never instantiates stable class
    terms for ``startup`` mode.  The returned adapter instantiates ``cls``
    once per (env, term) with a lightweight runtime cfg -- the official
    classes read only ``cfg.params`` in ``__init__`` -- and forwards each call
    as ``inst(env, env_ids, **params)`` with the mask converted to an index
    tensor. A stable configclass is unsuitable here because resolved params
    can contain non-deepcopyable simulator selectors.

    The per-term cache key uses ``id(params["asset_cfg"])``: the manager
    passes the same ``SceneEntityCfg`` object stored in ``term_cfg.params``
    on every call, and each YAML term owns its own instance -- so two terms
    sharing one adapter name (e.g. ``randomize_rigid_body_mass`` and
    ``randomize_rigid_body_mass_hand``) get separate class instances, as the
    stable manager would give them.
    """
    instances: dict[tuple[int, int], object] = {}

    def adapter(env: ManagerBasedEnv, env_mask: wp.array, **params):
        key = (id(env), id(params.get("asset_cfg")))
        if cls is _official_events.randomize_actuator_gains:
            # The stable gain randomizer intersects tensor actuator indices
            # with asset joint ids using torch.isin. Warp entity resolution
            # supplies a list; normalize a shallow copy for this torch call.
            asset_cfg = params.get("asset_cfg")
            if asset_cfg is not None and isinstance(asset_cfg.joint_ids, (list, tuple)):
                asset_cfg = copy.copy(asset_cfg)
                asset_cfg.joint_ids = torch.as_tensor(
                    asset_cfg.joint_ids, dtype=torch.long, device=env.device
                )
                params["asset_cfg"] = asset_cfg
        inst = instances.get(key)
        if inst is None:
            # EventTermCfg is a configclass and deep-copies params in __post_init__.
            # The experimental manager has already resolved SceneEntityCfg here;
            # it may hold ctypes-backed simulator selectors that cannot be deep-copied.
            # These class terms only consume cfg.params during construction, so pass
            # a lightweight runtime cfg instead of re-wrapping live params in a config.
            cfg = SimpleNamespace(func=cls, mode="startup", params=dict(params))
            inst = cls(cfg=cfg, env=env)
            instances[key] = inst
        inst(env, _mask_to_env_ids(env_mask), **params)

    adapter.__name__ = cls.__name__
    adapter._warp_protocol = True
    adapter.__doc__ = (
        f"Warp-convention adapter around the official ``{cls.__name__}`` class term "
        f"(same math, env_mask -> env_ids conversion, instance cached per env/term)."
    )
    return adapter


randomize_rigid_body_material = _stable_class_adapter(_official_events.randomize_rigid_body_material)
randomize_rigid_body_mass = _stable_class_adapter(_official_events.randomize_rigid_body_mass)
randomize_actuator_gains = _stable_class_adapter(_official_events.randomize_actuator_gains)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_mask: wp.array,
    asset_cfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize default joint positions (warp-convention port of the sonic torch event).

    Same math as :func:`gear_sonic.envs.manager_env.mdp.events.randomize_joint_default_pos`
    for the uniform ``add``/``abs`` cases (the only ones the tracking
    YAMLs use).  ``scale`` and non-uniform distributions would go through the
    official ``_randomize_prop_by_op`` in the torch event; they are rejected
    here until a task needs them (unitree's port made the same cut).

    The action kernel is switched to per-environment defaults after randomization.
    """
    if distribution != "uniform":
        raise ValueError(
            f"Unsupported distribution: {distribution!r}. Only 'uniform' is supported in the Warp port."
        )
    if operation not in ("add", "abs"):
        raise ValueError(
            f"Unsupported operation: {operation!r}. Only 'add'/'abs' are supported in the Warp port."
        )
    if pos_distribution_params is None:
        raise ValueError("pos_distribution_params is required in the Warp port.")

    asset = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term("joint_pos")
    if action_term._asset is not asset or not action_term.cfg.use_default_offset:
        raise ValueError("Joint default randomization requires the same asset and use_default_offset=True.")

    # (num_envs, num_joints) zero-copy torch view of the unified data layer
    default_pos = _to_torch(asset.data.default_joint_pos)

    # save nominal value for export (before randomization, like the torch event)
    asset.data.default_joint_pos_nominal = default_pos[0].clone()

    env_ids = _mask_to_env_ids(env_mask)

    # The Warp action normally adds a shared env-0 offset. Install this once
    # before graph capture so each environment instead reads its own defaults.
    # Installation is independent of whether this call randomizes any env.
    if not hasattr(action_term, "_default_joint_ids"):
        action_term._event_default_joint_pos = wp.from_torch(default_pos)
        action_term._default_joint_ids = action_term._joint_ids_wp
        if action_term._default_joint_ids is None:
            action_term._default_joint_ids = wp.array(
                range(action_term.action_dim), dtype=wp.int32, device=env.device
            )
        action_term.process_actions = MethodType(_process_per_env_default_actions, action_term)

    if env_ids.numel() == 0:
        return
    joint_ids = asset_cfg.joint_ids
    if not isinstance(joint_ids, slice):
        joint_ids = torch.as_tensor(list(joint_ids), dtype=torch.long, device=default_pos.device)

    lo, hi = pos_distribution_params
    pos = default_pos[env_ids].clone()
    current = pos[:, joint_ids]
    if operation == "add":
        pos[:, joint_ids] = current + torch.empty_like(current).uniform_(lo, hi)
    else:  # "abs"
        pos[:, joint_ids] = torch.empty_like(current).uniform_(lo, hi)
    default_pos[env_ids] = pos

    # The Warp action normally adds a shared env-0 offset. Install this once
    # before graph capture so each environment instead reads its own defaults.
    if not hasattr(action_term, "_default_joint_ids"):
        action_term._event_default_joint_pos = wp.from_torch(default_pos)
        action_term._default_joint_ids = action_term._joint_ids_wp
        if action_term._default_joint_ids is None:
            action_term._default_joint_ids = wp.array(
                range(action_term.action_dim), dtype=wp.int32, device=env.device
            )
        action_term.process_actions = MethodType(_process_per_env_default_actions, action_term)

    # Refresh the action term offset (not updated automatically).  Sonic's
    # torch event intersects action/asset joint names because the action term
    # may cover a subset/reordering of the asset joints; keep that logic.
    action_offset = _to_torch(action_term._offset)  # 1-D (action_dim,) wp.array

    asset_joint_names = asset.joint_names
    if isinstance(joint_ids, slice):
        randomized = set(asset_joint_names)
    else:
        randomized = {asset_joint_names[int(i)] for i in joint_ids}

    action_idx: list[int] = []
    asset_idx: list[int] = []
    for a_idx, name in enumerate(action_term._joint_names):
        if name in randomized:
            action_idx.append(a_idx)
            asset_idx.append(asset_joint_names.index(name))
    if action_idx:
        # Keep the 1-D IO descriptor offset in sync with env 0.
        action_offset[action_idx] = default_pos[0][asset_idx]
