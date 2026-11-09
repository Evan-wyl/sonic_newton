"""Warp-env wiring profile: swap a built torch env cfg to the experimental managers.

Stage 3a of the Newton/Warp migration (the Newton/Warp migration).
When ``manager_env.config.use_warp_env`` is set, the train entry constructs
``isaaclab_experimental.envs.ManagerBasedRLEnvWarp`` instead of the stable
``ManagerBasedRLEnv``.  The experimental managers isinstance-reject the stable
``isaaclab.managers.*TermCfg`` classes and call every term through the
Warp-first protocols, so the profile rewrites the already-built Hydra cfg in
place (no YAML forks -- the torch path stays the default and unchanged):

* actions → experimental ``JointActionCfg`` subclasses (official Warp ports,
  same fields incl. ``use_default_offset``);
* observations → experimental ``ObservationTermCfg`` with every torch term
  bridged through :func:`mdp_warp.torch_bridge.make_torch_obs_warp`; per-term
  (and group-level) ``history_length`` moves into
  :func:`mdp_warp.obs_history.make_history_obs` wrappers (the experimental
  manager raises ``NotImplementedError`` for any history_length > 0 and its
  concat fast-path requires all-history-free terms), and the classic noise
  cfgs move with them (the experimental manager silently skips classic noise
  objects, and re-noising whole history blocks every step would be wrong
  anyway);
* rewards / terminations → experimental term cfgs with bridged funcs (weights,
  ``time_out`` flags and params preserved verbatim);
* events → term funcs swapped to Warp-convention adapters (mask → env_ids);
  official ``ManagerTermBase`` classes go through
  :func:`mdp_warp.events._stable_class_adapter`, plain functions through the
  generic bridge.  The ``EventTermCfg`` class itself is shared (passthrough
  in the experimental package), so only ``func`` changes.

Terms that already follow a Warp protocol
(:func:`mdp_warp.torch_bridge.is_warp_term`) are kept as-is.

Stages 3b/3c/3d on top of the Stage-3a bridge:

* commands → exact ``TrackingCommand`` terms swap to
  :class:`mdp_warp.commands.TrackingCommandWarp` (STABLE torch CommandManager
  pipeline preserved; fills persistent buffers + Warp views once per step);
* observations → official experimental natives for the re-exported stable
  terms (base/joint/action/command) and ``mdp_warp.observations`` kernels /
  copy terms for the command-derived tokenizer & critic terms;
* rewards/terminations → official natives plus the ``mdp_warp.rewards`` /
  ``mdp_warp.terminations`` tracking ports;
* ``manager_call_config`` defaults to ``{"default": 1}`` (eager Warp managers).
  The switch also caps managers holding bridged Torch terms to eager mode.
"""

from __future__ import annotations

import inspect
import logging

from isaaclab.managers import SceneEntityCfg as _StableSceneEntityCfg
from isaaclab_experimental.envs.mdp import observations as _warp_native_obs, rewards as _warp_rewards
from isaaclab_experimental.envs.mdp.actions import (
    JointEffortActionCfg as WarpJointEffortActionCfg,
    JointPositionActionCfg as WarpJointPositionActionCfg,
)
from isaaclab_experimental.managers import (
    ObservationTermCfg as WarpObsTermCfg,
    RewardTermCfg as WarpRewardTermCfg,
    SceneEntityCfg as _WarpSceneEntityCfg,
    TerminationTermCfg as WarpTerminationTermCfg,
)

# sonic torch events that have a dedicated Warp-convention port in
# ``mdp_warp.events`` -- the generic bridge cannot run them: their torch
# indexing paths assume resolved explicit joint-id lists / per-env action
# offsets that the warp env's data layer resolves differently
import gear_sonic.envs.manager_env.mdp as _sonic_mdp

from . import (
    events as _warp_events,
    observations as _warp_obs,
    rewards as _warp_tracking_rewards,
    terminations as _warp_tracking_terminations,
)
from .commands import TrackingCommand, TrackingCommandWarpCfg
from .events import _stable_class_adapter
from .obs_history import make_history_obs
from .observations import make_command_copy_obs, native_with_noise
from .torch_bridge import (
    is_warp_term,
    make_torch_event_warp,
    make_torch_obs_warp,
    make_torch_reward_warp,
    make_torch_termination_warp,
)

_EVENT_WARP_PORTS = {
    _sonic_mdp.randomize_joint_default_pos: _warp_events.randomize_joint_default_pos,
}

# torch terms reading ``env.action_manager.action`` (wp.array under the warp
# env) through torch ops -- the official experimental mdp has native Warp
# kernels for them; route there instead of bridging.  Stage 3c adds the
# official natives for the remaining re-exported stable rewards (semantics
# identical: sonic never redefines them) and the sonic tracking-reward Warp
# ports (``mdp_warp.rewards``).
_REWARD_WARP_PORTS = {
    _sonic_mdp.action_rate_l2: _warp_rewards.action_rate_l2,
    _sonic_mdp.action_l2: _warp_rewards.action_l2,
    _sonic_mdp.joint_acc_l2: _warp_rewards.joint_acc_l2,
    _sonic_mdp.joint_pos_limits: _warp_rewards.joint_pos_limits,
    _sonic_mdp.undesired_contacts: _warp_rewards.undesired_contacts,
    _sonic_mdp.tracking_anchor_pos_error: _warp_tracking_rewards.tracking_anchor_pos_error,
    _sonic_mdp.tracking_anchor_ori_error: _warp_tracking_rewards.tracking_anchor_ori_error,
    _sonic_mdp.tracking_relative_body_pos_error: _warp_tracking_rewards.tracking_relative_body_pos_error,
    _sonic_mdp.tracking_relative_body_ori_error: _warp_tracking_rewards.tracking_relative_body_ori_error,
    _sonic_mdp.tracking_body_linvel_error: _warp_tracking_rewards.tracking_body_linvel_error,
    _sonic_mdp.tracking_body_angvel_error: _warp_tracking_rewards.tracking_body_angvel_error,
    _sonic_mdp.tracking_local_vr_5point_error: _warp_tracking_rewards.tracking_local_vr_5point_error,
    _sonic_mdp.anti_shake_ang_vel_l2: _warp_tracking_rewards.anti_shake_ang_vel_l2,
}

# sonic tracking-termination Warp ports (``mdp_warp.terminations``)
_TERMINATION_WARP_PORTS = {
    _sonic_mdp.exceeded_anchor_height: _warp_tracking_terminations.exceeded_anchor_height,
    _sonic_mdp.exceeded_anchor_ori: _warp_tracking_terminations.exceeded_anchor_ori,
    _sonic_mdp.exceeded_body_pos: _warp_tracking_terminations.exceeded_body_pos,
    _sonic_mdp.exceeded_body_height: _warp_tracking_terminations.exceeded_body_height,
    _sonic_mdp.tracking_time_out: _warp_tracking_terminations.tracking_time_out,
}

# re-exported stable obs terms with official experimental Warp natives
# (sonic re-exports them from isaaclab.envs.mdp verbatim)
_OBS_NATIVE_PORTS = {
    _sonic_mdp.base_lin_vel: _warp_native_obs.base_lin_vel,
    _sonic_mdp.base_ang_vel: _warp_native_obs.base_ang_vel,
    _sonic_mdp.joint_pos_rel: _warp_native_obs.joint_pos_rel,
    _sonic_mdp.joint_vel_rel: _warp_native_obs.joint_vel_rel,
    _sonic_mdp.last_action: _warp_native_obs.last_action,
    _sonic_mdp.generated_commands: _warp_native_obs.generated_commands,
}


def _kernel_obs_port(factory_name: str):
    def make(term_params, noise):
        return getattr(_warp_obs, factory_name)(term_params=term_params, noise=noise)

    return make


# sonic command-derived obs terms with dedicated Warp kernels / copy terms in
# ``mdp_warp.observations``.  Copy terms read the persistent buffers the
# TrackingCommandWarp fills once per step (buffer attr on the command).
_OBS_WARP_PORTS = {
    _sonic_mdp.gravity_dir: _kernel_obs_port("make_gravity_dir"),
    _sonic_mdp.motion_anchor_pos_b: _kernel_obs_port("make_motion_anchor_pos_b"),
    _sonic_mdp.motion_anchor_ori_b: _kernel_obs_port("make_motion_anchor_ori_b"),
    _sonic_mdp.robot_body_pos_b: _kernel_obs_port("make_robot_body_pos_b"),
    _sonic_mdp.robot_body_ori_b: _kernel_obs_port("make_robot_body_ori_b"),
    _sonic_mdp.vr_3point_local_target: _kernel_obs_port("make_vr_3point_local_target"),
    _sonic_mdp.vr_3point_local_orn_target: _kernel_obs_port("make_vr_3point_local_orn_target"),
    _sonic_mdp.command_z: lambda p, n: make_command_copy_obs(
        "_copy_command_z", term_params=p, noise=n
    ),
    _sonic_mdp.command_z_multi_future: lambda p, n: make_command_copy_obs(
        "_copy_command_z_multi_future", term_params=p, noise=n
    ),
    _sonic_mdp.command_multi_future: lambda p, n: make_command_copy_obs(
        "_copy_command_multi_future", term_params=p, noise=n
    ),
    _sonic_mdp.command_multi_future_lower_body: lambda p, n: make_command_copy_obs(
        "_copy_command_multi_future_lower_body", term_params=p, noise=n
    ),
    _sonic_mdp.motion_anchor_ori_b_mf: lambda p, n: make_command_copy_obs(
        "_copy_root_rot_dif_l_multi_future", term_params=p, noise=n
    ),
    _sonic_mdp.smpl_root_ori_b_mf: lambda p, n: make_command_copy_obs(
        "_copy_smpl_root_quat_w_dif_l_multi_future", term_params=p, noise=n
    ),
    _sonic_mdp.smpl_joints_multi_future_local: lambda p, n: make_command_copy_obs(
        "_copy_smpl_joints_multi_future_local", term_params=p, noise=n
    ),
    _sonic_mdp.joint_pos_multi_future_select_joints_for_smpl: lambda p, n: make_command_copy_obs(
        "_copy_joint_pos_multi_future_for_smpl", term_params=p, noise=n
    ),
    # generic getattr(command, obs_name) accessor -- reads the persistent
    # command attribute (e.g. the one-hot episode_encoder_index, int64 -> f32)
    _sonic_mdp.get_command_obs: lambda p, n: make_command_copy_obs(
        str(p.get("obs_name", "command")), term_params=p, noise=n
    ),
}

logger = logging.getLogger(__name__)

# stable -> experimental SceneEntityCfg: the official Warp natives read
# resolve-time attrs (joint_mask, joint_ids_wp) that only exist on the
# experimental class

_SCENE_CFG_SHARED_FIELDS = (
    "name",
    "joint_names",
    "joint_ids",
    "fixed_tendon_names",
    "fixed_tendon_ids",
    "body_names",
    "body_ids",
    "object_collection_names",
    "object_collection_ids",
    "preserve_order",
)


def _to_warp_scene_cfgs(params: dict) -> dict:
    """Rewrite stable ``SceneEntityCfg`` param values to the experimental class."""
    out = {}
    for key, value in params.items():
        if isinstance(value, _StableSceneEntityCfg) and not isinstance(
            value, _WarpSceneEntityCfg
        ):
            fields = {
                f: getattr(value, f)
                for f in _SCENE_CFG_SHARED_FIELDS
                if hasattr(value, f)
            }
            out[key] = _WarpSceneEntityCfg(**fields)
        else:
            out[key] = value
    return out

_GROUP_RESERVED_FIELDS = (
    "enable_corruption",
    "concatenate_terms",
    "history_length",
    "flatten_history_dim",
    "concatenate_dim",
    "history_order",
)

# stable action cfg class name -> experimental counterpart
_ACTION_CLASS_MAP = {
    "JointPositionActionCfg": WarpJointPositionActionCfg,
    "JointEffortActionCfg": WarpJointEffortActionCfg,
}


def _copy_fields(old_cfg, new_cls, exclude=()):
    """Build ``new_cls`` carrying every shared field of ``old_cfg``."""
    kwargs = {
        k: v
        for k, v in old_cfg.__dict__.items()
        if k in new_cls.__dataclass_fields__ and k not in exclude
    }
    return new_cls(**kwargs)


def _swap_actions(actions_cfg) -> list[str]:
    swapped = []
    if actions_cfg is None:
        return swapped
    for name, term_cfg in list(actions_cfg.__dict__.items()):
        if term_cfg is None:
            continue
        cls_name = type(term_cfg).__name__
        new_cls = _ACTION_CLASS_MAP.get(cls_name)
        if new_cls is None:
            # custom action terms need a Warp ActionTerm port (REF 3c); refuse
            # loudly instead of failing inside the manager with a confusing error
            raise TypeError(
                f"[warp-profile] action term '{name}' ({cls_name}) has no experimental counterpart "
                "mapped yet -- port it to isaaclab_experimental.managers.ActionTerm first."
            )
        if type(term_cfg) is new_cls:
            continue
        # class_type must NOT be copied: the stable cfg's ResolvableString
        # resolves to the stable class, and the experimental ActionManager
        # isinstance-rejects it
        setattr(actions_cfg, name, _copy_fields(term_cfg, new_cls, exclude=("class_type",)))
        swapped.append(name)
    return swapped


def _swap_observations(observations_cfg) -> list[str]:
    swapped = []
    if observations_cfg is None:
        return swapped
    for group_name, group_cfg in list(observations_cfg.__dict__.items()):
        if group_cfg is None:
            continue
        corruption = getattr(group_cfg, "enable_corruption", True)
        group_history = getattr(group_cfg, "history_length", None)
        if group_history is not None:
            # per-term history replaces the group knob (concat fast-path needs
            # history_length == 0 everywhere)
            group_cfg.history_length = None
        for term_name, term_cfg in list(group_cfg.__dict__.items()):
            if term_name in _GROUP_RESERVED_FIELDS or term_cfg is None:
                continue
            func = term_cfg.func
            history_length = term_cfg.history_length or 0
            if group_history is not None:
                history_length = int(group_history)
            noise = term_cfg.noise if corruption else None

            # resolve ResolvableString before the identity maps (torch_bridge
            # pattern; re-exported stable terms otherwise masquerade as warp)
            resolved = func
            if isinstance(resolved, str):
                from isaaclab.utils import string_to_callable

                resolved = string_to_callable(str(resolved))

            native_port = _OBS_NATIVE_PORTS.get(resolved)
            warp_port = _OBS_WARP_PORTS.get(resolved)
            if native_port is not None:
                # official experimental native: history wraps the ring; a
                # classic noise cfg on a history-free term must be baked in
                # (the experimental manager silently skips classic cfgs)
                if history_length > 0:
                    new_func = make_history_obs(native_port, history_length, noise=noise)
                elif noise is not None:
                    new_func = native_with_noise(native_port, noise)
                else:
                    new_func = native_port
            elif warp_port is not None:
                # sonic Warp port (kernel or copy term); noise enters the ring
                # per frame when history is active, else the term bakes it in
                new_func = warp_port(
                    _to_warp_scene_cfgs(dict(term_cfg.params)),
                    noise if history_length == 0 else None,
                )
                if history_length > 0:
                    new_func = make_history_obs(new_func, history_length, noise=noise)
            elif is_warp_term(func):
                new_func = func
                # a real Warp term with history still needs the ring wrapper
                if history_length > 0:
                    new_func = make_history_obs(func, history_length, noise=noise)
            else:
                bridged = make_torch_obs_warp(func, params=term_cfg.params)
                if history_length > 0:
                    # noise enters the ring once per frame (torch semantics)
                    new_func = make_history_obs(bridged, history_length, noise=noise)
                else:
                    # classic noise cfgs are silently skipped by the
                    # experimental manager -- the bridge applies them itself
                    new_func = make_torch_obs_warp(func, noise=noise, params=term_cfg.params)

            new_cfg = _copy_fields(term_cfg, WarpObsTermCfg)
            new_cfg.func = new_func
            if native_port is not None or warp_port is not None:
                new_cfg.params = _to_warp_scene_cfgs(dict(term_cfg.params))
            new_cfg.noise = None
            new_cfg.history_length = 0
            setattr(group_cfg, term_name, new_cfg)
            swapped.append(f"{group_name}/{term_name}")
    return swapped


def _swap_rewards(rewards_cfg) -> list[str]:
    swapped = []
    if rewards_cfg is None:
        return swapped
    for term_name, term_cfg in list(rewards_cfg.__dict__.items()):
        if term_cfg is None:
            continue
        func = term_cfg.func
        if isinstance(func, str):
            # configclass ResolvableString: resolve before the identity maps
            from isaaclab.utils import string_to_callable

            func = string_to_callable(str(func))
        if inspect.isclass(func):
            raise NotImplementedError(
                f"Warp profile does not support stateful term {term_name!r} ({func.__name__}); "
                "use use_warp_env=false for this configuration."
            )
        new_cfg = _copy_fields(term_cfg, WarpRewardTermCfg)
        warp_port = _REWARD_WARP_PORTS.get(func)
        if warp_port is not None:
            new_cfg.func = warp_port
            new_cfg.params = _to_warp_scene_cfgs(dict(term_cfg.params))
            swapped.append(f"{term_name}(warp-port)")
        else:
            new_cfg.func = func if is_warp_term(func) else make_torch_reward_warp(func)
            swapped.append(term_name)
        setattr(rewards_cfg, term_name, new_cfg)
    return swapped


def _swap_terminations(terminations_cfg) -> list[str]:
    swapped = []
    if terminations_cfg is None:
        return swapped
    for term_name, term_cfg in list(terminations_cfg.__dict__.items()):
        if term_cfg is None:
            continue
        func = term_cfg.func
        if isinstance(func, str):
            # configclass ResolvableString: resolve before the identity maps
            from isaaclab.utils import string_to_callable

            func = string_to_callable(str(func))
        if inspect.isclass(func):
            raise NotImplementedError(
                f"Warp profile does not support stateful term {term_name!r} ({func.__name__}); "
                "use use_warp_env=false for this configuration."
            )
        new_cfg = _copy_fields(term_cfg, WarpTerminationTermCfg)
        warp_port = _TERMINATION_WARP_PORTS.get(func)
        if warp_port is not None:
            new_cfg.func = warp_port
            new_cfg.params = _to_warp_scene_cfgs(dict(term_cfg.params))
            swapped.append(f"{term_name}(warp-port)")
        else:
            new_cfg.func = func if is_warp_term(func) else make_torch_termination_warp(func)
            swapped.append(term_name)
        setattr(terminations_cfg, term_name, new_cfg)
    return swapped


def _swap_events(events_cfg) -> list[str]:
    swapped = []
    if events_cfg is None:
        return swapped
    for term_name, term_cfg in list(events_cfg.__dict__.items()):
        if term_cfg is None or term_name.startswith("_"):
            continue
        func = term_cfg.func
        if isinstance(func, str):
            # configclass ResolvableString (str subclass): resolve before the
            # class/plain-function discrimination below
            from isaaclab.utils import string_to_callable

            func = string_to_callable(str(func))
            term_cfg.func = func
        if is_warp_term(func):
            continue
        warp_port = _EVENT_WARP_PORTS.get(func)
        if warp_port is not None:
            term_cfg.func = warp_port
            swapped.append(f"{term_name}(warp-port)")
            continue
        if inspect.isclass(func):
            # official ManagerTermBase event (e.g. randomize_rigid_body_material)
            adapter = _stable_class_adapter(func)
            # present the class __call__ signature (minus self) to the
            # manager's static param check (torch_bridge pattern)
            try:
                sig = inspect.signature(func.__call__)
                params = list(sig.parameters.values())[1:]  # drop self
                adapter.__signature__ = sig.replace(parameters=params)
            except (TypeError, ValueError):
                pass
        else:
            adapter = make_torch_event_warp(func)
        term_cfg.func = adapter
        swapped.append(term_name)
    return swapped


def _swap_commands(commands_cfg) -> list[str]:
    """Swap exact ``TrackingCommand`` terms to the Warp-consumable subclass.

    Stage 3b: ``TrackingCommandWarp`` keeps the STABLE torch CommandManager
    pipeline (resampling, adaptive sampling, metrics host mirror) but fills
    persistent buffers + Warp views once per step, which every Stage-3c
    obs/reward/termination kernel reads.  Only exact-class terms swap --
    subclasses (ForceTrackingCommand & friends) carry their own fields and
    stay bridged.
    """
    swapped = []
    if commands_cfg is None:
        return swapped
    for name, term_cfg in list(commands_cfg.__dict__.items()):
        if term_cfg is None or name.startswith("_"):
            continue
        cls = getattr(term_cfg, "class_type", None)
        if isinstance(cls, str):
            from isaaclab.utils import string_to_callable

            cls = string_to_callable(str(cls))
        if cls is None:
            continue
        if cls is TrackingCommand:
            setattr(
                commands_cfg,
                name,
                _copy_fields(term_cfg, TrackingCommandWarpCfg, exclude=("class_type",)),
            )
            swapped.append(f"{name}(TrackingCommandWarp)")
    return swapped


def apply_warp_env_profile(env_cfg, manager_call_config=None) -> None:
    """Rewrite a built ``ModularTrackingEnvCfg`` for the Warp env class.

    Args:
        env_cfg: the instantiated env cfg (Hydra-built, stable term cfgs).
        manager_call_config: optional ``ManagerCallSwitch`` config (dict of
            manager name -> mode, 0/1/2).  Defaults to ``{"default": 1}``
            (eager Warp managers; set 2 to request CUDA-graph capture): stages 3b/3c ported every
            observation/reward/termination term on the sonic tracking surface
            to Warp kernels or official experimental natives, and the switch
            auto-caps any manager that still holds a bridged torch term
            (non-capturable) down to eager mode 1.
    """
    commands = _swap_commands(getattr(env_cfg, "commands", None))
    actions = _swap_actions(getattr(env_cfg, "actions", None))
    observations = _swap_observations(getattr(env_cfg, "observations", None))
    rewards = _swap_rewards(getattr(env_cfg, "rewards", None))
    terminations = _swap_terminations(getattr(env_cfg, "terminations", None))
    events = _swap_events(getattr(env_cfg, "events", None))
    # Recent IsaacLab versions adapt the cfg again inside the Warp env
    # constructor. Recognize our Warp implementations and generated adapters
    # as native, so the frontend does not search for same-named core twins.
    from isaaclab_experimental.envs.frontend import WarpFrontend

    sonic_warp_prefix = "gear_sonic.envs.manager_env.mdp_warp."
    if sonic_warp_prefix not in WarpFrontend.WARP_ROOT_PREFIXES:
        WarpFrontend.WARP_ROOT_PREFIXES += (sonic_warp_prefix,)

    env_cfg.manager_call_config = dict(manager_call_config or {"default": 1})
    print("[warp-profile] ManagerBasedRLEnvWarp term profile applied:")
    print(f"  - commands     : {', '.join(commands) or '(none)'}")
    print(f"  - actions      : {', '.join(actions) or '(none)'}")
    print(f"  - observations : {len(observations)} terms ({', '.join(observations[:6])}"
          f"{' ...' if len(observations) > 6 else ''})")
    print(f"  - rewards      : {len(rewards)} terms, terminations: {len(terminations)} terms")
    print(f"  - events       : {', '.join(events) or '(none)'}")
    print(f"  - manager_call_config: {env_cfg.manager_call_config}")
