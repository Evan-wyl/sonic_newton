"""Warp-kernel MDP terms for the sonic tracking envs (Isaac Lab 3.0).

Ported/adapted from unitree_rl_lab's ``tasks/mimic/mdp_warp`` package (see
the Newton/Warp migration Phase 3).  The terms mirror
:mod:`gear_sonic.envs.manager_env.mdp` one-to-one; everything that has an
official Warp-first override in :mod:`isaaclab_experimental.envs.mdp` is
used from there, and only the tracking/motion-dependent observations,
rewards, terminations, events plus the Warp-consumable command term live
here.

Staged rollout (each stage keeps the torch path runnable side-by-side):

1. infrastructure: shared XYZW quaternion math (:mod:`.warp_utils`) and
   device-side per-term observation history (:func:`make_history_obs`) --
   the experimental Warp ObservationManager rejects ``history_length > 0``,
   and sonic's policy group needs per-term H=10;
2. term ports: observations / rewards / terminations / events for the
   G1 and H2 tracking experiments, plus ``TrackingCommandWarp`` (STABLE
   torch command first with persistent wp views, steady-state kernels
   second);
3. wiring: env-class switch (``ManagerBasedRLEnvWarp``) in the train/smoke
   entrypoints and experimental ``*TermCfg`` variants of the Hydra term
   YAMLs (the experimental managers isinstance-reject the stable
   ``isaaclab.managers.*TermCfg`` classes -- both subclass the same
   ``ManagerTermBaseCfg`` but are siblings).
"""

from .obs_history import make_history_obs, register_env_hint, resolve_dones
from .torch_bridge import (
    is_warp_term,
    make_torch_event_warp,
    make_torch_obs_warp,
    make_torch_reward_warp,
    make_torch_termination_warp,
)
