"""Load trusted SONIC checkpoints while resolving project classes in this package."""

import pickle
import sys

import torch


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        for old, new in (
            ("gear_sonic_newton", "gear_sonic"),
            ("groot.rl.agents.modules.modules", "gear_sonic.trl.modules.base_module"),
            ("groot.rl.agents", "gear_sonic.trl"),
            ("groot.rl.trl", "gear_sonic.trl"),
            ("groot.rl.envs", "gear_sonic.envs"),
            ("groot.rl.utils", "gear_sonic.utils"),
        ):
            if module == old or module.startswith(old + "."):
                module = new + module[len(old):]
                break
        return super().find_class(module, name)


# Required by torch's loader for legacy serialization headers.
load = pickle.load


def load_checkpoint(path, *, map_location="cpu"):
    """Remap pickled project class paths as well as YAML configuration paths.

    Like the existing training loader, this loads full training metadata and
    must only be used for trusted checkpoints.
    """
    return torch.load(
        path, map_location=map_location, weights_only=False,
        pickle_module=sys.modules[__name__],
    )


def load_policy_weights(policy, checkpoint):
    """Require a complete actor state; migrate only known std parameter names."""
    state = checkpoint.get("actor_model_state_dict", checkpoint.get("policy_state_dict"))
    if state is None:
        raise ValueError("Checkpoint has no recognized policy state dictionary.")
    state = dict(state)
    uses_std = "std" in policy.state_dict()
    if uses_std and "std" not in state and "log_std" in state:
        state["std"] = state.pop("log_std").exp()
    elif not uses_std and "log_std" not in state and "std" in state:
        state["log_std"] = state.pop("std").log()
    return policy.load_state_dict(state, strict=True)
