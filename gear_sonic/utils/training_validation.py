"""Opt-in numerical checks and evidence for short PPO training runs."""

from collections.abc import Mapping
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


def assert_finite(value, label):
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite(child, f"{label}[{index}]")
    elif isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite {label}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Non-finite {label}")


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        assert_finite(value, name)
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_steps(optimizer):
    return sorted({int(state["step"]) for state in optimizer.state.values() if "step" in state})


def json_values(value):
    if isinstance(value, Mapping):
        return {key: json_values(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(child) for child in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


class TrainingValidationCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.logs = []

    def on_train_begin(self, args, state, control, model, optimizer, lr_scheduler, accelerator, **kwargs):
        self._train_started = time.perf_counter()
        model = accelerator.unwrap_model(model)
        self.report = {
            "completed": False, "start_global_step": state.global_step,
            "start_actor_hash": parameter_hash(model.policy),
            "start_critic_hash": parameter_hash(model.value_model),
            "start_optimizer_steps": optimizer_steps(optimizer),
            "start_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "start_scheduler_epoch": lr_scheduler.state_dict().get("last_epoch"),
        }
        env = kwargs.get("env")
        if env is not None and env.config.get("record_physics_properties", False):
            from gear_sonic.utils.lab_api import asset_tensor
            import warp as wp
            robot = env.env.scene["robot"]
            mu = wp.to_torch(env.env.sim.physics_manager.get_model().shape_material_mu)
            mass = asset_tensor(robot, "body_mass")
            torso = robot.body_names.index("torso_link")
            offset = (asset_tensor(robot, "default_joint_pos") - robot._sonic_nominal_joint_pos).abs().max()
            self.report["physics_properties"] = {
                "mjwarp_njmax": env.env.sim.physics_manager._solver.mjw_data.njmax,
                "friction_min_max": [mu.min().item(), mu.max().item()],
                "torso_masses_kg": mass[:, torso].tolist(),
                "max_default_joint_offset_rad": offset.item(),
                "terrain_type": env.config.terrain_type,
                "active_events": env.env.event_manager.active_terms,
            }
            if hasattr(robot, "_sonic_nominal_body_com"):
                com_offset = asset_tensor(robot, "body_com_pos_b") - robot._sonic_nominal_body_com
                self.report["physics_properties"]["torso_com_offset_m"] = com_offset[:, torso].tolist()

    def on_log(self, args, state, control, logs, **kwargs):
        assert_finite(logs, "training_log")
        self.logs.append(json_values({"global_step": state.global_step, **logs}))
        self._write(state.global_step)

    def on_train_end(self, args, state, control, model, optimizer, lr_scheduler, accelerator, **kwargs):
        model = accelerator.unwrap_model(model)
        self.report.update({
            "completed": True, "final_global_step": state.global_step,
            "final_actor_hash": parameter_hash(model.policy),
            "final_critic_hash": parameter_hash(model.value_model),
            "final_optimizer_steps": optimizer_steps(optimizer),
            "final_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "final_scheduler_epoch": lr_scheduler.state_dict().get("last_epoch"),
        })
        self.report["actor_updated"] = self.report["start_actor_hash"] != self.report["final_actor_hash"]
        self.report["critic_updated"] = self.report["start_critic_hash"] != self.report["final_critic_hash"]
        self.report["training_wall_seconds"] = time.perf_counter() - self._train_started
        if accelerator.device.type == "cuda":
            self.report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(accelerator.device)
            self.report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(accelerator.device)
        if not self.report["actor_updated"] or not self.report["critic_updated"]:
            raise AssertionError("Actor and critic must both update during the short training test")
        assert_finite(optimizer.state_dict(), "optimizer_state")
        self._write(state.global_step)

    def _write(self, step):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = {**self.report, "logs": self.logs}
        (self.output_dir / f"training_validation_step_{step:06d}.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
