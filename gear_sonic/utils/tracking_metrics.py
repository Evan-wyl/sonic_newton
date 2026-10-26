"""Evaluation telemetry captured through Lab's recorder lifecycle before reset."""

from isaaclab.managers import RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm
from isaaclab.utils import configclass
import torch

from gear_sonic.utils.lab_api import asset_tensor


class TrackingMetricsRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.tracking_eval_samples = []

    def record_post_step(self):
        env = self._env
        command = env.command_manager.get_term("motion")
        sensor = env.scene["contact_forces"]
        values = {
            "body_position_error_m": (command.body_pos_w - command.robot_body_pos_w).norm(dim=-1).mean(dim=-1),
            "joint_position_error_rad": (command.joint_pos - command.robot_joint_pos).abs().mean(dim=-1),
            "root_height_m": asset_tensor(env.scene["robot"], "root_pos_w")[:, 2],
            "max_contact_force_n": asset_tensor(sensor, "net_forces_w").norm(dim=-1).max(dim=-1).values,
            "episode_control_steps": env.episode_length_buf,
            "motion_id": command.motion_ids,
            "terminated": env.termination_manager.terminated,
            "time_out": env.termination_manager.time_outs,
        }
        for name, value in values.items():
            if not torch.isfinite(value).all():
                raise AssertionError(f"Non-finite pre-reset evaluation metric: {name}")
        sample = {key: value.detach().cpu().tolist() for key, value in values.items()}
        sample["termination_terms"] = {
            key: env.termination_manager.get_term(key).detach().cpu().tolist()
            for key in env.termination_manager.active_terms
        }
        env.tracking_eval_samples.append(sample)
        return None, None


@configclass
class TrackingEvalRecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    evaluation = RecorderTermCfg(class_type=TrackingMetricsRecorder)
