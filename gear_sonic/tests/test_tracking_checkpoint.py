"""Check serialized class isolation and strict actor loading without a GPU."""

from io import BytesIO
import sys
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import torch

from gear_sonic.utils.average_meters import TensorAverageMeter
from gear_sonic.utils.checkpoint_io import load_checkpoint, load_policy_weights
from gear_sonic.utils.config_utils import resolve_motion_paths


class CheckpointTests(unittest.TestCase):
    def test_legacy_project_pickle_uses_new_package(self):
        source = BytesIO()
        torch.save({"metadata": TensorAverageMeter(), "tensor": torch.ones(2)}, source)
        source.seek(0)
        legacy = BytesIO()
        with zipfile.ZipFile(source) as original, zipfile.ZipFile(legacy, "w") as rewritten:
            for entry in original.infolist():
                data = original.read(entry.filename)
                if entry.filename.endswith("data.pkl"):
                    data = data.replace(b"gear_sonic.utils.average_meters", b"gear_sonic_newton.utils.average_meters")
                rewritten.writestr(entry, data)
        legacy.seek(0)
        restored = load_checkpoint(legacy)
        self.assertIs(type(restored["metadata"]), TensorAverageMeter)
        torch.testing.assert_close(restored["tensor"], torch.ones(2))
        self.assertNotIn("gear_sonic_newton.utils.average_meters", sys.modules)

    def test_std_conversion_preserves_input(self):
        policy = torch.nn.Module()
        policy.register_parameter("std", torch.nn.Parameter(torch.ones(2)))
        old = {"actor_model_state_dict": {"log_std": torch.tensor([0., -1.])}}
        load_policy_weights(policy, old)
        torch.testing.assert_close(policy.std, torch.tensor([0., -1.]).exp())
        self.assertIn("log_std", old["actor_model_state_dict"])

    def test_missing_or_incompatible_weights_fail(self):
        policy = torch.nn.Linear(3, 2)
        with self.assertRaises(ValueError):
            load_policy_weights(policy, {"state": {}})
        with self.assertRaises(RuntimeError):
            load_policy_weights(policy, {"policy_state_dict": {"weight": torch.zeros(2, 3)}})
        with self.assertRaises(RuntimeError):
            load_policy_weights(policy, {"policy_state_dict": {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)}})

    def test_data_paths_are_resolved_without_mutating_saved_config(self):
        original = OmegaConf.create({"commands": {"motion": {"motion_lib_cfg": {
            "motion_file": "data/motion_lib_bones_seed/robot_filtered",
            "smpl_motion_file": "zeros",
        }}}})
        resolved = resolve_motion_paths(original)
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(resolved.commands.motion.motion_lib_cfg.motion_file,
                         str((root / "data/motion_lib_bones_seed/robot_filtered").resolve()))
        self.assertEqual(resolved.commands.motion.motion_lib_cfg.smpl_motion_file, "zeros")
        self.assertEqual(original.commands.motion.motion_lib_cfg.motion_file,
                         "data/motion_lib_bones_seed/robot_filtered")

    def test_metrics_capture_terminal_state_before_reset(self):
        from gear_sonic.utils.tracking_metrics import TrackingMetricsRecorder
        command = SimpleNamespace(
            body_pos_w=torch.ones(2, 1, 3), robot_body_pos_w=torch.zeros(2, 1, 3),
            joint_pos=torch.ones(2, 1), robot_joint_pos=torch.zeros(2, 1),
            motion_ids=torch.tensor([0, 1]),
        )
        scene = {
            "robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.ones(2, 3))),
            "contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.ones(2, 1, 3))),
        }
        termination = SimpleNamespace(
            terminated=torch.tensor([True, False]), time_outs=torch.tensor([False, True]),
            active_terms=["height"], get_term=lambda name: torch.tensor([True, False]),
        )
        env = SimpleNamespace(
            scene=scene, command_manager=SimpleNamespace(get_term=lambda name: command),
            termination_manager=termination, episode_length_buf=torch.tensor([4, 8]),
            tracking_eval_samples=[],
        )
        recorder = object.__new__(TrackingMetricsRecorder)
        recorder._env = env
        recorder.record_post_step()
        command.body_pos_w.zero_()
        termination.terminated.zero_()
        self.assertEqual(env.tracking_eval_samples[0]["terminated"], [True, False])
        self.assertAlmostEqual(env.tracking_eval_samples[0]["body_position_error_m"][0], 3 ** .5, places=6)


if __name__ == "__main__":
    unittest.main()
