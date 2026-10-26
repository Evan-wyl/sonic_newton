"""CPU regression checks for full training-state restore and finite checks."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from gear_sonic_newton.trl.trainer.ppo_trainer import TRLPPOTrainer, episode_totals
from gear_sonic_newton.utils.training_validation import assert_finite, parameter_hash, json_values


class TrainingValidationTests(unittest.TestCase):
    def test_motion_resample_uses_reset_observations_and_clears_partial_episodes(self):
        from gear_sonic_newton.trl.callbacks.im_resample_callback import ImResampleCallback
        trainer = object.__new__(TRLPPOTrainer)
        reset_obs = {"actor_obs": torch.ones(2, 3)}
        trainer.env = SimpleNamespace(resample_motion=lambda: reset_obs)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.cur_reward_sum = torch.full((2,), 10.)
        trainer.cur_episode_length = torch.full((2,), 25.)
        old = {"actor_obs": torch.zeros(2, 3)}
        self.assertIs(trainer._consume_motion_resample_reset(old), old)
        callback = ImResampleCallback(motion_resample_frequency=2)
        callback.on_step_end(None, SimpleNamespace(global_step=1), None,
            env=trainer.env, accelerator=trainer.accelerator)
        actual = trainer._consume_motion_resample_reset(old)
        torch.testing.assert_close(actual["actor_obs"], reset_obs["actor_obs"])
        self.assertEqual(trainer.cur_reward_sum.count_nonzero().item(), 0)
        self.assertEqual(trainer.cur_episode_length.count_nonzero().item(), 0)
        self.assertIs(trainer._consume_motion_resample_reset(actual), actual)

    def test_episode_totals_empty_nested_and_mixed_history(self):
        self.assertEqual(episode_totals([], []), (0., 0., 0))
        self.assertEqual(episode_totals([[1., 2.], [3., 4.]], [12., 7.]), (10., 19., 2))
        self.assertEqual(episode_totals([[[1., 2.]], [[3., 4.]]], [[12.], [7.]]), (10., 19., 2))
        self.assertEqual(episode_totals([[[1., 2.]], [3., 4.]], [[12.], 7.]), (10., 19., 2))

    def test_nested_nonfinite_is_rejected(self):
        assert_finite({"advantages": torch.ones(2, 4), "values": [0.0]}, "rollout")
        for value in [float("inf"), {"loss": torch.tensor(float("nan"))}]:
            with self.assertRaises(FloatingPointError):
                assert_finite(value, "test")

    def test_parameter_update_and_detached_logs(self):
        module = torch.nn.Linear(2, 1)
        before = parameter_hash(module)
        with torch.no_grad():
            module.weight.add_(0.1)
        self.assertNotEqual(before, parameter_hash(module))
        buffer = torch.tensor([1., 2.])
        saved = json_values({"metric": buffer})
        buffer.zero_()
        self.assertEqual(saved["metric"], [1., 2.])

    def test_full_resume_preserves_adam_scheduler_and_group_lrs(self):
        def make_trainer():
            trainer = object.__new__(TRLPPOTrainer)
            trainer.model = SimpleNamespace(policy=torch.nn.Linear(2, 1), value_model=torch.nn.Linear(2, 1))
            trainer.optimizer = torch.optim.Adam([
                {"params": trainer.model.policy.parameters(), "lr": .001},
                {"params": trainer.model.value_model.parameters(), "lr": .002},
            ])
            trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.)
            trainer.accelerator = SimpleNamespace(device="cpu", unwrap_model=lambda model: model)
            trainer.args = SimpleNamespace(learning_rate=.001)
            trainer.state = SimpleNamespace(global_step=0)
            trainer.env = SimpleNamespace(load_env_state_dict=lambda state: setattr(trainer, "restored_env", state))
            return trainer
        original = make_trainer()
        loss = sum(parameter.square().sum() for module in [original.model.policy, original.model.value_model]
                   for parameter in module.parameters())
        loss.backward()
        original.optimizer.step()
        original.lr_scheduler.step()
        saved = {
            "policy_state_dict": original.model.policy.state_dict(),
            "value_state_dict": original.model.value_model.state_dict(),
            "optimizer_state_dict": original.optimizer.state_dict(),
            "lr_scheduler_state_dict": original.lr_scheduler.state_dict(),
            "state": SimpleNamespace(global_step=7, tot_timesteps=448),
            "args": original.args, "env_state_dict": {"marker": 123},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(saved, path)
            restored = make_trainer()
            restored.load_checkpoint(path, resume=True)
        self.assertEqual(restored.state.global_step, 7)
        self.assertEqual(restored.state.tot_timesteps, 448)
        self.assertEqual([g["lr"] for g in restored.optimizer.param_groups], [.001, .002])
        self.assertEqual(restored.lr_scheduler.state_dict(), original.lr_scheduler.state_dict())
        self.assertEqual(restored.restored_env, {"marker": 123})
        for key, value in original.optimizer.state_dict()["state"].items():
            for field, tensor in value.items():
                torch.testing.assert_close(restored.optimizer.state_dict()["state"][key][field], tensor)
        self.assertEqual(parameter_hash(restored.model.policy), parameter_hash(original.model.policy))


if __name__ == "__main__":
    unittest.main()
