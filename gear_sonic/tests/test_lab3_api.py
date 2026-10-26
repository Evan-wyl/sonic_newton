"""CPU checks of the motion/checkpoint contract at the Lab 3.0 boundary.

Run with the Isaac Lab Python; no Kit, GPU, robot asset import or motion dataset
is required. These checks do not validate a physical simulation.
"""

import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import warp as wp

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.utils.warp.proxy_array import ProxyArray

from gear_sonic.utils import lab_api as api, lab_math as m


class TestLab3Boundary(unittest.TestCase):
    def test_rotation_and_frame_geometry(self):
        zero = torch.zeros(2)
        yaw = torch.tensor([math.pi / 2, -math.pi / 2])
        q = m.quat_from_euler_xyz(zero, zero, yaw)
        vector = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        expected = torch.tensor([[0., 1., 0.], [0., -1., 0.]])
        torch.testing.assert_close(m.quat_apply(q, vector), expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(m.quat_apply_inverse(q, expected), vector, atol=1e-6, rtol=0)
        torch.testing.assert_close(m.quat_apply_yaw(q, vector), expected, atol=1e-6, rtol=0)
        identity = torch.tensor([[1., 0., 0., 0.]]).expand(2, -1)
        torch.testing.assert_close(m.quat_mul(q, m.quat_inv(q)), identity, atol=1e-6, rtol=0)
        torch.testing.assert_close(m.quat_conjugate(q), m.quat_inv(q))
        torch.testing.assert_close(m.quat_error_magnitude(q, -q), zero, atol=1e-6, rtol=0)
        matrix = m.matrix_from_quat(q)
        torch.testing.assert_close((matrix @ vector.unsqueeze(-1)).squeeze(-1), expected, atol=1e-6, rtol=0)
        origin = torch.tensor([[3., 4., 5.], [-2., 3., 1.]])
        relative_position, relative_q = m.subtract_frame_transforms(origin, q, origin + expected, q)
        torch.testing.assert_close(relative_position, vector, atol=1e-6, rtol=0)
        torch.testing.assert_close(relative_q, identity, atol=1e-6, rtol=0)
        torch.testing.assert_close(m.axis_angle_from_quat(q).norm(dim=-1), yaw.abs())
        # A non-yaw rotation prevents a WXYZ/XYZW mistake from hiding in a
        # planar-only example: 120 degrees about (1,1,1) cycles x -> y -> z.
        mixed_q = torch.full((3, 4), .5)
        basis = torch.eye(3)
        torch.testing.assert_close(m.quat_apply(mixed_q, basis), basis[[1, 2, 0]])
        torch.testing.assert_close(m.xyzw_to_wxyz(m.wxyz_to_xyzw(q)), q)
        self.assertEqual(api.config_quat_xyzw((1., 0., 0., 0.)), (0., 0., 0., 1.))

    def test_real_proxy_preserves_writable_nonrotation_buffer(self):
        buffer = wp.zeros((3, 2), dtype=wp.float32, device="cpu")
        proxy = ProxyArray(buffer)
        asset = SimpleNamespace(data=SimpleNamespace(default_joint_pos=proxy))
        values = api.asset_tensor(asset, "default_joint_pos")
        values[1, 0] = 2.5
        self.assertEqual(proxy.torch[1, 0].item(), 2.5)
        self.assertEqual(proxy.torch[0, 0].item(), 0)

    def test_partial_reset_round_trip_and_no_input_mutation(self):
        # This in-memory asset honors Lab's real signatures, selected-row writes,
        # and XYZW storage, allowing a format/index contract check without physics.
        class Asset:
            def __init__(self):
                self.pose = torch.zeros(4, 7)
                self.velocity = torch.zeros(4, 6)
                self.position = torch.zeros(4, 3)
                self.joint_velocity = torch.zeros(4, 3)

            def write_root_pose_to_sim_index(self, *, root_pose, env_ids=None):
                self.pose[env_ids] = root_pose

            def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids=None):
                self.velocity[env_ids] = root_velocity

            def write_joint_position_to_sim_index(self, *, position, joint_ids=None, env_ids=None):
                self.position[env_ids[:, None], joint_ids] = position

            def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids=None, env_ids=None):
                self.joint_velocity[env_ids[:, None], joint_ids] = velocity

        asset = Asset()
        env_ids = torch.tensor([3, 1])
        q = m.quat_from_euler_xyz(torch.tensor([.3, -.5]), torch.tensor([.2, .7]), torch.tensor([.9, -.2]))
        state = torch.cat((torch.randn(2, 3), q, torch.randn(2, 6)), dim=-1)
        original = state.clone()
        api.write_root_state(asset, state, env_ids=env_ids)
        asset.data = SimpleNamespace(root_quat_w=ProxyArray(wp.from_torch(asset.pose[:, 3:7].contiguous(), dtype=wp.quatf)))
        torch.testing.assert_close(api.asset_tensor(asset, "root_quat_w")[env_ids], q)
        torch.testing.assert_close(asset.velocity[env_ids], state[:, 7:])
        torch.testing.assert_close(state, original)
        torch.testing.assert_close(asset.pose[[0, 2]], torch.zeros(2, 7))
        joint_ids = torch.tensor([2, 0])
        position, velocity = torch.randn(2, 2), torch.randn(2, 2)
        api.write_joint_state(asset, position, velocity, env_ids=env_ids, joint_ids=joint_ids)
        torch.testing.assert_close(asset.position[env_ids[:, None], joint_ids], position)
        torch.testing.assert_close(asset.joint_velocity[env_ids[:, None], joint_ids], velocity)
        torch.testing.assert_close(asset.position[:, 1], torch.zeros(4))
        # Catch keyword drift against the selected Lab API rather than a mock alone.
        for name, kwargs in [
            ("write_root_pose_to_sim_index", {"root_pose": asset.pose, "env_ids": env_ids}),
            ("write_root_velocity_to_sim_index", {"root_velocity": asset.velocity, "env_ids": env_ids}),
            ("write_joint_position_to_sim_index", {"position": position, "joint_ids": joint_ids, "env_ids": env_ids}),
            ("write_joint_velocity_to_sim_index", {"velocity": velocity, "joint_ids": joint_ids, "env_ids": env_ids}),
        ]:
            inspect.signature(getattr(BaseArticulation, name)).bind(asset, **kwargs)

    def test_robot_configs_keep_order_and_action_scales(self):
        from gear_sonic.envs.manager_env.robots import g1, h2
        for cfg in [g1.G1_CYLINDER_MODEL_12_DEX_CFG, h2.H2_CFG]:
            self.assertEqual(cfg.joint_ordering, "physx")
            self.assertEqual(cfg.body_ordering, "physx")
        self.assertEqual(g1.G1_CYLINDER_MODEL_12_DEX_CFG.actuators["feet"].joint_effort_limit, 50)
        self.assertEqual(g1.G1_MODEL_12_ACTION_SCALE[".*_ankle_pitch_joint"], .25 * 50 / g1.G1_CYLINDER_MODEL_12_DEX_CFG.actuators["feet"].stiffness)

    def test_training_config_composition(self):
        from hydra import compose, initialize_config_dir
        from gear_sonic.utils.config_utils import register_rl_resolvers
        register_rl_resolvers()
        register_rl_resolvers()
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        for exp in ["sonic_release", "sonic_v1_1", "sonic_bones_seed", "sonic_h2"]:
            with self.subTest(exp=exp), initialize_config_dir(config_dir=config_dir):
                cfg = compose(config_name="base", overrides=[f"+exp=manager/universal_token/all_modes/{exp}"])
                self.assertEqual(cfg.base_dir, "logs_rl_newton")
                self.assertEqual(cfg.manager_env._target_, "gear_sonic.envs.manager_env.modular_tracking_env_cfg.ModularTrackingEnvCfg")
                self.assertTrue(cfg.algo.config.actor._target_.startswith("gear_sonic."))
                self.assertIn("motion", cfg.manager_env.commands)
                self.assertIn("gravity_dir", cfg.manager_env.observations.policy)
                self.assertIn("base_com", cfg.manager_env.events)
                self.assertIn("_target_", cfg.manager_env.actions)
                self.assertIn("_target_", cfg.manager_env.commands)
                self.assertIn("_target_", cfg.manager_env.rewards)
                self.assertIn("_target_", cfg.manager_env.terminations)
                self.assertIn("g1_recon", cfg.algo.config.actor.backbone.aux_loss_func)


if __name__ == "__main__":
    unittest.main()
