"""Validate backend selection and asset isolation without starting a simulator."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from gear_sonic_newton.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG
from gear_sonic_newton.envs.newton_config import (
    configure_newton_robot, make_newton_simulation, newton_launcher_args,
)


class NewtonConfigurationTests(unittest.TestCase):
    def test_solver_selection(self):
        cfg = make_newton_simulation(device="cpu")
        self.assertIsInstance(cfg.physics, NewtonCfg)
        self.assertIsInstance(cfg.physics.solver_cfg, MJWarpSolverCfg)
        self.assertEqual(cfg.physics.num_substeps, 2)

    def test_asset_conversion_does_not_mutate_source(self):
        original = G1_CYLINDER_MODEL_12_DEX_CFG
        original_path = original.spawn.asset_path
        original_props = original.spawn.articulation_props
        cfg = configure_newton_robot(original)
        self.assertIsInstance(cfg.spawn.articulation_props, NewtonArticulationCfg)
        self.assertTrue(Path(cfg.spawn.asset_path).is_absolute())
        self.assertTrue(cfg.spawn.run_multi_physics_conversion)
        self.assertEqual(cfg.joint_ordering, original.joint_ordering)
        self.assertEqual(original.spawn.asset_path, original_path)
        self.assertIs(original.spawn.articulation_props, original_props)
        self.assertIsNot(cfg.spawn, original.spawn)

    def test_headless_launcher_and_unsupported_camera(self):
        config = SimpleNamespace(
            manager_env=SimpleNamespace(config={}), headless=True, num_envs=2, seed=0,
        )
        args = newton_launcher_args(config, "cuda:0")
        self.assertFalse(args.require_kit)
        self.assertFalse(args.enable_cameras)
        self.assertEqual(args.visualizer, [])
        from isaaclab.app.sim_launcher import scan
        scan(make_newton_simulation(device="cpu"), args)
        config.manager_env.config["enable_cameras"] = True
        with self.assertRaises(NotImplementedError):
            newton_launcher_args(config, "cuda:0")


if __name__ == "__main__":
    unittest.main()
