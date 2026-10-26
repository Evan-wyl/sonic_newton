"""Run the project's G1 on a plane with Newton/MJWarp, without Kit or a policy.

From the repository root:
    .env_isaaclab/bin/python -m gear_sonic.scripts.smoke_newton_g1
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import GroundPlaneCfg, SimulationContext
from isaaclab.utils import configclass, instantiate

from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG, G1_ISAACLAB_JOINTS
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
from gear_sonic.envs.newton_config import configure_newton_robot, make_newton_simulation
from gear_sonic.utils.lab_api import asset_tensor, write_joint_state, write_root_state
from gear_sonic.utils.lab_math import xyzw_to_wxyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--fixed-base", action="store_true", help="Isolate PD response without a balance policy.")
    parser.add_argument("--output-dir", default="logs_rl_newton/mjwarp_smoke")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 200:
        parser.error("Use at least two environments and 200 steps to test partial reset and contact.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sim_cfg = make_newton_simulation(device=args.device, log_dir=str(output))

    @configclass
    class G1SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=GroundPlaneCfg())
        robot = configure_newton_robot(G1_CYLINDER_MODEL_12_DEX_CFG, cache_dir=output / "usd_cache")
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3,
            track_air_time=True, force_threshold=10.0,
        )

    scene_cfg = G1SceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    scene_cfg.robot.spawn.fix_base = args.fixed_base
    result = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "newton", "solver": "mjwarp", "device": args.device,
        "num_envs": args.num_envs, "requested_steps": args.steps,
        "dt": sim_cfg.dt, "solver_substeps": sim_cfg.physics.num_substeps,
        "policy_loaded": False, "kit_required": False, "passed": False,
        "fixed_base": args.fixed_base,
    }
    try:
        with launch_simulation(sim_cfg, args):
            sim = SimulationContext(sim_cfg)
            try:
                scene = instantiate(scene_cfg)
                sim.reset()
                robot = scene["robot"]
                sensor = scene["contact_forces"]
                if not type(robot).__module__.startswith("isaaclab_newton."):
                    raise AssertionError(f"Unexpected articulation implementation: {type(robot)}")
                result["articulation_type"] = f"{type(robot).__module__}.{type(robot).__name__}"
                result["joint_names"] = robot.joint_names
                result["body_names"] = robot.body_names
                assert robot.joint_names == G1_ISAACLab_ORDER, "Public joint order differs from SONIC's mapping"
                assert robot.body_names == G1_ISAACLAB_JOINTS, "Public body order differs from SONIC's mapping"
                result["public_order_matches_sonic"] = True
                default_pose = robot.data.default_root_pose.torch.clone()
                default_pose[:, :3] += scene.env_origins
                default_state = torch.cat((default_pose[:, :3], xyzw_to_wxyz(default_pose[:, 3:7]), robot.data.default_root_vel.torch), dim=-1)
                joint_pos = asset_tensor(robot, "default_joint_pos").clone()
                joint_vel = asset_tensor(robot, "default_joint_vel").clone()

                def reset(env_ids):
                    write_joint_state(robot, joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
                    write_root_state(robot, default_state[env_ids], env_ids=env_ids)
                    scene.reset(env_ids)

                reset(torch.arange(args.num_envs, device=sim.device))
                target = joint_pos.clone()
                elbow = robot.joint_names.index("left_elbow_joint")
                min_height, max_force = float("inf"), 0.0
                partial_reset_checked = False
                equilibrium_elbow = None
                for step in range(args.steps):
                    if step == args.steps // 2:
                        # Compare against the settled position, not the nominal
                        # pose: finite PD gains under gravity have static error.
                        equilibrium_elbow = asset_tensor(robot, "joint_pos")[:, elbow].clone()
                        target[:, elbow] += 0.05
                        selected = torch.tensor([args.num_envs - 1], device=sim.device)
                        untouched = asset_tensor(robot, "joint_pos")[0].clone()
                        reset(selected)
                        probe_state = default_state[selected].clone()
                        probe_state[:, 3:7] = torch.tensor(
                            [math.cos(.125), 0., 0., math.sin(.125)], device=sim.device
                        )
                        write_root_state(robot, probe_state, env_ids=selected)
                        torch.testing.assert_close(asset_tensor(robot, "joint_pos")[selected], joint_pos[selected], atol=1e-5, rtol=0)
                        torch.testing.assert_close(asset_tensor(robot, "joint_pos")[0], untouched, atol=1e-5, rtol=0)
                        torch.testing.assert_close(asset_tensor(robot, "root_quat_w")[selected], probe_state[:, 3:7], atol=1e-5, rtol=0)
                        partial_reset_checked = True
                    robot.actuators.target_command.set_position_index(value=target)
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(sim_cfg.dt)
                    positions = asset_tensor(robot, "joint_pos")
                    root = asset_tensor(robot, "root_pos_w")
                    force = sensor.data.net_normal_forces_w.torch
                    for values in (positions, asset_tensor(robot, "joint_vel"), root, force):
                        if not torch.isfinite(values).all():
                            raise AssertionError(f"Non-finite state/contact at step {step}")
                    min_height = min(min_height, root[:, 2].min().item())
                    max_force = max(max_force, force.norm(dim=-1).max().item())
                elbow_response = (asset_tensor(robot, "joint_pos")[:, elbow] - equilibrium_elbow).mean().item()
                result.update({
                    "completed_steps": args.steps, "partial_reset_checked": partial_reset_checked,
                    "min_root_height_m": min_height, "max_normal_force_n": max_force,
                    "elbow_target_delta_rad": 0.05, "elbow_response_rad": elbow_response,
                    "equilibrium_elbow_rad": equilibrium_elbow.tolist(),
                    "final_elbow_rad": asset_tensor(robot, "joint_pos")[:, elbow].tolist(),
                    "final_root_height_m": asset_tensor(robot, "root_pos_w")[:, 2].tolist(),
                    "final_max_joint_target_error_rad": (asset_tensor(robot, "joint_pos") - target).abs().max().item(),
                })
                if not args.fixed_base and max_force <= 1.0:
                    raise AssertionError("No meaningful ground contact measured.")
                result["standing_height_maintained"] = min_height >= 0.35
                # A free humanoid with no balance policy can fall physically.
                # Assess PD independently with the fixed-base experiment.
                if args.fixed_base and not 0.01 < elbow_response < 0.09:
                    raise AssertionError("Elbow did not follow the small PD target change.")
                result["passed"] = True
            finally:
                SimulationContext.clear_instance()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
