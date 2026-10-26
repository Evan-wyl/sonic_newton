"""Exercise the complete SONIC tracking task without updating model weights.

--synthetic-reference is an explicit plumbing test, never a policy benchmark.
Use real motion/SMPL inputs and --checkpoint for checkpoint assessment.
"""

import argparse
from datetime import datetime, timezone
import json
import io
from pathlib import Path

from easydict import EasyDict
from hydra import compose, initialize_config_dir
import joblib
import numpy as np
from omegaconf import OmegaConf
import torch

from gear_sonic.envs.newton_config import newton_launcher_args
from gear_sonic.train_agent_trl import create_manager_env
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.lab_api import asset_tensor
from gear_sonic.trl.utils.common import custom_instantiate
from gear_sonic.utils.obs_utils import get_group_term_obs_shape
from gear_sonic.utils.checkpoint_io import load_checkpoint, load_policy_weights


def build_policy(config, env, checkpoint=None):
    """Use the evaluation entry's dimension contract, with strict weight loading."""
    example = env.reset(flatten_dict_obs=False)
    for group, key in [("policy", "actor_obs"), ("critic", "critic_obs")]:
        size = env.env.observation_space[group].shape[-1]
        env.config.obs.obs_dims[key] = size
        env.config.robot.algo_obs_dim_dict[key] = size
    for group in env.env.observation_space:
        if group not in ("policy", "critic"):
            dims, names, total = get_group_term_obs_shape(example, group)
            env.config.obs.group_obs_dims[group] = dims
            env.config.obs.group_obs_names[group] = names
            env.config.obs.obs_dims[group] = total
            env.config.robot.algo_obs_dim_dict[group] = total
    env.config.robot.actions_dim = env.env.action_manager.total_action_dim
    policy = custom_instantiate(
        config.algo.config.actor, env_config=env.config, algo_config=config.algo.config,
        module_dim_dict=config.algo.config.get("module_dim", {}),
        backbone_kwargs={}, _resolve=False,
    ).to(env.device)
    policy.eval()
    if checkpoint:
        # Register the existing TRL compatibility aliases before unpickling
        # training-state metadata; evaluation itself does not create a trainer.
        from gear_sonic.trl.trainer import ppo_trainer  # noqa: F401
        saved = load_checkpoint(checkpoint, map_location=env.device)
        load_policy_weights(policy, saved)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def synthetic_reference(config, output):
    """Create a static robot reference; SMPL zeros are explicitly diagnostic."""
    from gear_sonic.envs.manager_env.robots.g1 import (
        G1_CYLINDER_MODEL_12_DEX_CFG, G1_ISAACLAB_TO_MUJOCO_DOF, G1_MUJOCO_TO_ISAACLAB_DOF,
    )
    from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    batch = Humanoid_Batch(EasyDict(OmegaConf.to_container(config.manager_env.commands.motion.motion_lib_cfg)))
    defaults = torch.zeros(29)
    import re
    for pattern, value in G1_CYLINDER_MODEL_12_DEX_CFG.init_state.joint_pos.items():
        for index, name in enumerate(G1_ISAACLab_ORDER):
            if re.fullmatch(pattern, name):
                defaults[index] = value
    mujoco_positions = defaults[G1_ISAACLAB_TO_MUJOCO_DOF]
    torch.testing.assert_close(mujoco_positions[G1_MUJOCO_TO_ISAACLAB_DOF], defaults)
    frames = 501
    pose = np.zeros((frames, 30, 3), dtype=np.float32)
    pose[:, 1:] = (mujoco_positions[:, None] * batch.dof_axis).numpy()
    trans = np.zeros((frames, 3), dtype=np.float32)
    trans[:, 2] = 0.76
    path = output / "synthetic_reference.pkl"
    joblib.dump({"diagnostic_static_pose": {
        "root_trans_offset": trans, "pose_aa": pose, "fps": 50,
    }}, path)
    return str(path)


def assert_finite(values, label):
    if isinstance(values, dict):
        for key, value in values.items():
            assert_finite(value, f"{label}.{key}")
    elif isinstance(values, torch.Tensor) and not torch.isfinite(values).all():
        raise AssertionError(f"Non-finite {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file")
    parser.add_argument("--smpl-motion-file")
    parser.add_argument("--motion-key", action="append", help="Limit evaluation to named motion clips; repeatable.")
    parser.add_argument("--synthetic-reference", action="store_true")
    policy_options = parser.add_mutually_exclusive_group()
    policy_options.add_argument("--checkpoint", help="Old policy weights; requires its saved config.")
    policy_options.add_argument("--untrained-policy", action="store_true", help="Diagnostic forward pass, no learned weights.")
    parser.add_argument("--checkpoint-config", help="Defaults to config.yaml beside the checkpoint.")
    parser.add_argument("--encoder", choices=["g1", "teleop", "smpl"], default="g1")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="logs_rl_newton/tracking_validation")
    args = parser.parse_args()
    if bool(args.motion_file) == args.synthetic_reference:
        parser.error("Choose --motion-file or --synthetic-reference explicitly.")
    if args.num_envs < 2 or args.steps < 2:
        parser.error("Use at least two environments and steps.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    register_rl_resolvers()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        config = compose(config_name="base", overrides=[
            "+exp=manager/universal_token/all_modes/sonic_release",
            f"num_envs={args.num_envs}", "headless=True",
            "manager_env.config.terrain_type=plane", f"experiment_dir={output}",
        ])
    if args.checkpoint:
        # Hugging Face snapshots use file symlinks; find config next to the
        # user-visible checkpoint, not next to its blob-store target.
        checkpoint_path = Path(args.checkpoint).absolute()
        config_path = Path(args.checkpoint_config) if args.checkpoint_config else checkpoint_path.parent / "config.yaml"
        if not checkpoint_path.is_file() or not config_path.is_file():
            parser.error("Checkpoint and its saved configuration must both exist.")
        raw = config_path.read_text()
        for old, new in [
            ("groot.rl.trl.", "gear_sonic.trl."),
            ("groot.rl.envs.", "gear_sonic.envs."),
            ("groot.rl.utils.", "gear_sonic.utils."),
            ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
            ("groot.rl.agents.", "gear_sonic.trl."),
            ("groot/rl/data/", "gear_sonic/data/"),
            ("gear_sonic.", "gear_sonic."), ("gear_sonic/", "gear_sonic/"),
            ("assets/bm/unitree_description/", "assets/robot_description/"),
            ("isaaclab.utils.noise.AdditiveUniformNoiseCfg", "isaaclab.utils.noise.UniformNoiseCfg"),
        ]:
            raw = raw.replace(old, new)
        config = OmegaConf.load(io.StringIO(raw))
        config.num_envs = args.num_envs
        config.headless = True
        config.checkpoint = str(checkpoint_path)
        config.experiment_dir = str(output)
        config.manager_env.config.terrain_type = "plane"
        for key in ("enable_cameras", "render_results", "render_ego", "overview_camera"):
            config.manager_env.config[key] = False
    config.manager_env.config.num_envs = args.num_envs
    config.sim_type = "newton_mjwarp"
    config.multi_gpu = False
    config.use_wandb = False
    if args.encoder == "smpl" and not args.synthetic_reference and not args.smpl_motion_file:
        parser.error("The SMPL encoder requires --smpl-motion-file.")
    config.manager_env.commands.motion.motion_lib_cfg.motion_file = (
        synthetic_reference(config, output) if args.synthetic_reference else str(Path(args.motion_file).resolve())
    )
    config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = (
        "zeros" if args.synthetic_reference else str(Path(args.smpl_motion_file).resolve()) if args.smpl_motion_file else None
    )
    config.manager_env.commands.motion.motion_lib_cfg.multi_thread = False
    if args.motion_key:
        config.manager_env.commands.motion.filter_motion_keys = args.motion_key
        config.manager_env.commands.motion.motion_lib_cfg.filter_motion_keys = args.motion_key
    config.manager_env.commands.motion.debug_vis = False
    config.manager_env.commands.motion.cat_upper_body_poses = False
    config.manager_env.commands.motion.freeze_frame_aug = False
    config.manager_env.commands.motion.encoder_sample_probs = {
        key: float(key == args.encoder) for key in ("g1", "teleop", "smpl")
    }
    # Stage 6 will validate randomization. Record the exclusions in the report.
    disabled_events = [key for key in config.manager_env.events if not key.startswith("_")]
    for key in disabled_events:
        config.manager_env.events[key] = None
    config.manager_env.recorders = {
        "_target_": "gear_sonic.utils.tracking_metrics.TrackingEvalRecordersCfg",
    }
    OmegaConf.save(config, output / "config.yaml")
    result = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
        "synthetic_reference": args.synthetic_reference, "checkpoint_loaded": False,
        "action_source": "zero_normalized_joint_actions", "training_updates": 0,
        "disabled_events": disabled_events, "num_envs": args.num_envs,
        "requested_control_steps": args.steps,
        "encoder": args.encoder,
        "motion_keys_requested": args.motion_key,
        "metric_sampling": "post_physics_before_autoreset",
        "checkpoint": str(Path(args.checkpoint).absolute()) if args.checkpoint else None,
        "motion_file": config.manager_env.commands.motion.motion_lib_cfg.motion_file,
        "smpl_motion_file": config.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file,
    }
    env = None
    try:
        env = create_manager_env(config, args.device, newton_launcher_args(config, args.device))
        obs = env.reset()
        assert_finite(obs, "initial_observation")
        result["observation_shapes"] = {key: list(value.shape) for key, value in obs.items()}
        result["observation_terms"] = env.env.observation_manager.active_terms
        command = env.env.command_manager.get_term("motion")
        if args.encoder == "smpl" and not args.synthetic_reference:
            if not command.motion_lib.motion_has_smpl.all():
                raise ValueError("Some selected robot motions have no matching SMPL data.")
        result["dt_control_s"] = env.env.step_dt
        result["sim_dt_s"] = env.env.cfg.sim.dt
        result["solver_substeps"] = env.env.cfg.sim.physics.num_substeps
        result["solver_cfg_type"] = type(env.env.cfg.sim.physics.solver_cfg).__name__
        result["articulation_type"] = f"{type(env.env.scene['robot']).__module__}.{type(env.env.scene['robot']).__name__}"
        robot = env.env.scene["robot"]
        result["joint_names"] = robot.joint_names
        from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
        assert robot.joint_names == G1_ISAACLab_ORDER
        action_term = env.env.action_manager.get_term("joint_pos")
        probe = torch.zeros((args.num_envs, robot.num_joints), device=args.device)
        probe[0, 0] = 0.25
        probe[-1, -1] = -0.25
        scale_cfg = env.env.cfg.actions.joint_pos.scale
        import re
        scales = torch.zeros(robot.num_joints, device=args.device)
        for pattern, value in scale_cfg.items():
            for index, name in enumerate(robot.joint_names):
                if re.fullmatch(pattern, name):
                    scales[index] = value
        assert (scales > 0).all(), "Missing positive action scale"
        action_term.process_actions(probe)
        expected = asset_tensor(robot, "default_joint_pos") + probe * scales
        torch.testing.assert_close(action_term.processed_actions, expected)
        result["action_scale_contract_checked"] = True
        result["action_scale_by_joint"] = scales.tolist()
        result["default_joint_position_rad"] = asset_tensor(robot, "default_joint_pos")[0].tolist()
        policy = build_policy(config, env, args.checkpoint) if args.checkpoint or args.untrained_policy else None
        result["checkpoint_loaded"] = bool(args.checkpoint)
        result["action_source"] = "checkpoint" if args.checkpoint else "untrained_policy" if policy is not None else result["action_source"]
        if policy is not None:
            result["policy_parameters"] = sum(parameter.numel() for parameter in policy.parameters())
        command.is_evaluating = True
        obs = env.reset()
        reset_count = 0
        rewards = []
        errors = []
        dones = torch.zeros(args.num_envs, device=args.device, dtype=torch.long)
        for step in range(args.steps):
            with torch.inference_mode():
                actions = policy.act_inference(obs, cur_dones=dones) if policy is not None else torch.zeros(
                    (args.num_envs, env.env.action_manager.total_action_dim), device=args.device
                )
            assert_finite(actions, "actions")
            obs, reward, dones, extras = env.step({"actions": actions})
            assert_finite(obs, "observation")
            assert_finite(reward, "reward")
            assert_finite(asset_tensor(robot, "joint_pos"), "joint_pos")
            rewards.append(reward.mean().item())
            reset_count += dones.count_nonzero().item()
            errors.extend(env.env.tracking_eval_samples[-1]["body_position_error_m"])
        samples = env.env.tracking_eval_samples
        failures = sum(sum(sample["terminated"]) for sample in samples)
        timeouts = sum(sum(sample["time_out"]) for sample in samples)
        ended = sum(sum(a or b for a, b in zip(sample["terminated"], sample["time_out"])) for sample in samples)
        term_counts = {
            name: sum(sum(sample["termination_terms"][name]) for sample in samples)
            for name in samples[0]["termination_terms"]
        }
        result.update({
            "passed": True, "completed_control_steps": args.steps,
            "auto_reset_count": reset_count, "mean_reward": float(np.mean(rewards)),
            "mean_body_position_error_m": float(np.mean(errors)),
            "p95_body_position_error_m": float(np.percentile(errors, 95)),
            "mean_joint_position_error_rad": float(np.mean([s["joint_position_error_rad"] for s in samples])),
            "task_failure_count": failures, "time_out_count": timeouts,
            "ended_episode_count": ended,
            "failure_fraction_of_ended_episodes": failures / ended if ended else None,
            "unfinished_episode_count": args.num_envs - sum(a or b for a, b in zip(samples[-1]["terminated"], samples[-1]["time_out"])),
            "termination_term_counts": term_counts,
            "loaded_motion_keys": [str(key) for key in command.motion_lib.curr_motion_keys],
            "max_contact_force_n": max(max(s["max_contact_force_n"]) for s in samples),
        })
        (output / "metrics.json").write_text(json.dumps(samples) + "\n")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None:
            env.close()
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
