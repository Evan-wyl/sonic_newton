"""Stage 6 isolated tests of Newton properties, Jacobians and object dynamics."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import warp as wp
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass, instantiate

from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG
from gear_sonic.envs.newton_config import configure_newton_robot, make_newton_simulation
from gear_sonic.envs.physics_adapter import get_linear_jacobian, set_world_forces
from gear_sonic.utils.lab_api import asset_tensor, write_joint_state
from gear_sonic.utils.lab_math import quat_apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-base", action="store_true")
    parser.add_argument("--output-dir", default="logs_rl_newton/step6_features")
    add_launcher_args(parser)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve();output.mkdir(parents=True, exist_ok=True)
    cfg = make_newton_simulation(device=args.device, log_dir=str(output))

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        robot = configure_newton_robot(G1_CYLINDER_MODEL_12_DEX_CFG, cache_dir=output/"usd_cache")
        table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(size=(1., 1., .5),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=20.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2., 0., .25)),
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(size=(.1, .1, .1),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=False),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(2., 0., .8)),
        )
        cube_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Cube", history_length=3,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"])

    scene_cfg = SceneCfg(num_envs=2, env_spacing=5.)
    scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    scene_cfg.robot.spawn.fix_base = args.fixed_base
    result = {"passed": False, "fixed_base": args.fixed_base}
    try:
        with launch_simulation(cfg, args):
            sim = sim_utils.SimulationContext(cfg)
            try:
                scene = instantiate(scene_cfg);sim.reset()
                robot = scene["robot"];cube = scene["cube"]
                env = SimpleNamespace(scene=scene, sim=sim, device=sim.device, num_envs=2)
                solver = sim.physics_manager._solver
                solver_mass_before = wp.to_torch(solver.mjw_model.body_mass).clone()
                solver_inertia_before = wp.to_torch(solver.mjw_model.body_inertia).clone()
                solver_mu_before = wp.to_torch(solver.mjw_model.geom_friction).clone()
                q = asset_tensor(robot, "default_joint_pos").clone();v = torch.zeros_like(q)
                write_joint_state(robot, q, v)
                from gear_sonic.envs.manager_env.mdp.commands import ForceTrackingCommand, ForceTrackingCommandCfg
                force_cfg = ForceTrackingCommandCfg()
                joints = list(force_cfg.waist_joints) + list(force_cfg.left_eef_deps) + list(force_cfg.right_eef_deps)
                bodies = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
                eps = .001
                errors = []
                origin_fd = []
                for body in bodies:
                    body_id = robot.body_names.index(body)
                    for offset in [(0., 0., 0.), (.18, -.025, .01)]:
                        write_joint_state(robot, q, v)
                        jac = get_linear_jacobian(robot, body, joints, point_offset=offset)
                        columns = []
                        for column, name in enumerate(joints):
                            probes = []
                            for sign in [1, -1]:
                                probe = q.clone()
                                probe[:, robot.joint_names.index(name)] += sign * eps
                                write_joint_state(robot, probe, v)
                                probes.append(asset_tensor(robot, "body_pos_w")[:, body_id].clone() + quat_apply(
                                    asset_tensor(robot, "body_quat_w")[:, body_id],
                                    torch.tensor(offset, device=sim.device).expand(2, 3)))
                            finite_diff = (probes[0] - probes[1]) / (2 * eps)
                            columns.append(finite_diff)
                            errors.append((finite_diff - jac[:, :, column]).abs().max().item())
                        if offset == (0., 0., 0.):
                            origin_fd.append(torch.stack(columns, dim=-1))
                write_joint_state(robot, q, v)
                assert max(errors) < .002, max(errors)
                result["jacobian_max_absolute_error_m_per_rad"] = max(errors)
                result["jacobian_joint_order"] = joints
                result["jacobian_bodies"] = bodies
                # Validate the project's actual 6x17 assembly against both
                # end-effector finite differences, without inventing a controller.
                command = object.__new__(ForceTrackingCommand)
                command._env = env
                command.robot = robot
                command.vr_3point_body_indices = [robot.body_names.index(body) for body in bodies]
                for field, names in [("waist_deps_ids", force_cfg.waist_joints),
                                     ("left_eef_deps_ids", force_cfg.left_eef_deps),
                                     ("right_eef_deps_ids", force_cfg.right_eef_deps)]:
                    setattr(command, field, [robot.joint_names.index(name) for name in names])
                matrix_error = (command.matrix_M - torch.cat(origin_fd, dim=1)).abs().max().item()
                assert matrix_error < .002, matrix_error
                result["compliance_matrix_fd_max_error"] = matrix_error
                from isaaclab.envs.mdp import randomize_rigid_body_mass, randomize_rigid_body_material
                selected = torch.tensor([0],device=sim.device)
                from gear_sonic.envs.manager_env.mdp.events import randomize_joint_default_pos
                action = SimpleNamespace(_joint_names=robot.joint_names, _offset=q.clone())
                env.action_manager = SimpleNamespace(get_term=lambda name: action)
                joint_cfg = SceneEntityCfg("robot", joint_names=["left_elbow_joint"]);joint_cfg.resolve(scene)
                for _ in range(2):
                    randomize_joint_default_pos(env,selected,joint_cfg,(.01,.01),"add")
                defaults = asset_tensor(robot,"default_joint_pos")
                torch.testing.assert_close(defaults[0,joint_cfg.joint_ids],q[0,joint_cfg.joint_ids]+.01)
                torch.testing.assert_close(defaults[1],q[1])
                torch.testing.assert_close(action._offset,defaults)
                result["joint_default_subset_offset_and_nonaccumulation_checked"] = True
                body_cfg=SceneEntityCfg("robot",body_names=["torso_link"]);body_cfg.resolve(scene)
                # Compare the changed GPU mass matrix with an independent CPU
                # MuJoCo recomputation of exactly the same per-world model.
                import copy
                import numpy as np
                import mujoco
                from gear_sonic.envs.manager_env.mdp.events import randomize_rigid_body_com
                com_before = asset_tensor(robot,"body_com_pos_b").clone()
                mass_com_before = asset_tensor(robot,"body_mass").clone()
                inertia_com_before = asset_tensor(robot,"body_inertia").clone()
                ipos_before = solver.mjw_model.body_ipos.numpy().copy()
                for _ in range(2):
                    randomize_rigid_body_com(env,selected,{"x":(.02,.02),"y":(-.03,-.03),"z":(.04,.04)},body_cfg)
                com_after = asset_tensor(robot,"body_com_pos_b")
                expected = com_before.clone()
                expected[0,body_cfg.body_ids] += torch.tensor([.02,-.03,.04],device=sim.device)
                torch.testing.assert_close(com_after,expected)
                torch.testing.assert_close(asset_tensor(robot,"body_mass"),mass_com_before)
                torch.testing.assert_close(asset_tensor(robot,"body_inertia"),inertia_com_before)
                ipos_after = solver.mjw_model.body_ipos.numpy()
                np.testing.assert_allclose(ipos_after[1],ipos_before[1],atol=1e-7)
                assert np.max(np.abs(ipos_after[0]-ipos_before[0])) > .03
                matrix_errors=[]; physical_matrix_errors=[]
                for world in range(2):
                    cpu_model=copy.copy(solver.mj_model)
                    for field in ("body_ipos","body_iquat","body_mass","body_inertia","dof_armature"):
                        getattr(cpu_model,field)[:] = getattr(solver.mjw_model,field).numpy()[world]
                    cpu_data=mujoco.MjData(cpu_model)
                    mujoco.mj_setConst(cpu_model,cpu_data)
                    cpu_data.qpos[:] = solver.mjw_data.qpos.numpy()[world]
                    mujoco.mj_forward(cpu_model,cpu_data)
                    gpu_matrix=solver.mjw_data.M.numpy()[world]
                    matrix_errors.append(float(np.max(np.abs(gpu_matrix-cpu_data.M))))
                    physical = np.abs(cpu_data.M) < 1e6  # exclude artificial table locking armature
                    physical_matrix_errors.append(float(np.max(np.abs(gpu_matrix[physical]-cpu_data.M[physical]))))
                    np.testing.assert_allclose(gpu_matrix,cpu_data.M,atol=2e-4,rtol=2e-4)
                    np.testing.assert_allclose(solver.mjw_model.dof_invweight0.numpy()[world],cpu_model.dof_invweight0,atol=2e-3,rtol=2e-3)
                result["com_subset_nonaccumulation_mass_inertia_preserved"] = True
                result["com_mjwarp_cpu_mass_matrix_max_error"] = max(matrix_errors)
                result["com_physical_mass_matrix_max_error"] = max(physical_matrix_errors)
                result["com_offset_m"] = [.02,-.03,.04]
                mass_before=asset_tensor(robot,"body_mass").clone()
                inertia_before=asset_tensor(robot,"body_inertia").clone()
                mass_cfg=EventTermCfg(func=randomize_rigid_body_mass,mode="startup",params={
                    "asset_cfg":body_cfg,"mass_distribution_params":(1.1,1.1),"operation":"scale","min_mass":1e-6})
                mass_term=randomize_rigid_body_mass(mass_cfg,env)
                for _ in range(2):
                    mass_term(env,selected,body_cfg,(1.1,1.1),"scale")
                mass_after=asset_tensor(robot,"body_mass")
                torch.testing.assert_close(mass_after[0,body_cfg.body_ids],mass_before[0,body_cfg.body_ids]*1.1)
                torch.testing.assert_close(mass_after[1],mass_before[1])
                torch.testing.assert_close(asset_tensor(robot,"body_inertia")[0,body_cfg.body_ids],inertia_before[0,body_cfg.body_ids]*1.1)
                torch.testing.assert_close(asset_tensor(robot,"body_inertia")[1],inertia_before[1])
                result["mass_subset_and_nonaccumulation_checked"] = True
                result["inertia_subset_and_nonaccumulation_checked"] = True
                material_cfg=EventTermCfg(func=randomize_rigid_body_material,mode="startup",params={
                    "asset_cfg":body_cfg,"static_friction_range":(.7,.7),"dynamic_friction_range":(.1,.1),
                    "restitution_range":(0.,0.),"num_buckets":64})
                material_term=randomize_rigid_body_material(material_cfg,env)
                model=sim.physics_manager.get_model()
                mu_before=wp.to_torch(model.shape_material_mu).clone()
                material_term(env,selected,**material_cfg.params)
                mu_after=wp.to_torch(model.shape_material_mu).clone()
                assert (mu_before!=mu_after).any()
                changed_worlds = wp.to_torch(model.shape_world)[mu_before!=mu_after].unique().tolist()
                assert changed_worlds == [0], changed_worlds
                result["material_changed_worlds"] = changed_worlds
                result["material_model_changed"] = True
                target=q.clone();max_contact=0.
                for step in range(300):
                    robot.actuators.target_command.set_position_index(value=target)
                    scene.write_data_to_sim();sim.step(render=False);scene.update(cfg.dt)
                    for values in [asset_tensor(robot,"joint_pos"),asset_tensor(cube,"root_pos_w")]:
                        assert torch.isfinite(values).all()
                    forces=scene["cube_contact"].data.force_matrix_w.torch
                    max_contact=max(max_contact,forces.norm(dim=-1).max().item())
                height=asset_tensor(cube,"root_pos_w")[:,2].clone()
                assert torch.all((height-.55).abs()<.03),height
                assert max_contact>1.,max_contact
                result["object_rest_height_m"] = height.tolist()
                result["object_table_max_force_n"] = max_contact
                solver_mass_after=wp.to_torch(solver.mjw_model.body_mass)
                solver_mu_after=wp.to_torch(solver.mjw_model.geom_friction)
                assert (solver_mass_before!=solver_mass_after).any()
                assert (solver_mu_before!=solver_mu_after).any()
                assert (solver_inertia_before != wp.to_torch(solver.mjw_model.body_inertia)).any()
                result["mjwarp_mass_changed"] = True
                result["mjwarp_inertia_changed"] = True
                result["mjwarp_friction_changed"] = True
                # Controlled sliding: a low-friction table in both worlds,
                # low/high cube friction, identical initial horizontal speeds.
                def material(asset_name, mu, ids):
                    entity=SceneEntityCfg(asset_name);entity.resolve(scene)
                    params={"asset_cfg":entity,"static_friction_range":(mu,mu),
                        "dynamic_friction_range":(mu,mu),"restitution_range":(0.,0.),"num_buckets":1}
                    term=randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material,
                        mode="startup",params=params),env)
                    term(env,ids,**params)
                material("table",.05,None)
                material("cube",.05,selected)
                material("cube",1.,torch.tensor([1],device=sim.device))
                start_x=asset_tensor(cube,"root_pos_w")[:,0].clone()
                velocity=torch.zeros(2,6,device=sim.device);velocity[:,0]=1.
                cube.write_root_velocity_to_sim_index(root_velocity=velocity)
                for _ in range(60):
                    scene.write_data_to_sim();sim.step(render=False);scene.update(cfg.dt)
                displacement=asset_tensor(cube,"root_pos_w")[:,0]-start_x
                assert displacement[0]>displacement[1]+.03,displacement
                result["low_high_friction_slide_distance_m"] = displacement.tolist()
                cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(velocity))
                # Lift only env 0 with a world force exceeding mg, clear it explicitly.
                force=torch.zeros(1,1,3,device=sim.device);force[:,:,2]=20.
                set_world_forces(cube,force,env_ids=selected)
                for step in range(50):
                    scene.write_data_to_sim();sim.step(render=False);scene.update(cfg.dt)
                lifted=asset_tensor(cube,"root_pos_w")[:,2].clone()
                set_world_forces(cube,torch.zeros_like(force),env_ids=selected)
                assert lifted[0]>height[0]+.1 and abs(lifted[1]-height[1])<.03,lifted
                result["object_subset_force_lift_height_m"] = lifted.tolist()
                result["solver_type"] = type(sim.physics_manager._solver).__name__
                result["passed"] = True
            finally:
                sim_utils.SimulationContext.clear_instance()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
