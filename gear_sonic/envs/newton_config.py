"""Newton/MJWarp configuration shared by SONIC scenes and task environments."""

from pathlib import Path
from argparse import Namespace

from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.utils import clone
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_newton.sim.schemas import NewtonArticulationCfg


def make_newton_physics(*, njmax=512):
    """Use the selected Lab release's locomotion solver/contact configuration."""
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=njmax,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )


def make_newton_simulation(*, device="cuda:0", dt=0.005, log_dir=None):
    return SimulationCfg(
        device=device, dt=dt, physics=make_newton_physics(),
        render_interval=4, log_dir=log_dir,
    )


def newton_launcher_args(config, device):
    """Launcher options for the current no-camera Newton training path."""
    options = config.manager_env.config
    if any(options.get(key, False) for key in (
        "enable_cameras", "render_results", "render_ego", "overview_camera"
    )):
        raise NotImplementedError("Newton camera workflows are not enabled yet; use a no-camera configuration.")
    return Namespace(
        device=str(device), headless=bool(config.headless),
        visualizer=[] if config.headless else ["newton_gl"],
        enable_cameras=False, require_kit=False,
        num_envs=config.num_envs, seed=config.seed,
    )


def configure_newton_robot(robot_cfg, *, cache_dir=None):
    """Retain SONIC mechanics/gains and public ordering, replace PhysX schemas.

    Standalone conversion generates multi-physics USD; it does not start Kit.
    Cylinders remain cylinders: the release's importer no longer implements
    replace_cylinders_with_capsules. Validate this geometry difference explicitly.
    """
    # The standalone importer authors PhysX USD metadata even when Newton is
    # the solver. Register codeless schemas, without importing a PhysX runtime.
    from pxr import Plug, Usd
    schemas = Path(__file__).resolve().parents[1] / "data/assets/usd_schemas"
    Plug.Registry().RegisterPlugins(str(schemas / "plugInfo.json"))
    if not Usd.SchemaRegistry().GetTypeFromSchemaTypeName("PhysxJointAPI"):
        raise RuntimeError("The standalone G1 importer requires the bundled PhysxJointAPI USD schema.")
    cfg = clone(robot_cfg)
    cfg.spawn.asset_path = str(Path(cfg.spawn.asset_path).resolve())
    cfg.spawn.rigid_props = UsdPhysicsRigidBodyCfg(kinematic_enabled=False)
    cfg.spawn.articulation_props = NewtonArticulationCfg(self_collision_enabled=True)
    cfg.spawn.self_collision = True
    cfg.spawn.replace_cylinders_with_capsules = False
    cfg.spawn.run_asset_transformer = True
    cfg.spawn.run_multi_physics_conversion = True
    if cache_dir is not None:
        cfg.spawn.usd_dir = str(Path(cache_dir).resolve())
    return cfg
