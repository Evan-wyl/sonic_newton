import math
from pathlib import Path

from omegaconf import OmegaConf


def resolve_motion_paths(manager_config):
    """Resolve task resource paths against the repository, independent of Hydra cwd."""
    config = manager_config.copy()
    commands = config.get("commands", {})
    motion = commands.get("motion")
    if motion is None or motion.get("motion_lib_cfg") is None:
        return config
    library = motion.motion_lib_cfg
    root = Path(__file__).resolve().parents[2]
    for key in ("motion_file", "smpl_motion_file", "soma_motion_file", "object_motion_file"):
        value = library.get(key)
        if value and value not in ("dummy", "zeros"):
            path = Path(value).expanduser()
            library[key] = str((root / path).resolve())
    if library.get("asset") and library.asset.get("assetRoot"):
        library.asset.assetRoot = str((root / Path(library.asset.assetRoot).expanduser()).resolve())
    return config

def register_rl_resolvers():

    # Register independently and idempotently: a duplicate must not prevent
    # subsequent resolvers from being installed during train/eval composition.
    resolvers = {
        "eval": eval,
        "if": lambda pred, a, b: a if pred else b,
        "eq": lambda x, y: x.lower() == y.lower(),
        "sqrt": lambda x: math.sqrt(float(x)),
        "sum": lambda x: sum(x),
        "ceil": lambda x: math.ceil(x),
        "int": lambda x: int(x),
        "len": lambda x: len(x),
        "sum_list": lambda x: sum(x),
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
