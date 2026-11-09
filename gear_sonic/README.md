# gear_sonic

An independent copy for **Isaac Lab 3.0 → Newton → MJWarp** training.

## Optional Warp Environment Managers

The experimental Isaac Lab Warp environment managers can run the adapted command, observation, reward, and termination calculations on top of the existing Newton/MJWarp physics backend. The Torch managers remain the default. To enable the Warp path, add these overrides to a training or evaluation command:

```bash
manager_env.config.use_warp_env=true \
manager_env.config.manager_call_config.default=1
```

`default=1` runs the Warp managers eagerly. After checking observations, rewards, resets, and checkpoint behavior, use `default=2` to request CUDA Graph capture. Managers with Torch bridge terms still run eagerly. This path requires an Isaac Lab installation that includes `isaaclab_experimental`.

Add `--use-warp-env` to the tracking validation script to exercise the Warp path. Run it again without the flag to compare `result.json` and `metrics.json` from the two runs. The script also accepts `--warp-manager-mode 2` to request CUDA Graph capture, `--disable-observation-corruption` for deterministic comparisons, and `--save-policy-trace` to save the first three policy observations and actions. Training and evaluation instantiate Isaac Lab's `ManagerBasedRLEnvWarp` directly. Only this validation script uses `SonicManagerBasedRLEnvWarp`: its metrics recorder reads task state, so the subclass skips the recorder's extra observation pass to prevent advancing custom history buffers twice.

With a synthetic static reference, the eager Warp and Torch paths each completed 100 control steps across two environments. Both recorded four automatic resets and the same termination sequence; their observation shapes matched. The largest difference in the recorded per-step body position error was about `2.5e-6` m. Warp mode 2 also completed 100 steps with four resets. After keeping every reward body-subset buffer alive for the command lifetime, a 4096-environment mode 2 run completed two training iterations (196608 environment steps), and a 32-environment validation run completed 100 steps with 64 automatic resets. Long-run convergence remains untested.

The released SONIC checkpoint was evaluated on a real motion clip with the G1, teleop, and SMPL encoders. With observation corruption disabled, the first three policy actions differed from the Torch path by at most about `6e-6` with the validation recorder adjustment and the WXYZ VR orientation output fix. Warp mode 2 completed a 20-step checkpoint run. These evaluations do not establish training convergence or performance across the full motion dataset.

Robot state uses Lab's XYZW quaternion order. The motion library and existing Torch task interfaces retain WXYZ; the Warp command converts quaternions at these boundaries. Height maps, using the reference root orientation as the anchor, and stateful reward or termination terms are not supported by the current Warp adaptation.

The independent copy, public API adaptation, minimal scene, full motion tracking environment, and a small GPU evaluation of an existing checkpoint have been completed. `utils/lab_api.py` and `utils/lab_math.py` adapt Lab 3.0 ProxyArray and XYZW data to SONIC's Torch and WXYZ interfaces while preserving the motion library and policy contracts. Hydra configuration composition has also been adapted.

`envs/newton_config.py` provides the Newton/MJWarp configuration and standalone URDF import. The entry points use `launch_simulation` without starting Kit. The full task, a small evaluation of an existing policy, short PPO training, and resume have passed. Some randomization, Jacobian calculations, basic object response, and short training on generated terrain have also been checked. **Center of mass randomization, the complete compliance and HOI tasks, rendering, and long term convergence have not been validated.** See the [minimal scene](../migration_baselines/step3/README.md), [policy evaluation](../migration_baselines/step4/README.md), [short training](../migration_baselines/step5/README.md), and [step 6 status](../migration_baselines/step6/README.md).

Minimal scene commands (GPU required; see the linked records for dependency setup):

```bash
.env_isaaclab/bin/python -m gear_sonic.scripts.smoke_newton_g1 \
  --num-envs 2 --steps 400 --fixed-base --device cuda:0 \
  --output-dir logs_rl_newton/mjwarp_smoke_fixed
.env_isaaclab/bin/python -m gear_sonic.scripts.smoke_newton_g1 \
  --num-envs 2 --steps 400 --device cuda:0 \
  --output-dir logs_rl_newton/mjwarp_smoke_floating
```

Step 2 API checks:

```bash
.env_isaaclab/bin/python -W ignore -m unittest discover \
  -s gear_sonic/tests -p 'test_lab3_api.py' -v
```

These tests cover rotations and coordinate transforms, a real CPU ProxyArray, partial environment state writes, G1/H2 configurations, and four Hydra training configurations. They do not require Kit or a GPU.

## Package Independence

- Python imports, Hydra `_target_` and `func` references, script calls, and package version references use `gear_sonic`.
- Code, robot models, assets, and human data under the package's `data/` directory have been copied; there are no links to the original package.
- Setuptools package discovery is limited to `gear_sonic` and its subpackages.
- The repository root `data/` directory is explicitly external training data linked with a user authorized symlink. Packaged robot resources use `gear_sonic/data/`.
- Module and resource paths are remapped before loading old configurations. The evaluation loader also remaps known old module prefixes in pickles to this package, avoiding imports from the original package during deserialization. Unknown external classes are outside this compatibility guarantee.

Install from the repository root:

```bash
uv pip install --python .env_isaaclab/bin/python \
  --overrides migration_baselines/step3/isaaclab-overrides.txt \
  -e 'gear_sonic[evaluation]'
```

Dependencies have been adapted to the current Lab environment with Python 3.12, NumPy 2, Torch 2.11, and Hydra 1.4. Isaac Lab, Newton, and the standalone asset importer must be prepared in advance. Core PPO training and resume have passed in that environment; other optional human model tools in the `training` extra have not been fully validated.

## Step 4 Evaluation

The default G1 configuration uses `data/motion_lib_bones_seed/robot_filtered` and `data/smpl_filtered`. Relative asset paths are resolved from the repository root when creating the environment, so Hydra's working directory does not affect them. See the [step 4 record](../migration_baselines/step4/README.md) for data links and commands to reproduce evaluation with a real checkpoint.

`scripts/validate_newton_tracking.py` supports `--checkpoint`, `--motion-file`, `--smpl-motion-file`, `--motion-key`, `--encoder`, and `--use-warp-env`. It saves stepwise error, contact, and termination records before resets. `--synthetic-reference` and `--untrained-policy` are diagnostic options and must not be treated as existing policy performance results.

## Entry Points and Outputs

The default MJWarp per environment constraint capacity, `njmax`, is 1024. The previous value of 512 overflowed when a large run needed more than 532 constraint rows. You can override it with `manager_env.config.mjwarp_njmax=<value>`; larger values use more GPU memory. Restart the training process after changing it; an active solver does not update automatically.

Short training uses `+exp=manager/newton_short_train`. A run with four environments and three updates was saved and resumed for two more updates. Actor and critic parameters changed, and Adam and scheduler state continued correctly. See the [step 5 report](../migration_baselines/step5/README.md) for configuration, commands, checkpoint paths, and numerical limits.

Run the entry points from the repository root:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=16 headless=True

python gear_sonic/eval_agent_trl.py \
  checkpoint=/path/to/checkpoint.pt +headless=True
```

| Output | Default location |
| --- | --- |
| Training logs and checkpoints | `logs_rl_newton/` |
| Single evaluation logs, exports, and default renders | `logs_eval_newton/` |
| Continuous evaluation metrics and completion marker | `eval_newton/` in the selected experiment directory |
| Continuous evaluation log | `eval_exp_newton.log` in the selected experiment directory |
| Continuous evaluation Hydra files | `logs_eval_newton/monitor/` |
| Data collection output | `outputs_newton/` |

Explicit output or resume directories take precedence over these defaults. Continuous evaluation requires an explicit experiment directory to monitor. Copying the package does not guarantee format compatibility for existing models, motion libraries, or other repository tools.

## Acknowledgments

We would like to acknowledge the following project, from which parts of the code in this repository are derived: [GR00T-WholeBodyControl](https://github.com/PoplarCrystal/GR00T-WholeBodyControl).
