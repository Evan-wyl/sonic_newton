# gear_sonic_newton

用于实现 **Isaac Lab 3.0 → Newton → MJWarp** 训练的独立副本。

当前已完成独立复制、第 2 步公共 API 适配、第 3 步最小场景，以及第 4 步完整动作跟踪环境和官方旧 checkpoint 的小规模 GPU 评估。`utils/lab_api.py` 和 `utils/lab_math.py` 转换 Lab 3.0 的 ProxyArray / XYZW 与 SONIC 原有 Torch / WXYZ 数据，保留动作库和策略契约；配置组合已适配新版 Hydra。

`envs/newton_config.py` 提供 Newton/MJWarp 配置与独立 URDF 导入，入口已接入 `launch_simulation`，不启动 Kit。完整任务、旧策略小样本评估、短 PPO 训练和恢复均已通过。第 6 步已验证部分随机化、Jacobian、基础物体响应及生成地形短训练；**质心随机化、完整柔顺/HOI 任务、渲染和长期收敛尚未验收**。详见[最小场景](../migration_baselines/step3/README.md)、[策略评估](../migration_baselines/step4/README.md)、[短训练](../migration_baselines/step5/README.md)和[第 6 步状态](../migration_baselines/step6/README.md)。

最小场景命令（使用 GPU，依赖安装说明见上述记录）：

```bash
.env_isaaclab/bin/python -m gear_sonic_newton.scripts.smoke_newton_g1 \
  --num-envs 2 --steps 400 --fixed-base --device cuda:0 \
  --output-dir logs_rl_newton/mjwarp_smoke_fixed
.env_isaaclab/bin/python -m gear_sonic_newton.scripts.smoke_newton_g1 \
  --num-envs 2 --steps 400 --device cuda:0 \
  --output-dir logs_rl_newton/mjwarp_smoke_floating
```

第 2 步检查命令：

```bash
.env_isaaclab/bin/python -W ignore -m unittest discover \
  -s gear_sonic_newton/tests -p 'test_lab3_api.py' -v
```

测试覆盖旋转与坐标变换、真实 CPU ProxyArray、部分环境的状态写入、G1/H2 配置与四套 Hydra 训练配置。无需启动 Kit 或 GPU。

## 独立性

- Python 导入、Hydra `_target_` / `func`、脚本调用、包版本引用均使用 `gear_sonic_newton`。
- `data/` 下的代码、机器人模型、资产及人体数据已完整复制；没有指向原包的软链接或硬链接。
- 两个包分别安装、分别修改。原包的 setuptools 包发现范围已收紧，避免将新包一并打包。
- 项目根目录下的 `data/` 是显式外部训练数据，使用用户授权的软链接；包内机器人资源仍使用本包，不依赖原 `gear_sonic`。
- 旧配置加载前重映射模块和资源路径；评估 loader 同时重映射 pickle 中已知的旧项目模块前缀到本包，避免反序列化时加载原包。未知外部类不在此兼容保证内。

从仓库根目录安装：

```bash
uv pip install --python .env_isaaclab/bin/python \
  --overrides migration_baselines/step3/isaaclab-overrides.txt \
  -e 'gear_sonic_newton[evaluation]'
```

依赖已适配当前 Lab 的 Python 3.12、NumPy 2、Torch 2.11 和 Hydra 1.4；Isaac Lab、Newton 及独立资产导入器需预先准备。当前环境已通过核心 PPO 训练与恢复，`training` extra 中其他可选人体模型工具未全面验收。

## 第 4 步评估

默认 G1 配置使用 `data/motion_lib_bones_seed/robot_filtered` 和 `data/smpl_filtered`。创建环境时以仓库根目录解析相对资源路径，避免 Hydra 工作目录影响。数据链接和真实 checkpoint 的复现命令见[第 4 步记录](../migration_baselines/step4/README.md)。

`scripts/validate_newton_tracking.py` 支持 `--checkpoint`、`--motion-file`、`--smpl-motion-file`、`--motion-key` 和 `--encoder`，保存重置前的逐步误差、接触与终止记录。`--synthetic-reference` / `--untrained-policy` 只用于诊断，不能用作旧策略表现结果。

## 入口与输出

MJWarp 每环境约束容量 `njmax` 默认设为 512，覆盖已观察到的 424 行需求。若更复杂的接触仍触发 `nefc overflow`，可在训练命令中使用 `++manager_env.config.mjwarp_njmax=1024`；容量增大会增加显存开销。修改后需要重启训练进程，已启动的求解器不会自动更新。

短训练使用 `+exp=manager/newton_short_train`。已验证 4 环境、3 轮更新后保存并恢复追加 2 轮，actor/critic 参数实际改变，Adam 和 scheduler 正确接续。配置、命令、checkpoint 路径及数值限制见[第 5 步报告](../migration_baselines/step5/README.md)。

从仓库根目录使用新入口，完成后端迁移后再进行训练验证：

```bash
python gear_sonic_newton/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  num_envs=16 headless=True

python gear_sonic_newton/eval_agent_trl.py \
  checkpoint=/path/to/checkpoint.pt +headless=True
```

| 内容 | 新包默认位置 |
| --- | --- |
| 训练日志与 checkpoint | `logs_rl_newton/` |
| 单次评估日志、导出及默认渲染 | `logs_eval_newton/` |
| 持续评估指标和完成标记 | 指定实验目录内的 `eval_newton/` |
| 持续评估日志 | 指定实验目录内的 `eval_exp_newton.log` |
| 持续评估 Hydra 文件 | `logs_eval_newton/monitor/` |
| 数据采集输出 | `outputs_newton/` |

显式指定输出目录或训练恢复目录时，以用户配置为准。持续评估需要显式指定待监控实验目录。已有模型、动作库和其他仓库级工具的格式兼容性，不由包复制保证。
