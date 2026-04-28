# MetaDrive 接入说明

## 简介

当前 MetaDrive 接入是 r2Dreamer 的第一版 SafeMetaDrive 适配层，目标是先跑通低维状态观测训练：

- 使用 SafeMetaDrive 的 `LidarStateObservation`，在 r2Dreamer 里暴露为 `obs["state"]`。
- 动作空间为连续二维动作 `[steering, throttle/brake]`，范围 `[-1, 1]`。
- Dreamer 仍然只优化环境原始 reward；安全成本暂时只作为日志指标。
- MetaDrive 源码默认引用 `/home/ac/@Lyz-Code/safeRL-metadrive/metadrive`。

## Hydra 训练接口

最小训练冒烟：

```bash
python train.py env=metadrive model=size12M device=cuda model.compile=False trainer.steps=200 env.env_num=1 env.eval_episode_num=1
```

正式训练：

```bash
python -u train.py env=metadrive model=size12M device=cuda:0 buffer.storage_device=cuda:0 model.compile=False
```

更长训练可以直接覆盖步数，例如 300 万环境步：

```bash
python -u train.py env=metadrive model=size12M device=cuda:0 buffer.storage_device=cuda:0 model.compile=False trainer.steps=3000000
```

默认每 100K 环境步保存一次 checkpoint：

- `checkpoints/step_000100000.pt`
- `checkpoints/step_000200000.pt`
- `checkpoints/index.jsonl`
- `checkpoints/latest.pt`
- `latest.pt`

`latest.pt` 保留在 run 根目录中，兼容旧的加载脚本。`checkpoints/index.jsonl` 记录每次保存的 step、update 数和原因。默认 `trainer.checkpoint_keep=0` 表示保留所有周期 checkpoint；如果磁盘空间紧张，可以设置例如 `trainer.checkpoint_keep=20` 只保留最近 20 个 `step_*.pt`。

默认配置在 `configs/env/metadrive.yaml`：

- `task: metadrive_safe`
- `train_start_seed: 100`
- `train_num_scenarios: 50`
- `eval_start_seed: 1000`
- `eval_num_scenarios: 10`
- `encoder.mlp_keys: '^state$'`
- `encoder.cnn_keys: '$^'`

## Python 接口

单环境 smoke test 可以直接使用：

```python
import numpy as np

from envs.metadrive import make_metadrive_env

env = make_metadrive_env(split="train", seed=0, num_scenarios=1, start_seed=100)
obs = env.reset()
print(obs["state"].shape)

action = np.zeros(env.action_space.shape, dtype=np.float32)
obs, reward, done, info = env.step(action)
print(reward, done, info.get("cost"))

env.close()
```

## 观测和日志字段

训练观测：

- `state`: MetaDrive 低维状态，当前默认形状为 `(259,)`。
- `is_first`: episode 首帧标记。
- `is_last`: episode 结束或截断标记。
- `is_terminal`: 环境 terminated 标记，不包含纯 time-limit 截断。

评估日志字段：

- `log_cost`
- `log_event_cost`
- `log_risk_field_cost`
- `log_risk_field_event_equivalent_cost`
- `log_success`
- `log_safe_success`
- `log_safe_route_completion`
- `log_route_completion`
- `log_crash_vehicle`
- `log_crash_object`
- `log_out_of_road`
- `log_max_step`

这些字段会被 `OnlineTrainer.eval()` 自动聚合成 `episode/eval_*` 指标。

其中 `episode/eval_safe_success` 是二值安全到达率：每个 episode 结束时只记录一次
`arrive_dest and no crash and no out_of_road`，因此在 `eval_episode_num` 较小时曲线会按
`1 / eval_episode_num` 的粒度跳变。`episode/eval_route_completion` 是连续的路线完成度；
`episode/eval_safe_route_completion` 则是在没有 crash/out_of_road 的 episode 上记录连续完成度，
发生安全事件时记为 `0`，适合观察安全约束下的渐进提升。

## 当前已知边界

- 第一版不启用图像观测。
- 第一版不实现 Lagrangian、cost critic 或安全约束优化。
- `dreamerpro` 当前依赖图像增强，不建议和 state-only MetaDrive 配置一起使用。
- MetaDrive/Panda3D 使用 native 资源，训练结束时必须显式关闭并行环境；`train.py` 已在 `finally` 中处理。
