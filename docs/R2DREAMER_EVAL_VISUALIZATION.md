# R2-Dreamer 评估与可视化说明

## 快速评估

```bash
conda activate r2dreamer
./test_latest.sh 10
```

脚本会自动找到最新 `final.pt`，从同一 run 目录的 `.hydra/config.yaml` 加载训练配置，并写出：

- `eval_results/latest/episodes.jsonl`
- `eval_results/latest/summary.json`
- `eval_results/latest/summary.csv`
- `eval_results/latest/training_curves.png`

也可以显式指定 checkpoint：

```bash
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 10 --out eval_results/r2
```

## R2-Dreamer vs DreamerV3/Dreamer

必须使用相同 `env.task` 的 checkpoint。工具默认会拒绝跨任务比较，例如 `metadrive_safe` 不能直接和 `dmc_walker_walk` 比。

```bash
python evaluate_compare.py \
  --run r2=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
  --run dreamer=/path/to/metadrive_dreamer/checkpoints/final.pt \
  --episodes 10 \
  --out eval_results/metadrive_compare
```

输出包括：

- `comparison_summary.csv/json`: 各模型最终指标表
- `final_performance.png`: reward、success、safe_success、route_completion、crash/out_of_road 对比
- `return_stability.png`: episode return 稳定性
- `safety_tradeoff.png`: 路线完成度、安全成功率、成本的权衡
- `representation_losses.png`: R2 的 `train/loss/barlow` 与 Dreamer 的 `train/loss/image` 等表征训练信号

## 报告解读重点

R2-Dreamer 与 DreamerV3 的核心差异不只是最终 reward：

- DreamerV3/Dreamer 使用 decoder 重构观测，训练日志中会出现 `train/loss/image`，可做 open-loop reconstruction，但计算和显存成本更高。
- R2-Dreamer 去掉 decoder 和数据增强，用潜在特征与 encoder embedding 的冗余减少目标训练表征，日志中体现为 `train/loss/barlow`。
- MetaDrive 报告应同时展示 `reward`、`route_completion`、`safe_success`、`cost`、`crash_vehicle`、`out_of_road` 和 `risk_field_cost`，突出性能、安全性和稳定性的综合优势。

当前 MetaDrive 配置是 state-only 观测，没有 `image` 键；因此 `--save-video` 只有在图像观测环境中才会生成 MP4。
