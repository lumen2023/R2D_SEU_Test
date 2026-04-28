# 模型评估和可视化指南

## 快速开始

### 方法1: 使用简易测试脚本（推荐）

**最简单的方式** - 测试最新的检查点：

```bash
# 测试最新保存的模型（5个episodes）
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt

# 测试更多episodes
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 10
```

输出示例：
```
============================================================
📂 检查点: logdir/2026-04-23/10-45-06/checkpoints/final.pt
============================================================

🔨 初始化环境...
🧠 初始化模型...
📥 加载检查点...
✅ 模型加载成功!

🎬 开始评估 (5 episodes)...

  Episode  1/5: Reward= 425.32, Length= 1000
  Episode  2/5: Reward= 387.15, Length=  998
  Episode  3/5: Reward= 412.78, Length= 1000
  Episode  4/5: Reward= 398.92, Length=  999
  Episode  5/5: Reward= 401.23, Length= 1000

============================================================
📊 评估结果
============================================================
  平均奖励:  405.08 ± 13.75
  奖励范围:  [387.15, 425.32]
  平均步数:  999.4 ± 0.9
============================================================
```

---

## 方法2: 使用TensorBoard查看训练曲线

在训练期间或训练完成后实时查看结果：

```bash
# 启动TensorBoard（替换logdir路径为您的实验目录）
tensorboard --logdir logdir/2026-04-23/10-45-06

# 然后在浏览器中打开：http://localhost:6006
```

### TensorBoard关键指标

- **episode/score**: 训练环境中的总奖励
- **episode/length**: 每个episode的步数
- **episode/eval_score**: 评估环境中的平均奖励
- **episode/eval_length**: 评估环境中的平均步数
- **train/loss/dyn**: 动力学模型损失（应该递减）
- **train/loss/rep**: 表征学习损失（应该收敛）
- **train/kl**: KL散度（应该在自由比特以上）

---

## 方法3: 完整的评估脚本（advanced）

使用更完整的评估工具进行深入分析：

```bash
# 基础评估
python evaluate.py checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt

# 评估多个episodes并保存视频
python evaluate.py \
    checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    num_episodes=10 \
    save_video=true \
    video_dir=eval_videos/

# 在不同环境上评估
python evaluate.py \
    checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    env=dmc_vision \
    env.task=cheetah_run \
    num_episodes=5

# 实时渲染（需要图形界面和OpenCV）
python evaluate.py \
    checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    render=true \
    num_episodes=3
```

---

## 方法4: 在训练期间进行评估

训练脚本已经集成了定期评估。您可以通过配置来控制：

```bash
# 更频繁地进行评估
python train.py logdir=runs/my_exp trainer.eval_every=5000

# 评估时运行更多episodes
python train.py logdir=runs/my_exp trainer.eval_episode_num=50
```

---

## 理解评估指标

### 主要指标

| 指标 | 含义 | 解释 |
|------|------|------|
| **reward_mean** | 平均奖励 | 模型性能的主要指标，越高越好 |
| **reward_std** | 奖励标准差 | 性能的稳定性，越低越稳定 |
| **length_mean** | 平均episode长度 | 模型能坚持多久，通常到达环境限制 |
| **success_rate** | 成功率 | 完成任务的比例（任务相关） |

### 任务特定指标

**MetaDrive 相关:**
- `log_safe_success`: 安全到达目标的比例
- `log_route_completion`: 路线完成度（0-1）
- `log_safe_route_completion`: 安全路线完成度

**Atari 相关:**
- `log_lives`: 剩余生命数

---

## 常见问题

### Q: 如何加载特定的检查点？

A: 使用以下任何一种方式：

```bash
# 方式1: 直接指定路径
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt

# 方式2: 使用最新链接
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/latest.pt

# 方式3: 指定step
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/step_000500000.pt
```

### Q: 如何查看可用的检查点？

A:
```bash
# 列出所有检查点
ls -lh logdir/2026-04-23/10-45-06/checkpoints/

# 查看检查点索引
cat logdir/2026-04-23/10-45-06/checkpoints/index.jsonl | tail -5
```

### Q: 如何比较多个模型？

A:
```bash
# 创建对比脚本
for ckpt in logdir/*/checkpoints/final.pt; do
    echo "Testing $ckpt"
    python test_checkpoint.py "$ckpt" 5
done
```

### Q: 评估很慢，如何加速？

A:
```bash
# 使用更小的模型
python test_checkpoint.py checkpoint.pt 3

# 使用CPU（某些情况下更快）
python test_checkpoint.py checkpoint.pt device=cpu

# 减少环境数量
python test_checkpoint.py checkpoint.pt env.num_envs=1
```

### Q: 如何保存和分享评估结果？

A:
```bash
# 将结果导出为JSON
python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 10 > results.txt

# 或使用evaluate脚本
python evaluate.py checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    num_episodes=20 save_video=true video_dir=eval_results/
```

---

## 可视化模型效果

### 选项1: TensorBoard（推荐）
最简单，开箱即用：
```bash
tensorboard --logdir logdir/2026-04-23/10-45-06
```

### 选项2: W&B在线可视化
需要登录，但功能更强大：
```bash
# 启用W&B
python train.py env=dmc_vision wandb.enabled=true wandb.project=my_project

# 查看结果: https://wandb.ai/
```

### 选项3: 本地视频保存
```bash
python evaluate.py \
    checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt \
    save_video=true \
    video_dir=my_videos/

# 播放视频
vlc my_videos/episode_000.mp4
# 或
ffplay my_videos/episode_000.mp4
```

---

## 高级用法

### 比较不同表征学习方法

```bash
# 比较不同的模型
for loss in r2dreamer dreamer infonce dreamerpro; do
    echo "Testing model with $loss"
    checkpoint="logdir/test_$loss/checkpoints/final.pt"
    python test_checkpoint.py "$checkpoint" 10
done
```

### 分析模型的鲁棒性

```bash
# 在不同任务上评估同一模型
python test_checkpoint.py checkpoint.pt \
    env=dmc_vision env.task=walker_walk

python test_checkpoint.py checkpoint.pt \
    env=dmc_vision env.task=cheetah_run

python test_checkpoint.py checkpoint.pt \
    env=dmc_vision env.task=humanoid_stand
```

### 可视化模型的想象轨迹

在 `evaluate.py` 中启用 `video_pred_log`：
```bash
python evaluate.py checkpoint=checkpoint.pt video_pred_log=true
```

这将显示模型预测的未来帧与实际帧的对比。

---

## 依赖安装

### 基础（已包含）
```bash
pip install -r requirements.txt
```

### 可选：视频保存
```bash
pip install opencv-python
```

### 可选：在线可视化
```bash
pip install wandb
wandb login
```

### 可选：更好的视频播放
```bash
sudo apt-get install ffmpeg vlc
```

---

## 疑难排除

### 错误: "CUDA out of memory"
解决方案：
```bash
# 使用CPU评估
python test_checkpoint.py checkpoint.pt device=cpu

# 或减少batch size
python evaluate.py checkpoint=checkpoint.pt model.batch_size=2
```

### 错误: "检查点不存在"
检查路径：
```bash
# 列出可用的检查点
find logdir -name "*.pt" -type f

# 检查最新的
ls -ltr logdir/*/checkpoints/final.pt | tail -1
```

### 警告: "低帧率"
通常不是问题，可以忽略。如需更高帧率：
```bash
# 使用compile优化（CUDA only）
python train.py model.compile=true

# 减少训练频率
python train.py trainer.train_ratio=0.5
```

---

## 相关文件

- `test_checkpoint.py` - 简易测试脚本
- `evaluate.py` - 完整评估工具
- `trainer.py` - 训练循环和评估逻辑
- `dreamer.py` - 模型定义和推理

更多信息请查看 [主README](README.md) 和 [快速开始](GETTING_STARTED.md)。
