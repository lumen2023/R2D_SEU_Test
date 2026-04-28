# R2-Dreamer 快速参考指南

## 🚀 5分钟快速开始

### 安装
```bash
pip install -r requirements.txt
```

### 运行示例
```bash
# DMC视觉任务（推荐起点）
python train.py env=dmc_vision env.task=walker_walk model=size50M

# Atari 100k
python train.py env=atari100k model=size12M model.rep_loss=r2dreamer

# Crafter
python train.py env=crafter model=size25M
```

---

## 📋 常用命令速查

### 切换表征方法
```bash
# Dreamer (重构)
model.rep_loss=dreamer

# R2-Dreamer (Barlow Twins) - 推荐
model.rep_loss=r2dreamer

# InfoNCE (对比学习)
model.rep_loss=infonce

# DreamerPro (原型学习)
model.rep_loss=dreamerpro
```

### 调整模型大小
```bash
model=size12M    # 小模型，适合Atari 100k
model=size25M    # 中小模型
model=size50M    # 标准模型 - 推荐
model=size100M   # 大模型
model=size200M   # 超大模型
model=size400M   # 巨型模型
```

### 修改训练时长
```bash
trainer.steps=500000     # 50万步
trainer.steps=1000000    # 100万步（默认）
trainer.steps=5000000    # 500万步
```

### 自定义随机种子
```bash
seed=0    # 默认
seed=42   # 可复现
seed=123  # 不同初始化
```

---

## 🔧 常见问题快速修复

### ❌ CUDA Out of Memory
```bash
# 方案1: 减小批次
trainer.batch_size=8 trainer.batch_length=32

# 方案2: 使用小模型
model=size12M

# 方案3: 减少并行环境
env.amount=2
```

### ❌ 训练不稳定/发散
```bash
# 降低学习率
model.lr=1e-5

# 增加KL自由比特
model.kl_free=2.0

# 调整AGC裁剪
model.agc=0.1
```

### ❌ 评估分数低
```bash
# 增加训练时间
trainer.steps=2000000

# 增加探索
model.act_entropy=1e-3

# 尝试不同表征方法
model.rep_loss=infonce
```

### ❌ 速度慢
```bash
# 启用编译加速
model.compile=true

# 减少更新频率
trainer.train_ratio=1024

# 增加并行环境
env.amount=8
```

---

## 📊 关键指标解读

### TensorBoard 指标

#### 训练性能
| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `episode/score` | 回合奖励 | 越高越好 |
| `episode/length` | 回合长度 | 取决于环境 |
| `train/opt/loss` | 总损失 | 应逐渐下降 |
| `train/opt/lr` | 学习率 | 预热后稳定 |

#### 世界模型质量
| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `train/loss/dyn` | 动力学损失 | < 10 |
| `train/loss/rep` | 表征损失 | ≈ kl_free |
| `train/dyn_entropy` | 先验熵 | > 0 |
| `train/rep_entropy` | 后验熵 | > 0 |

#### Actor-Critic
| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `train/loss/policy` | 策略损失 | 波动正常 |
| `train/loss/value` | 价值损失 | < 100 |
| `train/action_entropy` | 动作熵 | 逐渐下降 |
| `train/adv` | 优势均值 | ≈ 0 |
| `train/ret` | 回报均值 | 逐渐上升 |

#### 优化器状态
| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `train/opt/grad_norm` | 梯度范数 | < 10 |
| `train/opt/grad_scale` | 缩放因子 | 2^10-2^16 |
| `train/opt/update_rms` | 参数更新幅度 | 稳定 |

---

## 🎯 环境特定建议

### DeepMind Control Suite (DMC)
```yaml
# 推荐配置
env: dmc_vision
model: size50M
model.rep_loss: r2dreamer
trainer.steps: 1000000
env.amount: 4
```

**预期性能**:
- Walker Walk: 900-1000
- Cheetah Run: 800-900
- Humanoid Stand: 600-700

### Atari 100k
```yaml
# 推荐配置
env: atari100k
model: size12M
model.rep_loss: dreamer
trainer.steps: 100000  # 100K环境步
env.repeat: 4
```

**注意**: 数据效率是关键，不要训练太久

### Crafter
```yaml
# 推荐配置
env: crafter
model: size25M
model.rep_loss: infonce
trainer.steps: 5000000  # 需要更多步数
```

**评估指标**: 成就率（achievement rate）

### Meta-World
```yaml
# 推荐配置
env: metaworld
env.task: ml_push
model: size50M
model.rep_loss: r2dreamer
```

---

## 💡 调参技巧

### 学习率
```yaml
# 太小：训练慢
model.lr: 1e-5

# 合适：稳定收敛
model.lr: 4e-5  # 默认

# 太大：可能发散
model.lr: 1e-4
```

### KL自由比特
```yaml
# 太小：后验崩溃
model.kl_free: 0.1

# 合适：平衡
model.kl_free: 1.0  # 默认

# 太大：表示能力弱
model.kl_free: 5.0
```

### 想象长度
```yaml
# 太短：规划不足
model.imag_horizon: 5

# 合适：权衡
model.imag_horizon: 15  # 默认

# 太长：累积误差
model.imag_horizon: 30
```

### 动作熵
```yaml
# 太小：探索不足
model.act_entropy: 1e-4

# 合适：平衡探索利用
model.act_entropy: 3e-4  # 默认

# 太大：随机行动
model.act_entropy: 1e-3
```

---

## 🔍 调试检查清单

### 训练前
- [ ] GPU内存充足（至少12GB）
- [ ] 正确安装依赖
- [ ] 配置文件无误
- [ ] 随机种子设置

### 训练中（前1000步）
- [ ] 损失在下降
- [ ] KL散度 ≈ kl_free
- [ ] 梯度范数 < 10
- [ ] 无NaN/Inf值

### 训练中（10000步）
- [ ] 评估分数在提升
- [ ] 动作熵逐渐下降
- [ ] 回报分布合理
- [ ] 视频预测清晰

### 训练后
- [ ] 保存了checkpoint
- [ ] 记录了完整日志
- [ ] 生成了评估视频
- [ ] 性能符合预期

---

## 📁 文件输出说明

训练后 `logdir` 目录结构：
```
runs/exp_name/
├── console.log              # 控制台输出
├── metrics.jsonl            # JSON格式指标
├── config.yaml              # Hydra配置
├── latest.pt                # 最新checkpoint
├── events.out.tfevents.*    # TensorBoard日志
└── videos/                  # 视频文件夹（如有）
```

### 加载Checkpoint
```python
import torch
from dreamer import Dreamer

# 加载
checkpoint = torch.load("runs/exp_name/latest.pt")
agent.load_state_dict(checkpoint["agent_state_dict"])

# 恢复优化器状态
from tools import recursively_load_optim_state_dict
recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
```

---

## 🆘 获取帮助

### 文档
- [完整README](../README_zh.md) - 详细的项目介绍
- [架构详解](project_architecture.md) - 深入的技术细节
- [张量形状](tensor_shapes.md) - 所有张量的维度说明

### 社区
- GitHub Issues: 报告bug或提问
- Email: your-email@example.com

### 引用
如果本项目对您的研究有帮助，请引用：
```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar et al.},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

---

## ⚡ 高级技巧

### 多实验并行
```bash
# Terminal 1
python train.py seed=0 logdir=runs/exp1

# Terminal 2
python train.py seed=1 logdir=runs/exp2

# Terminal 3
python train.py seed=2 logdir=runs/exp3
```

### 超参数搜索
```bash
# 使用Hydra的多运行模式
python train.py -m \
  model.lr=1e-5,3e-5,1e-4 \
  model.kl_free=0.5,1.0,2.0 \
  ++experiment.name=sweep_$(date +%s)
```

### 从中间恢复训练
```python
# 在train.py的main函数中添加
if (logdir / "latest.pt").exists():
    checkpoint = torch.load(logdir / "latest.pt")
    agent.load_state_dict(checkpoint["agent_state_dict"])
    tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
    print(f"Resumed from step {replay_buffer.count()}")
```

### 自定义日志
```python
# 在trainer.py的eval方法中添加
self.logger.scalar("custom_metric", custom_value)
self.logger.image("attention_map", attention_img)
self.logger.histogram("latent_dist", latent_values)
```

---

**Happy Training! 🎉**
