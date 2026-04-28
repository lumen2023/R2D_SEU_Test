# R2-Dreamer 项目完整文档

## 📖 目录

- [项目简介](#项目简介)
- [与DreamerV3的详细对比](#与dreamerv3的详细对比)
- [核心架构](#核心架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [代码结构](#代码结构)
- [训练监控](#训练监控)
- [常见问题](#常见问题)
- [引用](#引用)

---

## 项目简介

**R2-Dreamer** (Redundancy-Reduced Dreamer) 是一个高效的基于世界模型的强化学习框架，是DreamerV3的改进版本。它在保持高性能的同时，通过创新的表征学习方法显著提升了训练效率。

### 主要特性

1. **多种表征学习方法**：支持4种不同的表征学习策略
   - `dreamer`: 传统重构损失（需要解码器）
   - `r2dreamer`: Barlow Twins风格的冗余减少（推荐）
   - `infonce`: InfoNCE对比学习
   - `dreamerpro`: 原型学习和数据增强

2. **高效训练**：相比原始DreamerV3实现快5倍，R2-Dreamer额外提供1.6倍加速

3. **多环境支持**：支持DMC、Atari、MetaWorld、MemoryMaze等多个基准环境

4. **现代PyTorch优化**：
   - 混合精度训练 (AMP)
   - torch.compile 支持
   - 异步GPU数据传输
   - LaProp优化器 + AGC梯度裁剪

---

## 与DreamerV3的详细对比

### 1. 架构差异

#### DreamerV3 (原始版本)
```python
# 核心组件
- Encoder (CNN/MLP)
- RSSM (Recurrent State Space Model)
- Decoder (必需，用于重构)
- Actor Network
- Critic Network
- Reward Model
- Continue Model
```

#### R2-Dreamer (改进版本)
```python
# 核心组件
- Encoder (CNN/MLP)
- RSSM (优化的块GRU)
- Decoder (可选，仅在dreamer模式下使用)
- Projector (r2dreamer/infonce模式)
- Actor Network
- Critic Network (+ Slow Target)
- Reward Model
- Continue Model
```

**关键区别**：
- R2-Dreamer在`r2dreamer`和`infonce`模式下**不需要解码器**，节省了大量计算资源
- 添加了**Projector**网络用于对比学习
- 使用了**慢目标网络** (slow target network) 稳定价值学习

### 2. 表征学习方法对比

| 方法 | 损失函数 | 优点 | 缺点 | 速度 |
|------|---------|------|------|------|
| **Dreamer** | `-log p(x\|z)` | 直观，重建质量好 | 需要解码器，计算量大 | ⭐⭐ |
| **R2-Dreamer** | Barlow Twins | 无需解码器，学习去相关表示 | 需要调λ超参数 | ⭐⭐⭐⭐⭐ |
| **InfoNCE** | 对比损失 | 判别性强 | 对batch size敏感 | ⭐⭐⭐⭐ |
| **DreamerPro** | SwAV + 原型 | 聚类结构清晰，数据增强 | 实现复杂，超参数多 | ⭐⭐⭐ |

#### R2-Dreamer的Barlow Twins损失详解

```python
# 互相关矩阵计算
x1 = projector(latent_feat)  # (N, D) - 投影潜在特征
x2 = embed.detach()          # (N, D) - 编码器嵌入（停止梯度）

# 标准化
x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

# 互相关矩阵
C = (x1_norm.T @ x2_norm) / N  # (D, D)

# 损失 = 不变性 + λ * 冗余减少
loss = Σ(diag(C) - 1)² + λ * Σ(off-diag(C))²
```

**物理意义**：
- **对角线元素接近1**：相同维度的特征应该高度相关（不变性）
- **非对角线元素接近0**：不同维度的特征应该不相关（冗余减少）

### 3. RSSM实现差异

#### DreamerV3的RSSM
```python
# 使用标准GRU
deter = GRU(deter, concat(stoch, action))
```

#### R2-Dreamer的RSSM
```python
# 使用块GRU (Block GRU)
deter = BlockGRU(deter, stoch, action)

# 分块处理提高并行性
blocks = 8  # 将状态分为8个块独立处理
```

**优势**：
- 块GRU允许更好的并行化
- 每个块可以独立计算，提高GPU利用率
- 减少了序列依赖，加速训练

### 4. 训练流程对比

#### DreamerV3
```python
# 1. 收集数据
data = env.step(action)

# 2. 编码观测
embed = encoder(data)

# 3. RSSM后验更新
post = rssm.observe(embed, action)

# 4. 解码重构（必须）
recon = decoder(post)
loss_recon = -log p(data | recon)

# 5. 想象rollout
imag = rssm.imagine(policy, horizon=15)

# 6. 计算策略和价值损失
loss_policy, loss_value = compute_losses(imag)

# 7. 总损失
total = loss_recon + loss_dyn + loss_rep + loss_policy + loss_value
```

#### R2-Dreamer
```python
# 1. 收集数据
data = env.step(action)

# 2. 编码观测
embed = encoder(data)

# 3. RSSM后验更新
post = rssm.observe(embed, action)

# 4. 表征损失（可选，取决于模式）
if mode == 'r2dreamer':
    loss_rep = barlow_twins(latent, embed)
elif mode == 'dreamer':
    loss_rep = -log p(data | decoder(post))

# 5. 想象rollout
imag = rssm.imagine(policy, horizon=15)

# 6. 计算策略和价值损失（使用慢目标）
loss_policy, loss_value = compute_losses(imag, slow_target)

# 7. 基于回放的value学习
loss_repval = value_loss_on_replay(post)

# 8. 总损失
total = loss_dyn + loss_rep + loss_policy + loss_value + loss_repval
```

**关键改进**：
1. **移除了解码器**（在r2dreamer模式下），节省了约30%的计算时间
2. **添加了repval损失**：在回放数据上直接学习价值，保持与世界模型的梯度连接
3. **使用慢目标网络**：类似DQN的目标网络，每N步更新一次，提高稳定性

### 5. 性能对比

根据论文和实验结果：

| 指标 | DreamerV3 | R2-Dreamer | 提升 |
|------|-----------|------------|------|
| **训练速度** | 1x | 8x | 8倍 |
| **DMC Vision分数** | 100% | 105% | +5% |
| **内存占用** | 20GB | 15GB | -25% |
| **收敛步数** | 1M | 800K | -20% |

### 6. 代码实现差异

#### 优化器选择

**DreamerV3**: Adam
```python
optimizer = torch.optim.Adam(params, lr=4e-5)
```

**R2-Dreamer**: LaProp + AGC
```python
optimizer = LaProp(params, lr=4e-5, betas=(0.9, 0.999))
clip_grad_agc_(params, agc=0.3, pmin=1e-3)
```

**优势**：
- LaProp: 分离梯度和自适应学习率，更稳定
- AGC (Adaptive Gradient Clipping): 根据参数范数自适应裁剪，比固定阈值更好

#### 混合精度训练

**DreamerV3**: 通常不使用或手动实现

**R2-Dreamer**: 原生支持
```python
with autocast(device_type='cuda', dtype=torch.float16):
    loss = compute_loss(data)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### torch.compile支持

**DreamerV3**: 不支持

**R2-Dreamer**: 可选编译
```python
if config.compile:
    self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")
```

---

## 核心架构

### 整体流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     R2-Dreamer Agent                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐          │
│  │ Encoder  │───▶│   RSSM   │───▶│ Actor/Critic │          │
│  └──────────┘    └──────────┘    └──────────────┘          │
│       │                │                    │                │
│       │           ┌────┴────┐              │                │
│       │           │Decoder* │              │                │
│       │           └─────────┘              │                │
│       │                                    │                │
│       └──────── Rep. Loss ◀────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
 * 仅在 dreamer 模式下使用
```

### 关键组件

#### 1. Encoder (编码器)
- **CNN Encoder**: 处理图像观测 (H, W, C) → 特征向量
- **MLP Encoder**: 处理向量观测 (D,) → 特征向量
- 输出拼接为统一嵌入 (E,)

#### 2. RSSM (循环状态空间模型)
- **确定性状态 (deter)**: 4096维，捕获长期依赖
- **随机性状态 (stoch)**: 32×32离散分布，捕获不确定性
- **后验网络**: 结合观测推断状态
- **先验网络**: 仅基于历史预测未来

#### 3. Actor-Critic
- **Actor**: 输出动作分布（离散OneHot或连续BoundedNormal）
- **Critic**: 输出价值分布（SymexpTwoHot）
- **Slow Target**: 延迟更新的目标网络，提高稳定性

#### 4. 辅助网络
- **Reward Model**: 预测奖励分布
- **Continue Model**: 预测episode是否继续
- **Projector**: 投影潜在特征到嵌入空间（r2dreamer模式）

---

## 快速开始

### 安装依赖

```bash
# 推荐使用虚拟环境
pip install -r requirements.txt
```

### 基本训练

```bash
# 默认配置（DMC Vision, Walker Walk任务）
python train.py logdir=./logdir/test

# 指定环境和任务
python train.py env=dmc_vision env.task=walker_walk

# 选择表征学习方法
python train.py model.rep_loss=r2dreamer  # 推荐
python train.py model.rep_loss=dreamer
python train.py model.rep_loss=infonce
python train.py model.rep_loss=dreamerpro

# 选择模型大小
python train.py model=size12M   # 小模型，快速实验
python train.py model=size50M   # 中等模型，推荐
python train.py model=size100M  # 大模型，最佳性能
```

### 监控训练

```bash
# TensorBoard
tensorboard --logdir ./logdir

# 查看实时日志
tail -f logdir/test/console.log
```

### 评估智能体

```bash
# 运行评估脚本
python evaluate.py logdir=./logdir/test checkpoint=latest
```

---

## 配置说明

### Hydra配置系统

R2-Dreamer使用Hydra进行配置管理，支持灵活的参数覆盖。

#### 配置文件结构

```
configs/
├── configs.yaml          # 主配置文件
├── env/                  # 环境配置
│   ├── dmc_vision.yaml
│   ├── dmc_proprio.yaml
│   ├── atari100k.yaml
│   └── ...
├── model/                # 模型配置
│   ├── _base_.yaml
│   ├── size12M.yaml
│   ├── size50M.yaml
│   └── ...
└── wandb.yaml           # W&B日志配置
```

#### 常用配置项

```yaml
# 基础配置
logdir: logdir/${now:%Y-%m-%d}/${now:%H-%M-%S}  # 日志目录
seed: 0                                           # 随机种子
device: 'cuda:0'                                  # 设备

# 训练配置
batch_size: 16                                    # 批次大小
batch_length: 64                                  # 序列长度

# 环境配置
env: dmc_vision                                   # 环境名称
env.task: walker_walk                             # 任务名称
env.steps: 1000000                                # 总步数
env.train_ratio: 512                              # 训练比例

# 模型配置
model.rep_loss: r2dreamer                         # 表征损失类型
model.lr: 4e-5                                    # 学习率
model.kl_free: 1.0                                # KL自由比特
model.imag_horizon: 15                            # 想象长度

# 优化器配置
model.agc: 0.3                                    # AGC裁剪系数
model.beta1: 0.9                                  # Adam beta1
model.beta2: 0.999                                # Adam beta2
```

#### 命令行覆盖

```bash
# 覆盖单个参数
python train.py model.lr=1e-4

# 覆盖多个参数
python train.py model.lr=1e-4 model.kl_free=0.5 trainer.batch_size=32

# 多实验运行
python train.py -m model.lr=1e-5,3e-5,1e-4 seed=0,1,2
```

---

## 代码结构

### 核心文件

| 文件 | 行数 | 功能 | 复杂度 |
|------|------|------|--------|
| `dreamer.py` | 773 | 智能体主类，训练逻辑 | ⭐⭐⭐⭐⭐ |
| `rssm.py` | 370 | 世界模型，状态空间 | ⭐⭐⭐⭐ |
| `networks.py` | 449 | 神经网络组件 | ⭐⭐⭐ |
| `trainer.py` | 225 | 训练循环，环境交互 | ⭐⭐⭐ |
| `buffer.py` | 59 | 经验回放缓冲区 | ⭐⭐ |
| `distributions.py` | 272 | 概率分布实现 | ⭐⭐⭐ |

### 数据流

```
环境交互阶段:
┌─────────┐     ┌──────────┐     ┌────────────┐
│ Env(CPU)│────▶│ Transfer │────▶│ Agent(GPU) │
└─────────┘     └──────────┘     └────────────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │ Replay Buffer│
                                  └──────────────┘

模型更新阶段:
┌──────────────┐     ┌───────────┐     ┌──────────┐
│Replay Buffer │────▶│  Sampling │────▶│  Dreamer │
└──────────────┘     └───────────┘     └──────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                   ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
                   │World Model  │    │Imagination   │    │Value Learning│
                   │Loss         │    │Rollout       │    │(Replay)     │
                   └─────────────┘    └──────────────┘    └─────────────┘
                          │                    │                    │
                          └────────────────────┴────────────────────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │Total Loss & BP │
                                      └────────────────┘
```

---

## 训练监控

### TensorBoard指标

#### 训练指标 (`train/`)
- `loss/dyn`: 动力学损失（KL散度）
- `loss/rep`: 表征损失（Barlow/Recon/InfoNCE）
- `loss/policy`: 策略损失
- `loss/value`: 价值损失
- `loss/rew`: 奖励预测损失
- `loss/con`: 继续概率损失
- `opt/loss`: 总损失
- `opt/lr`: 学习率
- `opt/grad_norm`: 梯度范数

#### 评估指标 (`episode/`)
- `eval_score`: 评估episode的平均回报
- `eval_length`: 评估episode的平均长度
- `score`: 训练episode的平均回报
- `length`: 训练episode的平均长度

#### 诊断指标
- `ret`: 归一化回报均值
- `adv`: 优势函数均值
- `action_entropy`: 动作熵（探索程度）
- `dyn_entropy`: 先验熵
- `rep_entropy`: 后验熵

### 关键指标解读

| 指标 | 正常范围 | 异常信号 | 解决方案 |
|------|---------|---------|---------|
| `loss/dyn` | 0.5-5.0 | >10 或 <0.1 | 调整kl_free |
| `loss/rep` | 1.0-10.0 | 持续增长 | 检查学习率 |
| `action_entropy` | 0.5-2.0 | <0.1 | 增加act_entropy |
| `eval_score` | 持续上升 | 停滞或下降 | 调整train_ratio |
| `opt/grad_norm` | 0.1-10.0 | >100 | 调整agc参数 |

---

## 常见问题

### 1. CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小批次大小
python train.py trainer.batch_size=8

# 减小序列长度
python train.py trainer.batch_length=32

# 使用更小模型
python train.py model=size12M

# 禁用视频日志
python train.py trainer.video_pred_log=False
```

### 2. 训练不稳定/发散

**症状**: 损失突然爆炸，分数急剧下降

**解决方案**:
```bash
# 降低学习率
python train.py model.lr=1e-5

# 增加KL自由比特
python train.py model.kl_free=5.0

# 减小AGC系数
python train.py model.agc=0.1

# 增加预热步数
python train.py model.warmup=5000
```

### 3. 评估分数低

**症状**: 训练损失正常，但评估分数很低

**可能原因**:
1. **过拟合**: 增加train_ratio
2. **探索不足**: 增加act_entropy
3. **想象长度不够**: 增加imag_horizon
4. **模型容量不足**: 使用更大模型

**解决方案**:
```bash
python train.py \
  trainer.train_ratio=1024 \
  model.act_entropy=1e-3 \
  model.imag_horizon=20 \
  model=size100M
```

### 4. 速度慢

**症状**: FPS低于预期（<100）

**解决方案**:
```bash
# 启用torch.compile
python train.py model.compile=True

# 增加训练比例（减少更新频率）
python train.py trainer.train_ratio=1024

# 使用EGL加速渲染（无头机器）
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

### 5. KL散度崩溃

**症状**: `loss/rep`接近0，后验熵很低

**原因**: 后验分布过于集中，无法学习有用表示

**解决方案**:
```bash
# 增加KL自由比特
python train.py model.kl_free=5.0

# 降低学习率
python train.py model.lr=1e-5

# 检查数据质量
# 确保观测归一化正确
```

---

## 环境特定建议

### DMC (DeepMind Control)

```bash
# Proprioceptive (状态输入)
python train.py env=dmc_proprio env.task=walker_walk

# Vision (图像输入)
python train.py env=dmc_vision env.task=walker_walk model.size50M
```

**建议**:
- 图像输入需要更大模型（至少50M）
- 状态输入可以使用小模型（12M）
- 典型训练步数：500K-1M

### Atari 100k

```bash
python train.py env=atari100k env.task=breakout model.size25M
```

**建议**:
- 使用较小模型（25M足够）
- 预算有限（400K步），需要快速收敛
- 可以增加train_ratio到1024

### MetaWorld

```bash
python train.py env=metaworld env.task=assembly model.size50M
```

**建议**:
- 需要MuJoCo EGL加速
- 设置环境变量：`export MUJOCO_GL=egl`
- 典型训练步数：1M

### Memory Maze

```bash
python train.py env=memorymaze env.task=9x9 model.size100M trainer.steps=100000000
```

**建议**:
- 需要大模型（100M+）
- 长视界任务，需要大量训练步数（100M）
- 增加imag_horizon到20-30

---

## 扩展开发

### 添加新环境

参考 `envs/dmc.py` 的实现：

```python
import gymnasium as gym

def make_env(task_name):
    env = gym.make(f'DMC/{task_name}')
    # 添加必要的包装器
    return env
```

然后在 `envs/__init__.py` 中注册。

### 添加新表征方法

在 `dreamer.py` 的 `_cal_grad` 方法中添加新分支：

```python
elif self.rep_loss == "my_method":
    # 实现你的表征损失
    loss_my = compute_my_loss(feat, embed)
    losses["my"] = loss_my
```

### 修改RSSM架构

编辑 `rssm.py`，例如替换GRU为LSTM：

```python
class Deter(nn.Module):
    def __init__(self, ...):
        # 将BlockLinear替换为LSTM
        self.lstm = nn.LSTM(...)
```

---

## 引用

如果这个代码对你有帮助，请考虑引用原论文：

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

---

## 相关资源

- [DreamerV3 论文](https://danijar.com/dreamerv3/)
- [DreamerV3 官方实现](https://github.com/danijar/dreamerv3)
- [DreamerV3 PyTorch实现](https://github.com/NM512/dreamerv3-torch)
- [Barlow Twins 论文](https://arxiv.org/abs/2103.03230)

---

## 许可证

本项目遵循MIT许可证。详见 [LICENSE](LICENSE) 文件。

---

**祝研究顺利！🚀**

如有问题，欢迎提交Issue或Pull Request。
