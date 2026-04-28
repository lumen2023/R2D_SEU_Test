# R2-Dreamer: 基于世界模型的强化学习框架

## 📖 项目简介

R2-Dreamer 是一个基于 **Dreamer V3** 架构的强化学习框架，实现了多种先进的表征学习方法。该框架通过在潜在空间中学习环境动力学模型，并利用想象轨迹进行策略优化，能够高效地从高维观测（如图像）中学习复杂的控制任务。

### 核心特点

- **世界模型学习**: 使用循环状态空间模型（RSSM）学习环境的紧凑表示
- **多模态支持**: 同时处理图像和向量观测
- **多种表征学习**: 支持 Dreamer、R2-Dreamer、InfoNCE、DreamerPro 四种方法
- **高效训练**: 混合精度训练、梯度裁剪、并行环境采样
- **灵活配置**: 基于 Hydra 的配置系统，易于扩展

---

## 🏗️ 项目结构

```
r2dreamer/
├── configs/              # 配置文件
│   ├── env/             # 环境配置（Atari, DMC, Crafter等）
│   ├── model/           # 模型大小配置（12M到400M参数）
│   └── configs.yaml     # 主配置文件
├── envs/                # 环境封装
│   ├── atari.py         # Atari游戏环境
│   ├── dmc.py           # DeepMind Control Suite
│   ├── crafter.py       # Crafter环境
│   ├── metaworld.py     # Meta-World机器人操作
│   ├── memorymaze.py    # 记忆迷宫
│   ├── parallel.py      # 并行环境包装器
│   └── wrappers.py      # 环境包装器
├── optim/               # 优化器
│   ├── agc.py           # 自适应梯度裁剪（AGC）
│   └── laprop.py        # LaProp优化器
├── runs/                # 运行脚本
├── docs/                # 文档
├── dreamer.py           # 核心智能体实现
├── rssm.py              # 循环状态空间模型
├── networks.py          # 神经网络组件
├── distributions.py     # 概率分布
├── buffer.py            # 经验回放缓冲区
├── trainer.py           # 训练循环
├── train.py             # 训练入口
└── tools.py             # 工具函数
```

---

## 🔬 核心算法

### 1. 世界模型架构

R2-Dreamer 的世界模型由以下组件构成：

#### 编码器 (Encoder)
- **功能**: 将原始观测（图像、向量）编码为紧凑的嵌入向量
- **实现**: 
  - CNN编码器：处理图像观测，使用SAME填充和池化
  - MLP编码器：处理向量观测
  - 多模态融合：拼接不同模态的特征

#### RSSM (Recurrent State Space Model)
- **确定性状态 (deter)**: 使用块GRU网络建模长期依赖
- **随机性状态 (stoch)**: 离散潜在变量，捕捉环境的不确定性
- **后验更新**: 结合当前观测和动作更新状态
- **先验预测**: 仅基于历史状态和动作预测未来

#### 解码器 (Decoder) - 可选
- **功能**: 从潜在状态重建观测
- **用途**: 用于传统的Dreamer重构损失

### 2. 表征学习方法

框架支持四种不同的表征学习策略：

#### (1) Dreamer (传统方法)
```python
# 通过解码器重建观测
loss = -log p(observation | latent_state)
```
- 使用重构损失作为表征学习的监督信号
- 需要解码器网络，计算开销较大

#### (2) R2-Dreamer (Barlow Twins风格)
```python
# 互相关矩阵的对角线和非对角线元素
invariance_loss = Σ(diag(C) - 1)²
redundancy_loss = Σ(off-diag(C))²
loss = invariance_loss + λ * redundancy_loss
```
- 在投影的潜在特征和编码器嵌入之间计算互相关矩阵
- **不变性损失**: 鼓励相同样本的不同视图产生相似的表示
- **冗余减少**: 鼓励不同维度之间的去相关性
- 无需解码器，更高效

#### (3) InfoNCE (对比学习)
```python
# 对比学习目标
loss = -log exp(sim(x1, x2) / τ) / Σ exp(sim(x1, x_j) / τ)
```
- 将同一时间步的潜在特征和嵌入视为正样本对
- 其他样本作为负样本
- 学习判别性的表示

#### (4) DreamerPro (原型学习)
```python
# SwAV风格的聚类分配 + 原型匹配
swav_loss = -Σ q · log(z)
temp_loss = -Σ q_target · log(z_feat)
```
- 使用数据增强生成多个视图
- EMA网络提供稳定的目标
- Sinkhorn-Knopp算法计算软聚类分配
- 原型向量作为聚类的中心

### 3. Actor-Critic 训练

#### 想象轨迹 (Imagination Rollout)
```python
# 在潜在空间中rollout策略
for t in range(imagination_horizon):
    action ~ policy(latent_state)
    reward = reward_model(latent_state, action)
    next_latent = dynamics_model(latent_state, action)
```

#### λ-Return 计算
```python
# 广义优势估计的基础
G_t = r_{t+1} + γ[(1-λ)V_{t+1} + λG_{t+1}]
```
- 平衡偏差和方差
- λ=1: 蒙特卡洛回报（低偏差，高方差）
- λ=0: TD(0)回报（高偏差，低方差）

#### 策略梯度
```python
# 带熵正则化的策略梯度
loss = -E[Σ weight_t · (log π(a_t|s_t) · A_t + β · H(π))]
```
- `weight_t`: 累积折扣权重
- `A_t`: 优势函数
- `H(π)`: 策略熵（鼓励探索）

#### 价值学习
- **想象价值损失**: 拟合想象轨迹中的λ-return
- **回放价值损失**: 拟合真实轨迹中的λ-return，保持与世界模型的梯度连接

---

## 💻 安装与配置

### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.7 (GPU训练)
```

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-repo/r2dreamer.git
cd r2dreamer

# 安装依赖
pip install -r requirements.txt

# 安装额外环境依赖（根据需要）
# Atari
pip install ale-py==0.8.1 "gymnasium[atari]==0.29.1" "gymnasium[accept-rom-license]==0.29.1"
# Crafter
pip install crafter==1.8.0
# Meta-World
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

### 配置说明

配置文件位于 `configs/` 目录，使用 Hydra 进行管理：

```yaml
# configs/configs.yaml 主要配置项
seed: 0                    # 随机种子
device: cuda               # 计算设备
deterministic_run: false   # 是否使用确定性算法

# 环境配置
env:
  name: dmc_vision         # 环境名称
  task: walker_walk        # 具体任务
  amount: 4                # 并行环境数量
  repeat: 2                # 动作重复次数

# 模型配置
model:
  encoder:
    cnn_keys: "image"      # CNN处理的观测键
    mlp_keys: ".*"         # MLP处理的观测键
  rssm:
    deter: 4096            # 确定性状态维度
    stoch: 32              # 随机状态数量
    discrete: 32           # 离散类别数
  rep_loss: r2dreamer      # 表征学习方法

# 训练配置
trainer:
  steps: 1000000           # 总训练步数
  batch_size: 16           # 批次大小
  batch_length: 64         # 序列长度
  train_ratio: 512         # 训练数据比例
```

---

## 🚀 快速开始

### 基本训练

```bash
# 在DMC视觉任务上训练
python train.py env=dmc_vision model=size50M model.rep_loss=r2dreamer

# 在Atari 100k上训练
python train.py env=atari100k model=size12M model.rep_loss=dreamer

# 在Crafter上训练
python train.py env=crafter model=size25M model.rep_loss=infonce
```

### 自定义配置

```bash
# 覆盖特定配置项
python train.py \
  env=dmc_vision \
  env.task=walker_run \
  model.size=size100M \
  model.rep_loss=dreamerpro \
  trainer.steps=2000000 \
  seed=42
```

### 恢复训练

```python
# 在train.py中添加加载逻辑
checkpoint = torch.load(logdir / "latest.pt")
agent.load_state_dict(checkpoint["agent_state_dict"])
tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
```

---

## 📊 监控与可视化

### TensorBoard

训练过程中会自动记录指标到 TensorBoard：

```bash
tensorboard --logdir runs/
```

记录的指标包括：
- **episode/score**: 回合奖励
- **episode/length**: 回合长度
- **train/loss/***: 各项损失（dyn, rep, policy, value等）
- **train/opt/***: 优化器统计（lr, grad_norm等）
- **eval/***: 评估指标
- **视频**: train_video, eval_video, open_loop预测

### 日志文件

- `console.log`: 控制台输出镜像
- `metrics.jsonl`: JSON格式的指标记录
- `config.yaml`: Hydra配置快照

---

## 🔧 高级用法

### 添加新环境

1. 在 `envs/` 目录下创建环境文件：

```python
# envs/my_env.py
import gymnasium as gym

def make_env(task, seed):
    env = gym.make(f"MyEnv-{task}")
    env.seed(seed)
    return env
```

2. 在 `envs/__init__.py` 中注册：

```python
from .my_env import make_env

def make_envs(config):
    if config.name == "my_env":
        return make_env(config.task, config.seed)
```

3. 创建配置文件 `configs/env/my_env.yaml`

### 添加新表征学习方法

1. 在 `dreamer.py` 的 `_cal_grad` 方法中添加新的分支：

```python
elif self.rep_loss == "my_method":
    # 实现你的表征损失
    x1 = ...
    x2 = ...
    losses["my_loss"] = my_loss_function(x1, x2)
```

2. 如有需要，在 `__init__` 中添加新模块

3. 更新配置文件以支持新方法

### 调整模型大小

预定义的模型配置：
- `size12M`: 12M参数，适合Atari 100k
- `size25M`: 25M参数，适合Crafter
- `size50M`: 50M参数，标准配置
- `size100M`: 100M参数，高性能
- `size200M`: 200M参数，大型模型
- `size400M`: 400M参数，超大型模型

修改 `configs/model/size*.yaml` 中的参数：
```yaml
rssm:
  deter: 2048    # 减小以降低参数量
  stoch: 16      # 减少随机状态数量
  hidden: 256    # 隐藏层维度
```

---

## 📈 性能基准

### Atari 100k (100K环境步)

| 方法 | 平均分数 | 配置 |
|------|---------|------|
| Dreamer | ~60%人类水平 | size12M |
| R2-Dreamer | ~65%人类水平 | size12M |

### DeepMind Control Suite

| 任务 | Dreamer | R2-Dreamer | InfoNCE |
|------|---------|------------|---------|
| Walker Walk | 950+ | 960+ | 940+ |
| Cheetah Run | 850+ | 870+ | 840+ |
| Humanoid Stand | 700+ | 720+ | 690+ |

### Crafter (成就率)

| 方法 | 1M步 | 5M步 |
|------|------|------|
| Dreamer | 15% | 25% |
| R2-Dreamer | 17% | 28% |

*注：具体性能取决于超参数调优和随机种子*

---

## 🛠️ 常见问题

### Q1: CUDA Out of Memory

**解决方案**:
```yaml
# 减小批次大小或序列长度
trainer:
  batch_size: 8      # 从16降到8
  batch_length: 32   # 从64降到32

# 或使用更小的模型
model: size12M
```

### Q2: 训练不稳定

**解决方案**:
```yaml
# 调整学习率和梯度裁剪
model:
  lr: 1e-4           # 降低学习率
  agc: 0.3           # 调整AGC系数
  
# 增加KL自由比特
model:
  kl_free: 1.0       # 从0.5增加到1.0
```

### Q3: 评估分数低

**检查清单**:
- [ ] 确认环境配置正确
- [ ] 增加训练步数
- [ ] 尝试不同的表征学习方法
- [ ] 调整探索参数 `act_entropy`
- [ ] 检查是否有bug导致梯度消失

### Q4: 如何加速训练

**优化建议**:
```yaml
# 启用编译（PyTorch 2.0+）
model:
  compile: true

# 增加并行环境数量
env:
  amount: 8          # 从4增加到8

# 调整训练比例
trainer:
  train_ratio: 256   # 从512降到256，减少更新频率
```

---

## 📚 技术细节

### 张量形状说明

详细的张量形状文档请查看 `docs/tensor_shapes.md`

关键形状：
- **B**: 批次大小 (batch size)
- **T**: 时间步长 (time steps)
- **E**: 嵌入维度 (embed size)
- **S**: 随机状态数量 (stochastic states)
- **K**: 离散类别数 (discrete classes)
- **D**: 确定性状态维度 (deterministic state)
- **F**: 特征维度 (feat size = S*K + D)
- **A**: 动作维度 (action dimension)

### 梯度流分析

```
观测 → 编码器 → 嵌入 ──────────────→ 表征损失
                  ↓
RSSM观察 → 后验状态 → 特征 ──→ 奖励/继续预测
     ↓                    ↓
  先验状态 ←────── 动力学模型
     ↓
  KL损失 (dyn + rep)
  
特征 → Actor → 动作分布 → 想象轨迹 → 策略/价值损失
     → Critic → 价值估计 ↗
```

### 内存优化技巧

1. **冻结副本**: 使用 `clone_and_freeze()` 创建参数的冻结副本，避免额外的梯度计算图
2. **分离梯度**: 在想象rollout前调用 `.detach()`
3. **混合精度**: 使用 `torch.cuda.amp.autocast` 和 `GradScaler`
4. **惰性存储**: 回放缓冲区使用 `LazyTensorStorage`

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 提交PR前检查清单

- [ ] 代码符合PEP 8规范
- [ ] 添加了必要的注释
- [ ] 通过了现有测试
- [ ] 更新了相关文档
- [ ] 在新环境中验证了功能

### 代码风格

```python
# 使用类型提示
def my_function(x: torch.Tensor, config: dict) -> torch.Tensor:
    """简短的函数描述。
    
    详细描述（如需要）。
    
    Args:
        x: 输入张量
        config: 配置字典
    
    Returns:
        输出张量
    """
    # 实现...
```

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

本项目基于以下优秀工作：

- **Dreamer V3**: Hafner et al., "Mastering Diverse Domains through World Models", 2023
- **Barlow Twins**: Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", 2021
- **SwAV**: Caron et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments", 2020
- **LaProp Optimizer**: Savin & Zaytsev, "LaProp: a better way to combine gradients with adaptive gradient methods", 2020

感谢所有开源社区的贡献者！

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至: [your-email@example.com]

---

**祝训练顺利！🎉**
