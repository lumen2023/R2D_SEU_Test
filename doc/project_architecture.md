# R2-Dreamer 项目架构详解

## 📌 核心组件概览

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
│       │           │ Decoder │              │                │
│       │           └─────────┘              │                │
│       │                                    │                │
│       └──────── Rep. Loss ◀────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 关键模块说明

### 1. Dreamer 智能体 ([dreamer.py](file:///home/ac/@Lyz-Code/r2dreamer/dreamer.py))

**职责**: 整合所有组件，实现完整的训练和推理流程

**核心方法**:
- `act()`: 策略推理，根据观测选择动作
- `update()`: 从回放缓冲区采样并执行优化步骤
- `_cal_grad()`: **最核心的训练逻辑**，计算所有损失和梯度

**训练流程**:
```python
# 1. 编码观测
embed = encoder(obs)

# 2. RSSM后验更新
post_stoch, post_deter = rssm.observe(embed, action, initial)

# 3. 计算表征损失（4种方法可选）
if rep_loss == "r2dreamer":
    loss = barlow_twins_loss(latent_feat, embed)

# 4. 想象轨迹 rollout
imag_feat, imag_action = imagine(start_state, horizon)

# 5. 计算策略和价值损失
policy_loss = -log_prob * advantage + entropy_bonus
value_loss = -log_prob(return_target)

# 6. 反向传播和优化
total_loss.backward()
optimizer.step()
```

---

### 2. RSSM 世界模型 ([rssm.py](file:///home/ac/@Lyz-Code/r2dreamer/rssm.py))

**职责**: 学习环境的动力学模型，在潜在空间中预测未来

**状态表示**:
- **确定性状态 (deter)**: 4096维，捕获长期依赖
- **随机性状态 (stoch)**: 32个离散变量×32类别，捕获不确定性

**关键网络**:
- `Deter`: 块GRU网络，处理确定性状态转移
- `_obs_net`: 后验网络，结合观测推断状态
- `_img_net`: 先验网络，仅基于历史预测

**两种模式**:
```python
# 后验模式（训练时）- 使用观测
stoch, deter, logit = obs_step(prev_stoch, prev_deter, action, embed)

# 先验模式（想象时）- 不使用观测
stoch, deter = img_step(prev_stoch, prev_deter, action)
```

---

### 3. 神经网络组件 ([networks.py](file:///home/ac/@Lyz-Code/r2dreamer/networks.py))

#### MultiEncoder
```
输入: {image: (B,T,H,W,C), vector: (B,T,D)}
  ↓
CNN Encoder → 图像特征
MLP Encoder → 向量特征
  ↓
拼接 → (B, T, E)
```

#### ConvEncoder
```
Image (H,W,C)
  ↓
Conv + Pool × N  (空间↓, 通道↑)
  ↓
Flatten → (H'*W'*C')
```

#### MLPHead
```
Input
  ↓
MLP (Linear + RMSNorm + Act) × N
  ↓
Distribution Head (OneHot/BoundedNormal/TwoHot)
```

---

### 4. 概率分布 ([distributions.py](file:///home/ac/@Lyz-Code/r2dreamer/distributions.py))

**支持的分布类型**:

| 分布 | 用途 | 特点 |
|------|------|------|
| `OneHotDist` | 离散动作/状态 | Gumbel-Softmax重参数化 |
| `MultiOneHotDist` | 多离散动作 | 多个OneHot的组合 |
| `BoundedNormal` | 连续动作 | tanh约束到[-1,1] |
| `SymexpTwoHot` | 奖励/价值 | 对称指数双直方图 |
| `Binary` | 继续概率 | Bernoulli分布 |

**关键技巧**:
- **Straight-through Gumbel-Softmax**: 离散采样的可微近似
- **Unimix**: 混合均匀分布防止过置信
- **Symlog/Symexp**: 对称对数/指数变换处理大范围值

---

### 5. 经验回放缓冲区 ([buffer.py](file:///home/ac/@Lyz-Code/r2dreamer/buffer.py))

**特性**:
- 使用 TorchRL 的 `ReplayBuffer`
- `SliceSampler`: 采样连续的轨迹片段
- 存储潜在状态 (stoch, deter) 用于上下文
- 支持异步GPU传输

**数据结构**:
```python
{
    "image": (B, T, H, W, C),
    "action": (B, T, A),
    "reward": (B, T, 1),
    "is_first": (B, T),
    "is_last": (B, T),
    "stoch": (B, T, S, K),  # 缓存的潜在状态
    "deter": (B, T, D),      # 缓存的潜在状态
    "episode": (B, T)        # episode ID，防止跨episode采样
}
```

---

### 6. 训练器 ([trainer.py](file:///home/ac/@Lyz-Code/r2dreamer/trainer.py))

**OnlineTrainer 主循环**:

```python
while step < max_steps:
    # 1. 评估（定期）
    if should_eval(step):
        eval(agent)
    
    # 2. 环境交互（CPU）
    act_cpu = act.to("cpu")
    trans_cpu, done = envs.step(act_cpu)
    
    # 3. 异步传输到GPU
    trans = trans_cpu.to(agent.device, non_blocking=True)
    
    # 4. 策略推理（GPU）
    act, agent_state = agent.act(trans)
    
    # 5. 存储到回放缓冲区
    buffer.add_transition(trans)
    
    # 6. 模型更新
    if should_update():
        for _ in range(num_updates):
            metrics = agent.update(buffer)
    
    # 7. 记录日志
    if should_log():
        logger.write(metrics)
```

**设计亮点**:
- CPU环境交互与GPU训练重叠
- 非阻塞内存传输 (`non_blocking=True`)
- 预训练阶段 (`pretrain`)
- 自动FPS计算

---

## 🎯 四种表征学习方法对比

### 1. Dreamer (重构)
```python
# 需要解码器
decoder_output = decoder(post_stoch, post_deter)
loss = -log p(obs | decoder_output)
```
- ✅ 直观，重建质量好
- ❌ 计算开销大，需要解码器

### 2. R2-Dreamer (Barlow Twins)
```python
# 互相关矩阵
x1 = projector(latent_feat)  # (N, D)
x2 = embed.detach()          # (N, D)
C = corr(x1_norm, x2_norm)   # (D, D)
loss = Σ(diag(C)-1)² + λ·Σ(off-diag(C))²
```
- ✅ 高效，无需解码器
- ✅ 学习去相关表示
- ⚠️ 需要调λ超参数

### 3. InfoNCE (对比学习)
```python
# 对比损失
logits = x1 @ x2.T  # (N, N)
loss = CrossEntropy(logits / temperature, labels)
```
- ✅ 判别性强
- ❌ 需要大量负样本
- ⚠️ 对batch size敏感

### 4. DreamerPro (原型学习)
```python
# SwAV风格
ema_targets = sinkhorn(ema_scores)  # 软聚类分配
loss = -Σ targets · log(predictions)
```
- ✅ 聚类结构清晰
- ✅ 数据增强提高鲁棒性
- ❌ 实现复杂，超参数多

---

## 🔄 数据流详解

### 训练时的数据流

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

### 想象 Rollout 详细流程

```python
# 初始状态来自后验编码
start = (post_stoch[-1], post_deter[-1])

for t in range(imagination_horizon):
    # 1. 提取特征
    feat = concat(flat(stoch), deter)
    
    # 2. 策略采样动作
    action ~ policy(feat)
    
    # 3. 预测奖励和继续概率
    reward = reward_model(feat)
    cont = cont_model(feat)
    
    # 4. 价值估计
    value = value_model(feat)
    
    # 5. 先验状态转移（无观测）
    stoch, deter = prior_step(stoch, deter, action)
    
    # 存储到轨迹
    trajectory.append((feat, action, reward, cont, value))

# 计算λ-return
returns = lambda_return(rewards, values, cont)

# 计算优势
advantages = returns - values

# 策略梯度
policy_loss = -Σ log_prob(action) * advantage + entropy_bonus
```

---

## ⚙️ 关键超参数

### 模型架构
```yaml
rssm:
  deter: 4096        # 确定性状态维度（越大容量越高）
  stoch: 32          # 随机状态数量（越多表达能力越强）
  discrete: 32       # 离散类别数（影响KL散度）
  hidden: 512        # 隐藏层维度
  dyn_layers: 1      # 动力学网络层数
  blocks: 8          # GRU分块数
```

### 训练配置
```yaml
trainer:
  batch_size: 16         # 序列批次大小
  batch_length: 64       # 序列长度
  train_ratio: 512       # 每512环境步更新一次
  pretrain: 100          # 预训练步数
  
model:
  lr: 4e-5              # 学习率
  warmup: 1000          # 预热步数
  kl_free: 1.0          # KL自由比特
  act_entropy: 3e-4     # 动作熵系数
  lamb: 0.95            # TD(λ)的λ
  horizon: 333          # 折扣视界 (γ = 1 - 1/333 ≈ 0.997)
  imag_horizon: 15      # 想象轨迹长度
```

### 优化器
```yaml
optimizer: LaProp       # 自适应优化器
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
  
gradient_clipping: AGC  # 自适应梯度裁剪
  agc: 0.3              # 裁剪系数
  pmin: 1e-3            # 最小范数
```

---

## 🐛 调试技巧

### 1. 检查梯度流
```python
# 在_cal_grad末尾添加
for name, param in self.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")
```

### 2. 监控KL散度
```python
# KL应该保持在kl_free附近
if metrics["loss/rep"] < self.kl_free * 0.5:
    print("Warning: KL collapse!")
if metrics["loss/rep"] > self.kl_free * 10:
    print("Warning: KL explosion!")
```

### 3. 验证想象轨迹
```python
# 检查想象的价值是否合理
print(f"Imagined reward mean: {imag_reward.mean()}")
print(f"Imagined value mean: {imag_value.mean()}")
print(f"Return mean: {ret.mean()}")
# 三者应该在相似量级
```

### 4. 可视化潜在空间
```python
# 保存stoch的t-SNE投影
from sklearn.manifold import TSNE
stoch_flat = post_stoch.reshape(-1, post_stoch.shape[-1]*post_stoch.shape[-2])
tsne = TSNE(n_components=2).fit_transform(stoch_flat.cpu().numpy())
plt.scatter(tsne[:, 0], tsne[:, 1])
plt.savefig("latent_tsne.png")
```

---

## 📊 性能分析

### 计算瓶颈
1. **RSSM observe**: 序列处理无法并行 (~40%时间)
2. **Imagination rollout**: 逐步rollout (~30%时间)
3. **Encoder forward**: CNN计算量大 (~15%时间)
4. **Backward pass**: 梯度计算 (~15%时间)

### 内存占用
```
50M参数模型 (batch_size=16, batch_length=64):
- 模型参数: ~200 MB
- 激活值: ~2 GB
- 回放缓冲区: ~10 GB (取决于max_size)
- 总计: ~15-20 GB GPU内存
```

### 加速建议
```yaml
# 1. 启用编译 (PyTorch 2.0+)
model.compile: true

# 2. 减少序列长度（如果内存不足）
trainer.batch_length: 32

# 3. 增加训练比例（减少更新频率）
trainer.train_ratio: 1024

# 4. 使用更小的模型
model: size12M
```

---

## 🔬 实验建议

### 消融实验
1. **表征方法对比**: 在相同环境下比较4种方法
2. **模型规模**: 测试12M/25M/50M/100M的影响
3. **想象长度**: 尝试5/10/15/20步
4. **KL自由比特**: 测试0.1/0.5/1.0/5.0

### 超参数搜索
```bash
# 使用Hydra的多运行功能
python train.py -m \
  model.lr=1e-5,3e-5,1e-4 \
  model.kl_free=0.5,1.0,2.0 \
  trainer.train_ratio=256,512,1024
```

---

## 📝 扩展方向

### 1. 添加新的环境
参考 `envs/dmc.py` 的实现模式

### 2. 实现新的表征方法
在 `_cal_grad` 中添加新分支

### 3. 改进RSSM架构
- 尝试Transformer替代GRU
- 添加注意力机制
- 分层RSSM

### 4. 分布式训练
- 使用DDP进行多GPU训练
- 异步回放缓冲区
- 参数服务器架构

---

**祝研究顺利！🚀**
