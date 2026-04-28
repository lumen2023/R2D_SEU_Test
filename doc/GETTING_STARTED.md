# 快速开始指南

## 🚀 5分钟验证算法

### 第一步：安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# W&B（可选，用于在线可视化）
pip install wandb
wandb login  # 首次使用需要登录
```

### 第二步：运行最简单的测试

**推荐：使用本体感知任务（无图像，速度快）**

```bash
# 超快速测试（1万步，约5-10分钟）
python train.py \
  env=dmc_proprio \
  env.task=walker_walk \
  model=size12M \
  trainer.steps=10000 \
  logdir=runs/quick_test
```

**预期输出**:
```
Logdir runs/quick_test
Create envs.
Simulate agent.
W&B initialized: ...  (如果启用了W&B)
[1000] episode/score 45.2 / loss/dyn 3.5 / ...
[2000] episode/score 78.9 / loss/dyn 2.8 / ...
...
```

### 第三步：检查结果

```bash
# 查看TensorBoard
tensorboard --logdir=runs/quick_test

# 访问 http://localhost:6006
# 应该看到：
# - episode/score 曲线上升
# - loss/dyn 和 loss/rep 稳定
# - 无 NaN 或异常值
```

---

## 📊 启用 W&B 在线可视化

### 方法一：命令行启用

```bash
python train.py \
  env=dmc_proprio \
  env.task=walker_walk \
  wandb.enabled=true \
  wandb.project=my_first_project \
  wandb.name=test_run_1
```

### 方法二：修改配置文件

编辑 `configs/wandb.yaml`:
```yaml
enabled: true
project: my_project
name: null  # 自动生成名称
```

然后正常运行：
```bash
python train.py env=dmc_proprio
```

### 查看 W&B 结果

训练开始后，控制台会显示：
```
W&B initialized: bright-sunset-123 (abc123def)
```

访问显示的链接或 https://wandb.ai/ 查看：
- 实时指标曲线
- 超参数对比
- 系统资源监控
- （可选）视频回放

---

## 🔄 切换环境

### 从状态量切换到视觉

```bash
# 状态量输入（快，适合验证）
python train.py env=dmc_proprio env.task=walker_walk

# 视觉输入（慢，更真实）
python train.py env=dmc_vision env.task=walker_walk
```

### 可用环境列表

```bash
# DeepMind Control - 视觉
python train.py env=dmc_vision env.task=walker_walk
python train.py env=dmc_vision env.task=cheetah_run
python train.py env=dmc_vision env.task=humanoid_stand

# DeepMind Control - 状态量
python train.py env=dmc_proprio env.task=walker_walk
python train.py env=dmc_proprio env.task=cartpole_balance

# Atari 游戏
python train.py env=atari100k env.task=breakout
python train.py env=atari100k env.task=pong

# Crafter 生存游戏
python train.py env=crafter

# Meta-World 机器人操作
python train.py env=metaworld env.task=ml_push
```

---

## 🎲 切换模型大小

```bash
# 小模型（快，适合Atari 100k）
python train.py env=atari100k model=size12M

# 标准模型（推荐）
python train.py env=dmc_vision model=size50M

# 大模型（更强，但更慢）
python train.py env=dmc_vision model=size200M
```

---

## 🔧 常用配置组合

### 1. 快速验证（5-10分钟）

```bash
python train.py \
  env=dmc_proprio \
  env.task=walker_walk \
  model=size12M \
  trainer.steps=10000 \
  trainer.batch_size=4 \
  trainer.batch_length=16 \
  logdir=runs/quick_verify
```

### 2. 标准实验（几小时）

```bash
python train.py \
  env=dmc_vision \
  env.task=walker_walk \
  model=size50M \
  model.rep_loss=r2dreamer \
  trainer.steps=1000000 \
  seed=0 \
  logdir=runs/walker_standard
```

### 3. 带 W&B 的实验

```bash
python train.py \
  env=dmc_vision \
  model=size50M \
  wandb.enabled=true \
  wandb.project=dmc_experiments \
  wandb.name=walker_r2dreamer \
  wandb.tags=[dmc,r2dreamer,seed0]
```

### 4. 多实验并行（消融实验）

```bash
# 同时运行9个实验（3种方法 × 3个种子）
python train.py -m \
  env=dmc_vision \
  env.task=walker_walk \
  model.rep_loss=r2dreamer,dreamer,infonce \
  seed=0,1,2 \
  wandb.enabled=true \
  wandb.project=ablation_study
```

---

## ✅ 验证算法正常工作的检查清单

### 训练开始后1分钟内检查

```bash
# 观察控制台输出
tail -f runs/your_exp/console.log
```

**✅ 正常标志**:
- [ ] 没有报错或警告
- [ ] `loss/dyn` 在 1-10 之间
- [ ] `loss/rep` 接近 `kl_free`（默认1.0）
- [ ] `opt/grad_norm` < 10
- [ ] 没有出现 NaN 或 Inf

### 训练1000步后检查

```bash
tensorboard --logdir=runs/your_exp
```

**✅ 正常标志**:
- [ ] `episode/score` 开始有波动上升趋势
- [ ] `action_entropy` 从 ~3 逐渐下降
- [ ] `ret`（回报）逐渐增加
- [ ] 所有损失曲线平滑，无剧烈震荡

### 训练10000步后检查

**✅ 正常标志**:
- [ ] `episode/eval_score` 明显高于初始值
- [ ] Walker Walk 应达到 200-400 分
- [ ] 视频预测（如果使用视觉）开始清晰
- [ ] W&B 上可以看到完整的指标历史

---

## 🐛 常见问题

### Q1: 提示 "wandb not found"

**解决**:
```bash
pip install wandb
wandb login  # 注册/登录
```

或者禁用 W&B：
```bash
python train.py wandb.enabled=false
```

### Q2: CUDA Out of Memory

**解决**:
```bash
# 减小批次
python train.py trainer.batch_size=4 trainer.batch_length=16

# 使用更小模型
python train.py model=size12M

# 减少并行环境
python train.py env.amount=2
```

### Q3: 训练分数一直是0

**可能原因**:
1. 环境问题
2. 动作空间不匹配

**调试**:
```bash
# 测试环境
python -c "
from envs import make_envs
from omegaconf import OmegaConf
config = OmegaConf.load('configs/env/dmc_proprio.yaml')
envs, _, obs_space, act_space = make_envs(config)
print('Obs space:', obs_space)
print('Act space:', act_space)
obs, done = envs.step(envs.action_space.sample())
print('Reward:', obs['reward'])
"
```

### Q4: Loss 爆炸或出现 NaN

**解决**:
```bash
# 降低学习率
python train.py model.lr=1e-5

# 增加 KL 自由比特
python train.py model.kl_free=2.0

# 调整梯度裁剪
python train.py model.agc=0.1
```

---

## 📈 下一步

验证算法可行后，可以：

1. **增加训练步数**
   ```bash
   python train.py trainer.steps=1000000
   ```

2. **尝试不同表征方法**
   ```bash
   python train.py model.rep_loss=r2dreamer  # 或 dreamer, infonce, dreamerpro
   ```

3. **进行多种子实验**
   ```bash
   python train.py -m seed=0,1,2,3,4
   ```

4. **在更难的任务上测试**
   ```bash
   python train.py env=dmc_vision env.task=humanoid_walk
   ```

5. **阅读完整文档**
   - [README_zh.md](../README_zh.md) - 项目总览
   - [docs/wandb_and_config_guide.md](wandb_and_config_guide.md) - W&B和配置详解
   - [docs/project_architecture.md](project_architecture.md) - 架构深入

---

## 💡 小贴士

1. **首次运行建议用状态量任务**（`dmc_proprio`），速度快，容易验证
2. **启用 W&B 方便对比实验**，特别是多种子时
3. **使用 `-m` 模式运行多个实验**，自动管理日志目录
4. **定期检查 TensorBoard**，及早发现问题
5. **保存重要实验的 checkpoint**，方便后续分析

---

**祝实验顺利！🎉**

遇到问题？查看 [docs/wandb_and_config_guide.md](wandb_and_config_guide.md) 获取更详细的帮助。
