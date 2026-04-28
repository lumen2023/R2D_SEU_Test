# W&B 集成与配置切换指南

## 📊 WandB (Weights & Biases) 集成

### 1. 安装 W&B

```bash
pip install wandb
```

### 2. 登录 W&B

```bash
wandb login
# 输入你的 API key（从 https://wandb.ai/authorize 获取）
```

### 3. 修改代码以支持 W&B

#### 方法一：在 `tools.py` 中添加 W&B Logger

在 `tools.py` 的 `Logger` 类中添加 W&B 支持：

```python
# 在 tools.py 顶部添加
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class Logger:
    def __init__(self, logdir, filename="metrics.jsonl", use_wandb=False, wandb_config=None):
        self._logdir = logdir
        self._filename = filename
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}
        
        # W&B 初始化
        self._use_wandb = use_wandb and WANDB_AVAILABLE
        if self._use_wandb:
            wandb.init(
                project=wandb_config.get("project", "r2dreamer"),
                name=wandb_config.get("name", None),
                config=wandb_config.get("config", {}),
                dir=str(logdir),
                resume=wandb_config.get("resume", "allow")
            )
    
    def scalar(self, name, value):
        self._scalars[name] = float(value)
        # 同步到 W&B
        if self._use_wandb:
            wandb.log({name: value}, step=self._last_step)
    
    def write(self, step, fps=False):
        # ... 原有代码 ...
        
        # W&B 记录视频
        if self._use_wandb and self._videos:
            for name, value in self._videos.items():
                if np.issubdtype(value.dtype, np.floating):
                    value = np.clip(255 * value, 0, 255).astype(np.uint8)
                wandb.log({
                    name: wandb.Video(value, fps=16, format="mp4")
                }, step=step)
        
        # ... 原有代码 ...
        
        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}
    
    def finish(self):
        """清理资源"""
        if self._use_wandb:
            wandb.finish()
```

#### 方法二：在 `train.py` 中初始化 W&B

```python
# 在 train.py 的 main 函数中添加
@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    # ... 原有代码 ...
    
    # === 初始化 W&B ===
    use_wandb = config.get("wandb", {}).get("enabled", False)
    wandb_config = None
    if use_wandb:
        import wandb
        wandb_config = {
            "project": config.wandb.project,
            "name": config.wandb.name if hasattr(config.wandb, "name") else None,
            "config": OmegaConf.to_container(config, resolve=True),
            "resume": "allow"
        }
    
    logger = tools.Logger(logdir, use_wandb=use_wandb, wandb_config=wandb_config)
    
    # ... 训练代码 ...
    
    # 训练结束时关闭 W&B
    logger.finish()
```

### 4. 添加 W&B 配置

在 `configs/configs.yaml` 中添加：

```yaml
# W&B 配置
wandb:
  enabled: false          # 是否启用 W&B
  project: r2dreamer      # 项目名称
  entity: null            # 团队名称（可选，null 表示个人）
  name: null              # 实验名称（null 则自动生成）
  tags: []                # 标签列表
```

### 5. 运行带 W&B 的实验

```bash
# 启用 W&B
python train.py env=dmc_vision wandb.enabled=true wandb.project=my_project

# 自定义实验名称
python train.py env=dmc_vision wandb.enabled=true wandb.name=walker_r2dreamer

# 添加标签
python train.py env=dmc_vision wandb.enabled=true wandb.tags=[dmc,r2dreamer,exp1]
```

### 6. 查看结果

访问 https://wandb.ai/ 查看：
- 实时指标曲线
- 超参数对比
- 视频可视化
- 系统资源监控

---

## 🔄 切换环境

### 可用环境列表

| 环境 | 配置名 | 任务示例 | 观测类型 |
|------|--------|---------|---------|
| DeepMind Control (视觉) | `dmc_vision` | walker_walk, cheetah_run | 图像 + 向量 |
| DeepMind Control (本体) | `dmc_proprio` | walker_walk, cheetah_run | 仅向量 |
| Atari 100k | `atari100k` | breakout, pong | 图像 |
| Crafter | `crafter` | 默认 | 图像 |
| Meta-World | `metaworld` | ml_push, ml_pick_place | 向量 |
| Memory Maze | `memorymaze` | 9x9, 11x11 | 图像 |

### 切换环境命令

```bash
# 1. DMC 视觉任务（推荐起点）
python train.py env=dmc_vision env.task=walker_walk

# 2. DMC 本体感知任务（更快）
python train.py env=dmc_proprio env.task=walker_walk

# 3. Atari 游戏
python train.py env=atari100k env.task=breakout

# 4. Crafter 生存游戏
python train.py env=crafter

# 5. Meta-World 机器人操作
python train.py env=metaworld env.task=ml_push

# 6. Memory Maze 导航
python train.py env=memorymaze env.task=9x9
```

### 自定义环境参数

```bash
# 修改并行环境数量
python train.py env=dmc_vision env.amount=8

# 修改动作重复次数
python train.py env=atari100k env.repeat=4

# 修改随机种子
python train.py env=dmc_vision env.seed=42
```

---

## 🎲 切换模型

### 预定义模型配置

| 模型 | 参数量 | 适用场景 | 配置名 |
|------|--------|---------|--------|
| 超小模型 | ~12M | Atari 100k, 快速验证 | `size12M` |
| 小模型 | ~25M | Crafter, 中等任务 | `size25M` |
| 标准模型 | ~50M | 大多数任务（推荐） | `size50M` |
| 大模型 | ~100M | 复杂任务 | `size100M` |
| 超大模型 | ~200M | 高难度任务 | `size200M` |
| 巨型模型 | ~400M | 研究用途 | `size400M` |

### 切换模型命令

```bash
# 使用不同大小的模型
python train.py env=dmc_vision model=size12M   # 最快
python train.py env=dmc_vision model=size50M   # 推荐
python train.py env=dmc_vision model=size200M  # 最强
```

### 自定义模型参数

创建自定义配置文件 `configs/model/my_custom.yaml`:

```yaml
# 自定义模型配置
defaults:
  - _base_

rssm:
  deter: 2048        # 确定性状态维度（默认4096）
  stoch: 16          # 随机状态数量（默认32）
  discrete: 16       # 离散类别数（默认32）
  hidden: 256        # 隐藏层维度（默认512）
  dyn_layers: 1      # 动力学网络层数
  blocks: 4          # GRU分块数

encoder:
  cnn:
    depth: 32        # CNN基础深度
    mults: [1, 2, 3, 4]  # 每层倍数

actor:
  units: 256         # Actor隐藏单元
  layers: 2          # Actor层数

critic:
  units: 256         # Critic隐藏单元
  layers: 2          # Critic层数
```

使用自定义配置：

```bash
python train.py env=dmc_vision model=my_custom
```

### 动态覆盖单个参数

```bash
# 只修改RSSM的deter维度
python train.py env=dmc_vision model.rssm.deter=2048

# 修改学习率
python train.py env=dmc_vision model.lr=1e-4

# 修改表征学习方法
python train.py env=dmc_vision model.rep_loss=r2dreamer
```

---

## 🧪 验证算法可行性（状态量输入输出）

### 方法一：使用本体感知任务（最简单）

DMC Proprioceptive 任务只使用向量状态，无需图像处理：

```bash
# 快速验证（10分钟）
python train.py \
  env=dmc_proprio \
  env.task=walker_walk \
  model=size12M \
  model.rep_loss=r2dreamer \
  trainer.steps=50000 \
  logdir=runs/test_proprio
```

**预期结果**:
- 5万步内应该能看到分数提升
- Walker Walk 应达到 300-500 分
- 训练速度快（无CNN）

### 方法二：简化视觉任务

```bash
# 小规模视觉任务验证
python train.py \
  env=dmc_vision \
  env.task=cartpole_balance \
  model=size12M \
  trainer.steps=100000 \
  trainer.batch_size=8 \
  trainer.batch_length=32 \
  logdir=runs/test_vision
```

### 方法三：调试模式（单环境、少步数）

创建调试配置 `configs/debug.yaml`:

```yaml
defaults:
  - configs

seed: 0
device: cuda
deterministic_run: false

env:
  name: dmc_proprio
  task: walker_walk
  amount: 1          # 单环境
  repeat: 1

model:
  size: size12M
  rep_loss: r2dreamer
  lr: 3e-5

trainer:
  steps: 10000       # 仅1万步
  batch_size: 4
  batch_length: 16
  pretrain: 10
  eval_every: 2000
  eval_episode_num: 2
```

运行调试：

```bash
python train.py --config-name=debug logdir=runs/debug_test
```

### 检查点验证清单

训练开始后，检查以下指标确认算法正常工作：

#### 前1000步
```bash
# 查看日志
tail -f runs/test/console.log
```

✅ **正常标志**:
- [ ] `loss/dyn` < 10 （动力学损失合理）
- [ ] `loss/rep` ≈ `kl_free` （KL散度在控制范围内）
- [ ] `opt/grad_norm` < 10 （梯度未爆炸）
- [ ] 无 NaN/Inf 错误

#### 前10000步
✅ **正常标志**:
- [ ] `episode/score` 开始上升
- [ ] `action_entropy` 逐渐下降（从~3降到~1）
- [ ] `ret` (回报) 逐渐增加
- [ ] TensorBoard 曲线平滑

#### 评估阶段
```bash
# 查看评估分数
tensorboard --logdir=runs/test
```

✅ **正常标志**:
- [ ] `episode/eval_score` > 初始分数
- [ ] 视频预测清晰（如果使用视觉）
- [ ] 策略不再完全随机

---

## 🔍 常见问题排查

### Q1: 训练完全没反应（分数=0）

**可能原因**:
1. 环境问题
2. 动作空间不匹配
3. 奖励未正确传递

**解决方法**:
```bash
# 1. 测试环境是否正常
python -c "
from envs import make_envs
from omegaconf import OmegaConf
config = OmegaConf.load('configs/env/dmc_proprio.yaml')
train_envs, _, obs_space, act_space = make_envs(config)
obs, done = train_envs.step(train_envs.action_space.sample())
print('Obs keys:', obs.keys())
print('Reward:', obs['reward'])
print('Done:', done)
"

# 2. 检查动作范围
python -c "
from envs import make_envs
from omegaconf import OmegaConf
config = OmegaConf.load('configs/env/dmc_proprio.yaml')
_, _, _, act_space = make_envs(config)
print('Action space:', act_space)
"
```

### Q2: Loss 爆炸或 NaN

**解决方法**:
```bash
# 降低学习率
python train.py model.lr=1e-5

# 增加梯度裁剪
python train.py model.agc=0.1

# 增加 KL 自由比特
python train.py model.kl_free=2.0
```

### Q3: 内存不足

**解决方法**:
```bash
# 减小批次
python train.py trainer.batch_size=4 trainer.batch_length=16

# 使用更小模型
python train.py model=size12M

# 减少并行环境
python train.py env.amount=2
```

### Q4: 训练太慢

**加速方法**:
```bash
# 1. 启用编译（PyTorch 2.0+）
python train.py model.compile=true

# 2. 减少更新频率
python train.py trainer.train_ratio=1024

# 3. 使用本体感知而非视觉
python train.py env=dmc_proprio

# 4. 增加并行环境（如果有多个CPU核心）
python train.py env.amount=16
```

---

## 📈 完整实验示例

### 示例1: DMC Walker Walk 完整实验

```bash
# 标准配置
python train.py \
  env=dmc_vision \
  env.task=walker_walk \
  model=size50M \
  model.rep_loss=r2dreamer \
  trainer.steps=1000000 \
  seed=0 \
  logdir=runs/walker_r2dreamer_seed0 \
  wandb.enabled=true \
  wandb.project=dmc_experiments \
  wandb.name=walker_r2dreamer_s0
```

### 示例2: Atari Breakout 数据效率测试

```bash
# Atari 100k 设置
python train.py \
  env=atari100k \
  env.task=breakout \
  model=size12M \
  model.rep_loss=dreamer \
  trainer.steps=100000 \
  env.repeat=4 \
  seed=0 \
  logdir=runs/atari_breakout_100k
```

### 示例3: 消融实验（多种子）

```bash
# 使用 Hydra 多运行模式
python train.py -m \
  env=dmc_vision \
  env.task=walker_walk \
  model.rep_loss=r2dreamer,dreamer,infonce \
  seed=0,1,2 \
  logdir=runs/ablation_${now:%Y%m%d_%H%M%S}
```

这将运行 3×3=9 个实验，自动记录到不同的子目录。

---

## 🎯 快速参考命令

```bash
# ===== 最简验证（5分钟）=====
python train.py env=dmc_proprio trainer.steps=10000 logdir=runs/quick_test

# ===== 标准实验（几小时）=====
python train.py env=dmc_vision model=size50M trainer.steps=1000000

# ===== 带 W&B 的实验 =====
python train.py env=dmc_vision wandb.enabled=true wandb.project=my_proj

# ===== 多实验并行 =====
python train.py -m model.rep_loss=r2dreamer,dreamer seed=0,1,2

# ===== 恢复训练 =====
# （需要在 train.py 中添加加载逻辑）
```

---

## 📚 相关文档

- [主README](../README_zh.md) - 项目总览
- [快速参考](quick_reference.md) - 常用命令
- [架构详解](project_architecture.md) - 技术细节

---

**祝实验顺利！🚀**
