import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):
    """确定性状态转移网络（块GRU风格）
    
    使用分块线性层实现高效的GRU-style状态更新，将确定性状态
    划分为多个块，每个块独立处理后再组合。
    """
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)  # 分块数量
        self.dynlayers = int(dynlayers)  # 动力学网络层数
        act = getattr(torch.nn, act)
        
        # 输入投影层：分别处理确定性状态、随机状态和动作
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        
        # 隐藏层：多层块线性变换
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        
        # GRU风格的门控机制
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        
        # 形状转换辅助函数
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)  # (..., D) -> (..., G, D/G)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)  # (..., G, D/G) -> (..., D)

    def forward(self, stoch, deter, action):
        """确定性状态转移（块GRU风格）
        
        Args:
            stoch: 随机状态，形状 (B, S, K)
            deter: 确定性状态，形状 (B, D)
            action: 动作，形状 (B, A)
        
        Returns:
            next_deter: 下一个确定性状态，形状 (B, D)
        """
        # (B, S, K), (B, D), (B, A)
        B = action.shape[0]

        # 展平随机状态并归一化动作幅度
        # (B, S*K)
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()  # 防止动作过大
        
        # 投影输入到隐藏空间
        # (B, U)
        x0 = self._dyn_in0(deter)   # 确定性状态投影
        x1 = self._dyn_in1(stoch)   # 随机状态投影
        x2 = self._dyn_in2(action)  # 动作投影

        # 拼接投影后的输入并在块维度上广播
        # (B, 3*U)
        x = torch.cat([x0, x1, x2], -1)
        # (B, G, 3*U) - 扩展到G个块
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # 结合每块的确定性状态和每块的输入
        # (B, G, D/G + 3*U) -> (B, D + 3*U*G)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

        # 通过隐藏层处理
        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D) - GRU门控信号
        x = self._dyn_gru(x)

        # 按块分割GRU风格的门控信号
        # (B, G, 3*D/G)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)

        # (B, D) - 重置门、候选状态、更新门
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)      # 重置门：控制忘记多少旧信息
        cand = torch.tanh(reset * cand)   # 候选状态：新信息的候选
        update = torch.sigmoid(update - 1)  # 更新门：控制保留多少旧信息（减1使初始偏向更新）
        
        # GRU状态更新公式
        # (B, D)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):
    """循环状态空间模型 (Recurrent State Space Model)
    
    RSSM是Dreamer的核心组件，结合了确定性RNN和随机潜在变量来建模环境动力学。
    
    状态表示：
    - 确定性状态 (deter): 捕获长期依赖和可预测的动态
    - 随机性状态 (stoch): 捕获不确定性和多模态分布
    
    两种模式：
    - 后验 (Posterior): 使用观测更新状态（训练时）
    - 先验 (Prior): 仅基于历史预测未来（想象时）
    """
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)      # 随机状态数量
        self._deter = int(config.deter)      # 确定性状态维度
        self._hidden = int(config.hidden)    # 隐藏层维度
        self._discrete = int(config.discrete)  # 离散类别数
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)  # 均匀混合比例（防止过置信）
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)  # 后验网络层数
        self._img_layers = int(config.img_layers)  # 先验网络层数
        self._dyn_layers = int(config.dyn_layers)  # 动力学网络层数
        self._blocks = int(config.blocks)      # 分块数量
        
        # 特征维度 = 展平的随机状态 + 确定性状态
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        
        # 确定性状态转移网络
        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
        )

        # 后验网络：根据观测推断随机状态
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size  # 输入 = 确定性状态 + 观测嵌入
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        # 先验网络：仅基于确定性状态预测随机状态
        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)  # 权重初始化

    def initial(self, batch_size):
        """返回初始潜在状态
        
        Args:
            batch_size: 批次大小
        
        Returns:
            stoch: 初始随机状态，全零，形状 (B, S, K)
            deter: 初始确定性状态，全零，形状 (B, D)
        """
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """使用后验 rollout 处理观测序列
        
        Args:
            embed: 观测嵌入序列，形状 (B, T, E)
            action: 动作序列，形状 (B, T, A)
            initial: 初始状态元组 (stoch, deter)
            reset: 重置标志，形状 (B, T)，1表示episode开始
        
        Returns:
            stochs: 后验随机状态序列，形状 (B, T, S, K)
            deters: 确定性状态序列，形状 (B, T, D)
            logits: 后验logits序列，形状 (B, T, S, K)
        """
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]  # 序列长度
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        
        # 逐步处理后验更新
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        
        # 堆叠为序列
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """单步后验更新：结合观测推断状态
        
        Args:
            stoch: 上一时刻随机状态，形状 (B, S, K)
            deter: 上一时刻确定性状态，形状 (B, D)
            prev_action: 上一时刻动作，形状 (B, A)
            embed: 当前观测嵌入，形状 (B, E)
            reset: 重置标志，形状 (B,)
        
        Returns:
            stoch: 更新后的随机状态，形状 (B, S, K)
            deter: 更新后的确定性状态，形状 (B, D)
            logit: 后验logits，形状 (B, S, K)
        """
        # (B, S, K), (B, D), (B, A), (B, E), (B,)
        # 如果episode重新开始，则重置状态为零
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        # 确定性状态转移，然后基于观测嵌入计算后验logits
        # (B, D) - 更新确定性状态
        deter = self._deter_net(stoch, deter, prev_action)
        # (B, D + E) - 拼接确定性状态和观测嵌入
        x = torch.cat([deter, embed], dim=-1)
        # (B, S, K) - 后验logits
        logit = self._obs_net(x)

        # 通过straight-through Gumbel-Softmax采样离散随机状态
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """单步先验预测：无观测的状态预测
        
        Args:
            stoch: 上一时刻随机状态，形状 (B, S, K)
            deter: 上一时刻确定性状态，形状 (B, D)
            prev_action: 上一时刻动作，形状 (B, A)
        
        Returns:
            stoch: 预测的随机状态，形状 (B, S, K)
            deter: 更新的确定性状态，形状 (B, D)
        """
        # (B, D) - 确定性状态转移
        deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K) - 先验预测随机状态
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        """计算先验分布参数并采样随机状态
        
        Args:
            deter: 确定性状态，形状 (B, D)
        
        Returns:
            stoch: 采样的随机状态，形状 (B, S, K)
            logit: 先验logits，形状 (B, S, K)
        """
        # (B, S, K) - 通过先验网络计算logits
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()  # 采样
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions):
        """给定动作序列，rollout先验动力学
        
        Args:
            stoch: 初始随机状态，形状 (B, S, K)
            deter: 初始确定性状态，形状 (B, D)
            actions: 动作序列，形状 (B, T, A)
        
        Returns:
            stochs: 随机状态序列，形状 (B, T, S, K)
            deters: 确定性状态序列，形状 (B, T, D)
        """
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        stochs, deters = [], []
        
        # 逐步预测未来状态
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def get_feat(self, stoch, deter):
        """展平随机状态并与确定性状态拼接
        
        Args:
            stoch: 随机状态，形状 (B, S, K)
            deter: 确定性状态，形状 (B, D)
        
        Returns:
            feat: 拼接的特征向量，形状 (B, S*K + D)
        """
        # (B, S, K), (B, D)
        # (B, S*K) - 展平随机状态
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D) - 拼接
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        """计算KL散度损失（带自由比特阈值）
        
        KL散度用于正则化后验和先验分布，防止后验过度偏离先验。
        使用两个方向的KL散度：
        - rep_loss: KL(q||p)，鼓励后验接近先验（表征学习）
        - dyn_loss: KL(p||q)，鼓励先验接近后验（动力学学习）
        
        Args:
            post_logit: 后验logits，形状 (..., S, K)
            prior_logit: 先验logits，形状 (..., S, K)
            free: 自由比特阈值，低于此值的KL不被惩罚
        
        Returns:
            dyn_loss: 动力学损失，形状 (...)
            rep_loss: 表征损失，形状 (...)
        """
        kld = dists.kl  # KL散度计算函数
        # 表征损失：KL(q(z|x)||p(z|h))，分离先验梯度
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        # 动力学损失：KL(p(z|h)||q(z|x))，分离后验梯度
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        
        # 使用clip而非torch.clip以确保梯度不被回传
        rep_loss = torch.clip(rep_loss, min=free)  # 自由比特阈值
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
