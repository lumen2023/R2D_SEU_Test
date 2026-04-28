#!/usr/bin/env python3
"""
快速测试脚本 - 评估已保存的检查点模型

使用示例:
    python test_checkpoint_v2.py logdir/2026-04-23/10-45-06/checkpoints/final.pt
    python test_checkpoint_v2.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 10
"""
import sys
import pathlib
import torch
import warnings
import yaml
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))

from omegaconf import OmegaConf, DictConfig
from dreamer import Dreamer
from envs import make_envs
import tools


def load_config_from_checkpoint(checkpoint_path):
    """从检查点目录加载保存的配置"""
    checkpoint_path = pathlib.Path(checkpoint_path)
    hydra_dir = checkpoint_path.parent.parent / ".hydra"
    config_file = hydra_dir / "config.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = OmegaConf.create(config_dict)
    return config


def test_checkpoint(checkpoint_path, num_episodes=5):
    """测试单个检查点"""
    checkpoint_path = pathlib.Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"📂 检查点: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # 从检查点目录加载配置
    print("📋 加载配置...")
    try:
        config = load_config_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None
    
    print(f"   任务: {config.env.task}")
    print(f"   模型: {config.model.rep_loss}\n")
    
    device = torch.device(config.device)
    
    # 创建环境和模型
    print("🔨 初始化环境...")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)
    
    print("🧠 初始化模型...")
    agent = Dreamer(config.model, obs_space, act_space).to(device)
    
    # 加载检查点
    print(f"📥 加载检查点...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    
    print(f"✅ 模型加载成功!\n")
    
    # 评估
    print(f"🎬 开始评估 ({num_episodes} episodes)...\n")
    
    results = []
    
    with torch.no_grad():
        for ep in range(num_episodes):
            # 初始化
            envs = eval_envs
            done = torch.ones(envs.env_num, dtype=torch.bool, device=device)
            agent_state = agent.get_initial_state(envs.env_num)
            act = agent_state["prev_action"].clone()
            
            episode_reward = 0.0
            episode_length = 0
            max_steps = 1000
            
            # 运行episode
            # 注意: 对于某些环境，第一步的done可能为True
            step_count = 0
            while step_count < max_steps:
                # CPU上执行环境步进
                act_cpu = act.detach().to("cpu")
                done_cpu = done.detach().to("cpu")
                trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
                
                # 观测移回GPU
                trans = trans_cpu.to(device, non_blocking=True)
                done = done_cpu.to(device)
                
                # 策略推理
                act, agent_state = agent.act(trans, agent_state, eval=True)
                
                # 收集奖励
                episode_reward += float(trans["reward"][0, 0])
                
                # 只在done且不是刚开始时更新长度
                if step_count > 0 or not done[0]:
                    episode_length += 1
                
                step_count += 1
                
                # 如果done，重置done以继续
                if done[0] and step_count >= 1:
                    break
            
            results.append({
                "reward": episode_reward,
                "length": episode_length,
            })
            
            print(f"  Episode {ep+1:2d}/{num_episodes}: "
                  f"Reward={episode_reward:7.2f}, Length={episode_length:4d}")
    
    # 统计结果
    rewards = [r["reward"] for r in results]
    lengths = [r["length"] for r in results]
    
    import numpy as np
    print(f"\n{'='*60}")
    print(f"📊 评估结果")
    print(f"{'='*60}")
    print(f"  平均奖励:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  奖励范围:  [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    print(f"  平均步数:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"{'='*60}\n")
    
    # 清理
    train_envs.close()
    eval_envs.close()
    
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python test_checkpoint_v2.py <checkpoint_path> [num_episodes]")
        print("\n示例:")
        print("  python test_checkpoint_v2.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 5")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    test_checkpoint(checkpoint_path, num_episodes=num_episodes)
