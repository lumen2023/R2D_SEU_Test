"""R2-Dreamer 训练入口脚本

使用示例:
    # 基本训练
    python train.py env=dmc_vision model=size50M
    
    # 指定任务和环境
    python train.py env=dmc_vision env.task=walker_walk model.rep_loss=r2dreamer
    
    # 自定义日志目录和随机种子
    python train.py logdir=runs/my_exp seed=42
    
    # 多实验并行（Hydra多运行模式）
    python train.py -m model.lr=1e-5,3e-5,1e-4 seed=0,1,2
"""
import atexit
import contextlib
import json
import os
import pathlib
import shutil
import sys
import time
import warnings

import hydra
import torch

import tools
from buffer import Buffer
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")  # 忽略警告信息
sys.path.append(str(pathlib.Path(__file__).parent))  # 添加项目根目录到路径
# torch.backends.cudnn.benchmark = True  # CUDA基准测试（可选开启）
torch.set_float32_matmul_precision("high")  # 设置浮点运算精度为高性能模式


class CheckpointManager:
    """Save periodic checkpoints without cluttering the run directory."""

    def __init__(self, logdir, keep=0):
        self.logdir = pathlib.Path(logdir)
        self.checkpoint_dir = self.logdir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep = int(keep)
        self.index_path = self.checkpoint_dir / "index.jsonl"

    def save(self, agent, step, update_count=0, reason="periodic"):
        step = int(step)
        update_count = int(update_count)
        path = self._path_for(step, reason)
        payload = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": step,
            "update_count": update_count,
            "reason": reason,
            "saved_at": time.time(),
        }
        tmp_path = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        self._replace_with_link_or_copy(path, self.checkpoint_dir / "latest.pt")
        self._replace_with_link_or_copy(path, self.logdir / "latest.pt")
        self._append_index(path, step, update_count, reason)
        self._prune_periodic_checkpoints()
        print(f"Saved checkpoint: {path}")
        return path

    def _path_for(self, step, reason):
        if reason == "periodic":
            name = f"step_{step:09d}.pt"
        elif reason == "final":
            name = "final.pt"
        elif reason == "interrupt":
            name = f"interrupted_step_{step:09d}.pt"
        else:
            safe_reason = str(reason).replace("/", "_")
            name = f"{safe_reason}_step_{step:09d}.pt"
        return self.checkpoint_dir / name

    def _append_index(self, path, step, update_count, reason):
        record = {
            "step": int(step),
            "update_count": int(update_count),
            "reason": reason,
            "path": str(path.relative_to(self.logdir)),
            "time": time.time(),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _prune_periodic_checkpoints(self):
        if self.keep <= 0:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old_path in checkpoints[: max(0, len(checkpoints) - self.keep)]:
            with contextlib.suppress(FileNotFoundError):
                old_path.unlink()

    @staticmethod
    def _replace_with_link_or_copy(src, dst):
        tmp_dst = dst.with_name(dst.name + ".tmp")
        with contextlib.suppress(FileNotFoundError):
            tmp_dst.unlink()
        try:
            os.link(src, tmp_dst)
        except OSError:
            shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    """主训练函数
    
    Args:
        config: Hydra配置对象，包含所有超参数和设置
    """
    # === 初始化设置 ===
    tools.set_seed_everywhere(config.seed)  # 设置全局随机种子，保证可复现性
    if config.deterministic_run:
        tools.enable_deterministic_run()  # 启用确定性算法（速度较慢但完全可复现）
    
    # === 创建日志目录 ===
    logdir = pathlib.Path(config.logdir).expanduser()  # 展开用户路径（~符号）
    logdir.mkdir(parents=True, exist_ok=True)  # 递归创建目录

    # === 设置控制台日志镜像 ===
    # 将stdout/stderr同时输出到控制台和文件
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())  # 程序退出时关闭文件

    print("Logdir", logdir)

    # === 初始化日志记录器 ===
    # 检查是否启用 W&B
    use_wandb = hasattr(config, 'wandb') and config.wandb.get('enabled', False)
    wandb_config = None
    if use_wandb:
        from omegaconf import OmegaConf
        wandb_config = {
            "project": config.wandb.get("project", "r2dreamer"),
            "name": config.wandb.get("name", None),
            "config": OmegaConf.to_container(config, resolve=True),
            "resume": config.wandb.get("resume", "allow"),
            "tags": config.wandb.get("tags", [])
        }
    
    logger = tools.Logger(logdir, use_wandb=use_wandb, wandb_config=wandb_config)  # TensorBoard + JSON + W&B
    logger.log_hydra_config(config)  # 保存Hydra配置到TensorBoard

    # === 创建经验回放缓冲区 ===
    replay_buffer = Buffer(config.buffer)

    # === 创建环境 ===
    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    # === 创建智能体 ===
    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)  # 移动到指定设备（CPU/GPU）

    # === 创建训练器并开始训练 ===
    checkpoint_manager = CheckpointManager(
        logdir,
        keep=config.trainer.get("checkpoint_keep", 0),
    )
    policy_trainer = OnlineTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_envs,
        eval_envs,
        checkpoint_saver=checkpoint_manager.save,
    )
    try:
        train_state = policy_trainer.begin(agent)  # 进入主训练循环
        checkpoint_manager.save(
            agent,
            train_state["step"],
            train_state["update_count"],
            reason="final",
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving an interrupt checkpoint before exit.")
        checkpoint_manager.save(
            agent,
            policy_trainer.current_step,
            policy_trainer.update_count,
            reason="interrupt",
        )
        raise
    finally:
        # MetaDrive/Panda3D owns native resources in worker processes, so close
        # them explicitly instead of relying on Python interpreter shutdown.
        with contextlib.suppress(Exception):
            train_envs.close()
        with contextlib.suppress(Exception):
            eval_envs.close()
        # === 关闭 W&B ===
        logger.finish()


if __name__ == "__main__":
    main()
