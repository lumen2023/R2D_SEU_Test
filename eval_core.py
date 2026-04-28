#!/usr/bin/env python3
"""Shared evaluation and plotting utilities for trained R2-Dreamer checkpoints."""

import csv
import json
import pathlib
import sys
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))


FOCUS_METRICS = (
    "reward",
    "length",
    "success",
    "safe_success",
    "route_completion",
    "safe_route_completion",
    "cost",
    "event_cost",
    "crash_vehicle",
    "crash_object",
    "out_of_road",
    "risk_field_cost",
    "risk_field_event_equivalent_cost",
)


def load_training_config(checkpoint_path):
    """Load the Hydra config saved next to a checkpoint."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    run_dir = checkpoint_run_dir(checkpoint_path)
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到训练配置: {config_path}\n"
            "请确认 checkpoint 位于 run_dir/checkpoints/ 下，且 run_dir/.hydra/config.yaml 存在。"
        )
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    return config, config_path


def checkpoint_run_dir(checkpoint_path):
    checkpoint_path = pathlib.Path(checkpoint_path)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def checkpoint_metadata(checkpoint_path):
    """Return index.jsonl metadata for the checkpoint when available."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    index_path = checkpoint_path.parent / "index.jsonl"
    if not index_path.exists():
        return {}
    target_names = {checkpoint_path.name, str(pathlib.Path("checkpoints") / checkpoint_path.name)}
    last = {}
    with index_path.open() as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("path") in target_names:
                return item
            last = item
    return last


def get_algorithm(config):
    return str(OmegaConf.select(config, "model.rep_loss", default="unknown"))


def get_task(config):
    return str(OmegaConf.select(config, "env.task", default="unknown"))


def prepare_eval_config(config, num_episodes, device=None, reference_env=None):
    """Copy a training config and make only evaluation-time overrides."""
    config = deepcopy(config)
    OmegaConf.set_struct(config, False)
    config.env.eval_episode_num = int(num_episodes)
    if reference_env:
        for key, value in reference_env.items():
            if OmegaConf.select(config, f"env.{key}", default=None) is not None:
                OmegaConf.update(config, f"env.{key}", value, merge=True)
    if device:
        config.device = str(device)
        if OmegaConf.select(config, "env.device", default=None) is not None:
            config.env.device = str(device)
        if OmegaConf.select(config, "model.device", default=None) is not None:
            config.model.device = str(device)
        if OmegaConf.select(config, "buffer.device", default=None) is not None:
            config.buffer.device = str(device)
    return config


def reference_env_settings(config):
    """Fields that should stay identical across compared runs."""
    keys = (
        "seed",
        "eval_start_seed",
        "eval_num_scenarios",
        "time_limit",
        "action_repeat",
        "metadrive_source",
        "extra_config",
    )
    values = {}
    for key in keys:
        value = OmegaConf.select(config, f"env.{key}", default=None)
        if value is not None:
            values[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return values


class CheckpointEvaluator:
    def __init__(self, checkpoint_path, config):
        import torch

        self.checkpoint_path = pathlib.Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {self.checkpoint_path}")
        self.config = config
        self.torch = torch
        self.device = torch.device(config.device)
        self.train_envs = None
        self.eval_envs = None
        self.agent = None

    def setup(self):
        from dreamer import Dreamer
        from envs import make_envs

        print(f"🔨 初始化环境: {get_task(self.config)}")
        self.train_envs, self.eval_envs, obs_space, act_space = make_envs(self.config.env)

        print(f"🧠 初始化模型: {get_algorithm(self.config)}")
        self.agent = Dreamer(self.config.model, obs_space, act_space).to(self.device)

        print(f"📥 加载检查点: {self.checkpoint_path}")
        checkpoint = self.torch.load(self.checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.agent.eval()

    def close(self):
        if self.train_envs is not None:
            self.train_envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()

    def evaluate(self, save_video=False, video_dir=None, max_steps=None):
        if self.agent is None:
            self.setup()
        envs = self.eval_envs
        env_num = int(envs.env_num)
        max_steps = int(max_steps or OmegaConf.select(self.config, "env.time_limit", default=1000))
        video_dir = pathlib.Path(video_dir) if video_dir else None
        if video_dir:
            video_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎬 开始 vectorized 评估: {env_num} episodes")
        torch = self.torch
        done = torch.ones(env_num, dtype=torch.bool, device=self.device)
        once_done = torch.zeros(env_num, dtype=torch.bool, device=self.device)
        steps = torch.zeros(env_num, dtype=torch.int32, device=self.device)
        returns = torch.zeros(env_num, dtype=torch.float32, device=self.device)
        log_metrics = {}
        frames = [] if save_video else None
        agent_state = self.agent.get_initial_state(env_num)
        act = agent_state["prev_action"].clone()

        with torch.no_grad():
            while not bool(once_done.all()) and int(steps.max().item()) < max_steps:
                active_before = (~done) & (~once_done)
                steps += active_before
                act_cpu = act.detach().to("cpu")
                done_cpu = done.detach().to("cpu")
                trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
                trans = trans_cpu.to(self.device, non_blocking=True)
                done = done_cpu.to(self.device)

                if save_video and "image" in trans and not bool(once_done[0]):
                    overlay = {
                        "reward": float(returns[0].item()),
                        "step": int(steps[0].item()),
                    }
                    for key in ("log_route_completion", "log_cost", "log_risk_field_cost"):
                        if key in trans:
                            overlay[key[4:]] = _tensor_item(trans[key][0])
                    frames.append(_frame_with_overlay(trans["image"][0, 0], overlay))

                act, agent_state = self.agent.act(trans, agent_state, eval=True)
                returns += trans["reward"][:, 0] * ~once_done
                for key, value in trans.items():
                    if key.startswith("log_"):
                        if key not in log_metrics:
                            log_metrics[key] = torch.zeros_like(returns)
                        log_metrics[key] += value[:, 0] * ~once_done
                once_done |= done

        for key in ("log_success", "log_safe_success", "log_crash_vehicle", "log_crash_object", "log_out_of_road"):
            if key in log_metrics:
                log_metrics[key] = torch.clip(log_metrics[key], max=1.0)

        episodes = []
        for idx in range(env_num):
            item = {
                "episode": idx,
                "reward": float(returns[idx].item()),
                "length": int(steps[idx].item()),
                "truncated_by_evaluator": bool(not once_done[idx].item()),
            }
            for key, values in log_metrics.items():
                item[key[4:]] = float(values[idx].item())
            episodes.append(item)

        if save_video:
            if frames and video_dir:
                _save_video(frames, video_dir / "episode_000.mp4")
            else:
                print("  ⚠️  未保存视频：该环境观测中没有 image 键，或没有有效帧。")
        return episodes


def evaluate_checkpoint(
    checkpoint_path,
    num_episodes=10,
    device=None,
    save_video=False,
    video_dir=None,
    out_dir=None,
    run_name=None,
    fair_env=None,
):
    checkpoint_path = pathlib.Path(checkpoint_path)
    raw_config, config_path = load_training_config(checkpoint_path)
    config = prepare_eval_config(raw_config, num_episodes, device=device, reference_env=fair_env)
    run_name = run_name or checkpoint_run_dir(checkpoint_path).name

    print("\n" + "=" * 70)
    print(f"📂 Run: {run_name}")
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"📍 Config: {config_path}")
    print(f"🧪 Task: {get_task(config)} | Algorithm: {get_algorithm(config)}")
    print("=" * 70)

    evaluator = CheckpointEvaluator(checkpoint_path, config)
    try:
        episodes = evaluator.evaluate(save_video=save_video, video_dir=video_dir)
    finally:
        evaluator.close()

    metadata = checkpoint_metadata(checkpoint_path)
    summary = summarize_episodes(
        episodes,
        extra={
            "run": run_name,
            "algorithm": get_algorithm(config),
            "task": get_task(config),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "checkpoint_step": metadata.get("step", ""),
            "update_count": metadata.get("update_count", ""),
        },
    )

    if out_dir:
        write_eval_outputs(out_dir, episodes, summary)
        run_dir = checkpoint_run_dir(checkpoint_path)
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            plot_training_curves({"run": run_name, "metrics_path": metrics_path}, pathlib.Path(out_dir))
    return episodes, summary


def summarize_episodes(episodes, extra=None):
    summary = dict(extra or {})
    numeric_keys = sorted(
        {
            key
            for item in episodes
            for key, value in item.items()
            if key != "episode" and isinstance(value, (int, float, np.integer, np.floating, bool))
        }
    )
    for key in numeric_keys:
        values = np.array([float(item.get(key, 0.0)) for item in episodes], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean()) if len(values) else 0.0
        summary[f"{key}_std"] = float(values.std()) if len(values) else 0.0
        summary[f"{key}_min"] = float(values.min()) if len(values) else 0.0
        summary[f"{key}_max"] = float(values.max()) if len(values) else 0.0
    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("📊 评估结果")
    print("=" * 70)
    labels = {
        "reward": "平均奖励",
        "length": "平均步数",
        "success": "成功率",
        "safe_success": "安全成功率",
        "route_completion": "路线完成度",
        "safe_route_completion": "安全路线完成度",
        "cost": "成本",
        "crash_vehicle": "车辆碰撞率",
        "out_of_road": "出路率",
        "risk_field_cost": "风险场成本",
    }
    for key, label in labels.items():
        mean_key = f"{key}_mean"
        if mean_key not in summary:
            continue
        mean = float(summary[mean_key])
        std = float(summary.get(f"{key}_std", 0.0))
        if key in {"success", "safe_success", "route_completion", "safe_route_completion", "crash_vehicle", "out_of_road"}:
            print(f"  {label:14s}: {mean * 100:7.2f}% ± {std * 100:.2f}%")
        else:
            print(f"  {label:14s}: {mean:9.3f} ± {std:.3f}")
    print("=" * 70 + "\n")


def write_eval_outputs(out_dir, episodes, summary):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.jsonl").open("w") as f:
        for item in episodes:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_csv(out_dir / "summary.csv", [summary])
    print(f"💾 已写出评估结果: {out_dir}")


def write_summary_csv(path, summaries):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for item in summaries:
        for key in item.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for item in summaries:
            writer.writerow(item)


def read_jsonl(path):
    rows = []
    with pathlib.Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def plot_training_curves(run_info, out_dir):
    metrics_path = pathlib.Path(run_info["metrics_path"])
    rows = read_jsonl(metrics_path)
    if not rows:
        return
    out_dir = pathlib.Path(out_dir)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  未生成训练曲线：matplotlib 未安装。")
        return

    specs = [
        ("episode/eval_score", "Eval score"),
        ("episode/eval_route_completion", "Route completion"),
        ("episode/eval_safe_success", "Safe success"),
        ("episode/eval_cost", "Cost"),
        ("fps/fps", "FPS"),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(9, 12), sharex=True)
    for ax, (key, title) in zip(axes, specs):
        xs = [r["step"] for r in rows if key in r and "step" in r]
        ys = [r[key] for r in rows if key in r and "step" in r]
        ax.set_title(title)
        if xs:
            ax.plot(xs, ys, linewidth=1.8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("environment step")
    fig.suptitle(f"Training curves: {run_info['run']}")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def plot_comparison(summaries, episodes_by_run, run_metrics, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  未生成对比图：matplotlib 未安装。")
        return

    names = [s["run"] for s in summaries]
    _bar_grid(
        plt,
        out_dir / "final_performance.png",
        names,
        summaries,
        [
            ("reward_mean", "Reward"),
            ("success_mean", "Success"),
            ("safe_success_mean", "Safe success"),
            ("route_completion_mean", "Route completion"),
            ("crash_vehicle_mean", "Crash vehicle"),
            ("out_of_road_mean", "Out of road"),
        ],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [[ep.get("reward", 0.0) for ep in episodes_by_run[name]] for name in names]
    ax.boxplot(data, labels=names, showmeans=True)
    ax.set_title("Episode return stability")
    ax.set_ylabel("return")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "return_stability.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    route = [s.get("route_completion_mean", 0.0) for s in summaries]
    cost = [s.get("cost_mean", 0.0) for s in summaries]
    safe = [s.get("safe_success_mean", 0.0) for s in summaries]
    x = np.arange(len(names))
    width = 0.28
    ax1.bar(x - width, route, width, label="route completion")
    ax1.bar(x, safe, width, label="safe success")
    ax1.set_ylabel("rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax2 = ax1.twinx()
    ax2.bar(x + width, cost, width, label="cost", color="tab:red", alpha=0.65)
    ax2.set_ylabel("cost")
    ax1.set_title("Safety and route-completion trade-off")
    ax1.grid(True, axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "safety_tradeoff.png", dpi=160)
    plt.close(fig)

    _plot_representation_losses(plt, run_metrics, out_dir / "representation_losses.png")


def _bar_grid(plt, path, names, summaries, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.reshape(-1)
    for ax, (key, title) in zip(axes, metrics):
        values = [s.get(key, 0.0) for s in summaries]
        ax.bar(names, values)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_representation_losses(plt, run_metrics, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for run, metrics_path in run_metrics.items():
        rows = read_jsonl(metrics_path)
        for key in ("train/loss/barlow", "train/loss/image"):
            xs = [r["step"] for r in rows if key in r and "step" in r]
            ys = [r[key] for r in rows if key in r and "step" in r]
            if xs:
                ax.plot(xs, ys, label=f"{run}: {key.split('/')[-1]}", linewidth=1.8)
                plotted = True
    ax.set_title("Representation-objective signals")
    ax.set_xlabel("environment step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No train/loss/barlow or train/loss/image metrics found", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _tensor_item(value):
    if hasattr(value, "detach"):
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(np.asarray(value).reshape(-1)[0])


def _frame_with_overlay(frame, overlay):
    frame = _as_hwc_uint8(frame)
    try:
        import cv2
    except ImportError:
        return frame
    lines = [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in overlay.items()]
    out = frame.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(out, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    return out


def _as_hwc_uint8(frame):
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame


def _save_video(frames, video_path):
    try:
        import cv2
    except ImportError:
        print("  ⚠️  未保存视频：请先安装 opencv-python。")
        return
    frames = [_as_hwc_uint8(frame) for frame in frames]
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  ✅ 视频已保存: {video_path}")
