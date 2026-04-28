#!/usr/bin/env python3
"""Compare R2-Dreamer and Dreamer/DreamerV3 checkpoints on the same task."""

import argparse
import json
import pathlib
import sys

from eval_core import (
    checkpoint_run_dir,
    evaluate_checkpoint,
    get_algorithm,
    get_task,
    load_training_config,
    plot_comparison,
    print_summary,
    read_jsonl,
    reference_env_settings,
    write_summary_csv,
)


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use NAME=CHECKPOINT, for example r2=path/to/final.pt")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return name, pathlib.Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trained R2-Dreamer family checkpoints")
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME=CHECKPOINT. Repeat this flag.")
    parser.add_argument("--episodes", "-n", type=int, default=10, help="Number of eval episodes per run")
    parser.add_argument("--out", required=True, help="Output directory for summaries and plots")
    parser.add_argument("--device", default=None, help="Override saved device, for example cpu or cuda:0")
    parser.add_argument("--save-video", action="store_true", help="Save one rollout video per run when image observations exist")
    parser.add_argument(
        "--allow-task-mismatch",
        action="store_true",
        help="Allow comparing checkpoints trained on different env.task values. Not recommended.",
    )
    return parser.parse_args()


def load_run_configs(runs):
    infos = []
    for name, checkpoint in runs:
        config, config_path = load_training_config(checkpoint)
        infos.append(
            {
                "name": name,
                "checkpoint": checkpoint,
                "config": config,
                "config_path": config_path,
                "task": get_task(config),
                "algorithm": get_algorithm(config),
            }
        )
    return infos


def validate_tasks(infos, allow_mismatch=False):
    tasks = {info["task"] for info in infos}
    if len(tasks) <= 1 or allow_mismatch:
        return
    details = "\n".join(
        f"  - {info['name']}: task={info['task']} checkpoint={info['checkpoint']}" for info in infos
    )
    raise ValueError(
        "拒绝比较不同 env.task 的 checkpoint。请提供同任务 DreamerV3/Dreamer baseline，"
        "或显式加 --allow-task-mismatch。\n" + details
    )


def main():
    args = parse_args()
    if len(args.run) < 2:
        print("❌ 至少需要两个 --run 才能对比。")
        sys.exit(2)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        infos = load_run_configs(args.run)
        validate_tasks(infos, allow_mismatch=args.allow_task_mismatch)
    except Exception as exc:
        print(f"❌ 对比配置无效: {exc}")
        sys.exit(2)

    reference = infos[0]
    fair_env = reference_env_settings(reference["config"])
    print("🔒 公平评估环境设置来自第一个 run:")
    print(json.dumps({"run": reference["name"], "task": reference["task"], **fair_env}, ensure_ascii=False, indent=2))

    summaries = []
    episodes_by_run = {}
    run_metrics = {}
    for info in infos:
        run_out = out_dir / info["name"]
        video_dir = run_out / "videos" if args.save_video else None
        episodes, summary = evaluate_checkpoint(
            info["checkpoint"],
            num_episodes=args.episodes,
            device=args.device,
            save_video=args.save_video,
            video_dir=video_dir,
            out_dir=run_out,
            run_name=info["name"],
            fair_env=fair_env,
        )
        print_summary(summary)
        summaries.append(summary)
        episodes_by_run[info["name"]] = episodes
        metrics_path = checkpoint_run_dir(info["checkpoint"]) / "metrics.jsonl"
        if metrics_path.exists():
            run_metrics[info["name"]] = metrics_path

    write_summary_csv(out_dir / "comparison_summary.csv", summaries)
    with (out_dir / "comparison_summary.json").open("w") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    plot_comparison(summaries, episodes_by_run, run_metrics, out_dir)
    print(f"✅ 对比完成: {out_dir}")


if __name__ == "__main__":
    main()
