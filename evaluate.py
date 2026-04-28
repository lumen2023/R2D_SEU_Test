#!/usr/bin/env python3
"""Full checkpoint evaluation entrypoint.

This wrapper intentionally loads the saved `.hydra/config.yaml` from the
checkpoint run directory instead of composing the repository's current defaults.

Examples:
    python evaluate.py checkpoint=logdir/2026-04-23/10-45-06/checkpoints/final.pt num_episodes=10
    python evaluate.py checkpoint=... save_video=true video_dir=eval_results/r2/videos out=eval_results/r2
"""

import argparse
import pathlib
import sys

from eval_core import evaluate_checkpoint, print_summary


def _coerce_value(value):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def _split_hydra_style(argv):
    positional = []
    overrides = {}
    for item in argv:
        if "=" in item and not item.startswith("--"):
            key, value = item.split("=", 1)
            overrides[key.replace("-", "_")] = _coerce_value(value)
        else:
            positional.append(item)
    return positional, overrides


def parse_args(argv=None):
    positional, overrides = _split_hydra_style(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser(description="Evaluate a trained R2-Dreamer checkpoint")
    parser.add_argument("checkpoint_pos", nargs="?", help="Optional checkpoint path")
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint .pt file")
    parser.add_argument("--num-episodes", "-n", type=int, default=None, help="Number of vectorized eval episodes")
    parser.add_argument("--device", default=None, help="Override saved device, for example cpu or cuda:0")
    parser.add_argument("--save-video", action="store_true", help="Save rollout video when image observations exist")
    parser.add_argument("--video-dir", default=None, help="Directory for rollout videos")
    parser.add_argument("--out", default=None, help="Directory for episodes.jsonl, summary.csv/json, and plots")
    parser.add_argument("--run-name", default=None, help="Name written into summary outputs")
    args = parser.parse_args(positional)

    checkpoint = overrides.get("checkpoint") or args.checkpoint or args.checkpoint_pos
    num_episodes = overrides.get("num_episodes", args.num_episodes)
    save_video = bool(overrides.get("save_video", args.save_video))
    video_dir = overrides.get("video_dir", args.video_dir)
    out_dir = overrides.get("out", args.out)
    device = overrides.get("device", args.device)
    run_name = overrides.get("run_name", args.run_name)
    return checkpoint, int(num_episodes or 10), device, save_video, video_dir, out_dir, run_name


def main():
    checkpoint, num_episodes, device, save_video, video_dir, out_dir, run_name = parse_args()
    if not checkpoint:
        print("❌ 错误: 请通过 checkpoint=<path> 或 --checkpoint <path> 指定检查点文件")
        sys.exit(2)
    if save_video and video_dir is None:
        video_dir = str(pathlib.Path(out_dir or "eval_results") / "videos")

    try:
        _, summary = evaluate_checkpoint(
            checkpoint,
            num_episodes=num_episodes,
            device=device,
            save_video=save_video,
            video_dir=video_dir,
            out_dir=out_dir,
            run_name=run_name,
        )
        print_summary(summary)
    except Exception as exc:
        print(f"❌ 评估失败: {exc}")
        raise


if __name__ == "__main__":
    main()
