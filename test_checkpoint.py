#!/usr/bin/env python3
"""Evaluate a trained checkpoint with its saved Hydra config.

Examples:
    python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt 5
    python test_checkpoint.py logdir/2026-04-23/10-45-06/checkpoints/final.pt --num-episodes 10 --out eval_results/r2
"""

import argparse
import pathlib
import sys

from eval_core import evaluate_checkpoint, print_summary


def parse_args():
    parser = argparse.ArgumentParser(description="R2-Dreamer checkpoint evaluator")
    parser.add_argument("checkpoint", help="Path to a checkpoint .pt file")
    parser.add_argument("episodes", nargs="?", type=int, help="Backward-compatible positional episode count")
    parser.add_argument("--num-episodes", "-n", type=int, default=None, help="Number of vectorized eval episodes")
    parser.add_argument("--device", default=None, help="Override saved device, for example cpu or cuda:0")
    parser.add_argument("--save-video", action="store_true", help="Save rollout video when image observations exist")
    parser.add_argument("--video-dir", default=None, help="Directory for rollout videos")
    parser.add_argument("--out", default=None, help="Directory for episodes.jsonl, summary.csv/json, and plots")
    parser.add_argument("--run-name", default=None, help="Name written into summary outputs")
    return parser.parse_args()


def test_checkpoint(checkpoint_path, num_episodes=5, device=None, save_video=False, video_dir=None, out_dir=None):
    episodes, summary = evaluate_checkpoint(
        checkpoint_path,
        num_episodes=num_episodes,
        device=device,
        save_video=save_video,
        video_dir=video_dir,
        out_dir=out_dir,
    )
    print_summary(summary)
    return summary


def main():
    args = parse_args()
    checkpoint = pathlib.Path(args.checkpoint)
    num_episodes = args.num_episodes if args.num_episodes is not None else args.episodes
    num_episodes = int(num_episodes or 5)
    video_dir = args.video_dir
    if args.save_video and video_dir is None:
        video_dir = str(pathlib.Path(args.out or "eval_results") / "videos")

    try:
        _, summary = evaluate_checkpoint(
            checkpoint,
            num_episodes=num_episodes,
            device=args.device,
            save_video=args.save_video,
            video_dir=video_dir,
            out_dir=args.out,
            run_name=args.run_name,
        )
        print_summary(summary)
    except Exception as exc:
        print(f"❌ 评估失败: {exc}")
        raise


if __name__ == "__main__":
    main()
