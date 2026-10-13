from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms import PPOAgent, PPOConfig
from common import add_config_argument, add_environment_arguments, make_env, parse_args_with_config


EPISODE_FIELDS = [
    "seed",
    "mode",
    "episode",
    "success",
    "collision_free_success",
    "timeout",
    "return",
    "steps",
    "min_clearance",
    "final_target_distance",
    "any_collision",
    "any_obstacle_collision",
    "action_saturation_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a static-obstacle Panda PPO policy.")
    add_config_argument(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10017])
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_environment_arguments(parser)
    args, config = parse_args_with_config(parser, sections=("environment", "evaluation"))
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if args.num_obstacles != 1:
        parser.error("static obstacle testing currently requires exactly one obstacle")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    args.source_config = config
    return args


def wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def run_case(
    agent: PPOAgent,
    args: argparse.Namespace,
    seed: int,
    mode: str,
) -> tuple[list[dict], dict]:
    torch.manual_seed(seed)
    env = make_env(args, seed=seed, evaluation=True)
    rows: list[dict] = []
    try:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            episode_return = 0.0
            min_clearance = float("inf")
            saturation_count = 0
            action_components = 0
            any_collision = False
            any_obstacle_collision = False
            while not done:
                action, _, _ = agent.act(obs, deterministic=(mode == "deterministic"))
                obs, reward, done, info = env.step(action)
                episode_return += reward
                min_clearance = min(min_clearance, float(info["obstacle_clearance"]))
                any_collision |= bool(info["collision"])
                any_obstacle_collision |= bool(info["obstacle_collision"])
                saturation_count += int(np.count_nonzero(np.abs(action) >= 0.98))
                action_components += action.size

            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "episode": episode,
                    "success": int(info["success"]),
                    "collision_free_success": int(info["success"] and not any_collision),
                    "timeout": int(info["timeout"]),
                    "return": episode_return,
                    "steps": env.step_count,
                    "min_clearance": min_clearance,
                    "final_target_distance": float(info["target_distance"]),
                    "any_collision": int(any_collision),
                    "any_obstacle_collision": int(any_obstacle_collision),
                    "action_saturation_rate": saturation_count / action_components,
                }
            )
    finally:
        env.close()

    collision_free_successes = sum(int(row["collision_free_success"]) for row in rows)
    ci_low, ci_high = wilson_interval(collision_free_successes, len(rows))
    summary = {
        "seed": seed,
        "mode": mode,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_free_success_rate": collision_free_successes / len(rows),
        "collision_free_success_ci_low": ci_low,
        "collision_free_success_ci_high": ci_high,
        "any_collision_rate": float(np.mean([row["any_collision"] for row in rows])),
        "obstacle_collision_rate": float(np.mean([row["any_obstacle_collision"] for row in rows])),
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_min_clearance": float(np.mean([row["min_clearance"] for row in rows])),
        "action_saturation_rate": float(np.mean([row["action_saturation_rate"] for row in rows])),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    env = make_env(args, seed=args.seeds[0], evaluation=True)
    obs_dim, action_dim = env.obs_dim, env.action_dim
    env.close()
    agent = PPOAgent(obs_dim, action_dim, PPOConfig(), device=args.device)
    agent.load(str(args.checkpoint))

    modes = ("deterministic", "stochastic") if args.mode == "both" else (args.mode,)
    episode_rows: list[dict] = []
    summaries: list[dict] = []
    for seed in args.seeds:
        for mode in modes:
            rows, summary = run_case(agent, args, seed, mode)
            episode_rows.extend(rows)
            summaries.append(summary)
            print(
                f"seed={seed} mode={mode} collision_free_success="
                f"{summary['collision_free_success_rate']:.3f} "
                f"any_collision={summary['any_collision_rate']:.3f} "
                f"min_clearance={summary['mean_min_clearance']:.4f} "
                f"return={summary['mean_return']:.2f}"
            )

    output_dir = args.output_dir or ROOT / "logs_ppo" / "obstacle_result_test" / args.checkpoint.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "episodes": args.episodes,
                "seeds": args.seeds,
                "mode": args.mode,
                "source_config": args.source_config,
            },
            file,
            indent=2,
        )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
