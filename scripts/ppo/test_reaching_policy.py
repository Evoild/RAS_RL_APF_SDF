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
from envs import DEFAULT_JOINT_STEP_DEG, PandaObstacleEnv


DEFAULT_CHECKPOINT = ROOT / "checkpoints_ppo" / (
    "total-steps_4000000_rollout-steps_2048_success-threshold_0.02_"
    "joint-steps-deg_3-1-0.5_step-distances_0.1-0.03_"
    "joint-step-mode_fixed_grad-clip_shared_observation_ee-goal-16d_"
    "policy-lr_0.0002_epochs_10_target-kl_0.02_seed_17"
) / "best_success.pt"

EPISODE_FIELDS = [
    "threshold",
    "seed",
    "mode",
    "episode",
    "success",
    "timeout",
    "return",
    "steps",
    "min_target_distance",
    "final_target_distance",
    "any_collision",
    "any_ground_collision",
    "any_self_collision",
    "any_obstacle_collision",
    "action_saturation_rate",
]
SUMMARY_FIELDS = [
    "threshold",
    "seed",
    "mode",
    "episodes",
    "successes",
    "success_rate",
    "success_ci_low",
    "success_ci_high",
    "mean_return",
    "mean_steps",
    "mean_min_target_distance",
    "mean_final_target_distance",
    "any_collision_rate",
    "action_saturation_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible no-obstacle PPO reaching result test.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.02, 0.01, 0.005])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10017])
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--joint-step-deg", type=float, default=DEFAULT_JOINT_STEP_DEG)
    parser.add_argument("--adaptive-joint-step", action="store_true")
    parser.add_argument("--ee-goal-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")
    if not args.thresholds or any(not math.isfinite(x) or x <= 0 for x in args.thresholds):
        parser.error("thresholds must be finite positive values")
    if not math.isfinite(args.joint_step_deg) or args.joint_step_deg <= 0:
        parser.error("joint-step-deg must be a finite positive value")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
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
    threshold: float,
    seed: int,
    mode: str,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    torch.manual_seed(seed)
    env = PandaObstacleEnv(
        num_obstacles=0,
        success_threshold=threshold,
        seed=seed,
        max_steps=args.max_steps,
        max_joint_step=np.deg2rad(args.joint_step_deg),
        adaptive_joint_step=args.adaptive_joint_step,
        include_ee_goal_observation=args.ee_goal_observation,
        randomize_reset=False,
    )
    if (env.obs_dim, env.action_dim) != (agent.model.actor_mean[0].in_features, agent.model.log_std.numel()):
        env.close()
        raise ValueError("Checkpoint and observation mode have different dimensions")

    rows: list[dict[str, float | int | str]] = []
    try:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            episode_return = 0.0
            min_distance = float("inf")
            saturation_count = 0
            action_components = 0
            any_collision = False
            any_ground = False
            any_self = False
            any_obstacle = False
            while not done:
                action, _, _ = agent.act(obs, deterministic=(mode == "deterministic"))
                obs, reward, done, info = env.step(action)
                episode_return += reward
                min_distance = min(min_distance, float(info["target_distance"]))
                saturation_count += int(np.count_nonzero(np.abs(action) >= 0.98))
                action_components += action.size
                any_collision |= bool(info["collision"])
                any_ground |= bool(info["ground_collision"])
                any_self |= bool(info["self_collision"])
                any_obstacle |= bool(info["obstacle_collision"])

            rows.append(
                {
                    "threshold": threshold,
                    "seed": seed,
                    "mode": mode,
                    "episode": episode,
                    "success": int(info["success"]),
                    "timeout": int(info["timeout"]),
                    "return": episode_return,
                    "steps": env.step_count,
                    "min_target_distance": min_distance,
                    "final_target_distance": float(info["target_distance"]),
                    "any_collision": int(any_collision),
                    "any_ground_collision": int(any_ground),
                    "any_self_collision": int(any_self),
                    "any_obstacle_collision": int(any_obstacle),
                    "action_saturation_rate": saturation_count / action_components,
                }
            )
    finally:
        env.close()

    successes = sum(int(row["success"]) for row in rows)
    ci_low, ci_high = wilson_interval(successes, len(rows))
    summary: dict[str, float | int | str] = {
        "threshold": threshold,
        "seed": seed,
        "mode": mode,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_min_target_distance": float(np.mean([row["min_target_distance"] for row in rows])),
        "mean_final_target_distance": float(np.mean([row["final_target_distance"] for row in rows])),
        "any_collision_rate": float(np.mean([row["any_collision"] for row in rows])),
        "action_saturation_rate": float(np.mean([row["action_saturation_rate"] for row in rows])),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or ROOT / "logs_ppo" / "result_test" / args.checkpoint.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    obs_dim = 16 if args.ee_goal_observation else 10
    agent = PPOAgent(obs_dim, 7, PPOConfig(), device=args.device)
    agent.load(str(args.checkpoint))

    modes = ("deterministic", "stochastic") if args.mode == "both" else (args.mode,)
    episode_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []
    print("Threshold changes success reward and episode termination; results are separate evaluations.")
    for threshold in args.thresholds:
        for seed in args.seeds:
            for mode in modes:
                rows, summary = run_case(agent, args, threshold, seed, mode)
                episode_rows.extend(rows)
                summaries.append(summary)
                print(
                    f"threshold={threshold:.3f} seed={seed} mode={mode} "
                    f"success={summary['successes']}/{summary['episodes']} "
                    f"95%CI=[{summary['success_ci_low']:.3f},{summary['success_ci_high']:.3f}] "
                    f"return={summary['mean_return']:.2f} steps={summary['mean_steps']:.1f} "
                    f"min_dist={summary['mean_min_target_distance']:.4f} "
                    f"any_collision={summary['any_collision_rate']:.3f}"
                )

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump({**vars(args), "checkpoint": str(args.checkpoint)}, file, indent=2, default=str)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
