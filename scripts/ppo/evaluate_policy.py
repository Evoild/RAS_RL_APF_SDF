from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig
from common import add_config_argument, add_environment_arguments, make_env, parse_args_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Panda PPO policy with a reproducible task profile.")
    add_config_argument(parser)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ppo/best_success.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="both")
    add_environment_arguments(parser)
    args, _ = parse_args_with_config(parser, sections=("environment", "evaluation"))
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def evaluate(args: argparse.Namespace, deterministic: bool) -> None:
    env = make_env(args, seed=args.seed, evaluation=True)
    agent = PPOAgent(env.obs_dim, env.action_dim, PPOConfig(), device=args.device)
    agent.load(str(args.checkpoint))

    returns: list[float] = []
    successes: list[float] = []
    collision_free_successes: list[float] = []
    episode_collisions: list[float] = []
    minimum_clearances: list[float] = []
    actions: list[np.ndarray] = []
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            episode_return = 0.0
            step = 0
            any_collision = False
            min_clearance = float("inf")
            while not done:
                action, _, _ = agent.act(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                any_collision |= bool(info["collision"])
                min_clearance = min(min_clearance, float(info["obstacle_clearance"]))
                actions.append(action.copy())
                step += 1
                if args.render:
                    env.render()
            returns.append(episode_return)
            successes.append(float(info["success"]))
            collision_free_successes.append(float(info["success"] and not any_collision))
            episode_collisions.append(float(any_collision))
            minimum_clearances.append(min_clearance)
            label = "deterministic" if deterministic else "stochastic"
            print(
                f"mode={label} episode={episode} return={episode_return:.2f} "
                f"success={info['success']} collision_free={info['success'] and not any_collision} "
                f"steps={step} target_distance={info['target_distance']:.4f} "
                f"min_clearance={min_clearance:.4f} any_collision={any_collision}"
            )
    finally:
        env.close()

    action_saturation = float(np.mean(np.abs(np.asarray(actions)) >= 0.98)) if actions else 0.0
    label = "deterministic" if deterministic else "stochastic"
    print(
        f"summary mode={label} return={np.mean(returns):.2f} "
        f"success_rate={np.mean(successes):.3f} "
        f"collision_free_success_rate={np.mean(collision_free_successes):.3f} "
        f"any_collision_rate={np.mean(episode_collisions):.3f} "
        f"mean_min_clearance={np.mean(minimum_clearances):.4f} "
        f"action_saturation_rate={action_saturation:.3f}"
    )


def main() -> None:
    args = parse_args()
    modes = [args.mode] if args.mode != "both" else ["deterministic", "stochastic"]
    for mode in modes:
        evaluate(args, deterministic=(mode == "deterministic"))


if __name__ == "__main__":
    main()
