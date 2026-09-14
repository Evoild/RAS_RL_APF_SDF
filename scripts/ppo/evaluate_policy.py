from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig
from envs import DEFAULT_SUCCESS_THRESHOLD, PandaObstacleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a seven-joint Panda reaching PPO policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ppo/best_success.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--joint-step-deg", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="both")
    return parser.parse_args()


def evaluate(args: argparse.Namespace, deterministic: bool) -> None:
    env = PandaObstacleEnv(
        randomize_reset=False,
        success_threshold=args.success_threshold,
        max_joint_step=np.deg2rad(args.joint_step_deg),
        max_steps=args.max_steps,
    )
    agent = PPOAgent(
        env.obs_dim,
        env.action_dim,
        PPOConfig(),
        device=args.device,
    )
    agent.load(str(args.checkpoint))

    returns = []
    successes = []
    collisions = []
    actions = []
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            episode_return = 0.0
            step = 0
            while not done:
                action, _, _ = agent.act(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                collisions.append(float(info["collision"]))
                actions.append(action.copy())
                step += 1
                if args.render:
                    env.render()
            returns.append(episode_return)
            successes.append(float(info["success"]))
            label = "deterministic" if deterministic else "stochastic"
            print(
                f"mode={label} episode={episode} return={episode_return:.2f} success={info['success']} steps={step} "
                f"target_distance={info['target_distance']:.4f} clearance={info['obstacle_clearance']:.4f} "
                f"collision={info['collision']}"
            )
    finally:
        env.close()

    action_saturation = float(np.mean(np.abs(np.asarray(actions)) >= 0.98)) if actions else 0.0
    label = "deterministic" if deterministic else "stochastic"
    print(
        f"summary mode={label} return={np.mean(returns):.2f} success_rate={np.mean(successes):.2f} "
        f"collision_rate={np.mean(collisions):.3f} action_saturation_rate={action_saturation:.3f}"
    )


def main() -> None:
    args = parse_args()
    modes = [args.mode] if args.mode != "both" else ["deterministic", "stochastic"]
    for mode in modes:
        evaluate(args, deterministic=(mode == "deterministic"))


if __name__ == "__main__":
    main()
