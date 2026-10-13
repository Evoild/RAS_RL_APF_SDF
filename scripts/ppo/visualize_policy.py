from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig
from common import add_config_argument, add_environment_arguments, make_env, parse_args_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a Panda PPO policy with a reproducible task profile.")
    add_config_argument(parser)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ppo/best_success.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--stochastic", action="store_true")
    add_environment_arguments(parser)
    args, _ = parse_args_with_config(parser, sections=("environment", "evaluation"))
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def main() -> None:
    args = parse_args()
    env = make_env(args, seed=args.seed, evaluation=True)
    agent = PPOAgent(env.obs_dim, env.action_dim, PPOConfig(), device=args.device)
    agent.load(str(args.checkpoint))
    frame_dt = 1.0 / max(args.fps, 1e-6)

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            episode_return = 0.0
            step = 0
            any_collision = False
            min_clearance = float("inf")
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            while not done:
                started = time.time()
                action, _, _ = agent.act(obs, deterministic=not args.stochastic)
                action = np.clip(action * args.action_scale, -1.0, 1.0)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                any_collision |= bool(info["collision"])
                min_clearance = min(min_clearance, float(info["obstacle_clearance"]))
                env.render()
                if step % 10 == 0 or done:
                    print(
                        "step={:03d} reward={:8.3f} return={:9.3f} target_dist={:.4f} "
                        "clearance={:.4f} proximity_penalty={:.3f} collision_penalty={:.1f} "
                        "collision={} success={}".format(
                            step,
                            reward,
                            episode_return,
                            info["target_distance"],
                            info["obstacle_clearance"],
                            info["proximity_penalty"],
                            info["collision_penalty"],
                            info["collision"],
                            info["success"],
                        )
                    )
                elapsed = time.time() - started
                time.sleep(max(0.0, frame_dt - elapsed))
                step += 1
            print(
                f"episode_return={episode_return:.3f} success={info['success']} "
                f"collision_free={info['success'] and not any_collision} "
                f"any_collision={any_collision} min_clearance={min_clearance:.4f} "
                f"timeout={info['timeout']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
