from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from envs import (
    DEFAULT_JOINT_STEP_DEG,
    DEFAULT_MEDIUM_JOINT_STEP_DEG,
    DEFAULT_MEDIUM_STEP_DISTANCE,
    DEFAULT_NEAR_JOINT_STEP_DEG,
    DEFAULT_NEAR_STEP_DISTANCE,
    DEFAULT_SUCCESS_THRESHOLD,
    PandaObstacleEnv,
)


CONFIG_SECTIONS = ("environment", "ppo", "training", "evaluation")


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="JSON experiment profile.")


def add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=("reaching", "static_obstacle"), default="reaching")
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--obstacle-radius", type=float, default=0.04)
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--distance-reward-scale", type=float, default=1.0)
    parser.add_argument("--success-reward", type=float, default=100.0)
    parser.add_argument("--timeout-penalty", type=float, default=-10.0)
    parser.add_argument("--proximity-margin", type=float, default=0.05)
    parser.add_argument("--proximity-penalty-scale", type=float, default=0.0)
    parser.add_argument("--collision-penalty", type=float, default=0.0)
    parser.add_argument("--terminate-on-collision", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--path-offset-range", type=float, default=0.08)
    parser.add_argument("--min-safety-dist", type=float, default=0.3)
    parser.add_argument("--obstacle-obs-noise-std", type=float, default=0.001)
    parser.add_argument("--obstacle-disturb-prob", type=float, default=0.0)
    parser.add_argument("--obstacle-disturb-step", type=float, default=0.02)
    parser.add_argument("--joint-step-deg", type=float, default=DEFAULT_JOINT_STEP_DEG)
    parser.add_argument("--medium-joint-step-deg", type=float, default=DEFAULT_MEDIUM_JOINT_STEP_DEG)
    parser.add_argument("--near-joint-step-deg", type=float, default=DEFAULT_NEAR_JOINT_STEP_DEG)
    parser.add_argument("--medium-step-distance", type=float, default=DEFAULT_MEDIUM_STEP_DISTANCE)
    parser.add_argument("--near-step-distance", type=float, default=DEFAULT_NEAR_STEP_DISTANCE)
    parser.add_argument("--adaptive-joint-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ee-goal-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obstacle-relative-observation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--randomize-reset", action=argparse.BooleanOptionalAction, default=False)


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    sections: Sequence[str] = CONFIG_SECTIONS,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    known, _ = bootstrap.parse_known_args(argv)
    config = load_config(known.config) if known.config is not None else {}
    flat_config = flatten_config(config, sections)
    valid_destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in flat_config.items() if key in valid_destinations})
    args = parser.parse_args(argv)
    return args, config


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a JSON object")
    return config


def flatten_config(
    config: dict[str, Any],
    sections: Sequence[str] = CONFIG_SECTIONS,
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section in sections:
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Config section '{section}' must be a JSON object")
        for key, value in values.items():
            if key in flat:
                raise ValueError(f"Duplicate config key: {key}")
            flat[key] = value
    if "profile" in config:
        flat["profile"] = config["profile"]
    return flat


def make_env(args: argparse.Namespace, *, seed: int, evaluation: bool = False) -> PandaObstacleEnv:
    return PandaObstacleEnv(
        num_obstacles=args.num_obstacles,
        obstacle_radius=args.obstacle_radius,
        success_threshold=args.success_threshold,
        distance_reward_scale=args.distance_reward_scale,
        success_reward=args.success_reward,
        timeout_penalty=args.timeout_penalty,
        obstacle_obs_noise_std=args.obstacle_obs_noise_std,
        collision_margin=0.01,
        terminate_on_collision=args.terminate_on_collision,
        proximity_margin=args.proximity_margin,
        proximity_penalty_scale=args.proximity_penalty_scale,
        collision_penalty=args.collision_penalty,
        path_offset_range=args.path_offset_range,
        min_safety_dist=args.min_safety_dist,
        obstacle_disturb_prob=args.obstacle_disturb_prob,
        obstacle_disturb_step=args.obstacle_disturb_step,
        max_joint_step=np.deg2rad(args.joint_step_deg),
        medium_joint_step=np.deg2rad(args.medium_joint_step_deg),
        near_joint_step=np.deg2rad(args.near_joint_step_deg),
        medium_step_distance=args.medium_step_distance,
        near_step_distance=args.near_step_distance,
        adaptive_joint_step=args.adaptive_joint_step,
        include_ee_goal_observation=args.ee_goal_observation,
        include_obstacle_relative_observation=args.obstacle_relative_observation,
        max_steps=args.max_steps,
        randomize_reset=False if evaluation else args.randomize_reset,
        seed=seed,
    )


def observation_mode(args: argparse.Namespace) -> str:
    base_dim = 10 + (6 if args.ee_goal_observation else 0)
    obstacle_dim = args.num_obstacles * (6 if args.obstacle_relative_observation else 3)
    return f"{base_dim + obstacle_dim}d"


def config_snapshot(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": config.get("profile"),
        "config_path": str(args.config) if args.config is not None else None,
        "resolved_arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_config": config,
    }
