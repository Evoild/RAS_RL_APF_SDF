from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from tqdm import trange
except ImportError:
    def trange(start, stop, step, desc=None, dynamic_ncols=None):
        return range(start, stop, step)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig, RolloutBuffer
from common import (
    add_config_argument,
    add_environment_arguments,
    config_snapshot,
    make_env as make_configured_env,
    observation_mode,
    parse_args_with_config,
)


def parse_args() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description="Train PPO for seven-joint Panda reaching and obstacle avoidance.")
    add_config_argument(parser)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=4_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--policy-lr", type=float, default=2e-4)
    parser.add_argument("--value-lr", type=float, default=2e-4)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--separate-grad-clip", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_ppo"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs_ppo"))
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-hist-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every-steps", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    add_environment_arguments(parser)
    args, config = parse_args_with_config(parser)
    if args.init_checkpoint is not None:
        args.init_checkpoint = Path(args.init_checkpoint)
        if not args.init_checkpoint.is_file():
            parser.error(f"initialization checkpoint does not exist: {args.init_checkpoint}")
    if args.task == "reaching" and args.num_obstacles != 0:
        parser.error("task=reaching requires --num-obstacles 0")
    if args.task == "static_obstacle" and args.num_obstacles != 1:
        parser.error("the current physical static-obstacle stage requires --num-obstacles 1")
    return args, config


def make_env(args: argparse.Namespace):
    return make_configured_env(args, seed=args.seed)


def make_eval_env(args: argparse.Namespace):
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 10_000
    return make_configured_env(args, seed=eval_seed, evaluation=True)


def base_run_name(args: argparse.Namespace) -> str:
    success_threshold = format(args.success_threshold, ".6g")
    policy_lr = format(args.policy_lr, ".6g")
    target_kl = format(args.target_kl, ".6g")
    joint_step_mode = "adaptive" if args.adaptive_joint_step else "fixed"
    grad_clip_mode = "separate" if args.separate_grad_clip else "shared"
    task = args.task.replace("_", "-")
    obs_mode = observation_mode(args)
    return (
        f"{task}"
        f"_nobs{args.num_obstacles}"
        f"_r{format(args.obstacle_radius, '.6g')}"
        f"_obs{obs_mode.removesuffix('d')}"
        f"_{joint_step_mode}{format(args.joint_step_deg, '.6g')}deg"
        f"_grad{grad_clip_mode}"
        f"_s{success_threshold}"
        f"_p{format(args.proximity_penalty_scale, '.6g')}"
        f"_c{format(args.collision_penalty, '.6g')}"
        f"_steps{args.total_steps}"
        f"_roll{args.rollout_steps}"
        f"_lr{policy_lr}"
        f"_ep{args.train_epochs}"
        f"_kl{target_kl}"
        f"_seed{args.seed}"
    )


def resolve_run_dirs(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    tensorboard_root = args.tensorboard_dir if args.tensorboard_dir is not None else args.log_dir / "tensorboard"
    run_name = base_run_name(args)
    run_index = 2
    while True:
        tensorboard_dir = tensorboard_root / run_name
        checkpoint_run_dir = args.checkpoint_dir / run_name
        log_run_dir = args.log_dir / "runs" / run_name
        tensorboard_exists = tensorboard_dir.exists() and any(tensorboard_dir.iterdir())
        checkpoint_exists = checkpoint_run_dir.exists() and any(checkpoint_run_dir.iterdir())
        log_exists = log_run_dir.exists() and any(log_run_dir.iterdir())
        if not tensorboard_exists and not checkpoint_exists and not log_exists:
            return run_name, tensorboard_dir, checkpoint_run_dir, log_run_dir
        run_name = f"{base_run_name(args)}_run-{run_index:03d}"
        run_index += 1


def make_tensorboard_writer(args: argparse.Namespace, tensorboard_dir: Path):
    if args.no_tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard is not available. Install it with: python3 -m pip install tensorboard")
        return None

    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log_dir: {tensorboard_dir}")
    return SummaryWriter(log_dir=str(tensorboard_dir))


def write_tensorboard(
    writer,
    agent: PPOAgent,
    log: dict[str, float],
    action_batch: list[np.ndarray],
    obs: np.ndarray,
    update_index: int,
    hist_every: int,
) -> None:
    if writer is None:
        return

    step = int(log["steps"])
    scalar_groups = {
        "rollout": [
            "mean_return",
            "mean_episode_len",
            "success_rate",
            "collision_free_success_rate",
            "collision_rate",
            "ground_collision_rate",
            "self_collision_rate",
            "obstacle_collision_rate",
            "target_distance",
            "min_obstacle_distance",
            "mean_joint_step_deg",
            "proximity_penalty",
            "collision_penalty",
        ],
        "ppo": [
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "max_approx_kl",
            "clip_fraction",
            "update_epochs",
            "kl_early_stop",
            "actor_grad_norm",
            "critic_grad_norm",
        ],
        "policy": ["action_saturation_rate", "policy_std_mean"],
    }
    for group, fields in scalar_groups.items():
        for field in fields:
            value = log.get(field)
            if value is not None and np.isfinite(value):
                writer.add_scalar(f"{group}/{field}", float(value), step)

    if action_batch:
        actions = np.asarray(action_batch, dtype=np.float32)
        for index in range(actions.shape[1]):
            writer.add_scalar(f"actions/mean_{index:02d}", float(np.mean(actions[:, index])), step)
            writer.add_scalar(f"actions/std_{index:02d}", float(np.std(actions[:, index])), step)
            writer.add_scalar(f"actions/abs_mean_{index:02d}", float(np.mean(np.abs(actions[:, index]))), step)

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        mean = agent.model.action_mean(obs_tensor).squeeze(0).detach().cpu().numpy()
        log_std = torch.clamp(agent.model.log_std, agent.config.log_std_min, agent.config.log_std_max)
        std = torch.exp(log_std).detach().cpu().numpy()
    for index, value in enumerate(mean):
        writer.add_scalar(f"policy/mean_{index:02d}", float(value), step)
    for index, value in enumerate(std):
        writer.add_scalar(f"policy/std_{index:02d}", float(value), step)

    if hist_every > 0 and update_index % hist_every == 0:
        for name, parameter in agent.model.named_parameters():
            writer.add_histogram(f"parameters/{name}", parameter.detach().cpu(), step)
            if parameter.grad is not None:
                writer.add_histogram(f"gradients/{name}", parameter.grad.detach().cpu(), step)
    writer.flush()


def action_saturation_rate(actions: list[np.ndarray], threshold: float = 0.98) -> float:
    if not actions:
        return 0.0
    action_array = np.asarray(actions, dtype=np.float32)
    return float(np.mean(np.abs(action_array) >= threshold))


def policy_std_mean(agent: PPOAgent) -> float:
    with torch.no_grad():
        log_std = torch.clamp(agent.model.log_std, agent.config.log_std_min, agent.config.log_std_max)
        return float(torch.exp(log_std).mean().item())


def evaluate_agent(agent: PPOAgent, args: argparse.Namespace, deterministic: bool) -> dict[str, float]:
    env = make_eval_env(args)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    episode_collisions: list[float] = []
    step_collisions: list[float] = []
    actions: list[np.ndarray] = []
    final_target_distances: list[float] = []
    minimum_obstacle_distances: list[float] = []
    collision_free_successes: list[float] = []

    try:
        for _ in range(args.eval_episodes):
            obs = env.reset()
            done = False
            episode_return = 0.0
            episode_len = 0
            final_info = None
            any_collision = False
            min_obstacle_distance = float("inf")
            while not done:
                action, _, _ = agent.act(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                episode_len += 1
                step_collisions.append(float(info["collision"]))
                any_collision |= bool(info["collision"])
                min_obstacle_distance = min(min_obstacle_distance, float(info["min_obstacle_distance"]))
                actions.append(action.copy())
                final_info = info

            if final_info is None:
                continue
            returns.append(episode_return)
            lengths.append(episode_len)
            successes.append(float(final_info["success"]))
            episode_collisions.append(float(any_collision))
            collision_free_successes.append(float(final_info["success"] and not any_collision))
            final_target_distances.append(float(final_info["target_distance"]))
            minimum_obstacle_distances.append(min_obstacle_distance)
    finally:
        env.close()

    prefix = "eval_det" if deterministic else "eval_stoch"
    return {
        f"{prefix}_mean_return": float(np.mean(returns)) if returns else float("nan"),
        f"{prefix}_mean_episode_len": float(np.mean(lengths)) if lengths else float("nan"),
        f"{prefix}_success_rate": float(np.mean(successes)) if successes else 0.0,
        f"{prefix}_collision_free_success_rate": (
            float(np.mean(collision_free_successes)) if collision_free_successes else 0.0
        ),
        f"{prefix}_episode_collision_rate": float(np.mean(episode_collisions)) if episode_collisions else 0.0,
        f"{prefix}_collision_rate": float(np.mean(step_collisions)) if step_collisions else 0.0,
        f"{prefix}_action_saturation_rate": action_saturation_rate(actions),
        f"{prefix}_target_distance": float(np.mean(final_target_distances)) if final_target_distances else float("nan"),
        f"{prefix}_min_obstacle_distance": (
            float(np.mean(minimum_obstacle_distances)) if minimum_obstacle_distances else float("nan")
        ),
    }


def write_eval_tensorboard(writer, eval_metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in eval_metrics.items():
        if np.isfinite(value):
            writer.add_scalar(f"eval/{key.removeprefix('eval_')}", value, step)
    writer.flush()


def main() -> None:
    args, source_config = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name, tensorboard_dir, checkpoint_run_dir, log_run_dir = resolve_run_dirs(args)
    env = make_env(args)
    cfg = PPOConfig(
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        train_epochs=args.train_epochs,
        target_kl=args.target_kl,
        separate_grad_clip=args.separate_grad_clip,
    )
    agent = PPOAgent(
        env.obs_dim,
        env.action_dim,
        cfg,
        device=args.device,
    )
    if args.init_checkpoint is not None:
        migration = agent.initialize_from(str(args.init_checkpoint))
        print(f"Initialized checkpoint: {args.init_checkpoint} ({migration})")
    buffer = RolloutBuffer()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
    log_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {checkpoint_run_dir}")
    print(f"Log dir: {log_run_dir}")
    csv_path = log_run_dir / "metrics.csv"
    jsonl_path = log_run_dir / "metrics.jsonl"
    with (log_run_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config_snapshot(args, source_config), config_file, ensure_ascii=False, indent=2)
    log_fields = [
        "update",
        "steps",
        "episodes",
        "mean_return",
        "mean_episode_len",
        "success_rate",
        "collision_free_success_rate",
        "collision_rate",
        "ground_collision_rate",
        "self_collision_rate",
        "obstacle_collision_rate",
        "target_distance",
        "min_obstacle_distance",
        "mean_joint_step_deg",
        "proximity_penalty",
        "collision_penalty",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "max_approx_kl",
        "clip_fraction",
        "update_epochs",
        "kl_early_stop",
        "actor_grad_norm",
        "critic_grad_norm",
        "action_saturation_rate",
        "policy_std_mean",
    ]
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    jsonl_file = jsonl_path.open("w", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=log_fields)
    writer.writeheader()
    tb_writer = make_tensorboard_writer(args, tensorboard_dir)
    if tb_writer is not None:
        tb_writer.add_text("config/run_name", run_name, 0)
        tb_writer.add_text("config/checkpoint_dir", str(checkpoint_run_dir), 0)
        tb_writer.add_text("config/args", json.dumps(vars(args), default=str, indent=2), 0)
        tb_writer.add_text("config/ppo", json.dumps(cfg.__dict__, indent=2), 0)

    obs = env.reset()
    episode_return = 0.0
    episode_len = 0
    episode_any_collision = False
    completed_episodes = 0
    best_train_success_rate = -float("inf")
    best_train_return = -float("inf")
    best_train_success_return = -float("inf")
    best_eval_success_rate = -float("inf")
    best_eval_return = -float("inf")
    best_eval_success_return = -float("inf")
    next_eval_step = args.eval_every_steps if args.eval_every_steps > 0 else None

    try:
        progress = trange(0, args.total_steps, args.rollout_steps, desc="seven-joint-ppo", dynamic_ncols=True)
        for global_step in progress:
            rollout_returns: list[float] = []
            rollout_lengths: list[int] = []
            rollout_successes: list[float] = []
            rollout_collision_free_successes: list[float] = []
            rollout_collisions: list[float] = []
            rollout_ground_collisions: list[float] = []
            rollout_self_collisions: list[float] = []
            rollout_obstacle_collisions: list[float] = []
            rollout_actions: list[np.ndarray] = []
            rollout_joint_steps: list[float] = []
            rollout_proximity_penalties: list[float] = []
            rollout_collision_penalties: list[float] = []

            for _ in range(args.rollout_steps):
                action, log_prob, value = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                buffer.add(obs, action, log_prob, reward, done, value)
                obs = next_obs
                episode_return += reward
                episode_len += 1
                episode_any_collision |= bool(info["collision"])
                rollout_actions.append(action.copy())
                rollout_collisions.append(float(info["collision"]))
                rollout_ground_collisions.append(float(info.get("ground_collision", False)))
                rollout_self_collisions.append(float(info.get("self_collision", False)))
                rollout_obstacle_collisions.append(float(info.get("obstacle_collision", False)))
                rollout_joint_steps.append(float(info["effective_joint_step_deg"]))
                rollout_proximity_penalties.append(float(info["proximity_penalty"]))
                rollout_collision_penalties.append(float(info["collision_penalty"]))

                if done:
                    rollout_returns.append(episode_return)
                    rollout_lengths.append(episode_len)
                    rollout_successes.append(float(info["success"]))
                    rollout_collision_free_successes.append(float(info["success"] and not episode_any_collision))
                    completed_episodes += 1
                    obs = env.reset()
                    episode_return = 0.0
                    episode_len = 0
                    episode_any_collision = False

            _, _, last_value = agent.act(obs, deterministic=True)
            metrics = agent.update(buffer, last_value)
            buffer.clear()

            mean_return = float(np.mean(rollout_returns)) if rollout_returns else float("nan")
            mean_episode_len = float(np.mean(rollout_lengths)) if rollout_lengths else float("nan")
            mean_success = float(np.mean(rollout_successes)) if rollout_successes else 0.0
            mean_collision_free_success = (
                float(np.mean(rollout_collision_free_successes)) if rollout_collision_free_successes else 0.0
            )
            mean_collision = float(np.mean(rollout_collisions)) if rollout_collisions else 0.0
            mean_ground_collision = float(np.mean(rollout_ground_collisions)) if rollout_ground_collisions else 0.0
            mean_self_collision = float(np.mean(rollout_self_collisions)) if rollout_self_collisions else 0.0
            mean_obstacle_collision = float(np.mean(rollout_obstacle_collisions)) if rollout_obstacle_collisions else 0.0
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    ret=f"{mean_return:.2f}",
                    succ=f"{mean_success:.2f}",
                    coll=f"{mean_collision:.2f}",
                    dist=f"{info['target_distance']:.3f}",
                )

            update_index = global_step // args.rollout_steps
            log = {
                "update": update_index,
                "steps": global_step + args.rollout_steps,
                "episodes": completed_episodes,
                "mean_return": mean_return,
                "mean_episode_len": mean_episode_len,
                "success_rate": mean_success,
                "collision_free_success_rate": mean_collision_free_success,
                "collision_rate": mean_collision,
                "ground_collision_rate": mean_ground_collision,
                "self_collision_rate": mean_self_collision,
                "obstacle_collision_rate": mean_obstacle_collision,
                "target_distance": float(info["target_distance"]),
                "min_obstacle_distance": float(info["min_obstacle_distance"]),
                "mean_joint_step_deg": float(np.mean(rollout_joint_steps)),
                "proximity_penalty": float(np.mean(rollout_proximity_penalties)),
                "collision_penalty": float(np.mean(rollout_collision_penalties)),
                "action_saturation_rate": action_saturation_rate(rollout_actions),
                "policy_std_mean": policy_std_mean(agent),
                **metrics,
            }
            writer.writerow(log)
            csv_file.flush()
            jsonl_file.write(json.dumps(log, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            write_tensorboard(
                tb_writer,
                agent,
                log,
                rollout_actions,
                obs,
                update_index,
                args.tensorboard_hist_every,
            )

            if update_index % args.log_every == 0:
                print(json.dumps(log, ensure_ascii=False))

            train_selection_success = (
                mean_collision_free_success if args.task == "static_obstacle" else mean_success
            )
            if rollout_returns and (
                train_selection_success > best_train_success_rate
                or (
                    train_selection_success == best_train_success_rate
                    and mean_return > best_train_success_return
                )
            ):
                best_train_success_rate = train_selection_success
                best_train_success_return = mean_return
                agent.save(str(checkpoint_run_dir / "best_train_success.pt"))

            if rollout_returns and mean_return > best_train_return:
                best_train_return = mean_return
                agent.save(str(checkpoint_run_dir / "best_train_return.pt"))

            current_steps = int(log["steps"])
            if next_eval_step is not None and current_steps >= next_eval_step:
                eval_metrics = {
                    **evaluate_agent(agent, args, deterministic=True),
                    **evaluate_agent(agent, args, deterministic=False),
                }
                eval_log = {"steps": current_steps, "checkpoint": str(checkpoint_run_dir), **eval_metrics}
                print(json.dumps(eval_log, ensure_ascii=False))
                write_eval_tensorboard(tb_writer, eval_metrics, current_steps)

                eval_success_rate = eval_metrics[
                    "eval_det_collision_free_success_rate"
                    if args.task == "static_obstacle"
                    else "eval_det_success_rate"
                ]
                eval_return = eval_metrics["eval_det_mean_return"]
                if (
                    eval_success_rate > best_eval_success_rate
                    or (
                        eval_success_rate == best_eval_success_rate
                        and eval_return > best_eval_success_return
                    )
                ):
                    best_eval_success_rate = eval_success_rate
                    best_eval_success_return = eval_return
                    agent.save(str(checkpoint_run_dir / "best_success.pt"))

                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    agent.save(str(checkpoint_run_dir / "best_return.pt"))

                while next_eval_step is not None and next_eval_step <= current_steps:
                    next_eval_step += args.eval_every_steps
    finally:
        agent.save(str(checkpoint_run_dir / "final.pt"))
        csv_file.close()
        jsonl_file.close()
        if tb_writer is not None:
            tb_writer.close()
        env.close()


if __name__ == "__main__":
    main()
