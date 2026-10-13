from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig
from envs import (
    DEFAULT_JOINT_STEP_DEG,
    DEFAULT_MEDIUM_JOINT_STEP_DEG,
    DEFAULT_MEDIUM_STEP_DISTANCE,
    DEFAULT_NEAR_JOINT_STEP_DEG,
    DEFAULT_NEAR_STEP_DISTANCE,
    DEFAULT_SUCCESS_THRESHOLD,
    PandaObstacleEnv,
)


REWARD_FIELDS = [
    "distance_reward",
    "reward_success",
    "timeout_penalty",
]
DISTANCE_FIELDS = ["target_distance", "ee_obstacle_distance"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record reward terms from a PPO best.pt rollout.")
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/evoild/program/RL+DRF/rl/checkpoints_ppo/total-steps_60000000_rollout-steps_4096_seed_17_run-007/last.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs_ppo/reward_test"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--obstacle-radius", type=float, default=0.04)
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD)
    parser.add_argument("--joint-step-deg", type=float, default=DEFAULT_JOINT_STEP_DEG)
    parser.add_argument("--medium-joint-step-deg", type=float, default=DEFAULT_MEDIUM_JOINT_STEP_DEG)
    parser.add_argument("--near-joint-step-deg", type=float, default=DEFAULT_NEAR_JOINT_STEP_DEG)
    parser.add_argument("--medium-step-distance", type=float, default=DEFAULT_MEDIUM_STEP_DISTANCE)
    parser.add_argument("--near-step-distance", type=float, default=DEFAULT_NEAR_STEP_DISTANCE)
    parser.add_argument("--adaptive-joint-step", action="store_true")
    parser.add_argument("--ee-goal-observation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def write_reward_txt(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "step",
        *REWARD_FIELDS,
        "reward_sum",
        "env_reward",
        "cumulative_reward",
        "target_distance",
        "ee_obstacle_distance",
        "min_obstacle_distance",
        "effective_joint_step_deg",
        "success",
        "collision",
        "timeout",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_rewards(path: Path, rows: list[dict[str, float]]) -> Path:
    if plt is None:
        svg_path = path.with_suffix(".svg")
        write_reward_svg(svg_path, rows)
        return svg_path

    steps = [row["step"] for row in rows]
    fig, ax_reward = plt.subplots(figsize=(10, 6))
    for field in REWARD_FIELDS:
        ax_reward.plot(steps, [row[field] for row in rows], label=field, linewidth=1.8)
    ax_reward.plot(steps, [row["reward_sum"] for row in rows], label="reward_sum", linewidth=2.2, color="black")
    ax_reward.set_xlabel("step")
    ax_reward.set_ylabel("step reward")
    ax_reward.grid(True, alpha=0.3)

    ax_return = ax_reward.twinx()
    ax_return.plot(
        steps,
        [row["cumulative_reward"] for row in rows],
        label="cumulative_reward",
        linewidth=2.2,
        color="#f97316",
        linestyle="--",
    )
    ax_return.set_ylabel("cumulative reward")

    lines, labels = ax_reward.get_legend_handles_labels()
    return_lines, return_labels = ax_return.get_legend_handles_labels()
    ax_reward.legend(lines + return_lines, labels + return_labels)
    plt.title("Reward terms over rollout steps")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_distances(path: Path, rows: list[dict[str, float]]) -> Path:
    if plt is None:
        svg_path = path.with_suffix(".svg")
        write_distance_svg(svg_path, rows)
        return svg_path

    steps = [row["step"] for row in rows]
    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        [row["target_distance"] for row in rows],
        label="ee_target_distance",
        linewidth=2.0,
        color="#2563eb",
    )
    plt.plot(
        steps,
        [row["ee_obstacle_distance"] for row in rows],
        label="ee_obstacle_distance",
        linewidth=2.0,
        color="#f97316",
    )
    plt.xlabel("step")
    plt.ylabel("distance")
    plt.title("Distances over rollout steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def write_reward_svg(path: Path, rows: list[dict[str, float]]) -> None:
    width = 1000
    height = 600
    left = 70
    right = 170
    top = 40
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    series = [*REWARD_FIELDS, "reward_sum"]
    cumulative_series = "cumulative_reward"
    colors = {
        "distance_reward": "#2563eb",
        "reward_success": "#16a34a",
        "timeout_penalty": "#dc2626",
        "reward_sum": "#111827",
        "cumulative_reward": "#f97316",
    }
    steps = [row["step"] for row in rows]
    values = [row[field] for row in rows for field in series]
    cumulative_values = [row[cumulative_series] for row in rows]
    min_step = min(steps)
    max_step = max(steps)
    min_value = min(values)
    max_value = max(values)
    min_cumulative = min(cumulative_values)
    max_cumulative = max(cumulative_values)
    if min_step == max_step:
        max_step += 1
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    if min_cumulative == max_cumulative:
        min_cumulative -= 1.0
        max_cumulative += 1.0

    def x_pos(step: float) -> float:
        return left + (step - min_step) / (max_step - min_step) * plot_width

    def y_pos(value: float) -> float:
        return top + (max_value - value) / (max_value - min_value) * plot_height

    def cumulative_y_pos(value: float) -> float:
        return top + (max_cumulative - value) / (max_cumulative - min_cumulative) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">Reward terms over rollout steps</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151"/>',
        f'<line x1="{width - right}" y1="{top}" x2="{width - right}" y2="{height - bottom}" stroke="#f97316"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">step</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">step reward</text>',
        f'<text x="{width - 18}" y="{height / 2}" transform="rotate(90 {width - 18} {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#f97316">cumulative reward</text>',
    ]
    for frac in np.linspace(0.0, 1.0, 6):
        y = top + frac * plot_height
        value = max_value - frac * (max_value - min_value)
        cumulative_value = max_cumulative - frac * (max_cumulative - min_cumulative)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11" fill="#4b5563">{value:.3f}</text>'
        )
        lines.append(
            f'<text x="{width - right + 8}" y="{y + 4:.2f}" text-anchor="start" font-family="sans-serif" '
            f'font-size="11" fill="#f97316">{cumulative_value:.3f}</text>'
        )
    for index, field in enumerate(series):
        points = " ".join(f'{x_pos(row["step"]):.2f},{y_pos(row[field]):.2f}' for row in rows)
        stroke_width = 2.4 if field == "reward_sum" else 1.8
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[field]}" stroke-width="{stroke_width}"/>'
        )
        legend_y = top + 20 + index * 24
        lines.append(
            f'<line x1="{width - right + 25}" y1="{legend_y}" x2="{width - right + 55}" '
            f'y2="{legend_y}" stroke="{colors[field]}" stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'<text x="{width - right + 62}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{field}</text>'
        )
    cumulative_points = " ".join(
        f'{x_pos(row["step"]):.2f},{cumulative_y_pos(row[cumulative_series]):.2f}' for row in rows
    )
    lines.append(
        f'<polyline points="{cumulative_points}" fill="none" stroke="{colors[cumulative_series]}" '
        f'stroke-width="2.2" stroke-dasharray="7 5"/>'
    )
    legend_y = top + 20 + len(series) * 24
    lines.append(
        f'<line x1="{width - right + 25}" y1="{legend_y}" x2="{width - right + 55}" '
        f'y2="{legend_y}" stroke="{colors[cumulative_series]}" stroke-width="2.2" stroke-dasharray="7 5"/>'
    )
    lines.append(
        f'<text x="{width - right + 62}" y="{legend_y + 4}" font-family="sans-serif" '
        f'font-size="12">{cumulative_series}</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_distance_svg(path: Path, rows: list[dict[str, float]]) -> None:
    width = 1000
    height = 600
    left = 70
    right = 210
    top = 40
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    labels = {
        "target_distance": "ee_target_distance",
        "ee_obstacle_distance": "ee_obstacle_distance",
    }
    colors = {
        "target_distance": "#2563eb",
        "ee_obstacle_distance": "#f97316",
    }
    steps = [row["step"] for row in rows]
    values = [row[field] for row in rows for field in DISTANCE_FIELDS if np.isfinite(row[field])]
    if not values:
        values = [0.0, 1.0]
    min_step = min(steps)
    max_step = max(steps)
    min_value = min(values)
    max_value = max(values)
    if min_step == max_step:
        max_step += 1
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    def x_pos(step: float) -> float:
        return left + (step - min_step) / (max_step - min_step) * plot_width

    def y_pos(value: float) -> float:
        return top + (max_value - value) / (max_value - min_value) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">Distances over rollout steps</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">step</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">distance</text>',
    ]
    for frac in np.linspace(0.0, 1.0, 6):
        y = top + frac * plot_height
        value = max_value - frac * (max_value - min_value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11" fill="#4b5563">{value:.3f}</text>'
        )
    for index, field in enumerate(DISTANCE_FIELDS):
        finite_rows = [row for row in rows if np.isfinite(row[field])]
        if not finite_rows:
            continue
        points = " ".join(f'{x_pos(row["step"]):.2f},{y_pos(row[field]):.2f}' for row in finite_rows)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{colors[field]}" stroke-width="2.0"/>')
        legend_y = top + 20 + index * 24
        lines.append(
            f'<line x1="{width - right + 25}" y1="{legend_y}" x2="{width - right + 55}" '
            f'y2="{legend_y}" stroke="{colors[field]}" stroke-width="2.0"/>'
        )
        lines.append(
            f'<text x="{width - right + 62}" y="{legend_y + 4}" font-family="sans-serif" '
            f'font-size="12">{labels[field]}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = PandaObstacleEnv(
        num_obstacles=args.num_obstacles,
        obstacle_radius=args.obstacle_radius,
        success_threshold=args.success_threshold,
        max_joint_step=np.deg2rad(args.joint_step_deg),
        medium_joint_step=np.deg2rad(args.medium_joint_step_deg),
        near_joint_step=np.deg2rad(args.near_joint_step_deg),
        medium_step_distance=args.medium_step_distance,
        near_step_distance=args.near_step_distance,
        adaptive_joint_step=args.adaptive_joint_step,
        include_ee_goal_observation=args.ee_goal_observation,
        max_steps=args.max_steps,
        randomize_reset=False,
    )
    agent = PPOAgent(
        env.obs_dim,
        env.action_dim,
        PPOConfig(),
        device=args.device,
    )
    agent.load(str(args.checkpoint))

    rows: list[dict[str, float]] = []
    obs = env.reset()
    done = False
    step = 0
    episode_return = 0.0
    try:
        while not done:
            action, _, _ = agent.act(obs, deterministic=not args.stochastic)
            action = np.clip(action * args.action_scale, -1.0, 1.0)
            obs, reward, done, info = env.step(action)
            step += 1
            episode_return += reward
            reward_terms = {field: float(info[field]) for field in REWARD_FIELDS}
            reward_sum = sum(reward_terms.values())
            ee_obstacle_distance = (
                float(np.linalg.norm(env.data.site_xpos[env.ee_site_id] - env.obstacles[0]))
                if env.num_obstacles
                else float("nan")
            )
            rows.append(
                {
                    "step": step,
                    **reward_terms,
                    "reward_sum": reward_sum,
                    "env_reward": float(reward),
                    "cumulative_reward": float(episode_return),
                    "target_distance": float(info["target_distance"]),
                    "ee_obstacle_distance": ee_obstacle_distance,
                    "min_obstacle_distance": float(info["min_obstacle_distance"]),
                    "effective_joint_step_deg": float(info["effective_joint_step_deg"]),
                    "success": float(info["success"]),
                    "collision": float(info["collision"]),
                    "timeout": float(info["timeout"]),
                }
            )
            if args.render:
                env.render()
    finally:
        env.close()

    txt_path = args.output_dir / "reward_terms_best.txt"
    plot_path = plot_rewards(args.output_dir / "reward_terms_best.png", rows)
    distance_plot_path = plot_distances(args.output_dir / "distance_terms_best.png", rows)
    write_reward_txt(txt_path, rows)

    last = rows[-1]
    print(
        f"steps={step} return={episode_return:.3f} success={bool(last['success'])} "
        f"collision={bool(last['collision'])} timeout={bool(last['timeout'])} "
        f"txt={txt_path} reward_plot={plot_path} distance_plot={distance_plot_path}"
    )


if __name__ == "__main__":
    main()
