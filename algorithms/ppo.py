from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


LOG_PROB_EPS = 1e-6


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 2e-4
    value_lr: float = 2e-4
    train_epochs: int = 10
    minibatch_size: int = 2048
    target_kl: float | None = 0.02
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    separate_grad_clip: bool = False
    actor_hidden_sizes: tuple[int, ...] = (256, 128)
    critic_hidden_sizes: tuple[int, ...] = (256, 128)
    log_std_init: float = -0.7
    log_std_min: float = -5.0
    log_std_max: float = 0.0


def make_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_size))
        layers.append(nn.ReLU())
        last_dim = hidden_size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_sizes: Sequence[int],
        critic_hidden_sizes: Sequence[int],
        log_std_init: float,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.actor_mean = make_mlp(obs_dim, actor_hidden_sizes, action_dim)
        final_actor_layer = self.actor_mean[-1]
        nn.init.zeros_(final_actor_layer.weight)
        nn.init.zeros_(final_actor_layer.bias)
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), log_std_init))
        self.critic = make_mlp(obs_dim, critic_hidden_sizes, 1)

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(obs)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.action_mean(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []

    def add(self, obs, action: np.ndarray, log_prob, reward, done, value) -> None:
        self.observations.append(obs)
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.model = ActorCritic(
            obs_dim,
            action_dim,
            config.actor_hidden_sizes,
            config.critic_hidden_sizes,
            config.log_std_init,
            config.log_std_min,
            config.log_std_max,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.actor_mean.parameters(), "lr": config.policy_lr},
                {"params": [self.model.log_std], "lr": config.policy_lr},
                {"params": self.model.critic.parameters(), "lr": config.value_lr},
            ]
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.model.distribution(obs_tensor)
        raw_action = dist.mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        value = self.model.value(obs_tensor)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict[str, float]:
        cfg = self.config
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.asarray(buffer.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + cfg.gamma * values[step + 1] * nonterminal - values[step]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_np = np.nan_to_num(np.asarray(buffer.observations), nan=0.0, posinf=1e3, neginf=-1e3)
        actions_np = np.clip(np.nan_to_num(np.asarray(buffer.actions), nan=0.0), -1.0 + LOG_PROB_EPS, 1.0 - LOG_PROB_EPS).astype(np.float32)
        log_probs_np = np.nan_to_num(np.asarray(buffer.log_probs), nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(log_probs_np, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        indices = np.arange(len(buffer))
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
        }
        actor_params = list(self.model.actor_mean.parameters()) + [self.model.log_std]
        critic_params = list(self.model.critic.parameters())
        updates = 0
        epochs_used = 0
        max_approx_kl = 0.0
        kl_early_stop = False
        for epoch in range(cfg.train_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), cfg.minibatch_size):
                batch = indices[start : start + cfg.minibatch_size]
                dist = self.model.distribution(obs[batch])
                raw_actions = self._atanh(actions[batch])
                log_probs = self._squashed_log_prob(dist, raw_actions, actions[batch])
                log_ratio = log_probs - old_log_probs[batch]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages_t[batch]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages_t[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()

                values_pred = self.model.value(obs[batch])
                value_loss = nn.functional.mse_loss(values_pred, returns_t[batch])
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                if not torch.isfinite(loss):
                    continue

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean()
                approx_kl_value = float(approx_kl.item())
                max_approx_kl = max(max_approx_kl, approx_kl_value)
                if cfg.target_kl is not None and approx_kl_value > cfg.target_kl:
                    kl_early_stop = True
                    break

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad_norm = self._grad_norm(actor_params)
                critic_grad_norm = self._grad_norm(critic_params)
                if cfg.separate_grad_clip:
                    nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    self.model.log_std.clamp_(cfg.log_std_min, cfg.log_std_max)

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["approx_kl"] += approx_kl_value
                metrics["clip_fraction"] += float(clip_fraction.item())
                metrics["actor_grad_norm"] += actor_grad_norm
                metrics["critic_grad_norm"] += critic_grad_norm
                updates += 1
                epochs_used = epoch + 1

            if kl_early_stop:
                break

        for key in metrics:
            metrics[key] /= max(updates, 1)
        metrics["max_approx_kl"] = max_approx_kl
        metrics["update_epochs"] = float(epochs_used)
        metrics["kl_early_stop"] = float(kl_early_stop)
        return metrics

    @staticmethod
    def _grad_norm(parameters: list[nn.Parameter]) -> float:
        norms = [torch.linalg.vector_norm(param.grad.detach()) for param in parameters if param.grad is not None]
        return float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0

    @staticmethod
    def _atanh(action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0 + LOG_PROB_EPS, 1.0 - LOG_PROB_EPS)
        return 0.5 * (torch.log1p(action) - torch.log1p(-action))

    @staticmethod
    def _squashed_log_prob(dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(torch.clamp(1.0 - action.pow(2), min=LOG_PROB_EPS))
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

    def initialize_from(self, path: str) -> dict[str, int]:
        """Initialize a larger-observation policy while preserving learned behavior."""
        checkpoint = torch.load(path, map_location=self.device)
        source_state = checkpoint["model"]
        target_state = self.model.state_dict()
        expandable = {"actor_mean.0.weight", "critic.0.weight"}
        source_obs_dim = None

        for name, target in target_state.items():
            if name not in source_state:
                raise ValueError(f"Initialization checkpoint is missing parameter: {name}")
            source = source_state[name]
            if source.shape == target.shape:
                target_state[name] = source
                continue
            if (
                name in expandable
                and source.ndim == 2
                and target.ndim == 2
                and source.shape[0] == target.shape[0]
                and source.shape[1] < target.shape[1]
            ):
                expanded = torch.zeros_like(target)
                expanded[:, : source.shape[1]] = source
                target_state[name] = expanded
                source_obs_dim = int(source.shape[1])
                continue
            raise ValueError(
                f"Cannot initialize parameter {name}: checkpoint {tuple(source.shape)} "
                f"!= model {tuple(target.shape)}"
            )

        self.model.load_state_dict(target_state)
        return {
            "source_obs_dim": source_obs_dim if source_obs_dim is not None else self.obs_dim,
            "target_obs_dim": self.obs_dim,
        }
