# Pure RL Panda Reaching Baseline

当前先完成无障碍 reaching，让 deterministic evaluation 的 success rate 稳定起来；之后再逐步加入弱障碍、随机障碍和 APF。

## 第一阶段环境

环境实现在 `envs/panda_env.py`，训练入口是 `scripts/ppo/train_ppo.py`。

默认配置：

```text
num_obstacles       = 0
controlled_joints   = 7
observation_dim     = 10（7 个关节角 + 3 维目标位置）
action_dim          = 7
success_threshold   = 0.02 m
max_steps           = 300
frame_skip          = 10
```

目标采样与参考脚本 `rl_panda_reach_target_high_profile.py` 一致：

```text
x in [-0.5, 0.8]
y in [-0.5, 0.5]
z in [0.05, 0.3]
0.4 m < start-to-goal distance < 0.5 m
goal.x > 0.2 m
goal.z > 0.2 m
```

策略输出 7 维 `[-1, 1]` 动作，分别控制 Panda joint1 到 joint7，夹爪保持打开。当前仍使用增量关节控制：

```python
joint_target = current_joint_position + action * max_joint_step
```

默认 `max_joint_step = 3 deg`，可通过 `--joint-step-deg` 调整。参考脚本使用绝对关节目标；这里保留增量控制，以限制单步跳变并减小训练/部署动作语义差异。

## 奖励函数

第一阶段只保留目标距离、成功和超时三项奖励：

```text
total_reward =
    distance_reward
  + success_reward
  + timeout_penalty
```

```text
distance_reward = -distance_to_goal
success_reward  = +100（distance < 0.02 m）
timeout_penalty = -10（达到 300 步且未成功）
```

负距离奖励让策略每一步都倾向于减小末端到目标的距离，也避免停在起点持续累计正奖励。达到目标立即结束 episode；无障碍阶段碰撞不会提前结束。成功与超时互斥，不会在同一步同时加分和扣分。

`--success-threshold` 会同时控制成功判定、`+100` 成功奖励和成功终止。未传入时统一使用环境常量 `DEFAULT_SUCCESS_THRESHOLD=0.02`；新训练目录名会包含实际阈值，例如 `_success-threshold_0.02_`。

## Actor、Critic 与 PPO

Actor 和 Critic 使用独立网络，与参考脚本的网络宽度和激活函数一致：

```text
Actor:  obs -> 256 ReLU -> 128 ReLU -> 7
Critic: obs -> 256 ReLU -> 128 ReLU -> 1
```

PPO 核心配置：

```text
total_steps          = 40,000,000
rollout_steps        = 2,048
minibatch_size       = 2,048
train_epochs         = 10
policy_lr            = 2e-4
value_lr             = 2e-4
entropy_coef         = 0
value_coef           = 0.5
gamma                = 0.99
gae_lambda            = 0.95
clip_ratio           = 0.2
max_grad_norm         = 0.5
actor_hidden_sizes   = [256, 128]
critic_hidden_sizes  = [256, 128]
log_std_init          = -0.7
log_std_range         = [-5, 0]
```

参考脚本使用 24 个并行 SB3 环境；当前自定义 PPO 训练器仍是单环境，因此 `rollout_steps=2048` 表示每次更新共收集 2048 个 transition。`log_std_init=-0.7` 也不是 SB3 默认值，而是为增量动作保留适中的初始探索范围。

策略分布使用 tanh-squashed Gaussian：

```text
raw_action ~ Normal(mean, std)
action = tanh(raw_action)
```

PPO 更新时包含对应的 log-prob Jacobian 修正，训练动作与概率计算一致。

## 训练与评估

先用 200 万步检查训练链路和奖励走势：

```bash
python3 scripts/ppo/train_ppo.py \
  --total-steps 2000000 \
  --rollout-steps 2048 \
  --eval-every-steps 100000 \
  --eval-episodes 100 \
  --num-obstacles 0 \
  --success-threshold 0.02 \
  --max-steps 300 \
  --joint-step-deg 3
```

确认 success rate 能上升后，再把 `--total-steps` 提高到 `40000000`。训练默认不渲染、不录制视频。

独立评估：

```bash
python3 scripts/ppo/evaluate_policy.py \
  --checkpoint checkpoints_ppo/<run>/best_success.pt \
  --episodes 100 \
  --mode both \
  --success-threshold 0.02 \
  --max-steps 300 \
  --joint-step-deg 3
```

`--mode deterministic` 使用 actor mean，`--mode stochastic` 从策略方差采样，`--mode both` 会分别执行两种评估。模型选择以 deterministic success rate 为准。

可视化默认使用 deterministic policy：

```bash
python3 scripts/ppo/visualize_policy.py \
  --checkpoint checkpoints_ppo/<run>/best_success.pt \
  --episodes 3 \
  --success-threshold 0.02 \
  --max-steps 300 \
  --joint-step-deg 3
```

增加 `--stochastic` 可观察采样策略。

## TensorBoard

```bash
tensorboard --logdir logs_ppo/tensorboard --port 6006
```

浏览器打开 `http://localhost:6006`。重点观察：

```text
rollout/success_rate
rollout/target_distance
ppo/value_loss
ppo/approx_kl
policy/action_saturation_rate
policy/policy_std_mean
eval/eval_det_success_rate
eval/eval_stoch_success_rate
```

## 兼容性与后续阶段

环境已从 6 维动作改为 7 维动作，默认观测也从带障碍物的位置输入改为 10 维 reaching 输入。旧 checkpoint 的网络形状不兼容，不能直接用于当前环境，需要重新训练。

课程顺序：

1. 无障碍 reaching，先以 2 cm 阈值让 deterministic success rate 稳定。
2. 加入弱障碍和障碍接近惩罚。
3. 增加目标、障碍和初始状态随机化。
4. 依次将成功阈值收紧到 1 cm 和 5 mm，精细阶段把关节步长降到 1--2 度。
5. 在 pure RL baseline 稳定后加入 APF shaping 或 action prior 做对照实验。
