# Pure RL Panda Reaching Baseline

本项目当前阶段先建立 **无障碍 Panda reaching** 的纯 PPO baseline。目标是在不使用 APF、不加入障碍物奖励的条件下，让策略稳定完成末端到目标点的运动；确认 reaching 能力后，再逐步加入弱障碍、随机障碍和 APF 对照。

当前推荐基线来自第 8 次实验。核心结论是：保持奖励、PPO 和动作语义不变，仅将观测从 `qpos + goal` 扩展为包含末端位置和目标相对向量的 16 维观测后，2 cm deterministic success rate 从约 `1%` 提升到 `96%`。

## 1. Baseline 总配置

```yaml
experiment:
  task: no-obstacle reaching
  seed: 17
  total_steps: 4_000_000
  rollout_steps: 2_048
  eval_every_steps: 100_000
  eval_episodes: 100

environment:
  robot: Franka Panda
  controlled_joints: 7
  num_obstacles: 0
  max_episode_steps: 300
  frame_skip: 10
  success_threshold: 0.02  # m
  randomize_reset: false

observation:
  fields:
    - qpos[7]
    - goal[3]
    - ee_pos[3]
    - goal_minus_ee_pos[3]
  dimension: 16
  running_normalization: false

reward:
  distance_reward: -distance_to_goal
  success_reward: 100
  timeout_penalty: -10
  running_normalization: false
  advantage_normalization: per rollout

action:
  dimension: 7
  policy_range: [-1, 1]
  mode: incremental joint position
  joint_step_mode: fixed
  joint_step_deg: 3
  adaptive_joint_step: false
  environment_action_clip: [-1, 1]
  joint_limit_clip: true

network:
  actor_hidden_sizes: [256, 128]
  critic_hidden_sizes: [256, 128]
  activation: ReLU
  actor_critic_parameters: independent

policy_distribution:
  type: tanh-squashed diagonal Gaussian
  log_prob_jacobian_correction: true
  log_std_init: -0.7
  log_std_range: [-5, 0]

ppo:
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  minibatch_size: 2_048
  train_epochs: 10
  policy_lr: 0.0002
  value_lr: 0.0002
  entropy_coef: 0
  value_coef: 0.5
  max_grad_norm: 0.5
  grad_clip_mode: shared
  target_kl: 0.02
```

## 2. 环境与目标采样

环境实现在 `envs/panda_env.py`。PPO 控制 Panda 的 joint1 到 joint7，夹爪保持打开。当前 baseline 不生成障碍物：

```text
num_obstacles = 0
action_dim = 7
max_steps = 300
frame_skip = 10
```

目标点从以下工作空间采样：

```text
x in [-0.5, 0.8]
y in [-0.5, 0.5]
z in [0.05, 0.3]

0.4 m < initial_ee_to_goal_distance < 0.5 m
goal.x > 0.2 m
goal.z > 0.2 m
```

训练环境使用 seed `17`。训练内评估默认使用 `seed + 10000 = 10017`，并关闭 reset 随机化，从而让不同 checkpoint 在同一批目标上比较。

## 3. 状态输入与归一化

PPO baseline 使用 16 维观测：

```text
observation = concat(
    qpos[7],
    goal[3],
    ee_pos[3],
    goal - ee_pos[3],
)
```

其中：

- `qpos`：7 个受控关节角。
- `goal`：目标点世界坐标。
- `ee_pos`：末端 site 世界坐标。
- `goal - ee_pos`：从末端指向目标的相对向量。

第 7 次实验只使用 `qpos[7] + goal[3]`，网络需要从关节角自行近似正运动学，最佳 deterministic success rate 仅约 `1%`。第 8 次实验加入 `ee_pos` 和相对目标向量后，其他关键配置保持不变，最佳成功率提升到 `96%`。因此 16 维几何观测是当前 baseline 中最关键的有效改动。

当前 **没有启用 observation running normalization**。目标位置、末端位置和关节角直接输入网络。旧 10 维 checkpoint 与当前 16 维网络输入层不兼容；测试旧模型时必须显式传入 `--no-ee-goal-observation`。

## 4. 奖励塑形与归一化

当前奖励只包含三项：

```python
distance_reward = -distance_to_goal
success_reward = 100.0 if distance_to_goal < success_threshold else 0.0
timeout_penalty = -10.0 if timeout else 0.0

reward = distance_reward + success_reward + timeout_penalty
```

具体语义：

- 每一步使用负距离奖励，直接推动末端靠近目标。
- 距离小于 `success_threshold` 时给予 `+100`，并立即结束 episode。
- 300 步仍未成功时给予 `-10`，并结束 episode。
- 成功与超时互斥。
- 无障碍 baseline 中，碰撞不提供额外奖励或惩罚，也不会触发终止。

已删除的复合项包括直线奖励、路径偏离惩罚、姿态惩罚、动作平滑惩罚、接触惩罚、关节限位惩罚、进度奖励和障碍物奖励。这些项不属于当前 baseline。

当前 **没有启用 reward running normalization 或 reward scaling**。PPO 更新时只对每个 rollout 计算出的 advantage 做标准化：

```text
normalized_advantage = (advantage - mean) / (std + epsilon)
```

因此“奖励归一化”和“优势归一化”不能混为一谈。成功奖励仍以原始 `+100` 进入 return 和 Critic target，这也是成功率升高后 value loss 和 Critic 梯度增大的主要原因之一。

`--success-threshold` 同时控制成功统计、成功奖励和 episode 终止。改变阈值会改变实际 MDP，不能只把同一批轨迹按新阈值重新计数。

## 5. Actor、Critic 与动作分布

Actor 和 Critic 使用独立 MLP：

```text
Actor:  16 -> 256 ReLU -> 128 ReLU -> 7
Critic: 16 -> 256 ReLU -> 128 ReLU -> 1
```

策略使用 tanh-squashed Gaussian：

```text
raw_action ~ Normal(mean, std)
action = tanh(raw_action)
```

训练时 log probability 包含 tanh 变换对应的 Jacobian 修正，保证 PPO 优化使用的概率与环境实际执行动作一致。训练阶段从分布采样；deterministic evaluation 使用分布均值经过 tanh 后的动作。

初始 `log_std=-0.7`，并限制在 `[-5, 0]`。`entropy_coef=0`，策略探索由可学习的标准差和 PPO 目标共同决定，不额外添加 entropy bonus。

## 6. 动作输出与环境转换

Actor 输出 7 维归一化动作，经过 tanh 后位于 `[-1, 1]`。环境再次执行安全裁剪，然后转换为增量关节位置目标：

```python
action = clip(action, -1.0, 1.0)
joint_target = current_qpos + action * deg2rad(3.0)
joint_target = clip(joint_target, joint_lower_limit, joint_upper_limit)
```

所以单个控制 step 中，每个关节的最大目标增量为 `3°`。MuJoCo 在该目标下执行 `frame_skip=10` 个物理步。

当前 baseline 使用固定 3°，不传 `--adaptive-joint-step`。3°/1°/0.5° 的距离分段动作曾单独测试，但没有带来稳定提升，因此不纳入 baseline。动作已经经过 tanh 和环境安全裁剪，也不再增加更窄的硬裁剪范围。

## 7. PPO 更新配置

每收集 2048 个 transition 执行一次 PPO 更新。当前是单环境训练，不是 2048 个并行环境：

```text
rollout_steps = 2048
minibatch_size = 2048
train_epochs = 10
policy_lr = 2e-4
value_lr = 2e-4
gamma = 0.99
gae_lambda = 0.95
clip_ratio = 0.2
value_coef = 0.5
entropy_coef = 0
max_grad_norm = 0.5
target_kl = 0.02
```

Actor、Critic 和 `log_std` 使用同一个 Adam optimizer，但 Actor 与 Critic的学习率参数组分别设置。默认对全部参数执行共享梯度裁剪。独立 Actor/Critic 梯度裁剪对照没有改善成功率，因此未纳入 baseline。

当更新过程中 minibatch 的 `approx_kl` 已超过 `0.02` 时，停止该 rollout 剩余的 PPO epoch。该机制用于限制后续更新，不能撤销刚刚完成并导致 KL 上升的 optimizer step。

## 8. Baseline 训练命令

在 `rl/` 目录运行：

```bash
python3 scripts/ppo/train_ppo.py \
  --total-steps 4000000 \
  --rollout-steps 2048 \
  --policy-lr 0.0002 \
  --value-lr 0.0002 \
  --train-epochs 10 \
  --target-kl 0.02 \
  --eval-every-steps 100000 \
  --eval-episodes 100 \
  --num-obstacles 0 \
  --success-threshold 0.02 \
  --max-steps 300 \
  --joint-step-deg 3 \
  --ee-goal-observation \
  --seed 17
```

不要为 baseline 添加以下参数：

```text
--adaptive-joint-step
--separate-grad-clip
--randomize-reset
--no-ee-goal-observation
```

训练默认不渲染、不录制视频。模型选择以训练内 deterministic evaluation 的 success rate 为第一优先级，相同成功率下再比较 mean return。推荐使用 `best_success.pt`，不要直接用 `final.pt`。

## 9. Baseline Checkpoint

本次最优模型：

```text
checkpoints_ppo/
  total-steps_4000000_rollout-steps_2048_success-threshold_0.02_
  joint-steps-deg_3-1-0.5_step-distances_0.1-0.03_
  joint-step-mode_fixed_grad-clip_shared_observation_ee-goal-16d_
  policy-lr_0.0002_epochs_10_target-kl_0.02_seed_17/
  best_success.pt
```

目录名仍记录了 1°、0.5° 和距离分界参数，但 `joint-step-mode_fixed` 表示本次实验实际只使用 `joint-step-deg_3`。

## 10. 实验结果

### 10.1 训练过程

按每 100 万环境步统计的 rollout success rate：

| 训练区间 | Rollout success rate |
| --- | ---: |
| 0--1M | 约 6.7% |
| 1M--2M | 约 21.3% |
| 2M--3M | 约 50.5% |
| 3M--4M | 约 86.2% |

最后两个 rollout success rate 分别约为 `94.1%` 和 `86.7%`，平均奖励已经转正。

### 10.2 训练内评估

| 指标 | 最佳 checkpoint | Final checkpoint |
| --- | ---: | ---: |
| Deterministic success rate | 96% | 91% |
| Stochastic success rate | 96% | 91% |
| Deterministic mean return | 72.93 | 66.71 |

最佳结果出现在约 360 万步。训练到 400 万步后有轻微退化，因此模型发布和后续课程训练应从 `best_success.pt` 开始。

### 10.3 CPU 独立复评

在 seed `10017`、100 episodes、2 cm 阈值下：

| Checkpoint / mode | Success rate | Mean episode length |
| --- | ---: | ---: |
| `best_success.pt`, deterministic | 96/100 | 约 117 steps |
| `best_success.pt`, stochastic | 94/100 | 约 125 steps |
| `final.pt`, deterministic | 86/100 | - |

100 条 deterministic 轨迹都曾进入目标 5 cm 范围。训练内评估与 CPU 独立复评基本一致，说明当前 baseline 已显著缩小训练与 evaluation 的 success gap。

### 10.4 跨目标 Seed 泛化

`best_success.pt` 在 2 cm、deterministic、每组 100 episodes 下：

| Target seed | Success rate |
| --- | ---: |
| 10017 | 96% |
| 10018 | 100% |
| 10019 | 97% |
| 20017 | 96% |

这些结果说明策略对目标采样具有较好稳定性，但训练只使用一个训练 seed，因此尚不能代表跨训练 seed 的 PPO 方差。

### 10.5 更严格成功阈值诊断

使用同一个 2 cm baseline checkpoint，在 seed `10017` 下分别创建新环境测试：

| Success threshold | Deterministic success | Wilson 95% CI |
| --- | ---: | ---: |
| 2 cm | 96/100 | 约 [90.2%, 98.4%] |
| 1 cm | 71/100 | 约 [61.5%, 79.0%] |
| 5 mm | 23/100 | 约 [15.8%, 32.2%] |

该结果表明 2 cm reaching 已基本建立，但精确到达能力仍有提升空间。三档阈值是分别运行的评估；改变阈值同时改变成功奖励和终止条件。

### 10.6 PPO 与 Critic 诊断

```text
max value loss                  ~= 281
max critic grad norm            ~= 1107
max max_approx_kl               ~= 0.037
KL early stops                  = 97 / 1954 updates
final deterministic saturation ~= 31%
```

Critic loss 和梯度在成功 episode 大量出现后增大，原因之一是原始 `+100` 成功回报进入 value target。由于 success rate 同时从低水平持续提升到 96%，不能仅凭 Critic loss 数值增大判断训练失败。后续实验应同时观察成功率、mean return、KL、动作饱和率和 value loss。

## 11. 配置改动效果总结

| 改动 | 对照结果 | 是否纳入 baseline |
| --- | --- | --- |
| tanh-squashed Gaussian + Jacobian 修正 | 统一训练采样和评估均值动作的概率语义 | 是 |
| 奖励简化为负距离、成功、超时 | 去除停留和互相冲突的复合塑形项，策略开始有效靠近 | 是 |
| 2 cm 统一成功阈值 | 旧 5 mm 训练最佳约 8%；2 cm 旧基线达到 23% | 是 |
| `target_kl=0.02` | 极端 KL 从旧实验约 0.216 降到当前最大约 0.037 | 是 |
| 恢复 `policy_lr=2e-4`、10 epochs | 低学习率 1e-4、5 epochs 时最佳仅 1%，更新过弱 | 是 |
| 固定 3° 增量动作 | 动作语义稳定；分段动作未获得稳定收益 | 是 |
| 16 维几何观测 | 相同主要配置下，deterministic success 从约 1% 提升到 96% | 是，核心改动 |
| 独立 Actor/Critic 梯度裁剪 | 最佳约 1%，未改善 reaching | 否 |
| 状态 running normalization | 尚未做严格单变量实验 | 否 |
| 奖励 running normalization / scaling | 尚未做严格单变量实验 | 否 |
| 更窄动作硬裁剪 | tanh 与环境裁剪已经限幅，尚无必要 | 否 |

## 12. 测试与可视化

### 12.1 一键结果测试

默认加载上述 `best_success.pt`，测试 2 cm、1 cm、5 mm，分别运行 deterministic 和 stochastic 各 100 episodes：

```bash
python3 scripts/ppo/test_reaching_policy.py
```

只测试 2 cm deterministic，并覆盖多个目标 seed：

```bash
python3 scripts/ppo/test_reaching_policy.py \
  --thresholds 0.02 \
  --seeds 10017 10018 10019 20017 \
  --mode deterministic \
  --episodes 100
```

输出文件位于：

```text
logs_ppo/result_test/<checkpoint-run>/config.json
logs_ppo/result_test/<checkpoint-run>/episodes.csv
logs_ppo/result_test/<checkpoint-run>/summary.csv
```

### 12.2 独立评估

```bash
RUN=total-steps_4000000_rollout-steps_2048_success-threshold_0.02_joint-steps-deg_3-1-0.5_step-distances_0.1-0.03_joint-step-mode_fixed_grad-clip_shared_observation_ee-goal-16d_policy-lr_0.0002_epochs_10_target-kl_0.02_seed_17

python3 scripts/ppo/evaluate_policy.py \
  --checkpoint "checkpoints_ppo/$RUN/best_success.pt" \
  --episodes 100 \
  --mode both \
  --success-threshold 0.02 \
  --max-steps 300 \
  --joint-step-deg 3 \
  --ee-goal-observation
```

### 12.3 可视化

```bash
python3 scripts/ppo/visualize_policy.py \
  --checkpoint "checkpoints_ppo/$RUN/best_success.pt" \
  --episodes 3 \
  --success-threshold 0.02 \
  --max-steps 300 \
  --joint-step-deg 3 \
  --ee-goal-observation
```

默认使用 deterministic policy；增加 `--stochastic` 可观察采样动作。

## 13. TensorBoard

```bash
tensorboard --logdir logs_ppo/tensorboard --port 6006
```

浏览器打开 `http://localhost:6006`。建议重点观察：

```text
rollout/success_rate
rollout/mean_return
rollout/target_distance
ppo/value_loss
ppo/approx_kl
ppo/max_approx_kl
ppo/clip_fraction
ppo/update_epochs
ppo/kl_early_stop
ppo/actor_grad_norm
ppo/critic_grad_norm
policy/action_saturation_rate
policy/policy_std_mean
eval/det_success_rate
eval/stoch_success_rate
```

## 14. 下一阶段

1. 将当前 `best_success.pt` 固定为无障碍 2 cm reaching baseline。
2. 增加 checkpoint resume 与配置校验，再以 1 cm 阈值进行课程微调。
3. 完成 1 cm/5 mm 精度对照后，逐步加入弱障碍和障碍接近惩罚。
4. 增加目标、障碍和初始状态随机化，并使用多个训练 seed 报告均值与方差。
5. pure RL obstacle avoidance 稳定后，再加入 APF shaping 或 action prior 做严格对照。

当前所有成功率仅代表 `num_obstacles=0` 的 reaching，不代表已经具备避障能力。
