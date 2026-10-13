# 项目改进记录

## 记录规则

后续每次对环境、奖励、Actor/Critic、PPO 算法、训练参数或评估流程进行实质修改时，都必须在本文件末尾追加一节，标题统一使用 `第x次改进`。每节至少记录改进目的、修改内容、关键配置、验证方式和实验结果；尚未实施的设想不计入改进编号。

## 第1次改进

### 改进目的

解决 PPO 训练时从高斯分布采样、评估时使用均值所产生的动作语义不一致，并提高训练和评估差异的可观测性。

### 改进内容

- Actor 改为 tanh-squashed Gaussian，动作天然限制在 `[-1, 1]`。
- PPO 更新加入 tanh 变换对应的 log-prob Jacobian 修正，避免执行动作与概率计算不一致。
- 训练内评估同时执行 deterministic 和 stochastic 两种模式。
- 独立评估脚本增加 `deterministic`、`stochastic` 和 `both` 模式。
- 增加 `policy_std_mean` 和 `action_saturation_rate` 日志。
- 关节动作由完整关节范围映射改为受限的增量控制。

### 关键配置

```text
action = tanh(raw_action)
raw_action ~ Normal(mean, std)
max_joint_step = 3 deg
```

### 验证与结果

语法检查、Actor 前向、PPO 更新以及 deterministic/stochastic 评估链路均通过。旧绝对关节动作 checkpoint 与新动作语义不再具有可比性。

## 第2次改进

### 改进目的

建立无障碍纯 RL reaching baseline，先验证 Panda 能否稳定到达目标，再考虑障碍物和 APF。

### 改进内容

- 默认障碍物数量改为 `0`。
- 控制关节由 joint1--joint6 扩展为 joint1--joint7。
- 默认观测改为 7 个关节角和 3 维目标位置，共 10 维。
- 目标工作空间和采样约束与参考 reaching 脚本对齐。
- Actor 和 Critic 改为相互独立的 `[256, 128]` ReLU 网络。
- PPO 的 rollout、batch、epoch 和学习率按参考配置调整。

### 关键配置

```text
num_obstacles = 0
observation_dim = 10
action_dim = 7
actor_hidden_sizes = [256, 128]
critic_hidden_sizes = [256, 128]
rollout_steps = 2048
minibatch_size = 2048
train_epochs = 10
policy_lr = 2e-4
value_lr = 2e-4
entropy_coef = 0
```

### 验证与结果

环境 reset/step、目标采样、7 维 Actor/Critic 和完整训练入口均通过测试。使用参考复合奖励训练 200 万步时 success rate 仍为 `0`，策略倾向于输出小动作并停留在起点附近。

## 第3次改进

### 改进目的

去除会鼓励策略停留在起点或限制有效探索的复合奖励，只保留直接 reaching 目标。

### 改进内容

- 删除直线奖励、偏离惩罚、姿态惩罚、动作平滑惩罚、接触惩罚、关节限位惩罚、进度奖励和障碍物奖励。
- 距离奖励改为每步负的末端目标距离。
- 成功时增加明确的 `+100` 奖励。
- 未成功且达到最大步数时增加 `-10` 超时惩罚。
- PPO 和 DDPG reward test 脚本同步为相同的三项奖励日志。

### 关键配置

```text
distance_reward = -distance_to_goal
success_reward = +100
timeout_penalty = -10
total_reward = distance_reward + success_reward + timeout_penalty
```

### 验证与结果

分别验证了普通步骤、成功步骤和超时步骤的奖励求和，短 PPO 训练和奖励绘图均通过。采用 5 mm 成功阈值训练 600 万步后，最佳 deterministic success rate 为 `8%`；策略已能靠近目标，但存在动作过大、接近后离开的现象。

## 第4次改进

### 改进目的

建立 success threshold 课程接口，并确保训练、训练内评估、独立评估、可视化和 reward test 使用完全一致的成功判定。

### 改进内容

- 默认成功阈值从 `0.005 m` 放宽为 `0.02 m`。
- 增加统一常量 `DEFAULT_SUCCESS_THRESHOLD=0.02`。
- 所有 PPO 脚本统一支持 `--success-threshold` 和 `--max-steps`。
- 显式传入阈值时使用传入值；未传入时使用环境统一默认值。
- 成功阈值同时控制 success 统计、`+100` 成功奖励和 episode 成功终止。
- visualize 的默认最大步数由 12000 改为与训练和评估一致的 300。
- 新训练目录名加入实际 success threshold，避免不同阈值实验重名。

### 关键配置

```text
success_threshold = 0.02 m
max_steps = 300
joint_step_deg = 3
run_name includes success-threshold
```

### 验证与结果

验证了未传参数时所有脚本均使用 `0.02`，显式传入 `0.005` 时均使用 `0.005`。在约 1 cm 距离下，2 cm 阈值会触发 `+100` 并结束 episode，5 mm 阈值不会触发。

2 cm 阈值重新训练 600 万步后的结果：

```text
best deterministic success rate = 23%
best step = 3,500,032
best deterministic mean return = -42.46
final deterministic success rate = 4%
final deterministic mean return = -68.60
```

训练约 350 万步后开始退化。最大 `approx_kl=0.216`、最大 `clip_fraction=0.613`，说明部分 PPO 更新幅度过大；最优策略动作饱和率约 `20%`，后续应优先稳定 PPO 更新并改善近目标精细控制。

## 第5次改进

### 改进目的

针对 2 cm reaching 实验在约 350 万步达到峰值后退化，以及固定 3° 关节增量导致策略接近目标后容易越过成功区域的问题，同时稳定 PPO 更新并提升近目标精细控制能力。

### 改进内容

- Actor 学习率由 `2e-4` 降为 `1e-4`，每个 rollout 的 PPO epoch 由 `10` 降为 `5`。
- 增加 `target_kl=0.02`；更新前若 minibatch 的 `approx_kl` 超过阈值，立即停止当前 rollout 的后续 PPO epoch。
- 增加远、中、近三档自适应关节增量，按动作执行前的末端目标距离选择 3°、1° 或 0.5°。
- 训练、训练内评估、独立评估、可视化和 reward test 统一传入三档步长及距离分界。
- 日志新增 `max_approx_kl`、`update_epochs`、`kl_early_stop` 和 `mean_joint_step_deg`。
- 实验目录名加入三档步长及距离分界，避免不同动作语义的运行重名。
- 默认总训练步数调整为 400 万步，先验证是否能保持上一次实验在 350 万步附近出现的性能峰值。

### 关键配置

```text
policy_lr = 1e-4
value_lr = 2e-4
train_epochs = 5
target_kl = 0.02

distance >= 0.10 m:          joint_step = 3.0 deg
0.03 m <= distance < 0.10 m: joint_step = 1.0 deg
distance < 0.03 m:           joint_step = 0.5 deg
```

### 验证与结果

语法检查通过；环境分别在 20 cm、5 cm 和 2 cm 目标距离下正确选择 3°、1° 和 0.5°。构造 `approx_kl=1.135` 的异常更新后，PPO 正确设置 `kl_early_stop=1` 且未执行该 minibatch；64 步完整训练测试正常生成日志和 checkpoint。

400 万步长训练结果不理想：最佳 deterministic success rate 仅为 `1%`（800,768 步），最终为 `0%`，最佳 deterministic mean return 为 `-108.02`。虽然最大 `approx_kl` 从旧实验的 `0.216` 降到 `0.026`，最大 `clip_fraction` 从 `0.613` 降到 `0.147`，但前 100 万步平均 `approx_kl` 仅 `0.0012`，整体更新过弱。训练期间平均关节步长约 `2.99°`，说明策略很少进入 10 cm 精细控制范围，自适应步长尚未成为主要影响因素。逐步复评还发现新 `best_success.pt` 和 `final.pt` 的 100 个 episode 都曾发生碰撞，而当前 episode collision 指标只读取最后一步状态，后续需要修正统计口径。下一次应恢复 `policy_lr=2e-4` 和 `train_epochs=10`，同时保留 `target_kl=0.02` 与自适应步长。

## 第6次改进

### 改进目的

修正第5次实验中过于保守的 PPO 更新，在保留 KL 安全上限的同时恢复旧配置已经验证过的 reaching 学习速度。

### 改进内容

- Actor 学习率由 `1e-4` 恢复为 `2e-4`。
- 每个 rollout 的训练 epoch 由 `5` 恢复为 `10`。
- 保留 `target_kl=0.02`，在更新幅度异常时提前停止剩余 epoch。
- 保持 Critic 学习率、entropy 系数、网络结构、奖励和自适应步长不变，避免一次改变过多变量。
- 训练入口新增 PPO 参数选项，运行目录名加入 policy LR、epoch 和 target KL。

### 关键配置

```text
total_steps = 4,000,000
rollout_steps = 2,048
minibatch_size = 2,048
policy_lr = 2e-4
value_lr = 2e-4
train_epochs = 10
target_kl = 0.02
entropy_coef = 0
```

### 验证与结果

参数修改已完成。32 步端到端训练测试通过，日志确认执行 `10` 个 epoch，`max_approx_kl=0.0106`，未触发 KL 提前停止；运行目录正确包含 policy LR、epoch 和 target KL。

400 万步结果：最佳 deterministic success rate 为 `6%`（3,901,440 步），最终为 `0%`；最佳 stochastic success rate 为 `3%`。前四个百万步窗口的训练 success rate 分别约 `0.23%`、`0.35%`、`0.53%` 和 `0.43%`，仍远低于旧固定 3° 实验。最大 `max_approx_kl=0.051`，KL 提前停止 87 / 1,954 次；最大 value loss 为 `153`。独立对照旧 `best_success.pt`：相同 100 个目标、固定 3° 成功 `23%`，分段步长成功 `12%`。这一对照不能证明分段控制是全部训练差异的原因，但说明动作语义改变对已学策略有显著影响，下一轮应先恢复固定步长单独验证。

## 第7次改进

### 改进目的

隔离第6次训练中分段关节增量对 reaching 成功率的影响，并观测 Actor/Critic 共享梯度裁剪是否被 Critic 大梯度主导。

### 改进内容

- PPO 训练及配套独立评估、可视化和 reward test 默认使用固定 `3°` 关节增量；`--adaptive-joint-step` 可显式恢复原三档控制。
- 保持 `policy_lr=2e-4`、`train_epochs=10`、`target_kl=0.02`、奖励和 10 维观测不变。
- 增加裁剪前 `actor_grad_norm`、`critic_grad_norm` 日志；可选 `--separate-grad-clip` 独立裁剪两组参数，但默认仍使用原共享裁剪。
- 运行目录名加入动作模式与梯度裁剪模式，避免不同实验混淆。
- 状态归一化、相对目标观测和奖励尺度调整暂不启用，待固定步长对照完成后逐项测试。动作已由 tanh 限幅，不做额外硬裁剪。

### 关键配置

```text
joint_step_mode = fixed
joint_step_deg = 3
grad_clip = shared
max_grad_norm = 0.5
policy_lr = 2e-4
value_lr = 2e-4
train_epochs = 10
target_kl = 0.02
observation_dim = 10
```

### 验证与结果

语法检查、默认固定步长/共享裁剪和显式分段步长/独立裁剪两条 32 步端到端训练路径均通过。默认测试日志的 `mean_joint_step_deg=3.0`、`update_epochs=10`；裁剪前 Actor 梯度范数约 `2.11`，Critic 约 `23.69`。

固定 3°、共享裁剪的 400 万步训练结果仍不理想：最佳及最终 deterministic success rate 均为 `1%`，最佳 deterministic mean return 为 `-109.40`；最佳 stochastic success rate 为 `1%`。对比第6次分段步长实验的最佳 `6%` 和旧固定步长基线的最佳 `23%`，单独恢复固定步长没有恢复性能。相同 100 个评估目标复评 `best_success.pt`：平均轨迹最小目标距离为 `0.209 m`，仅 `14%` 到过 10 cm 内，`6%` 到过 5 cm 内，`38%` 的 episode 至少发生过一次碰撞。失败主要发生在到达目标附近之前，而非近目标越界。

本轮最大 `max_approx_kl=0.039`，KL 提前停止 `126/1,954` 次；最大 value loss 为 `62.2`。裁剪前 Actor 梯度范数均值约 `1.96`，Critic 约 `25.88`，所有 update 的 Critic 范数都大于 Actor，典型共享裁剪缩放系数约 `0.02`。这说明共享裁剪常由 Critic 主导，但 Adam 对梯度尺度较不敏感，因此不能仅凭范数断言 Actor 参数更新被相同比例压制。下一轮应在固定 3°、相同 PPO/奖励/观测配置下只启用 `--separate-grad-clip` 做对照，再根据结果决定是否测试相对目标观测或奖励尺度。

独立裁剪对照已完成，同为固定 3°、400 万步、相同 seed 和 PPO 参数。最佳 deterministic/stochastic success rate 均为 `1%`，最终均为 `0%`；共享裁剪的最佳/最终 deterministic 为 `1%/1%`。独立裁剪的最终 deterministic mean return 为 `-132.10`，比共享裁剪的 `-119.38` 更低。KL 提前停止从 `126` 降到 `68` 次，但最大 `clip_fraction` 从 `0.179` 增到 `0.184`，没有获得更好的 reaching。独立裁剪 `best_success.pt` 在 CPU 上对相同 100 目标复评为 `0/100` 成功，平均轨迹最小距离 `0.315 m`，只有 `6%` 曾进入 10 cm；`final.pt` 没有轨迹进入 10 cm。TensorBoard 的最佳 `1/100` 是单次训练内评估，可能使用 CUDA；与 CPU 复评的差异可能包含数值设备差异。这两个小样本结果都不足以证明稳定成功。

独立裁剪不是本项目当前的有效改进。当前 `qpos + goal` 在理论上可通过正运动学推得目标相对向量，但网络必须自行学会这一映射。下一步优先保持固定 3°、共享裁剪、原奖励和 PPO 参数，只对观测增加末端位置与目标相对向量做单变量实验；运行时观测统计/归一化应单独排在之后，且评估必须使用训练保存的同一统计量。动作已经由 tanh 限幅，不做额外硬裁剪；统一奖励缩放也不应在尚未验证观测改动前被视为主要解决手段。

## 第8次改进

### 改进目的

在不改变奖励、PPO 更新和关节动作语义的前提下，直接向策略提供 reaching 几何信息，减少网络从关节角自行近似正运动学和目标相对方向的负担。

### 改进内容

- PPO 默认观测由 10 维扩展到 16 维：保留 `qpos[7]` 和世界坐标 `goal[3]`，追加 MuJoCo 末端 site 世界坐标 `ee_pos[3]` 及 `goal - ee_pos[3]`。
- 环境类仍默认 10 维，避免直接影响 DDPG；PPO 训练、训练内评估、独立评估、可视化和 reward test 默认使用 16 维，可用 `--no-ee-goal-observation` 选择旧 10 维。
- 运行目录名增加 `observation_ee-goal-16d` 或 `observation_legacy-10d`，避免不同输入层 checkpoint 混淆。
- 保持固定 3°、共享梯度裁剪、`target_kl=0.02`、原三项奖励和其他 PPO 超参数不变。
- 暂不启用在线状态归一化、奖励缩放或额外动作硬裁剪，确保本轮仅比较观测信息。

### 关键配置

```text
observation = [qpos(7), goal(3), ee_pos(3), goal-ee_pos(3)]
observation_dim = 16
action_dim = 7
joint_step_mode = fixed
grad_clip = shared
policy_lr = 2e-4
value_lr = 2e-4
train_epochs = 10
target_kl = 0.02
```

### 验证与结果

reset 和 step 的 16 维观测通过形状及 `goal-ee_pos` 数值一致性测试；相对向量范数与奖励中的目标距离一致。16 维新模式和 10 维旧模式各完成 32 步训练测试，目录与 checkpoint 互不覆盖；两种模式的独立评估、16 维 reward test 均通过。旧 10 维 checkpoint 必须显式传 `--no-ee-goal-observation` 才能加载。

400 万步长训练已经完成。训练端按每百万步统计的 rollout success rate 约为 `6.7%`、`21.3%`、`50.5%`、`86.2%`；最后两个 rollout 分别为 `94.1%` 和 `86.7%`，平均奖励已经转正。训练内 100 episode 评估的最佳 deterministic/stochastic success rate 均为 `96%`（约 360 万步），最终均为 `91%`；最佳 deterministic mean return 为 `72.93`，最终为 `66.71`。对照上一轮固定 3°、共享裁剪、10 维观测的最佳 `1%`，本轮在相同 PPO/奖励/动作配置下出现明确提升；这支持直接提供末端与相对目标信息的做法，但单一 seed 对照仍不足以估计跨训练种子的方差。

在 CPU 上对 `best_success.pt` 使用训练内评估目标 seed 复评，deterministic success 为 `96/100`、stochastic 为 `94/100`，平均 episode 长度分别约 `117` 和 `125` 步，100 条轨迹都曾进入目标 5 cm 范围。`final.pt` 在 CPU 上 deterministic 为 `86/100`，说明 360 万步后的策略仍有一定退化；应把 `best_success.pt` 而不是 `final.pt` 作为第一阶段基线。对最优模型更换目标 seed `10017/10018/10019/20017` 的 100 episode deterministic success 分别为 `96%/100%/97%/96%`，当前无障碍 2 cm reaching 对目标采样具有较好的稳定性。

本轮 value loss 最大约 `281`，Critic 裁剪前梯度范数最大约 `1107`，随成功回报和短 episode 大量出现而升高；这不能单凭数值判定训练失稳，因为评估 success 同时持续提升。最大 `max_approx_kl=0.037`，KL 提前停止 `97/1,954` 次；最终 deterministic 动作饱和率约 `31%`，后续仍需监控训练更长时的策略保持情况。

使用同一 `best_success.pt` 在 seed `10017` 下只改变成功阈值进行诊断：`2 cm/1 cm/5 mm` 的 deterministic success 分别为 `96%/71%/23%`。更严格阈值会改变成功奖励和 episode 终止时点，因此这些值是新判定口径的独立评估，不是同一批 2 cm episode 的简单重标记。下一阶段宜先把 1 cm 阈值作为课程目标；但当前训练脚本没有 checkpoint resume 入口，真正从最优模型继续微调前需要先补上恢复训练与配置校验，不能把严格阈值评估误称为微调。之后再评估 5 mm；当前成绩仅针对 `num_obstacles=0` 的 reaching，不证明避障能力。

## 第9次改进

### 改进目的

提供可复跑的独立无障碍 reaching 结果测试，统一输出不同成功阈值、目标 seed 和动作模式下的成功率、回报、轨迹精度与碰撞情况。

### 改进内容

- 新增 `scripts/ppo/test_reaching_policy.py`，默认加载第8次实验的 16 维 `best_success.pt`。
- 默认按 2 cm、1 cm、5 mm 阈值分别运行 deterministic 和 stochastic 各 100 个 episode；可指定多个目标 seed、checkpoint 和测试规模。
- 每个配置保存逐 episode `episodes.csv`、汇总 `summary.csv` 与 `config.json`；汇总增加 Wilson 95% 成功率区间。
- 轨迹最小/最终目标距离、动作饱和率与是否曾发生碰撞均按完整 episode 统计，不再依赖最后一步碰撞状态。
- 保持模型权重、环境奖励和 PPO 训练参数不变；旧 10 维 checkpoint 可用 `--no-ee-goal-observation` 测试。

### 关键配置

```text
checkpoint = 16d best_success.pt
num_obstacles = 0
success_thresholds = [0.02, 0.01, 0.005] m
episodes_per_case = 100
seed = 10017
mode = both
device = cpu
joint_step_mode = fixed, 3 deg
```

### 验证与结果

语法检查、16 维与旧 10 维 checkpoint 加载、两阈值双模式冒烟测试、CSV 行数和字段检查均通过。100 episode deterministic 正式验证：2 cm `96/100`（95% 区间约 `[90.2%, 98.4%]`）、1 cm `71/100`（约 `[61.5%, 79.0%]`）、5 mm `23/100`（约 `[15.8%, 32.2%]`），与第8次手工复评一致。各阈值单独执行仿真并改变成功终止判定，不能当作同一轨迹的重新标注。

## 第10次改进

### 改进目的

在保留无障碍 reaching baseline 的同时，引入可独立复现的单静态障碍训练阶段，避免复制并维护两套 PPO 训练器，并保证训练、评估和可视化使用同一环境配置。

### 改进内容

- 新增 `configs/ppo/reaching_baseline.json` 和 `configs/ppo/static_obstacle_stage1.json`，分别固定无障碍 baseline 与单静态障碍配置。
- 新增 `scripts/ppo/common.py`，统一 JSON profile 解析、CLI 覆盖、环境构造、观测维度命名和配置快照；训练参数与评估参数按 section 隔离。
- 保留 `train.sh`、`eval.sh`、`visualize.sh` 作为无障碍入口；新增对应的三个 obstacle shell 入口。
- 静态障碍观测由 16 维扩展到 22 维，追加障碍物世界坐标和 `obstacle_pos-ee_pos`。
- 环境奖励新增二次接近惩罚与碰撞惩罚，并支持碰撞立即终止；无障碍 profile 将两项尺度设为零，保持原三项奖励。
- 障碍物每次 reset 沿路径附近重新采样，episode 内不扰动；将静态阶段 `min_safety_dist` 从不适合当前路径长度的 0.3 m 调整为 0.08 m。
- 修正 clearance 已减球半径后碰撞阈值又重复加入半径的问题；改用 MuJoCo `mj_geomDistance` 计算 23 个机器人碰撞 geom 到球形障碍表面的精确最小距离，不再混入工作台距离或使用关节锚点近似。
- PPO 增加 16→22 维初始化迁移：Actor/Critic 第一层前 16 列复制，新增 6 列清零，其余参数必须严格同形。
- 训练日志改为 `logs_ppo/runs/<run_name>/` 独立目录并保存 `config.json`；静态障碍模型按 deterministic `collision_free_success_rate` 选择。
- 新增 `test_obstacle_policy.py`，输出无碰撞成功率及 Wilson 区间、碰撞率、超时率、最小 clearance、回报和动作饱和率。

### 关键配置

```text
task = static_obstacle
num_obstacles = 1
obstacle_radius = 0.03 m
obstacle_disturb_prob = 0
min_safety_dist = 0.08 m
observation_dim = 22
proximity_margin = 0.08 m
proximity_penalty_scale = 2
collision_penalty = -20
terminate_on_collision = true
init_checkpoint = reaching 16d best_success.pt
```

### 验证与结果

Python 编译、JSON 解析、shell 语法和 `git diff --check` 均通过。使用项目 `mujoco` Conda 环境完成无障碍与静态障碍各 32-step CPU 端到端训练：baseline 正确使用 16 维网络且新增奖励均为零；静态障碍正确使用 22 维网络，并完成旧 checkpoint 的 16→22 维迁移。两条训练均生成隔离的 checkpoint、metrics 和配置快照。

数值测试确认迁移前后的 deterministic action 和 value 在相同前 16 维输入下保持一致；22 维相对障碍向量满足 `obstacle_pos-ee_pos`；障碍位置在 episode 内不变；总奖励等于各分量之和；强制零 clearance 时接近惩罚为 `-2` 且正确识别障碍碰撞。旧 16 维 baseline checkpoint 仍可被重构后的评估入口严格加载。

静态障碍评估和结果测试各完成 1 episode 冒烟，正确输出 `collision_free_success_rate`、`any_collision_rate` 和 `mean_min_clearance`。本次尚无正式静态障碍长训练结果；下一步运行 `./scripts/train_obstacle.sh` 后再分析奖励尺度和避障效果。

精确 clearance 修改后再次完成 32-step 静态障碍训练，CPU 速度约为 80 step/s。连续 200 次 reset 采样中没有初始碰撞或起终点安全距离违规，最小初始障碍表面 clearance 约为 1.72 cm。
