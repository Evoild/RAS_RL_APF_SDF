#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="total-steps_4000000_rollout-steps_2048_success-threshold_0.02_joint-steps-deg_3-1-0.5_step-distances_0.1-0.03_joint-step-mode_fixed_grad-clip_shared_observation_ee-goal-16d_policy-lr_0.0002_epochs_10_target-kl_0.02_seed_17"
DEFAULT_CHECKPOINT="${PROJECT_ROOT}/checkpoints_ppo/${RUN_NAME}/best_success.pt"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  printf 'Checkpoint not found: %s\n' "${CHECKPOINT}" >&2
  exit 1
fi

python3 scripts/ppo/evaluate_policy.py \
  --config configs/ppo/reaching_baseline.json \
  --checkpoint "${CHECKPOINT}" \
  --episodes 100 \
  --mode both \
  "$@"
