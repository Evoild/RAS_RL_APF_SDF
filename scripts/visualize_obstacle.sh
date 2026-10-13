#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_NAME="static-obstacle_nobs1_r0.03_obs22_fixed3deg_gradshared_s0.02_p2_c-20_steps4000000_roll2048_lr0.0002_ep10_kl0.02_seed17"
DEFAULT_CHECKPOINT="${PROJECT_ROOT}/checkpoints_ppo/${RUN_NAME}/best_success.pt"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  printf 'Checkpoint not found: %s\nSet CHECKPOINT=/path/to/best_success.pt when using a run suffix.\n' "${CHECKPOINT}" >&2
  exit 1
fi

python3 scripts/ppo/visualize_policy.py \
  --config configs/ppo/static_obstacle_stage1.json \
  --checkpoint "${CHECKPOINT}" \
  --episodes 3 \
  "$@"
