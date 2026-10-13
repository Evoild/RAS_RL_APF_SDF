#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

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
  --seed 17 \
  "$@"
