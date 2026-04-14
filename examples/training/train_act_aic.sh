#!/usr/bin/env bash

set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-slobot/aic}"
JOB_NAME="${JOB_NAME:-act_slobot_aic}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_REVISION="${DATASET_REVISION:-}"
POLICY_PATH="${POLICY_PATH:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

cmd=(
  uv run lerobot-train
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--policy.device=${POLICY_DEVICE}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  cmd+=("--dataset.root=${DATASET_ROOT}")
fi

if [[ -n "${DATASET_REVISION}" ]]; then
  cmd+=("--dataset.revision=${DATASET_REVISION}")
fi

if [[ -n "${POLICY_PATH}" ]]; then
  cmd+=("--policy.path=${POLICY_PATH}")
else
  cmd+=("--policy.type=act")
fi

if [[ -n "${POLICY_REPO_ID}" ]]; then
  cmd+=("--policy.repo_id=${POLICY_REPO_ID}")
  cmd+=("--policy.push_to_hub=true")
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
