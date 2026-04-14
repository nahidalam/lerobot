#!/usr/bin/env bash

set -euo pipefail

SOURCE_DATASET_REPO_ID="${SOURCE_DATASET_REPO_ID:-slobot/aic}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-}"
SOURCE_DATASET_REVISION="${SOURCE_DATASET_REVISION:-}"
PREPARE_ACT_DATASET="${PREPARE_ACT_DATASET:-true}"
ACT_DATASET_REPO_ID="${ACT_DATASET_REPO_ID:-slobot/aic_act}"
ACT_DATASET_ROOT="${ACT_DATASET_ROOT:-outputs/datasets/act_ready_slobot_aic}"
JOB_NAME="${JOB_NAME:-act_slobot_aic}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
POLICY_PATH="${POLICY_PATH:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

if [[ "${PREPARE_ACT_DATASET}" == "true" ]]; then
  prep_cmd=(
    uv run python examples/training/prepare_aic_act_dataset.py
    "--source-repo-id=${SOURCE_DATASET_REPO_ID}"
    "--prepared-repo-id=${ACT_DATASET_REPO_ID}"
    "--output-dir=${ACT_DATASET_ROOT}"
  )

  if [[ -n "${SOURCE_DATASET_ROOT}" ]]; then
    prep_cmd+=("--source-root=${SOURCE_DATASET_ROOT}")
  fi

  if [[ -n "${SOURCE_DATASET_REVISION}" ]]; then
    prep_cmd+=("--source-revision=${SOURCE_DATASET_REVISION}")
  fi

  printf 'Preparing ACT-ready dataset:\n'
  printf ' %q' "${prep_cmd[@]}"
  printf '\n'
  "${prep_cmd[@]}"
fi

cmd=(
  uv run lerobot-train
  "--dataset.repo_id=${ACT_DATASET_REPO_ID}"
  "--dataset.root=${ACT_DATASET_ROOT}"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
  "--policy.device=${POLICY_DEVICE}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
)

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
