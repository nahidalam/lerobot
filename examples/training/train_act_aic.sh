#!/usr/bin/env bash

set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_DATASET_REPO_ID:-slobot/aic}}"
DATASET_ROOT="${DATASET_ROOT:-${SOURCE_DATASET_ROOT:-}}"
DATASET_REVISION="${DATASET_REVISION:-${SOURCE_DATASET_REVISION:-}}"
ACT_KEEP_CAMERAS="${ACT_KEEP_CAMERAS:-all}"
TASK_ID_KEY="${TASK_ID_KEY:-auto}"
JOB_NAME="${JOB_NAME:-act_slobot_aic}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
CONFIG_OUTPUT="${CONFIG_OUTPUT:-}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-act-aic}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
POLICY_PATH="${POLICY_PATH:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
CHECK_ONLY="${CHECK_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_HF_CHECK="${SKIP_HF_CHECK:-false}"

cmd=(
  uv run python examples/training/act_aic_finetune.py
  "--dataset-repo-id=${DATASET_REPO_ID}"
  "--keep-cameras=${ACT_KEEP_CAMERAS}"
  "--task-id-key=${TASK_ID_KEY}"
  "--job-name=${JOB_NAME}"
  "--output-dir=${OUTPUT_DIR}"
  "--policy-device=${POLICY_DEVICE}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  cmd+=("--dataset-root=${DATASET_ROOT}")
fi

if [[ -n "${DATASET_REVISION}" ]]; then
  cmd+=("--revision=${DATASET_REVISION}")
fi

if [[ -n "${CONFIG_OUTPUT}" ]]; then
  cmd+=("--config-output=${CONFIG_OUTPUT}")
fi

if [[ -n "${POLICY_PATH}" ]]; then
  cmd+=("--policy-path=${POLICY_PATH}")
fi

if [[ "${WANDB_ENABLE}" == "true" ]]; then
  cmd+=("--wandb")
  cmd+=("--wandb-project=${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=("--wandb-entity=${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_MODE}" ]]; then
    cmd+=("--wandb-mode=${WANDB_MODE}")
  fi
fi

if [[ -n "${POLICY_REPO_ID}" ]]; then
  cmd+=("--policy-repo-id=${POLICY_REPO_ID}")
  cmd+=("--push-to-hub")
elif [[ "${PUSH_TO_HUB}" == "true" ]]; then
  cmd+=("--push-to-hub")
fi

if [[ "${CHECK_ONLY}" == "true" ]]; then
  cmd+=("--check-only")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=("--dry-run")
fi

if [[ "${SKIP_HF_CHECK}" == "true" ]]; then
  cmd+=("--skip-hf-check")
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
