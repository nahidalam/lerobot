#!/usr/bin/env bash

set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-${SOURCE_DATASET_REPO_ID:-slobot/aic}}"
DATASET_ROOT="${DATASET_ROOT:-${SOURCE_DATASET_ROOT:-}}"
DATASET_REVISION="${DATASET_REVISION:-${SOURCE_DATASET_REVISION:-main}}"
PI05_KEEP_CAMERAS="${PI05_KEEP_CAMERAS:-all}"
TASK_ID_KEY="${TASK_ID_KEY:-none}"
JOB_NAME="${JOB_NAME:-pi05_slobot_aic}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
CONFIG_OUTPUT="${CONFIG_OUTPUT:-}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_DTYPE="${POLICY_DTYPE:-bfloat16}"
POLICY_PATH="${POLICY_PATH:-lerobot/pi05_base}"
NORMALIZATION="${NORMALIZATION:-mean_std}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-3000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-500}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-pi05-aic}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
CHECK_ONLY="${CHECK_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_HF_CHECK="${SKIP_HF_CHECK:-false}"

cmd=(
  uv run python examples/training/pi05_aic_finetune.py
  "--dataset-repo-id=${DATASET_REPO_ID}"
  "--revision=${DATASET_REVISION}"
  "--keep-cameras=${PI05_KEEP_CAMERAS}"
  "--task-id-key=${TASK_ID_KEY}"
  "--job-name=${JOB_NAME}"
  "--output-dir=${OUTPUT_DIR}"
  "--policy-device=${POLICY_DEVICE}"
  "--policy-dtype=${POLICY_DTYPE}"
  "--policy-path=${POLICY_PATH}"
  "--normalization=${NORMALIZATION}"
  "--batch-size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--num-workers=${NUM_WORKERS}"
  "--log-freq=${LOG_FREQ}"
  "--save-freq=${SAVE_FREQ}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  cmd+=("--dataset-root=${DATASET_ROOT}")
fi

if [[ -n "${CONFIG_OUTPUT}" ]]; then
  cmd+=("--config-output=${CONFIG_OUTPUT}")
fi

if [[ "${COMPILE_MODEL}" == "true" ]]; then
  cmd+=("--compile-model")
else
  cmd+=("--no-compile-model")
fi

if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  cmd+=("--gradient-checkpointing")
else
  cmd+=("--no-gradient-checkpointing")
fi

if [[ "${FREEZE_VISION_ENCODER}" == "true" ]]; then
  cmd+=("--freeze-vision-encoder")
else
  cmd+=("--no-freeze-vision-encoder")
fi

if [[ "${TRAIN_EXPERT_ONLY}" == "true" ]]; then
  cmd+=("--train-expert-only")
else
  cmd+=("--no-train-expert-only")
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
