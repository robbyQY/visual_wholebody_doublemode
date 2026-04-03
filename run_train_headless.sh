#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/visual_wholebody"
SCRIPT_DIR="${ROOT_DIR}/low-level/legged_gym/scripts"

PROJ_NAME="b1z1-low"
EXPTID="train_default"
TASK="b1z1"
MAX_ITERATIONS="10"

LOG_DIR="${ROOT_DIR}/low-level/logs/${PROJ_NAME}/${EXPTID}"
LOG_FILE="${LOG_DIR}/train.log"

DISABLE_WANDB=true

mkdir -p "${LOG_DIR}"

if [[ "${DISABLE_WANDB}" == true ]]; then
  export WANDB_DISABLED=true
  export WANDB_SILENT=true
fi

timestamp_log() {
  while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
  done
}

{
  echo "==== Training Config ===="
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "SCRIPT_DIR=${SCRIPT_DIR}"
  echo "PROJ_NAME=${PROJ_NAME}"
  echo "EXPTID=${EXPTID}"
  echo "TASK=${TASK}"
  echo "MAX_ITERATIONS=${MAX_ITERATIONS:-<default>}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "DISABLE_WANDB=${DISABLE_WANDB}"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo
} | timestamp_log > "${LOG_FILE}"

cd "${SCRIPT_DIR}"
python train.py \
  --proj_name "${PROJ_NAME}" \
  --exptid "${EXPTID}" \
  --task "${TASK}" \
  $([[ -n "${MAX_ITERATIONS}" ]] && echo --max_iterations "${MAX_ITERATIONS}") \
  2>&1 | timestamp_log >> "${LOG_FILE}"
