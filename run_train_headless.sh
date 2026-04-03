#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/visual_wholebody"
SCRIPT_DIR="${ROOT_DIR}/low-level/legged_gym/scripts"

PROJ_NAME="b1z1-low"
EXPTID="train_default"
TASK="b1z1"
MAX_ITERATIONS="10"
NUM_ENVS=""
MIXED_HEIGHT_REFERENCE=false
TRUNK_FOLLOW_RATIO=""

LOG_DIR="${ROOT_DIR}/low-level/logs/${PROJ_NAME}/${EXPTID}"
LOG_FILE="${LOG_DIR}/train.log"

DISABLE_WANDB=true

mkdir -p "${LOG_DIR}"

if [[ "${DISABLE_WANDB}" == true ]]; then
  export WANDB_DISABLED=true
  export WANDB_SILENT=true
fi

NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"

if [[ -n "${NUM_ENVS}" ]]; then
  TOTAL_NUM_ENVS="${NUM_ENVS}"
else
  TOTAL_NUM_ENVS="<config>"
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  DISTRIBUTED=true
else
  DISTRIBUTED=false
fi

if [[ "${TOTAL_NUM_ENVS}" != "<config>" && "${NUM_GPUS}" -gt 0 ]]; then
  if (( TOTAL_NUM_ENVS % NUM_GPUS == 0 )); then
    NUM_ENVS_PER_GPU="$(( TOTAL_NUM_ENVS / NUM_GPUS ))"
  else
    NUM_ENVS_PER_GPU="<invalid: not divisible>"
  fi
else
  NUM_ENVS_PER_GPU="<resolved in Python>"
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
  echo "NUM_ENVS=${TOTAL_NUM_ENVS}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "DISABLE_WANDB=${DISABLE_WANDB}"
  echo "MIXED_HEIGHT_REFERENCE=${MIXED_HEIGHT_REFERENCE}"
  echo "TRUNK_FOLLOW_RATIO=${TRUNK_FOLLOW_RATIO:-<config>}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "DISTRIBUTED=${DISTRIBUTED}"
  echo "NUM_ENVS_PER_GPU=${NUM_ENVS_PER_GPU}"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo
} | timestamp_log > "${LOG_FILE}"

cd "${SCRIPT_DIR}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCH_CMD=(
    torchrun
    --standalone
    --nnodes=1
    --nproc_per_node "${NUM_GPUS}"
    train.py
  )
else
  LAUNCH_CMD=(
    python
    train.py
  )
fi

{
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "LAUNCH_CMD=${LAUNCH_CMD[*]}"
} | timestamp_log >> "${LOG_FILE}"

"${LAUNCH_CMD[@]}" \
  --proj_name "${PROJ_NAME}" \
  --exptid "${EXPTID}" \
  --task "${TASK}" \
  $([[ "${MIXED_HEIGHT_REFERENCE}" == true ]] && echo --mixed_height_reference) \
  $([[ -n "${TRUNK_FOLLOW_RATIO}" ]] && echo --trunk_follow_ratio "${TRUNK_FOLLOW_RATIO}") \
  $([[ -n "${NUM_ENVS}" ]] && echo --num_envs "${NUM_ENVS}") \
  $([[ -n "${MAX_ITERATIONS}" ]] && echo --max_iterations "${MAX_ITERATIONS}") \
  2>&1 | timestamp_log >> "${LOG_FILE}"
