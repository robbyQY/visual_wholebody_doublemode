#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/visual_wholebody"
SCRIPT_DIR="${ROOT_DIR}/low-level/legged_gym/scripts"
LOG_ROOT="/data/logs"

PROJ_NAME="b1z1-low"
EXPTID="train_default"
TASK="b1z1"
MAX_ITERATIONS="10"
NUM_ENVS=""
MIXED_HEIGHT_REFERENCE=false
TRUNK_FOLLOW_RATIO=""
TRAIN_MODE="fresh"      # Training mode: fresh | resume | load
LOAD_EXPTID=""          # only used when TRAIN_MODE=load
LOAD_CKPT="-1"          # only used when TRAIN_MODE=load

DISABLE_WANDB=false
OBSERVE_GAIT_COMMANDS=true

export LEGGED_GYM_LOG_ROOT="${LOG_ROOT}"

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

TRAIN_MODE_ARGS=()
case "${TRAIN_MODE}" in
  fresh)
    ;;
  resume)
    TRAIN_MODE_ARGS+=(--train_mode resume)
    ;;
  load)
    if [[ -z "${LOAD_EXPTID}" ]]; then
      echo "LOAD_EXPTID must be set when TRAIN_MODE=load"
      exit 1
    fi
    TRAIN_MODE_ARGS+=(--train_mode load --load_exptid "${LOAD_EXPTID}" --checkpoint "${LOAD_CKPT}")
    ;;
  *)
    echo "Unsupported TRAIN_MODE=${TRAIN_MODE}. Expected one of: fresh, resume, load"
    exit 1
    ;;
esac

"${LAUNCH_CMD[@]}" \
  --proj_name "${PROJ_NAME}" \
  --exptid "${EXPTID}" \
  --task "${TASK}" \
  "${TRAIN_MODE_ARGS[@]}" \
  $([[ "${OBSERVE_GAIT_COMMANDS}" == true ]] && echo --observe_gait_commands) \
  $([[ "${MIXED_HEIGHT_REFERENCE}" == true ]] && echo --mixed_height_reference) \
  $([[ -n "${TRUNK_FOLLOW_RATIO}" ]] && echo --trunk_follow_ratio "${TRUNK_FOLLOW_RATIO}") \
  $([[ -n "${NUM_ENVS}" ]] && echo --num_envs "${NUM_ENVS}") \
  $([[ -n "${MAX_ITERATIONS}" ]] && echo --max_iterations "${MAX_ITERATIONS}")
