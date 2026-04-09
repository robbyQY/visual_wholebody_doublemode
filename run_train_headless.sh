#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/visual_wholebody"
SCRIPT_DIR="${ROOT_DIR}/low-level/legged_gym/scripts"
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="/data/logs"

PROJ_NAME="b1z1-low"
EXPTID="train_default"
TASK="b1z1"
MAX_ITERATIONS="10"
NUM_ENVS=""
MIXED_HEIGHT_REFERENCE=false
TRUNK_FOLLOW_RATIO=""
OMNIDIRECTIONAL_POS_Y=false
EE_GOAL_OBS_MODE="command"  # command | arm_base_target
LIN_VEL_X_MIN_SCHEDULE=()
LIN_VEL_X_MAX_SCHEDULE=()
ANG_VEL_YAW_SCHEDULE=()
TRACKING_LIN_VEL_MAX_SCHEDULE=()
TRACKING_ANG_VEL_SCHEDULE=()
MIXING_SCHEDULE=()
PRIV_REG_COEF_SCHEDULE=()
TRAIN_MODE="fresh"      # Training mode: fresh | resume | load
LOAD_EXPTID=""          # only used when TRAIN_MODE=load
LOAD_CKPT="-1"          # only used when TRAIN_MODE=load
TRAIN_LOG_EVERY="100"
NOHUP_BACKGROUND=false

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

timestamp_stderr_to_file() {
  local target_file="$1"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
  done >> "${target_file}"
}

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

CURRICULUM_ARGS=()
if (( ${#LIN_VEL_X_MIN_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--lin_vel_x_min_schedule "$(IFS=,; echo "${LIN_VEL_X_MIN_SCHEDULE[*]}")")
fi
if (( ${#LIN_VEL_X_MAX_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--lin_vel_x_max_schedule "$(IFS=,; echo "${LIN_VEL_X_MAX_SCHEDULE[*]}")")
fi
if (( ${#ANG_VEL_YAW_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--ang_vel_yaw_schedule "$(IFS=,; echo "${ANG_VEL_YAW_SCHEDULE[*]}")")
fi
if (( ${#TRACKING_LIN_VEL_MAX_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--tracking_lin_vel_max_schedule "$(IFS=,; echo "${TRACKING_LIN_VEL_MAX_SCHEDULE[*]}")")
fi
if (( ${#TRACKING_ANG_VEL_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--tracking_ang_vel_schedule "$(IFS=,; echo "${TRACKING_ANG_VEL_SCHEDULE[*]}")")
fi
if (( ${#MIXING_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--mixing_schedule "$(IFS=,; echo "${MIXING_SCHEDULE[*]}")")
fi
if (( ${#PRIV_REG_COEF_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--priv_reg_coef_schedule "$(IFS=,; echo "${PRIV_REG_COEF_SCHEDULE[*]}")")
fi

ERROR_LOG="${SH_DIR}/error.log"

TRAIN_CMD=(
  "${LAUNCH_CMD[@]}"
  --proj_name "${PROJ_NAME}"
  --exptid "${EXPTID}"
  --task "${TASK}"
  "${TRAIN_MODE_ARGS[@]}"
  "${CURRICULUM_ARGS[@]}"
  --train_log_every "${TRAIN_LOG_EVERY}"
  --ee_goal_obs_mode "${EE_GOAL_OBS_MODE}"
)

if [[ "${OBSERVE_GAIT_COMMANDS}" == true ]]; then
  TRAIN_CMD+=(--observe_gait_commands)
fi
if [[ "${MIXED_HEIGHT_REFERENCE}" == true ]]; then
  TRAIN_CMD+=(--mixed_height_reference)
fi
if [[ -n "${TRUNK_FOLLOW_RATIO}" ]]; then
  TRAIN_CMD+=(--trunk_follow_ratio "${TRUNK_FOLLOW_RATIO}")
fi
if [[ "${OMNIDIRECTIONAL_POS_Y}" == true ]]; then
  TRAIN_CMD+=(--omnidirectional_pos_y)
fi
if [[ -n "${NUM_ENVS}" ]]; then
  TRAIN_CMD+=(--num_envs "${NUM_ENVS}")
fi
if [[ -n "${MAX_ITERATIONS}" ]]; then
  TRAIN_CMD+=(--max_iterations "${MAX_ITERATIONS}")
fi

if [[ "${NOHUP_BACKGROUND}" == true ]]; then
  nohup "${TRAIN_CMD[@]}" > /dev/null 2> >(timestamp_stderr_to_file "${ERROR_LOG}") &
  echo "Started background training (PID=$!)."
  echo "stderr -> ${ERROR_LOG}"
else
  "${TRAIN_CMD[@]}" 2> >(timestamp_stderr_to_file "${ERROR_LOG}")
fi
