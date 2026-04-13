#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/visual_wholebody"
SCRIPT_DIR="${ROOT_DIR}/low-level/legged_gym/scripts"
SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="/data/logs"
RDZV_PORT="${RDZV_PORT:-}"

PROJ_NAME="b1z1-low"
TASK="b1z1"
EXPTID="train"

# Training control
TRAIN_MODE="fresh"      # Training mode: fresh | resume | load
LOAD_EXPTID=""          # only used when TRAIN_MODE=load
LOAD_CKPT="-1"          # only used when TRAIN_MODE=load
MAX_ITERATIONS=""
NUM_ENVS=""
TRAIN_LOG_EVERY="100"

# Task / observation options
EE_GOAL_OBS_MODE="command"  # command | arm_base_target
OBSERVE_GAIT_COMMANDS=true
MIXED_HEIGHT_REFERENCE=false
TRUNK_FOLLOW_RATIO="0.5"
OMNIDIRECTIONAL_POS_Y=false
ENABLE_DYNAMIC_GAIT_FREQUENCY=false  # min/max gait frequency = 1.2/2.8

# Curriculum schedules
LIN_VEL_X_MIN_SCHEDULE=()
LIN_VEL_X_MAX_SCHEDULE=()
ANG_VEL_YAW_SCHEDULE=()
MIXING_SCHEDULE=()
PRIV_REG_COEF_SCHEDULE=()

# Runtime toggles
NOHUP_BACKGROUND=true
DISABLE_WANDB=false

EXPTID_SUFFIX=""
if [[ "${OMNIDIRECTIONAL_POS_Y}" == true ]]; then
  EXPTID_SUFFIX+="_omnidirec"
fi
if [[ "${ENABLE_DYNAMIC_GAIT_FREQUENCY}" == true ]]; then
  EXPTID_SUFFIX+="_adaptivegait"
fi
if [[ "${MIXED_HEIGHT_REFERENCE}" == true ]]; then
  EXPTID_SUFFIX+="_doublemode"
fi
EXPTID+="${EXPTID_SUFFIX}"

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

RUN_INSTANCE_ID="$(date +%Y%m%d_%H%M%S)_$$"

timestamp_stderr_to_file() {
  local target_file="$1"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${line}"
  done >> "${target_file}"
}

pick_free_port() {
  python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

pick_torchrun_flag() {
  local torchrun_help="$1"
  local dashed_flag="$2"
  local underscored_flag="$3"

  if [[ "${torchrun_help}" == *"${dashed_flag}"* ]]; then
    printf '%s' "${dashed_flag}"
  else
    printf '%s' "${underscored_flag}"
  fi
}

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  if [[ -z "${RDZV_PORT}" ]]; then
    RDZV_PORT="$(pick_free_port)"
  fi
  RDZV_ENDPOINT="127.0.0.1:${RDZV_PORT}"
  TORCHRUN_HELP="$(torchrun --help 2>&1 || true)"
  RDZV_BACKEND_FLAG="$(pick_torchrun_flag "${TORCHRUN_HELP}" "--rdzv-backend" "--rdzv_backend")"
  RDZV_ENDPOINT_FLAG="$(pick_torchrun_flag "${TORCHRUN_HELP}" "--rdzv-endpoint" "--rdzv_endpoint")"
  RDZV_ID_FLAG="$(pick_torchrun_flag "${TORCHRUN_HELP}" "--rdzv-id" "--rdzv_id")"
  LAUNCH_CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node "${NUM_GPUS}"
    "${RDZV_BACKEND_FLAG}=c10d"
    "${RDZV_ENDPOINT_FLAG}" "${RDZV_ENDPOINT}"
    "${RDZV_ID_FLAG}" "${PROJ_NAME}-${EXPTID}-${RUN_INSTANCE_ID}"
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
if (( ${#MIXING_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--mixing_schedule "$(IFS=,; echo "${MIXING_SCHEDULE[*]}")")
fi
if (( ${#PRIV_REG_COEF_SCHEDULE[@]} > 0 )); then
  CURRICULUM_ARGS+=(--priv_reg_coef_schedule "$(IFS=,; echo "${PRIV_REG_COEF_SCHEDULE[*]}")")
fi

GAIT_FREQUENCY_ARGS=()
if [[ "${ENABLE_DYNAMIC_GAIT_FREQUENCY}" == true ]]; then
  GAIT_FREQUENCY_ARGS+=(--gait_frequency_min "1.2" --gait_frequency_max "2.8")
fi

ERROR_LOG="${SH_DIR}/error_${RUN_INSTANCE_ID}.log"

TRAIN_CMD=(
  "${LAUNCH_CMD[@]}"
  --proj_name "${PROJ_NAME}"
  --exptid "${EXPTID}"
  --task "${TASK}"
  "${TRAIN_MODE_ARGS[@]}"
  "${CURRICULUM_ARGS[@]}"
  "${GAIT_FREQUENCY_ARGS[@]}"
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
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    echo "Using rendezvous endpoint ${RDZV_ENDPOINT}"
  fi
  nohup "${TRAIN_CMD[@]}" > /dev/null 2> >(timestamp_stderr_to_file "${ERROR_LOG}") &
  echo "Started background training (PID=$!)."
  echo "stderr -> ${ERROR_LOG}"
else
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    echo "Using rendezvous endpoint ${RDZV_ENDPOINT}"
  fi
  "${TRAIN_CMD[@]}" 2> >(timestamp_stderr_to_file "${ERROR_LOG}")
fi
