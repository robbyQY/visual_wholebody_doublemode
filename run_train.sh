#!/usr/bin/env bash
set -euo pipefail

SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${SH_DIR}/run_train_lib.sh"

ROOT_DIR="/workspace/visual_wholebody"
SCRIPT_DIR="${ROOT_DIR}/low-level/legged_gym/scripts"
LOG_ROOT="/data/logs"
QUEUE_ROOT="${ROOT_DIR}/.run_train_queue"
QUEUE_LOCK_FILE="${QUEUE_ROOT}/queue.lock"
QUEUE_JOBS_DIR="${QUEUE_ROOT}/jobs"
RDZV_PORT_OVERRIDE="${RDZV_PORT:-}"

RUN_TRAIN_LIB_TEMPLATE="${SH_DIR}/run_train_lib.sh"
RUN_TRAIN_WORKER_TEMPLATE="${SH_DIR}/run_train_worker.sh"

TASK="b1z1"
PROJ_NAME=""

# Training control
TRAIN_MODE="fresh"      # fresh | resume | load
LOAD_EXPTID=""          # used when TRAIN_MODE=load or to select an exact resume run
LOAD_CKPT="-1"          # used when TRAIN_MODE=load
SEED=""
MAX_ITERATIONS=""
NUM_ENVS=""
REQUESTED_NUM_GPUS="1"
TRAIN_LOG_EVERY="100"
QUEUE_POLL_INTERVAL_S="15"
WANDB_GROUP=""

# Task / observation options
EE_GOAL_OBS_MODE="command"  # command | arm_base_target
ROBOT_ABLATION=""          # empty | none | legs | trunk | arm | mass | inertial | structure | legs-inertial, etc.; combine with "," or "+"
LEG_COLLISION_SCALE=""     # empty | e.g. 0.9; scales leg collision geometry independently of ROBOT_ABLATION
REWARD_SCALE_PRESET="legacy"  # legacy | height_flexible | robotlab_b2
OBSERVE_GAIT_COMMANDS=true
ENABLE_DYNAMIC_GAIT_FREQUENCY=false  # min/max gait frequency = 1.2/2.8
MIXED_HEIGHT_REFERENCE=true
TRUNK_FOLLOW_RATIO="0.5"
OMNIDIRECTIONAL_POS_Y=false
MOUNT_DEG=""               # 0 | 90 | 180 | 270
MOUNT_X=""                 # empty -> task config default
MOUNT_Y=""                 # empty -> task config default

# Batch mode
TASK_OPTIONS_CSV=""

# Curriculum schedules
LIN_VEL_X_MIN_SCHEDULE=()
LIN_VEL_X_MAX_SCHEDULE=()
ANG_VEL_YAW_SCHEDULE=()
NON_OMNI_GOAL_YAW_SCHEDULE=()
MIXING_SCHEDULE=()
PRIV_REG_COEF_SCHEDULE=()

# Runtime toggles
NOHUP_BACKGROUND=true
DISABLE_WANDB=false

TRAIN_SCALAR_VARS=(
  PROJ_NAME
  TASK
  TRAIN_MODE
  LOAD_EXPTID
  LOAD_CKPT
  SEED
  MAX_ITERATIONS
  NUM_ENVS
  REQUESTED_NUM_GPUS
  TRAIN_LOG_EVERY
  QUEUE_POLL_INTERVAL_S
  WANDB_GROUP
  EE_GOAL_OBS_MODE
  ROBOT_ABLATION
  LEG_COLLISION_SCALE
  REWARD_SCALE_PRESET
  OBSERVE_GAIT_COMMANDS
  MIXED_HEIGHT_REFERENCE
  TRUNK_FOLLOW_RATIO
  OMNIDIRECTIONAL_POS_Y
  MOUNT_DEG
  MOUNT_X
  MOUNT_Y
  ENABLE_DYNAMIC_GAIT_FREQUENCY
  NOHUP_BACKGROUND
  DISABLE_WANDB
)

capture_base_config() {
  local var_name
  for var_name in "${TRAIN_SCALAR_VARS[@]}"; do
    printf -v "BASE_${var_name}" '%s' "${!var_name}"
  done
}

restore_base_config() {
  local var_name
  local base_var_name
  for var_name in "${TRAIN_SCALAR_VARS[@]}"; do
    base_var_name="BASE_${var_name}"
    printf -v "${var_name}" '%s' "${!base_var_name}"
  done
  unset CSV_ROW_INDEX || true
}

capture_base_config

if [[ -n "${TASK_OPTIONS_CSV}" ]]; then
  BATCH_QUEUED_COUNT=0
  while IFS= read -r csv_row_assignments; do
    [[ -n "${csv_row_assignments}" ]] || continue
    restore_base_config
    eval "${csv_row_assignments}"
    enqueue_training_job
    BATCH_QUEUED_COUNT="$(( BATCH_QUEUED_COUNT + 1 ))"
  done < <(emit_csv_override_rows "${TASK_OPTIONS_CSV}")

  echo "Queued ${BATCH_QUEUED_COUNT} jobs from ${TASK_OPTIONS_CSV}."
else
  enqueue_training_job
fi
