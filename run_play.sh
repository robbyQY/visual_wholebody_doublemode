#!/usr/bin/env bash
set -euo pipefail

# Standard play entry:
# - uses `play.py`
# - keeps `TELEOP_MODE` off
# - command / EE-goal sampling semantics are restored from the training run metadata

GPU_ID="1"
ROOT_DIR="/workspace/visual_wholebody/low-level"
SCRIPT_DIR="${ROOT_DIR}/legged_gym/scripts"
LOG_ROOT="/data/logs"

PROJ_NAME="b1z1-low"
EXPTID="train_default"
CHECKPOINT="45000"
CKPT_DIR="${LOG_ROOT}/${PROJ_NAME}/${EXPTID}"
SRC_CKPT="${CKPT_DIR}/model_${CHECKPOINT}.pt"

HEADLESS=false
ACTION_DELAY_MODE="auto"  # auto | undelayed | delayed
EE_GOAL_OBS_MODE="command"  # command | arm_base_target (official ckpt)
USE_JIT=false

[[ -f "${SRC_CKPT}" ]] || { echo "Checkpoint not found: ${SRC_CKPT}"; exit 1; }
export LEGGED_GYM_LOG_ROOT="${LOG_ROOT}"

cd "${SCRIPT_DIR}"

python "play.py" \
  --exptid "${EXPTID}" \
  --task b1z1 \
  --proj_name "${PROJ_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --sim_device "cuda:${GPU_ID}" \
  --rl_device "cuda:${GPU_ID}" \
  $([[ "${HEADLESS}" == false ]] && echo --no-headless) \
  --action_delay_mode "${ACTION_DELAY_MODE}" \
  --ee_goal_obs_mode "${EE_GOAL_OBS_MODE}" \
  $([[ "${USE_JIT}" == true ]] && echo --use_jit)
