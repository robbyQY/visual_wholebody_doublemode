#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/visual_wholebody/low-level"
SCRIPT_DIR="${ROOT_DIR}/legged_gym/scripts"
PROJ_NAME="b1z1-low"
EXPTID="pretrained38000"
CHECKPOINT="38000"
SRC_CKPT="/data/model_${CHECKPOINT}.pt"
DST_DIR="${ROOT_DIR}/logs/${PROJ_NAME}/${EXPTID}"
DST_CKPT="${DST_DIR}/model_${CHECKPOINT}.pt"

USE_INTERFACE=false
HEADLESS=false
OBSERVE_GAIT_COMMANDS=true
TELEOP_MODE=true
USE_JIT=false

[[ -f "${SRC_CKPT}" ]] || { echo "Checkpoint not found: ${SRC_CKPT}"; exit 1; }
mkdir -p "${DST_DIR}"
cp -f "${SRC_CKPT}" "${DST_CKPT}"

cd "${SCRIPT_DIR}"
SCRIPT="play.py"
[[ "${USE_INTERFACE}" == true ]] && SCRIPT="b1z1_interface.py"

python "${SCRIPT}" \
  --exptid "${EXPTID}" \
  --task b1z1 \
  --proj_name "${PROJ_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  $([[ "${HEADLESS}" == false ]] && echo --no-headless) \
  $([[ "${OBSERVE_GAIT_COMMANDS}" == true ]] && echo --observe_gait_commands) \
  $([[ "${TELEOP_MODE}" == true ]] && echo --teleop_mode) \
  $([[ "${USE_JIT}" == true ]] && echo --use_jit)
