#!/usr/bin/env bash
set -euo pipefail

GPU_ID="0"
ROOT_DIR="/home/leakycauldron/visual_wholebody_doublemode/low-level"
SCRIPT_DIR="${ROOT_DIR}/legged_gym/scripts"
PYTHONPATH=/workspace/visual_wholebody_doublemode/low-level:/workspace/visual_wholebody_doublemode/third_party/rsl_rl

ROBOT_URDF_PATH="/home/leakycauldron/visual_wholebody_doublemode/b2z1/urdf/b2z1_isaacsim_mesh_axis_fixed.urdf"
CKPT_PATH="/home/leakycauldron/Downloads/ckpt/调权重_robotlab_等高系_2/model_25000.pt"

cd "${SCRIPT_DIR}"
# PYTHONPATH=/workspace/visual_wholebody_doublemode/low-level:/workspace/visual_wholebody_doublemode/third_party/rsl_rl \
/home/leakycauldron/IsaacLab/isaaclab.sh -p manip_loco_interface_isaaclab.py \
  --robot_urdf_path "${ROBOT_URDF_PATH}" \
  --ckpt_path "${CKPT_PATH}" \
  --num_envs 1 \
  --teleop_mode \
  --kit_args=--/rtx/verifyDriverVersion/enabled=false \
  --device "cuda:${GPU_ID}" \
  --enable_cameras
