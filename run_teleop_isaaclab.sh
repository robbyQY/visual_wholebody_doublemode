#!/usr/bin/env bash
set -euo pipefail

GPU_ID="0"
ROOT_DIR="/home/leakycauldron/visual_wholebody_doublemode/low-level"
SCRIPT_DIR="${ROOT_DIR}/legged_gym/scripts"
PYTHONPATH=/workspace/visual_wholebody_doublemode/low-level:/workspace/visual_wholebody_doublemode/third_party/rsl_rl

# 可选：直接指定URDF。留空时，脚本会根据下面 mount/ablation 参数自动生成。
ROBOT_URDF_PATH=""
BASE_ROBOT="b2z1"
MOUNT_DEG="0"
# MOUNT_XYZ=(0.2 0 0.09)  # 不设置时使用默认值
ROBOT_ABLATION=""         # 例如: "legs" / "trunk" / "arm" / "legs+inertial"
LEG_COLLISION_SCALE="1.0" # 例如: "0.9"

# CKPT_PATH="/home/leakycauldron/Downloads/ckpt/调权重_robotlab_等高系_2/model_25000.pt"
CKPT_PATH="/home/leakycauldron/Downloads/ckpt/测试全向_robotlab_等高系_左侧机械臂_2/model_19400.pt"
cd "${SCRIPT_DIR}"

CMD=(
  /home/leakycauldron/IsaacLab/isaaclab.sh -p manip_loco_interface_isaaclab.py
  --ckpt_path "${CKPT_PATH}"
  --num_envs 1
  --teleop_mode
  --base_robot "${BASE_ROBOT}"
  --mount_deg "${MOUNT_DEG}"
  --robot_ablation "${ROBOT_ABLATION}"
  --leg_collision_scale "${LEG_COLLISION_SCALE}"
  --kit_args=--/rtx/verifyDriverVersion/enabled=false
  --device "cuda:${GPU_ID}"
  --enable_cameras
)

if [[ -n "${ROBOT_URDF_PATH}" ]]; then
  CMD+=(--robot_urdf_path "${ROBOT_URDF_PATH}")
fi

"${CMD[@]}"
