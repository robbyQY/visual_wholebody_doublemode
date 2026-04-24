#!/usr/bin/env bash
set -euo pipefail

# 运行前提:
#   1. `HEADLESS=false`
#      只有在 viewer 打开时，下面这些 viewer 按键才会生效。
#   2. viewer 窗口必须处于焦点状态
#      如果终端或别的窗口在前台，按键不会发送给仿真器。
#
# 在 play 模式下仍然有效的 viewer / 仿真器按键:
#   ESC : 退出程序
#   SPACE : 暂停/继续仿真
#   F : 切换自由视角
#       - 关闭自由视角时，相机会持续跟随当前 `lookat_id` 对应的环境/机器人
#       - 开启自由视角时，可以手动拖动和浏览场景
#   [ / ] : 切换上一台 / 下一台机器人（或环境实例）
#   0~8 : 直接将相机跟随到编号 0~8 的环境实例
#   9 : 手动 reset 所有环境（当前脚本默认只有 1 个 env）
#   V : 切换 viewer sync
#       - sync 开启时，viewer 按实时方式刷新
#       - sync 关闭时，渲染/刷新节奏会变化，常用于提速观察
#   跟随视角球坐标控制（仅在 `F` 关闭时生效）
#     ← / → : 围绕目标调整方位角
#     ↑ / ↓ : 调整俯仰角
#     PageUp / PageDown : 缩小 / 增大跟随半径

GPU_ID="1"
ROOT_DIR="/workspace/visual_wholebody/low-level"
SCRIPT_DIR="${ROOT_DIR}/legged_gym/scripts"
LOG_ROOT="/data/logs"

TASK="b1z1"
PROJ_NAME="${TASK}-low"
EXPTID="train_default"
CHECKPOINT="45000"
CKPT_DIR="${LOG_ROOT}/${PROJ_NAME}/${EXPTID}"
SRC_CKPT="${CKPT_DIR}/model_${CHECKPOINT}.pt"

HEADLESS=false
VIEWER_DISPLAY_MODE="mesh"  # mesh | collision；只影响viewer显示，不影响实际动力学
ACTION_DELAY_MODE="auto"  # auto | undelayed | delayed
EE_GOAL_OBS_MODE=""  # empty (follow checkpoint) | command | arm_base_target (official ckpt)
ROBOT_ABLATION=""  # empty (follow checkpoint) | none | legs | trunk | arm | mass | inertial | structure
CURRICULUM_ITER=""  # empty (start from curriculum step 0) | e.g. 3000
CURRICULUM_PROGRESS=""  # empty | normalized progress in [0, 1], e.g. 0.6
TRUNK_FOLLOW_RATIO="0.0"  # 0.0~1.0
PRINT_FORCE_SENSOR_EVERY=""  # empty (disable) | e.g. 10
STATIC_DEFAULT_POSE=false
USE_JIT=false

if [[ "${STATIC_DEFAULT_POSE}" != true ]]; then
  [[ -f "${SRC_CKPT}" ]] || { echo "Checkpoint not found: ${SRC_CKPT}"; exit 1; }
fi
export LEGGED_GYM_LOG_ROOT="${LOG_ROOT}"

cd "${SCRIPT_DIR}"

python "play.py" \
  --exptid "${EXPTID}" \
  --task "${TASK}" \
  --proj_name "${PROJ_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --sim_device "cuda:${GPU_ID}" \
  --rl_device "cuda:${GPU_ID}" \
  $([[ "${HEADLESS}" == false ]] && echo --no-headless) \
  $([[ "${HEADLESS}" == false ]] && echo --viewer_display_mode "${VIEWER_DISPLAY_MODE}") \
  --action_delay_mode "${ACTION_DELAY_MODE}" \
  $([[ -n "${EE_GOAL_OBS_MODE}" ]] && echo --ee_goal_obs_mode "${EE_GOAL_OBS_MODE}") \
  $([[ -n "${ROBOT_ABLATION}" ]] && echo --robot_ablation "${ROBOT_ABLATION}") \
  $([[ -n "${TRUNK_FOLLOW_RATIO}" ]] && echo --trunk_follow_ratio "${TRUNK_FOLLOW_RATIO}") \
  $([[ -n "${PRINT_FORCE_SENSOR_EVERY}" ]] && echo --print_force_sensor_every "${PRINT_FORCE_SENSOR_EVERY}") \
  $([[ "${STATIC_DEFAULT_POSE}" == true ]] && echo --static_default_pose) \
  $([[ -n "${CURRICULUM_ITER}" ]] && echo --curriculum_iter "${CURRICULUM_ITER}") \
  $([[ -n "${CURRICULUM_PROGRESS}" ]] && echo --curriculum_progress "${CURRICULUM_PROGRESS}") \
  $([[ "${USE_JIT}" == true ]] && echo --use_jit)
