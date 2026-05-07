#!/usr/bin/env bash
set -euo pipefail

# 运行前提:
#   1. `HEADLESS=false`
#      只有在 viewer 打开时，Isaac Gym 才能接收键盘事件。
#   2. `TELEOP_MODE=true`
#      只有 teleop 模式开启时，下面这些机器人控制按键才会生效。
#   3. viewer 窗口必须处于焦点状态
#      如果终端或别的窗口在前台，按键不会发送给仿真器。
#
# 机器人控制按键:
#   底盘移动
#     W / S: 向前 / 向后 线速度增加0.05
#     A / D: 向左 / 向右 角速度增加0.05rad
#     Q : 线速度清零
#     E : 角速度清零
#
#   机械臂控制
#     G     : 切换 末端6DoF模式 / 关节模式
#             - 末端 -> 关节: 保持切换瞬间各关节状态为当前目标
#             - 关节 -> 末端: 将当前末端6DoF作为当前目标
#     末端6DoF模式（默认）
#       Y / H : x 方向 +0.05 / -0.05
#       U / J : y 方向 +0.05 / -0.05
#       I / K : z 方向 +0.05 / -0.05
#       Z / X : roll  +0.05 / -0.05
#       C / M : pitch +0.05 / -0.05
#       B / N : yaw   +0.05 / -0.05
#       L     : 恢复机械臂末端位姿默认值
#     关节模式（复用同一组按键）
#       Y / H : joint1 +0.05 / -0.05
#       U / J : joint2 +0.05 / -0.05
#       I / K : joint3 +0.05 / -0.05
#       Z / X : joint4 +0.05 / -0.05
#       C / M : joint5 +0.05 / -0.05
#       B / N : joint6 +0.05 / -0.05
#       L     : 恢复机械臂关节目标到默认值
#
#   夹爪控制
#     O / P : 张开0.05rad / 闭合0.05rad
#
#   mixed_height_reference模式控制（仅在该模式开启时生效）
#     R / T : z-invariant模式 / trunk-follow模式
#
# 仿真器 / viewer 常用按键:
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
TELEOP_INPUT_REGULARIZATION=false
ACTION_DELAY_MODE="auto"  # auto | undelayed | delayed
EE_GOAL_OBS_MODE=""  # empty (follow checkpoint) | command | arm_base_target (official ckpt)
ROBOT_ABLATION=""  # empty (follow checkpoint) | none | legs | trunk | arm | mass | inertial | structure | legs-inertial, etc.; combine with "," or "+"
LEG_COLLISION_SCALE=""  # empty (follow checkpoint) | e.g. 0.9
USE_JIT=false

EFFECTIVE_CHECKPOINT="${CHECKPOINT:--1}"

if [[ "${EFFECTIVE_CHECKPOINT}" != "-1" ]]; then
  [[ -f "${SRC_CKPT}" ]] || { echo "Checkpoint not found: ${SRC_CKPT}"; exit 1; }
fi
export LEGGED_GYM_LOG_ROOT="${LOG_ROOT}"

cd "${SCRIPT_DIR}"

python "manip_loco_interface.py" \
  --exptid "${EXPTID}" \
  --task "${TASK}" \
  --proj_name "${PROJ_NAME}" \
  --checkpoint "${EFFECTIVE_CHECKPOINT}" \
  --sim_device "cuda:${GPU_ID}" \
  --rl_device "cuda:${GPU_ID}" \
  $([[ "${HEADLESS}" == false ]] && echo --no-headless) \
  $([[ "${HEADLESS}" == false ]] && echo --viewer_display_mode "${VIEWER_DISPLAY_MODE}") \
  --teleop_mode \
  $([[ "${TELEOP_INPUT_REGULARIZATION}" == true ]] && echo --teleop_input_regularization) \
  --action_delay_mode "${ACTION_DELAY_MODE}" \
  $([[ -n "${EE_GOAL_OBS_MODE}" ]] && echo --ee_goal_obs_mode "${EE_GOAL_OBS_MODE}") \
  $([[ -n "${ROBOT_ABLATION}" ]] && echo --robot_ablation "${ROBOT_ABLATION}") \
  $([[ -n "${LEG_COLLISION_SCALE}" ]] && echo --leg_collision_scale "${LEG_COLLISION_SCALE}") \
  $([[ "${USE_JIT}" == true ]] && echo --use_jit)

