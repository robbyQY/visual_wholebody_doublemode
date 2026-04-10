#!/usr/bin/env bash
set -euo pipefail

# 运行前提:
#   1. `HEADLESS=false`
#      只有在 viewer 打开时，下面这些 viewer 按键才会生效。
#   2. viewer 窗口必须处于焦点状态
#      如果终端或别的窗口在前台，按键不会发送给仿真器。
#
# play 模式说明:
#   1. 使用 `play.py`
#   2. 不开启 `TELEOP_MODE`
#   3. 底盘 / 机械臂 / 夹爪的 teleop 控制按键在这里都无效
#   4. command / EE-goal 的自动采样语义会尽量从训练 run metadata 中恢复
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
