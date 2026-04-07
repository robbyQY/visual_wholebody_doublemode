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
#   机械臂末端位置控制
#     Y / H : 末端 x 方向 +0.05 / -0.05
#     U / J : 末端 y 方向 +0.05 / -0.05
#     I / K : 末端 z 方向 +0.05 / -0.05
#
#   机械臂末端姿态控制
#     Z / X : roll  +0.05 / -0.05
#     C / M : pitch +0.05 / -0.05
#     B / N : yaw   +0.05 / -0.05
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

USE_INTERFACE=true
HEADLESS=false
TELEOP_MODE=true
TELEOP_INPUT_REGULARIZATION=false
USE_JIT=false

[[ -f "${SRC_CKPT}" ]] || { echo "Checkpoint not found: ${SRC_CKPT}"; exit 1; }
export LEGGED_GYM_LOG_ROOT="${LOG_ROOT}"

cd "${SCRIPT_DIR}"
SCRIPT="play.py"
[[ "${USE_INTERFACE}" == true ]] && SCRIPT="b1z1_interface.py"

python "${SCRIPT}" \
  --exptid "${EXPTID}" \
  --task b1z1 \
  --proj_name "${PROJ_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --sim_device "cuda:${GPU_ID}" \
  --rl_device "cuda:${GPU_ID}" \
  $([[ "${HEADLESS}" == false ]] && echo --no-headless) \
  $([[ "${TELEOP_MODE}" == true ]] && echo --teleop_mode) \
  $([[ "${TELEOP_INPUT_REGULARIZATION}" == true ]] && echo --teleop_input_regularization) \
  $([[ "${USE_JIT}" == true ]] && echo --use_jit)
