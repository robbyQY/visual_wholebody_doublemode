#!/usr/bin/env bash

run_train_die() {
  echo "$*" >&2
  exit 1
}

get_total_gpu_count() {
  python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}

write_array_declaration() {
  local name="$1"
  shift
  printf '%s=(' "${name}"
  local item
  for item in "$@"; do
    printf '%q ' "${item}"
  done
  printf ')\n'
}

normalize_bool_value() {
  local raw_value="${1:-}"
  local normalized
  normalized="$(printf '%s' "${raw_value}" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in
    true|1|yes|y|on)
      printf 'true\n'
      ;;
    false|0|no|n|off)
      printf 'false\n'
      ;;
    *)
      return 1
      ;;
  esac
}

validate_optional_number() {
  local raw_value="${1:-}"
  local field_name="$2"
  if [[ -z "${raw_value}" ]]; then
    return 0
  fi
  if [[ "${raw_value}" =~ ^[-+]?[0-9]*\.?[0-9]+$ ]]; then
    return 0
  fi
  run_train_die "${field_name} must be numeric, got: ${raw_value}"
}

generate_train_exptid() {
  local reward_scale_preset="$1"
  local mixed_height_reference="$2"
  local trunk_follow_ratio="$3"
  local omnidirectional_pos_y="$4"
  local mount_deg="$5"
  local enable_dynamic_gait_frequency="$6"
  local exptid
  local -a exptid_parts=()

  case "${reward_scale_preset}" in
    height_flexible)
      exptid_parts+=("下蹲")
      ;;
    legacy)
      exptid_parts+=("前倾")
      ;;
    *)
      run_train_die "Unsupported REWARD_SCALE_PRESET=${reward_scale_preset}. Expected one of: legacy, height_flexible"
      ;;
  esac

  if [[ "${mixed_height_reference}" != true ]]; then
    exptid_parts+=("等高系")
  else
    case "${trunk_follow_ratio}" in
      0|0.0|0.00)
        exptid_parts+=("等高系")
        ;;
      1|1.0|1.00)
        exptid_parts+=("机体系")
        ;;
      *)
        exptid_parts+=("${trunk_follow_ratio}机体系")
        ;;
    esac
  fi

  if [[ "${omnidirectional_pos_y}" == true ]]; then
    exptid_parts+=("全向采样")
  fi

  case "${mount_deg}" in
    0)
      ;;
    90)
      exptid_parts+=("左侧机械臂")
      ;;
    180)
      exptid_parts+=("倒装机械臂")
      ;;
    270)
      exptid_parts+=("右侧机械臂")
      ;;
    *)
      run_train_die "Unsupported MOUNT_DEG=${mount_deg}. Expected one of: 0, 90, 180, 270"
      ;;
  esac

  if [[ "${enable_dynamic_gait_frequency}" == true ]]; then
    exptid_parts+=("动态步频")
  fi

  if (( ${#exptid_parts[@]} == 0 )); then
    run_train_die "Failed to auto-generate EXPTID."
  fi

  exptid="$(IFS=_; echo "${exptid_parts[*]}")"
  printf '%s\n' "${exptid}"
}

emit_csv_override_rows() {
  local csv_path="$1"
  python - "${csv_path}" <<'PY'
import csv
import pathlib
import shlex
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"TASK_OPTIONS_CSV not found: {path}")

column_aliases = {
    "PROJ_NAME": "PROJ_NAME",
    "proj_name": "PROJ_NAME",
    "项目名": "PROJ_NAME",
    "TASK": "TASK",
    "task": "TASK",
    "任务": "TASK",
    "TRAIN_MODE": "TRAIN_MODE",
    "train_mode": "TRAIN_MODE",
    "训练模式": "TRAIN_MODE",
    "LOAD_EXPTID": "LOAD_EXPTID",
    "load_exptid": "LOAD_EXPTID",
    "加载实验名": "LOAD_EXPTID",
    "LOAD_CKPT": "LOAD_CKPT",
    "load_ckpt": "LOAD_CKPT",
    "加载检查点": "LOAD_CKPT",
    "MAX_ITERATIONS": "MAX_ITERATIONS",
    "max_iterations": "MAX_ITERATIONS",
    "最大迭代数": "MAX_ITERATIONS",
    "NUM_ENVS": "NUM_ENVS",
    "num_envs": "NUM_ENVS",
    "环境数": "NUM_ENVS",
    "REQUESTED_NUM_GPUS": "REQUESTED_NUM_GPUS",
    "requested_num_gpus": "REQUESTED_NUM_GPUS",
    "gpu张数": "REQUESTED_NUM_GPUS",
    "GPU张数": "REQUESTED_NUM_GPUS",
    "TRAIN_LOG_EVERY": "TRAIN_LOG_EVERY",
    "train_log_every": "TRAIN_LOG_EVERY",
    "训练日志间隔": "TRAIN_LOG_EVERY",
    "WANDB_GROUP": "WANDB_GROUP",
    "wandb_group": "WANDB_GROUP",
    "wandb组别": "WANDB_GROUP",
    "wandb分组": "WANDB_GROUP",
    "group": "WANDB_GROUP",
    "EE_GOAL_OBS_MODE": "EE_GOAL_OBS_MODE",
    "ee_goal_obs_mode": "EE_GOAL_OBS_MODE",
    "末端目标观测模式": "EE_GOAL_OBS_MODE",
    "REWARD_SCALE_PRESET": "REWARD_SCALE_PRESET",
    "reward_scale_preset": "REWARD_SCALE_PRESET",
    "奖励预设": "REWARD_SCALE_PRESET",
    "OBSERVE_GAIT_COMMANDS": "OBSERVE_GAIT_COMMANDS",
    "observe_gait_commands": "OBSERVE_GAIT_COMMANDS",
    "观察步态指令": "OBSERVE_GAIT_COMMANDS",
    "MIXED_HEIGHT_REFERENCE": "MIXED_HEIGHT_REFERENCE",
    "mixed_height_reference": "MIXED_HEIGHT_REFERENCE",
    "混合高度参考": "MIXED_HEIGHT_REFERENCE",
    "TRUNK_FOLLOW_RATIO": "TRUNK_FOLLOW_RATIO",
    "trunk_follow_ratio": "TRUNK_FOLLOW_RATIO",
    "机体系比例": "TRUNK_FOLLOW_RATIO",
    "OMNIDIRECTIONAL_POS_Y": "OMNIDIRECTIONAL_POS_Y",
    "omnidirectional_pos_y": "OMNIDIRECTIONAL_POS_Y",
    "全向采样": "OMNIDIRECTIONAL_POS_Y",
    "MOUNT_DEG": "MOUNT_DEG",
    "mount_deg": "MOUNT_DEG",
    "机械臂角度": "MOUNT_DEG",
    "ENABLE_DYNAMIC_GAIT_FREQUENCY": "ENABLE_DYNAMIC_GAIT_FREQUENCY",
    "enable_dynamic_gait_frequency": "ENABLE_DYNAMIC_GAIT_FREQUENCY",
    "动态步频": "ENABLE_DYNAMIC_GAIT_FREQUENCY",
    "DISABLE_WANDB": "DISABLE_WANDB",
    "disable_wandb": "DISABLE_WANDB",
    "关闭wandb": "DISABLE_WANDB",
}
allowed_columns = sorted(set(column_aliases.values()))

with path.open("r", encoding="utf-8-sig", newline="") as f:
    filtered_lines = [
        line for line in f
        if line.strip() and not line.lstrip().startswith("#")
    ]

if not filtered_lines:
    raise SystemExit(f"TASK_OPTIONS_CSV is empty: {path}")

reader = csv.DictReader(filtered_lines)
if reader.fieldnames is None:
    raise SystemExit(f"TASK_OPTIONS_CSV must contain a header row: {path}")

normalized_fieldnames = [field.strip() for field in reader.fieldnames]
unknown_columns = [field for field in normalized_fieldnames if field and field not in column_aliases]
if unknown_columns:
    raise SystemExit(
        "Unsupported TASK_OPTIONS_CSV columns: "
        + ", ".join(unknown_columns)
        + ". Allowed canonical columns: "
        + ", ".join(sorted(allowed_columns))
    )

row_index = 0
for row_index, row in enumerate(reader, start=1):
    assignments = [f"CSV_ROW_INDEX={row_index}"]
    for raw_key, raw_value in row.items():
        input_key = raw_key.strip()
        if not input_key:
            continue
        key = column_aliases[input_key]
        value = (raw_value or "").strip()
        if value == "":
            continue
        assignments.append(f"{key}={shlex.quote(value)}")
    print("; ".join(assignments))

if row_index == 0:
    raise SystemExit(f"TASK_OPTIONS_CSV does not contain any data rows: {path}")
PY
}

normalize_current_train_config() {
  OBSERVE_GAIT_COMMANDS="$(normalize_bool_value "${OBSERVE_GAIT_COMMANDS}")" || run_train_die "OBSERVE_GAIT_COMMANDS must be a boolean"
  MIXED_HEIGHT_REFERENCE="$(normalize_bool_value "${MIXED_HEIGHT_REFERENCE}")" || run_train_die "MIXED_HEIGHT_REFERENCE must be a boolean"
  OMNIDIRECTIONAL_POS_Y="$(normalize_bool_value "${OMNIDIRECTIONAL_POS_Y}")" || run_train_die "OMNIDIRECTIONAL_POS_Y must be a boolean"
  ENABLE_DYNAMIC_GAIT_FREQUENCY="$(normalize_bool_value "${ENABLE_DYNAMIC_GAIT_FREQUENCY}")" || run_train_die "ENABLE_DYNAMIC_GAIT_FREQUENCY must be a boolean"
  NOHUP_BACKGROUND="$(normalize_bool_value "${NOHUP_BACKGROUND}")" || run_train_die "NOHUP_BACKGROUND must be a boolean"
  DISABLE_WANDB="$(normalize_bool_value "${DISABLE_WANDB}")" || run_train_die "DISABLE_WANDB must be a boolean"

  case "${EE_GOAL_OBS_MODE}" in
    command|arm_base_target)
      ;;
    *)
      run_train_die "Unsupported EE_GOAL_OBS_MODE=${EE_GOAL_OBS_MODE}. Expected one of: command, arm_base_target"
      ;;
  esac

  case "${REWARD_SCALE_PRESET}" in
    legacy|height_flexible)
      ;;
    *)
      run_train_die "Unsupported REWARD_SCALE_PRESET=${REWARD_SCALE_PRESET}. Expected one of: legacy, height_flexible"
      ;;
  esac

  case "${MOUNT_DEG}" in
    0|90|180|270)
      ;;
    *)
      run_train_die "Unsupported MOUNT_DEG=${MOUNT_DEG}. Expected one of: 0, 90, 180, 270"
      ;;
  esac

  case "${TRAIN_MODE}" in
    fresh|resume|load)
      ;;
    *)
      run_train_die "Unsupported TRAIN_MODE=${TRAIN_MODE}. Expected one of: fresh, resume, load"
      ;;
  esac

  if ! [[ "${REQUESTED_NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    run_train_die "REQUESTED_NUM_GPUS must be a positive integer, got: ${REQUESTED_NUM_GPUS}"
  fi
  if ! [[ "${TRAIN_LOG_EVERY}" =~ ^[1-9][0-9]*$ ]]; then
    run_train_die "TRAIN_LOG_EVERY must be a positive integer, got: ${TRAIN_LOG_EVERY}"
  fi
  if ! [[ "${QUEUE_POLL_INTERVAL_S}" =~ ^[1-9][0-9]*$ ]]; then
    run_train_die "QUEUE_POLL_INTERVAL_S must be a positive integer, got: ${QUEUE_POLL_INTERVAL_S}"
  fi

  validate_optional_number "${TRUNK_FOLLOW_RATIO}" "TRUNK_FOLLOW_RATIO"
  validate_optional_number "${MAX_ITERATIONS}" "MAX_ITERATIONS"
  validate_optional_number "${NUM_ENVS}" "NUM_ENVS"

  if [[ "${TRAIN_MODE}" == "load" && -z "${LOAD_EXPTID}" ]]; then
    run_train_die "LOAD_EXPTID must be set when TRAIN_MODE=load"
  fi
}

build_train_args() {
  local -a train_mode_args=()
  local -a curriculum_args=()
  local -a gait_frequency_args=()

  case "${TRAIN_MODE}" in
    fresh)
      ;;
    resume)
      train_mode_args+=(--train_mode resume)
      ;;
    load)
      train_mode_args+=(--train_mode load --load_exptid "${LOAD_EXPTID}" --checkpoint "${LOAD_CKPT}")
      ;;
  esac

  if (( ${#LIN_VEL_X_MIN_SCHEDULE[@]} > 0 )); then
    curriculum_args+=(--lin_vel_x_min_schedule "$(IFS=,; echo "${LIN_VEL_X_MIN_SCHEDULE[*]}")")
  fi
  if (( ${#LIN_VEL_X_MAX_SCHEDULE[@]} > 0 )); then
    curriculum_args+=(--lin_vel_x_max_schedule "$(IFS=,; echo "${LIN_VEL_X_MAX_SCHEDULE[*]}")")
  fi
  if (( ${#ANG_VEL_YAW_SCHEDULE[@]} > 0 )); then
    curriculum_args+=(--ang_vel_yaw_schedule "$(IFS=,; echo "${ANG_VEL_YAW_SCHEDULE[*]}")")
  fi
  if (( ${#NON_OMNI_GOAL_YAW_SCHEDULE[@]} > 0 )); then
    curriculum_args+=(--non_omni_pos_y_schedule "$(IFS=,; echo "${NON_OMNI_GOAL_YAW_SCHEDULE[*]}")")
  fi
  if (( ${#MIXING_SCHEDULE[@]} > 0 )); then
    curriculum_args+=(--mixing_schedule "$(IFS=,; echo "${MIXING_SCHEDULE[*]}")")
  fi
  if (( ${#PRIV_REG_COEF_SCHEDULE[@]} > 0 )); then
    curriculum_args+=(--priv_reg_coef_schedule "$(IFS=,; echo "${PRIV_REG_COEF_SCHEDULE[*]}")")
  fi

  if [[ "${ENABLE_DYNAMIC_GAIT_FREQUENCY}" == true ]]; then
    gait_frequency_args+=(--gait_frequency_min "1.2" --gait_frequency_max "2.8")
  fi

  TRAIN_ARGS=(
    --proj_name "${PROJ_NAME}"
    --exptid "${EXPTID}"
    --task "${TASK}"
    "${train_mode_args[@]}"
    "${curriculum_args[@]}"
    "${gait_frequency_args[@]}"
    --train_log_every "${TRAIN_LOG_EVERY}"
    --wandb_group "${WANDB_GROUP}"
    --ee_goal_obs_mode "${EE_GOAL_OBS_MODE}"
    --reward_scale_preset "${REWARD_SCALE_PRESET}"
    --mount_deg "${MOUNT_DEG}"
  )

  if [[ "${OBSERVE_GAIT_COMMANDS}" == true ]]; then
    TRAIN_ARGS+=(--observe_gait_commands)
  fi
  if [[ "${MIXED_HEIGHT_REFERENCE}" == true ]]; then
    TRAIN_ARGS+=(--mixed_height_reference)
  fi
  if [[ -n "${TRUNK_FOLLOW_RATIO}" ]]; then
    TRAIN_ARGS+=(--trunk_follow_ratio "${TRUNK_FOLLOW_RATIO}")
  fi
  if [[ "${OMNIDIRECTIONAL_POS_Y}" == true ]]; then
    TRAIN_ARGS+=(--omnidirectional_pos_y)
  fi
  if [[ -n "${NUM_ENVS}" ]]; then
    TRAIN_ARGS+=(--num_envs "${NUM_ENVS}")
  fi
  if [[ -n "${MAX_ITERATIONS}" ]]; then
    TRAIN_ARGS+=(--max_iterations "${MAX_ITERATIONS}")
  fi
}

prepare_training_submission() {
  normalize_current_train_config

  EXPTID="$(generate_train_exptid \
    "${REWARD_SCALE_PRESET}" \
    "${MIXED_HEIGHT_REFERENCE}" \
    "${TRUNK_FOLLOW_RATIO}" \
    "${OMNIDIRECTIONAL_POS_Y}" \
    "${MOUNT_DEG}" \
    "${ENABLE_DYNAMIC_GAIT_FREQUENCY}")"

  TOTAL_AVAILABLE_GPUS="$(get_total_gpu_count)"
  if ! [[ "${TOTAL_AVAILABLE_GPUS}" =~ ^[0-9]+$ ]]; then
    run_train_die "Failed to detect available GPUs. torch.cuda.device_count() returned: ${TOTAL_AVAILABLE_GPUS}"
  fi
  if (( TOTAL_AVAILABLE_GPUS < 1 )); then
    run_train_die "No CUDA GPUs detected."
  fi
  if (( REQUESTED_NUM_GPUS > TOTAL_AVAILABLE_GPUS )); then
    run_train_die "REQUESTED_NUM_GPUS=${REQUESTED_NUM_GPUS} exceeds detected GPU count ${TOTAL_AVAILABLE_GPUS}."
  fi

  if [[ -n "${NUM_ENVS}" ]]; then
    TOTAL_NUM_ENVS="${NUM_ENVS}"
  else
    TOTAL_NUM_ENVS="<config>"
  fi

  if (( REQUESTED_NUM_GPUS > 1 )); then
    DISTRIBUTED=true
  else
    DISTRIBUTED=false
  fi

  if [[ "${TOTAL_NUM_ENVS}" != "<config>" ]]; then
    if (( TOTAL_NUM_ENVS % REQUESTED_NUM_GPUS == 0 )); then
      NUM_ENVS_PER_GPU="$(( TOTAL_NUM_ENVS / REQUESTED_NUM_GPUS ))"
    else
      NUM_ENVS_PER_GPU="<invalid: not divisible>"
    fi
  else
    NUM_ENVS_PER_GPU="<resolved in Python>"
  fi

  build_train_args
}

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

read_file_or_empty() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    cat "${path}"
  fi
}

acquire_queue_lock() {
  exec 9>"${QUEUE_LOCK_FILE}"
  flock -x 9
}

release_queue_lock() {
  flock -u 9
}

cleanup_stale_jobs_locked() {
  local job_dir
  local state
  local launcher_pid
  local train_pid

  shopt -s nullglob
  for job_dir in "${QUEUE_JOBS_DIR}"/*; do
    [[ -d "${job_dir}" ]] || continue
    state="$(read_file_or_empty "${job_dir}/state")"
    case "${state}" in
      queued)
        launcher_pid="$(read_file_or_empty "${job_dir}/launcher_pid")"
        if [[ -n "${launcher_pid}" ]] && ! kill -0 "${launcher_pid}" 2>/dev/null; then
          printf 'cancelled\n' > "${job_dir}/state"
        fi
        ;;
      starting)
        launcher_pid="$(read_file_or_empty "${job_dir}/launcher_pid")"
        train_pid="$(read_file_or_empty "${job_dir}/train_pid")"
        if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
          :
        elif [[ -n "${launcher_pid}" ]] && kill -0 "${launcher_pid}" 2>/dev/null; then
          :
        else
          printf 'failed\n' > "${job_dir}/state"
        fi
        ;;
      running)
        train_pid="$(read_file_or_empty "${job_dir}/train_pid")"
        if [[ -z "${train_pid}" ]] || ! kill -0 "${train_pid}" 2>/dev/null; then
          printf 'failed\n' > "${job_dir}/state"
        fi
        ;;
    esac
  done
  shopt -u nullglob
}

find_queue_head_locked() {
  local job_dir
  local state

  shopt -s nullglob
  for job_dir in "${QUEUE_JOBS_DIR}"/*; do
    [[ -d "${job_dir}" ]] || continue
    state="$(read_file_or_empty "${job_dir}/state")"
    if [[ "${state}" == "queued" ]]; then
      printf '%s\n' "${job_dir}"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

reserve_gpus_if_available_locked() {
  local job_dir
  local state
  local assigned_gpu_csv
  local gpu_id
  local -a assigned_gpu_ids=()
  local -a free_gpu_ids=()
  local -A busy_gpu_ids=()

  shopt -s nullglob
  for job_dir in "${QUEUE_JOBS_DIR}"/*; do
    [[ -d "${job_dir}" ]] || continue
    state="$(read_file_or_empty "${job_dir}/state")"
    if [[ "${state}" != "starting" && "${state}" != "running" ]]; then
      continue
    fi
    assigned_gpu_csv="$(read_file_or_empty "${job_dir}/assigned_gpus")"
    [[ -n "${assigned_gpu_csv}" ]] || continue
    IFS=, read -r -a assigned_gpu_ids <<< "${assigned_gpu_csv}"
    for gpu_id in "${assigned_gpu_ids[@]}"; do
      [[ -n "${gpu_id}" ]] || continue
      busy_gpu_ids["${gpu_id}"]=1
    done
  done
  shopt -u nullglob

  for ((gpu_id=0; gpu_id<TOTAL_AVAILABLE_GPUS; gpu_id++)); do
    if [[ -z "${busy_gpu_ids[${gpu_id}]+x}" ]]; then
      free_gpu_ids+=("${gpu_id}")
    fi
  done

  if (( ${#free_gpu_ids[@]} < REQUESTED_NUM_GPUS )); then
    return 1
  fi

  assigned_gpu_ids=("${free_gpu_ids[@]:0:REQUESTED_NUM_GPUS}")
  printf '%s\n' "$(IFS=,; echo "${assigned_gpu_ids[*]}")" > "${JOB_DIR}/assigned_gpus"
  printf 'starting\n' > "${JOB_DIR}/state"
  printf '%(%Y-%m-%d %H:%M:%S)T\n' -1 > "${JOB_DIR}/started_at"
  return 0
}

write_job_snapshot_file() {
  local snapshot_file="$1"
  {
    printf 'ROOT_DIR=%q\n' "${ROOT_DIR}"
    printf 'SCRIPT_DIR=%q\n' "${SCRIPT_DIR}"
    printf 'LOG_ROOT=%q\n' "${LOG_ROOT}"
    printf 'QUEUE_ROOT=%q\n' "${QUEUE_ROOT}"
    printf 'QUEUE_LOCK_FILE=%q\n' "${QUEUE_LOCK_FILE}"
    printf 'QUEUE_JOBS_DIR=%q\n' "${QUEUE_JOBS_DIR}"
    printf 'JOB_ID=%q\n' "${JOB_ID}"
    printf 'JOB_DIR=%q\n' "${JOB_DIR}"
    printf 'JOB_ORDER=%q\n' "${JOB_ORDER}"
    printf 'RUN_INSTANCE_ID=%q\n' "${RUN_INSTANCE_ID}"
    printf 'PROJ_NAME=%q\n' "${PROJ_NAME}"
    printf 'TASK=%q\n' "${TASK}"
    printf 'EXPTID=%q\n' "${EXPTID}"
    printf 'REQUESTED_NUM_GPUS=%q\n' "${REQUESTED_NUM_GPUS}"
    printf 'TOTAL_AVAILABLE_GPUS=%q\n' "${TOTAL_AVAILABLE_GPUS}"
    printf 'QUEUE_POLL_INTERVAL_S=%q\n' "${QUEUE_POLL_INTERVAL_S}"
    printf 'WANDB_GROUP=%q\n' "${WANDB_GROUP}"
    printf 'ERROR_LOG=%q\n' "${ERROR_LOG}"
    printf 'QUEUE_LOG=%q\n' "${QUEUE_LOG}"
    printf 'BACKGROUND_MODE=%q\n' "${NOHUP_BACKGROUND}"
    printf 'DISABLE_WANDB=%q\n' "${DISABLE_WANDB}"
    printf 'LEGGED_GYM_LOG_ROOT_VALUE=%q\n' "${LOG_ROOT}"
    printf 'RDZV_PORT_OVERRIDE=%q\n' "${RDZV_PORT_OVERRIDE}"
    write_array_declaration TRAIN_ARGS "${TRAIN_ARGS[@]}"
  } > "${snapshot_file}"
}

write_job_metadata_file() {
  local metadata_file="$1"
  {
    printf 'JOB_ID=%q\n' "${JOB_ID}"
    printf 'JOB_ORDER=%q\n' "${JOB_ORDER}"
    printf 'RUN_INSTANCE_ID=%q\n' "${RUN_INSTANCE_ID}"
    printf 'EXPTID=%q\n' "${EXPTID}"
    printf 'REQUESTED_NUM_GPUS=%q\n' "${REQUESTED_NUM_GPUS}"
    printf 'TOTAL_AVAILABLE_GPUS=%q\n' "${TOTAL_AVAILABLE_GPUS}"
    printf 'DISTRIBUTED=%q\n' "${DISTRIBUTED}"
    printf 'NUM_ENVS=%q\n' "${TOTAL_NUM_ENVS}"
    printf 'NUM_ENVS_PER_GPU=%q\n' "${NUM_ENVS_PER_GPU}"
    printf 'ERROR_LOG=%q\n' "${ERROR_LOG}"
    printf 'QUEUE_LOG=%q\n' "${QUEUE_LOG}"
    printf 'SCRIPT_DIR=%q\n' "${SCRIPT_DIR}"
    printf 'LOG_ROOT=%q\n' "${LOG_ROOT}"
    printf 'QUEUE_POLL_INTERVAL_S=%q\n' "${QUEUE_POLL_INTERVAL_S}"
    printf 'WANDB_GROUP=%q\n' "${WANDB_GROUP}"
    printf 'NOHUP_BACKGROUND=%q\n' "${NOHUP_BACKGROUND}"
    printf 'TRAIN_MODE=%q\n' "${TRAIN_MODE}"
    printf 'LOAD_EXPTID=%q\n' "${LOAD_EXPTID}"
    printf 'LOAD_CKPT=%q\n' "${LOAD_CKPT}"
    printf 'MAX_ITERATIONS=%q\n' "${MAX_ITERATIONS}"
    printf 'TRAIN_LOG_EVERY=%q\n' "${TRAIN_LOG_EVERY}"
    printf 'EE_GOAL_OBS_MODE=%q\n' "${EE_GOAL_OBS_MODE}"
    printf 'REWARD_SCALE_PRESET=%q\n' "${REWARD_SCALE_PRESET}"
    printf 'OBSERVE_GAIT_COMMANDS=%q\n' "${OBSERVE_GAIT_COMMANDS}"
    printf 'MIXED_HEIGHT_REFERENCE=%q\n' "${MIXED_HEIGHT_REFERENCE}"
    printf 'TRUNK_FOLLOW_RATIO=%q\n' "${TRUNK_FOLLOW_RATIO}"
    printf 'OMNIDIRECTIONAL_POS_Y=%q\n' "${OMNIDIRECTIONAL_POS_Y}"
    printf 'MOUNT_DEG=%q\n' "${MOUNT_DEG}"
    printf 'ENABLE_DYNAMIC_GAIT_FREQUENCY=%q\n' "${ENABLE_DYNAMIC_GAIT_FREQUENCY}"
    write_array_declaration TRAIN_ARGS "${TRAIN_ARGS[@]}"
  } > "${metadata_file}"
}

enqueue_training_job() {
  prepare_training_submission

  RUN_INSTANCE_ID="$(date +%Y%m%d_%H%M%S)_$$"
  ERROR_LOG="${SH_DIR}/error_${RUN_INSTANCE_ID}.log"
  QUEUE_LOG="${SH_DIR}/queue_${RUN_INSTANCE_ID}.log"

  mkdir -p "${QUEUE_JOBS_DIR}"

  exec 9>"${QUEUE_LOCK_FILE}"
  flock -x 9
  local next_order_file="${QUEUE_ROOT}/next_order.txt"
  local next_order
  if [[ -f "${next_order_file}" ]]; then
    next_order="$(<"${next_order_file}")"
  else
    next_order="1"
  fi
  JOB_ORDER="${next_order}"
  printf '%s\n' "$(( next_order + 1 ))" > "${next_order_file}"
  JOB_ID="$(printf '%06d_%s' "${JOB_ORDER}" "${RUN_INSTANCE_ID}")"
  JOB_DIR="${QUEUE_JOBS_DIR}/${JOB_ID}"
  mkdir -p "${JOB_DIR}"
  printf '%s\n' "${JOB_ORDER}" > "${JOB_DIR}/order"
  printf 'queued\n' > "${JOB_DIR}/state"
  printf '%s\n' "${REQUESTED_NUM_GPUS}" > "${JOB_DIR}/requested_num_gpus"
  printf '%s\n' "${EXPTID}" > "${JOB_DIR}/exptid"
  printf '%s\n' "${ERROR_LOG}" > "${JOB_DIR}/error_log"
  printf '%s\n' "${QUEUE_LOG}" > "${JOB_DIR}/queue_log"
  printf '%(%Y-%m-%d %H:%M:%S)T\n' -1 > "${JOB_DIR}/enqueued_at"
  flock -u 9

  local job_snapshot_file="${JOB_DIR}/config_snapshot.sh"
  local job_metadata_file="${JOB_DIR}/job_metadata.sh"
  local job_lib_file="${JOB_DIR}/run_train_lib.sh"
  local job_worker_file="${JOB_DIR}/run_train_worker.sh"

  cp "${RUN_TRAIN_LIB_TEMPLATE}" "${job_lib_file}"
  cp "${RUN_TRAIN_WORKER_TEMPLATE}" "${job_worker_file}"
  chmod +x "${job_lib_file}" "${job_worker_file}"

  write_job_snapshot_file "${job_snapshot_file}"
  write_job_metadata_file "${job_metadata_file}"

  local launch_message_prefix="Queued training job ${JOB_ID}"
  if [[ -n "${CSV_ROW_INDEX:-}" ]]; then
    launch_message_prefix+=" (csv row ${CSV_ROW_INDEX})"
  fi

  if [[ "${NOHUP_BACKGROUND}" == true ]]; then
    nohup "${job_worker_file}" "${job_snapshot_file}" > "${QUEUE_LOG}" 2>&1 &
    local launcher_pid=$!
    echo "${launch_message_prefix} (launcher PID=${launcher_pid})."
  else
    echo "${launch_message_prefix}."
  fi
  echo "EXPTID=${EXPTID}"
  echo "REQUESTED_NUM_GPUS=${REQUESTED_NUM_GPUS}"
  echo "TOTAL_AVAILABLE_GPUS=${TOTAL_AVAILABLE_GPUS}"
  echo "NUM_ENVS=${TOTAL_NUM_ENVS}"
  echo "NUM_ENVS_PER_GPU=${NUM_ENVS_PER_GPU}"
  echo "DISTRIBUTED=${DISTRIBUTED}"
  echo "snapshot -> ${job_snapshot_file}"
  echo "metadata -> ${job_metadata_file}"
  if [[ "${NOHUP_BACKGROUND}" == true ]]; then
    echo "queue log -> ${QUEUE_LOG}"
  fi
  echo "stderr -> ${ERROR_LOG}"

  if [[ "${NOHUP_BACKGROUND}" != true ]]; then
    "${job_worker_file}" "${job_snapshot_file}"
  fi
}

run_queued_training_worker() {
  log_line() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
  }

  mkdir -p "${QUEUE_JOBS_DIR}"
  printf '%s\n' "$$" > "${JOB_DIR}/launcher_pid"

  export LEGGED_GYM_LOG_ROOT="${LEGGED_GYM_LOG_ROOT_VALUE}"
  if [[ "${DISABLE_WANDB}" == true ]]; then
    export WANDB_DISABLED=true
    export WANDB_SILENT=true
  fi

  log_line "Queue launcher started for ${EXPTID} (job ${JOB_ID}, order ${JOB_ORDER}, requested_gpus=${REQUESTED_NUM_GPUS})"

  local assigned_gpu_csv=""
  local current_state
  local queue_head
  while true; do
    acquire_queue_lock
    cleanup_stale_jobs_locked
    current_state="$(read_file_or_empty "${JOB_DIR}/state")"
    if [[ "${current_state}" != "queued" ]]; then
      release_queue_lock
      log_line "Job ${JOB_ID} left queued state unexpectedly: ${current_state}"
      exit 1
    fi

    queue_head="$(find_queue_head_locked || true)"
    if [[ "${queue_head}" != "${JOB_DIR}" ]]; then
      release_queue_lock
      sleep "${QUEUE_POLL_INTERVAL_S}"
      continue
    fi

    if reserve_gpus_if_available_locked; then
      assigned_gpu_csv="$(read_file_or_empty "${JOB_DIR}/assigned_gpus")"
      release_queue_lock
      break
    fi

    release_queue_lock
    sleep "${QUEUE_POLL_INTERVAL_S}"
  done

  if [[ -z "${assigned_gpu_csv}" ]]; then
    log_line "Failed to reserve GPUs for ${JOB_ID}"
    exit 1
  fi

  local -a launch_cmd=()
  if (( REQUESTED_NUM_GPUS > 1 )); then
    local rdzv_port
    local rdzv_endpoint
    local torchrun_help
    local rdzv_backend_flag
    local rdzv_endpoint_flag
    local rdzv_id_flag
    if [[ -n "${RDZV_PORT_OVERRIDE}" ]]; then
      rdzv_port="${RDZV_PORT_OVERRIDE}"
    else
      rdzv_port="$(pick_free_port)"
    fi
    rdzv_endpoint="127.0.0.1:${rdzv_port}"
    torchrun_help="$(torchrun --help 2>&1 || true)"
    rdzv_backend_flag="$(pick_torchrun_flag "${torchrun_help}" "--rdzv-backend" "--rdzv_backend")"
    rdzv_endpoint_flag="$(pick_torchrun_flag "${torchrun_help}" "--rdzv-endpoint" "--rdzv_endpoint")"
    rdzv_id_flag="$(pick_torchrun_flag "${torchrun_help}" "--rdzv-id" "--rdzv_id")"
    launch_cmd=(
      torchrun
      --nnodes=1
      --nproc_per_node "${REQUESTED_NUM_GPUS}"
      "${rdzv_backend_flag}=c10d"
      "${rdzv_endpoint_flag}" "${rdzv_endpoint}"
      "${rdzv_id_flag}" "${PROJ_NAME}-${EXPTID}-${RUN_INSTANCE_ID}"
      train.py
    )
    log_line "Using rendezvous endpoint ${rdzv_endpoint}"
  else
    launch_cmd=(
      python
      train.py
    )
  fi

  local -a train_cmd=(
    "${launch_cmd[@]}"
    "${TRAIN_ARGS[@]}"
  )

  log_line "Launching ${EXPTID} on GPUs ${assigned_gpu_csv}"

  if [[ "${BACKGROUND_MODE}" == true ]]; then
    (
      cd "${SCRIPT_DIR}"
      export CUDA_VISIBLE_DEVICES="${assigned_gpu_csv}"
      "${train_cmd[@]}"
    ) > /dev/null 2> >(timestamp_stderr_to_file "${ERROR_LOG}") &
  else
    (
      cd "${SCRIPT_DIR}"
      export CUDA_VISIBLE_DEVICES="${assigned_gpu_csv}"
      "${train_cmd[@]}"
    ) 2> >(timestamp_stderr_to_file "${ERROR_LOG}") &
  fi

  local train_pid=$!
  acquire_queue_lock
  printf '%s\n' "${train_pid}" > "${JOB_DIR}/train_pid"
  printf 'running\n' > "${JOB_DIR}/state"
  release_queue_lock

  set +e
  wait "${train_pid}"
  local train_exit_code=$?
  set -e

  acquire_queue_lock
  printf '%s\n' "${train_exit_code}" > "${JOB_DIR}/exit_code"
  printf '%(%Y-%m-%d %H:%M:%S)T\n' -1 > "${JOB_DIR}/finished_at"
  if (( train_exit_code == 0 )); then
    printf 'completed\n' > "${JOB_DIR}/state"
  else
    printf 'failed\n' > "${JOB_DIR}/state"
  fi
  release_queue_lock

  log_line "Training finished for ${EXPTID} with exit code ${train_exit_code}"
  exit "${train_exit_code}"
}
