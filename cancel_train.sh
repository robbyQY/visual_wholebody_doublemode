#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 <queue-log-suffix-pid>" >&2
  echo "Example: $0 12345  # for queue_20260420_153000_12345.log" >&2
  exit 1
fi

TARGET_PID="$1"
if ! [[ "${TARGET_PID}" =~ ^[0-9]+$ ]]; then
  echo "PID suffix must be numeric: ${TARGET_PID}" >&2
  exit 1
fi

LOG_ROOT="${LOG_ROOT:-/data/logs}"
QUEUE_ROOT="${QUEUE_ROOT:-${LOG_ROOT}/.run_train_queue}"
QUEUE_LOCK_FILE="${QUEUE_LOCK_FILE:-${QUEUE_ROOT}/queue.lock}"
QUEUE_JOBS_DIR="${QUEUE_JOBS_DIR:-${QUEUE_ROOT}/jobs}"
TRAIN_TERM_TIMEOUT_S="${TRAIN_TERM_TIMEOUT_S:-20}"
LAUNCHER_TERM_TIMEOUT_S="${LAUNCHER_TERM_TIMEOUT_S:-5}"

read_file_or_empty() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    cat "${path}"
  fi
}

timestamp_now() {
  printf '%(%Y-%m-%d %H:%M:%S)T\n' -1
}

acquire_queue_lock() {
  exec 9>"${QUEUE_LOCK_FILE}"
  flock -x 9
}

release_queue_lock() {
  flock -u 9
}

is_pid_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

wait_for_pid_exit() {
  local pid="$1"
  local timeout_s="$2"
  local steps=$(( timeout_s * 10 ))
  local step
  for ((step=0; step<steps; step++)); do
    if ! is_pid_alive "${pid}"; then
      return 0
    fi
    sleep 0.1
  done
  ! is_pid_alive "${pid}"
}

terminate_pid() {
  local pid="$1"
  local label="$2"
  local timeout_s="$3"

  if ! is_pid_alive "${pid}"; then
    return 0
  fi

  echo "[cancel] sending SIGTERM to ${label} PID ${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  if wait_for_pid_exit "${pid}" "${timeout_s}"; then
    return 0
  fi

  echo "[cancel] ${label} PID ${pid} did not exit in ${timeout_s}s, sending SIGKILL"
  kill -KILL "${pid}" 2>/dev/null || true
  wait_for_pid_exit "${pid}" 2 || true
}

collect_matching_jobs() {
  local job_dir
  local queue_log
  local queue_log_name

  MATCHED_JOB_DIRS=()
  shopt -s nullglob
  for job_dir in "${QUEUE_JOBS_DIR}"/*; do
    [[ -d "${job_dir}" ]] || continue
    queue_log="$(read_file_or_empty "${job_dir}/queue_log")"
    queue_log_name="$(basename "${queue_log:-}")"
    if [[ "${queue_log_name}" =~ _${TARGET_PID}\.log$ ]]; then
      MATCHED_JOB_DIRS+=("${job_dir}")
    fi
  done
  shopt -u nullglob
}

cancel_job() {
  local job_dir="$1"
  local job_id
  local state
  local exptid
  local launcher_pid
  local train_pid
  local queue_log
  local cancel_file

  job_id="$(basename "${job_dir}")"
  cancel_file="${job_dir}/cancel_requested"

  acquire_queue_lock
  state="$(read_file_or_empty "${job_dir}/state")"
  exptid="$(read_file_or_empty "${job_dir}/exptid")"
  launcher_pid="$(read_file_or_empty "${job_dir}/launcher_pid")"
  train_pid="$(read_file_or_empty "${job_dir}/train_pid")"
  queue_log="$(read_file_or_empty "${job_dir}/queue_log")"

  case "${state}" in
    cancelled|completed|failed)
      release_queue_lock
      echo "[skip] ${job_id} exptid=${exptid} state=${state}"
      return 0
      ;;
    queued)
      printf '%s\n' "$(timestamp_now)" > "${cancel_file}"
      printf 'cancelled\n' > "${job_dir}/state"
      printf '%(%Y-%m-%d %H:%M:%S)T\n' -1 > "${job_dir}/finished_at"
      release_queue_lock

      echo "[cancel] queued job ${job_id} exptid=${exptid}"
      echo "[cancel] queue log: ${queue_log}"
      if [[ -n "${launcher_pid}" ]]; then
        terminate_pid "${launcher_pid}" "launcher" "${LAUNCHER_TERM_TIMEOUT_S}"
      fi
      return 0
      ;;
    starting|running)
      printf '%s\n' "$(timestamp_now)" > "${cancel_file}"
      release_queue_lock

      echo "[cancel] ${state} job ${job_id} exptid=${exptid}"
      echo "[cancel] queue log: ${queue_log}"
      if [[ -n "${train_pid}" ]]; then
        terminate_pid "${train_pid}" "train" "${TRAIN_TERM_TIMEOUT_S}"
      fi

      if [[ -n "${launcher_pid}" ]] && is_pid_alive "${launcher_pid}"; then
        if ! wait_for_pid_exit "${launcher_pid}" "${LAUNCHER_TERM_TIMEOUT_S}"; then
          terminate_pid "${launcher_pid}" "launcher" "${LAUNCHER_TERM_TIMEOUT_S}"
        fi
      fi

      acquire_queue_lock
      state="$(read_file_or_empty "${job_dir}/state")"
      if [[ "${state}" == "starting" || "${state}" == "running" ]]; then
        if ! is_pid_alive "${train_pid}" && ! is_pid_alive "${launcher_pid}"; then
          printf 'cancelled\n' > "${job_dir}/state"
          printf '%(%Y-%m-%d %H:%M:%S)T\n' -1 > "${job_dir}/finished_at"
        fi
      fi
      release_queue_lock
      return 0
      ;;
    *)
      release_queue_lock
      echo "[warn] ${job_id} exptid=${exptid} has unknown state=${state}" >&2
      return 1
      ;;
  esac
}

if [[ ! -d "${QUEUE_JOBS_DIR}" ]]; then
  echo "Queue jobs directory not found: ${QUEUE_JOBS_DIR}" >&2
  exit 1
fi

mkdir -p "${QUEUE_ROOT}"
touch "${QUEUE_LOCK_FILE}"

collect_matching_jobs

if (( ${#MATCHED_JOB_DIRS[@]} == 0 )); then
  echo "No queued/running jobs matched queue log suffix pid ${TARGET_PID}" >&2
  exit 1
fi

echo "Matched ${#MATCHED_JOB_DIRS[@]} job(s) for queue log suffix pid ${TARGET_PID}:"
for job_dir in "${MATCHED_JOB_DIRS[@]}"; do
  echo "  $(basename "${job_dir}")"
done

for job_dir in "${MATCHED_JOB_DIRS[@]}"; do
  cancel_job "${job_dir}"
done

echo "Cancellation request completed."
