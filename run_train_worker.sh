#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 <config_snapshot.sh>" >&2
  exit 1
fi

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_FILE="$1"
LIB_FILE="${WORKER_DIR}/run_train_lib.sh"

if [[ ! -f "${LIB_FILE}" ]]; then
  echo "Missing worker library: ${LIB_FILE}" >&2
  exit 1
fi
if [[ ! -f "${SNAPSHOT_FILE}" ]]; then
  echo "Missing worker snapshot: ${SNAPSHOT_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${LIB_FILE}"
# shellcheck source=/dev/null
source "${SNAPSHOT_FILE}"

run_queued_training_worker
