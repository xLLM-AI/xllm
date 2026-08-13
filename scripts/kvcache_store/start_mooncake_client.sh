#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: start_mooncake_client.sh [start|run|stop|status]

Start an independent resource-owning Mooncake Store client that discovers the
active master through etcd.

Required for start/run:
  MOONCAKE_ETCD_ENDPOINTS  Raw etcd endpoints separated by semicolons.
  MOONCAKE_CLIENT_HOST     Reachable data endpoint, including its port.
                          Example: 10.0.2.21:50053

Optional:
  MOONCAKE_CLIENT_BIN          Explicit mooncake_client path.
  MOONCAKE_CLUSTER_ID          Default: xllm-mooncake
  MOONCAKE_CLIENT_PORT         Default: 50052
  MOONCAKE_GLOBAL_SEGMENT_SIZE Default: 4GB
  MOONCAKE_METADATA_SERVER     Default: P2PHANDSHAKE
  MOONCAKE_PROTOCOL            Default: tcp
  MOONCAKE_INSTANCE_NAME       Default: client-<port>
  MOONCAKE_LOG_DIR             Default: <repo>/logs/mooncake
  MOONCAKE_PID_FILE            Default: <log-dir>/<instance>.pid
  MOONCAKE_LOG_FILE            Default: <log-dir>/<instance>.log
  MOONCAKE_STOP_TIMEOUT_SECONDS Default: 30

Commands:
  start   Start in the background. This is the default.
  run     Run in the foreground.
  stop    Stop the process recorded in the PID file.
  status  Show process status.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_port() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be an integer: $value"
  (( value >= 1 && value <= 65535 )) || die "$name is out of range: $value"
}

resolve_client_binary() {
  local configured="${MOONCAKE_CLIENT_BIN:-}"
  if [[ -n "$configured" ]]; then
    [[ -x "$configured" ]] || die "MOONCAKE_CLIENT_BIN is not executable: $configured"
    printf '%s\n' "$configured"
    return
  fi

  local installed
  installed="$(command -v mooncake_client || true)"
  if [[ -n "$installed" ]]; then
    printf '%s\n' "$installed"
    return
  fi

  local candidate
  shopt -s nullglob
  for candidate in "$REPO_ROOT"/build/lib.*/xllm/mooncake_client; do
    if [[ -x "$candidate" ]]; then
      shopt -u nullglob
      printf '%s\n' "$candidate"
      return
    fi
  done
  shopt -u nullglob
  die "mooncake_client not found; set MOONCAKE_CLIENT_BIN"
}

normalize_etcd_endpoints() {
  local endpoints="$1"
  endpoints="${endpoints#etcd://}"
  [[ -n "$endpoints" ]] || die "MOONCAKE_ETCD_ENDPOINTS is required"
  [[ "$endpoints" != *"http://"* && "$endpoints" != *"https://"* ]] || \
    die "MOONCAKE_ETCD_ENDPOINTS must not include http:// or https://"
  printf '%s\n' "$endpoints"
}

read_pid() {
  [[ -r "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$pid"
}

running_pid() {
  local pid
  pid="$(read_pid || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local command_line
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$command_line" == *"mooncake_client"* ]] || return 1
  [[ "$command_line" == *"--port=$CLIENT_PORT"* ]] || return 1
  printf '%s\n' "$pid"
}

ACTION="${1:-start}"
case "$ACTION" in
  -h|--help|help)
    usage
    exit 0
    ;;
  start|run|stop|status) ;;
  *)
    usage >&2
    die "unknown command: $ACTION"
    ;;
esac

CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-50052}"
INSTANCE_NAME="${MOONCAKE_INSTANCE_NAME:-client-$CLIENT_PORT}"
LOG_DIR="${MOONCAKE_LOG_DIR:-$REPO_ROOT/logs/mooncake}"
PID_FILE="${MOONCAKE_PID_FILE:-$LOG_DIR/$INSTANCE_NAME.pid}"
LOG_FILE="${MOONCAKE_LOG_FILE:-$LOG_DIR/$INSTANCE_NAME.log}"

if [[ "$ACTION" == "status" ]]; then
  if pid="$(running_pid)"; then
    echo "$INSTANCE_NAME is running (pid=$pid, log=$LOG_FILE)"
    exit 0
  fi
  echo "$INSTANCE_NAME is not running"
  exit 1
fi

if [[ "$ACTION" == "stop" ]]; then
  pid="$(running_pid || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$PID_FILE"
    echo "$INSTANCE_NAME is not running"
    exit 0
  fi
  kill "$pid"
  stop_timeout="${MOONCAKE_STOP_TIMEOUT_SECONDS:-30}"
  [[ "$stop_timeout" =~ ^[0-9]+$ ]] || die "MOONCAKE_STOP_TIMEOUT_SECONDS must be an integer"
  for (( second=0; second<stop_timeout; second++ )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "$INSTANCE_NAME stopped"
      exit 0
    fi
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "$INSTANCE_NAME was force-stopped after ${stop_timeout}s"
  exit 0
fi

ETCD_ENDPOINTS="$(normalize_etcd_endpoints "${MOONCAKE_ETCD_ENDPOINTS:-}")"
CLIENT_HOST="${MOONCAKE_CLIENT_HOST:-}"
[[ -n "$CLIENT_HOST" ]] || die "MOONCAKE_CLIENT_HOST is required"
CLUSTER_ID="${MOONCAKE_CLUSTER_ID:-xllm-mooncake}"
[[ "$CLUSTER_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid MOONCAKE_CLUSTER_ID: $CLUSTER_ID"
validate_port MOONCAKE_CLIENT_PORT "$CLIENT_PORT"

CLIENT_BIN="$(resolve_client_binary)"
CLIENT_BIN_DIR="$(dirname "$CLIENT_BIN")"
PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
COMMAND=(
  "$CLIENT_BIN"
  "--host=$CLIENT_HOST"
  "--port=$CLIENT_PORT"
  "--global_segment_size=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-4GB}"
  "--master_server_address=etcd://$ETCD_ENDPOINTS"
  "--metadata_server=${MOONCAKE_METADATA_SERVER:-P2PHANDSHAKE}"
  "--protocol=$PROTOCOL"
)
RUNTIME_LIBRARY_PATH="$CLIENT_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
FORCE_TCP="${MC_FORCE_TCP:-}"
if [[ "$PROTOCOL" == "tcp" && -z "$FORCE_TCP" ]]; then
  FORCE_TCP=1
fi

if [[ "$ACTION" == "run" ]]; then
  export MC_STORE_CLUSTER_ID="$CLUSTER_ID"
  export MC_FORCE_TCP="$FORCE_TCP"
  export LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH"
  exec "${COMMAND[@]}"
fi

if pid="$(running_pid)"; then
  die "$INSTANCE_NAME is already running with pid $pid"
fi

mkdir -p "$LOG_DIR" "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
umask 027
nohup env \
  MC_STORE_CLUSTER_ID="$CLUSTER_ID" \
  MC_FORCE_TCP="$FORCE_TCP" \
  LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH" \
  "${COMMAND[@]}" >>"$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  tail -n 40 "$LOG_FILE" >&2 || true
  die "$INSTANCE_NAME exited during startup"
fi

echo "$INSTANCE_NAME started (pid=$pid, host=$CLIENT_HOST, service_port=$CLIENT_PORT, log=$LOG_FILE)"
