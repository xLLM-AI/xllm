#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/dsv4_py_cp2_feature_stack_logs}"
BASE_PORT="${BASE_PORT:-20050}"

pid_cmdline() {
  local pid="$1"
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
}

owned_pid() {
  local rank="$1" pid="$2" port=$((BASE_PORT + rank)) cmd
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  cmd="$(pid_cmdline "$pid")"
  [[ "$cmd" == *"--model_impl python"* &&
     ("$cmd" == *"--node_rank=$rank"* || "$cmd" == *"--node_rank $rank"*) &&
     ("$cmd" == *"--port=$port"* || "$cmd" == *"--port $port"*) ]]
}

declare -a owned=()
for rank in $(seq 0 7); do
  pid_file="$LOG_DIR/node_${rank}.pid"
  [[ -f "$pid_file" ]] || continue
  pid="$(<"$pid_file")"
  if owned_pid "$rank" "$pid"; then
    owned+=("$rank:$pid")
  elif kill -0 "$pid" 2>/dev/null; then
    echo "Refusing to kill pid=$pid: command line does not match rank $rank/port $((BASE_PORT + rank))." >&2
    echo "  $(pid_cmdline "$pid")" >&2
    exit 1
  fi
done

if ((${#owned[@]} == 0)); then
  echo "No owned xLLM Python processes found in $LOG_DIR"
  exit 0
fi

for item in "${owned[@]}"; do
  rank="${item%%:*}"
  pid="${item##*:}"
  echo "TERM rank=$rank pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
done

for _ in $(seq 1 30); do
  alive=0
  for item in "${owned[@]}"; do
    pid="${item##*:}"
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
    fi
  done
  if ((alive == 0)); then
    break
  fi
  sleep 1
done

for item in "${owned[@]}"; do
  rank="${item%%:*}"
  pid="${item##*:}"
  if kill -0 "$pid" 2>/dev/null; then
    echo "KILL rank=$rank pid=$pid"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

for rank in $(seq 0 7); do
  rm -f "$LOG_DIR/node_${rank}.pid"
done
echo "Stopped owned processes from $LOG_DIR"
