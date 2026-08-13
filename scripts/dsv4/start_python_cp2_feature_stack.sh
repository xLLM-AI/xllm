#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp}"
BINARY="${BINARY:-$ROOT_DIR/build/lib.linux-aarch64-cpython-311/xllm/xllm}"
DEVICE_BASE="${DEVICE_BASE:-8}"
BASE_PORT="${BASE_PORT:-20050}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1:10400}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/dsv4_py_cp2_feature_stack_logs}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

if [[ "$DEVICE_BASE" != "8" ]]; then
  echo "This script defaults to, and protects, physical cards 8-15 (DEVICE_BASE=8)." >&2
  exit 2
fi
if [[ ! -x "$BINARY" ]]; then
  echo "xLLM binary is not executable: $BINARY" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "Model directory does not exist: $MODEL" >&2
  exit 2
fi

set +u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -u
opp=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_xllm_math
if [[ -d "$opp" ]]; then
  export ASCEND_CUSTOM_OPP_PATH="$opp${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
fi

port_is_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" { found=1 } END { exit !found }'
    return $?
  fi
  (exec 3<>"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
}

pid_cmdline() {
  local pid="$1"
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
}

mkdir -p "$LOG_DIR"
for rank in $(seq 0 7); do
  pid_file="$LOG_DIR/node_${rank}.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(<"$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "A live process is recorded in $pid_file: pid=$pid" >&2
      echo "  $(pid_cmdline "$pid")" >&2
      exit 1
    fi
    rm -f "$pid_file"
  fi
  port=$((BASE_PORT + rank))
  if port_is_listening "$port"; then
    echo "Refusing to start: port $port is already listening." >&2
    exit 1
  fi
done

master_port="${MASTER_ADDR##*:}"
if [[ "$MASTER_ADDR" == *:* ]] && port_is_listening "$master_port"; then
  echo "Refusing to start: master port $master_port is already listening." >&2
  exit 1
fi

rm -f "$LOG_DIR"/node_*.exit
export PYTHONHASHSEED=0
export PYTHONIOENCODING=utf-8
export XLLM_PYTHON_MODEL_PATH="${XLLM_PYTHON_MODEL_PATH:-$ROOT_DIR/build/lib.linux-aarch64-cpython-311}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-0}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-43680}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-18466-18565}"
export ASCEND_GLOBAL_EVENT_ENABLE="${ASCEND_GLOBAL_EVENT_ENABLE:-0}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"

echo "Starting Python DeepSeek-V4 on physical cards 8-15"
echo "Model: $MODEL"
echo "Logs:  $LOG_DIR"

for rank in $(seq 0 7); do
  port=$((BASE_PORT + rank))
  log_file="$LOG_DIR/node_${rank}.log"
  : > "$log_file"
  ASCEND_RT_VISIBLE_DEVICES=$((DEVICE_BASE + rank)) nohup "$BINARY" \
    --model "$MODEL" \
    --model_id deepseek_v4 \
    --model_impl python \
    --host 127.0.0.1 \
    --port "$port" \
    --master_node_addr="$MASTER_ADDR" \
    --nnodes=8 \
    --node_rank="$rank" \
    --max_memory_utilization=0.8 \
    --max_cache_size=4294967296 \
    --host_blocks_factor=2 \
    --max_tokens_per_batch=2048 \
    --max_seqs_per_batch=16 \
    --block_size=128 \
    --communication_backend=hccl \
    --tool_call_parser=deepseekv4 \
    --enable_prefix_cache=true \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --python_graph_backend=aclgraph \
    --enable_graph_double_buffer=true \
    --npu_kernel_backend=TORCH \
    --ep_size=8 \
    --dp_size=1 \
    --cp_size=2 \
    > "$log_file" 2>&1 &
  echo "$!" > "$LOG_DIR/node_${rank}.pid"
  echo "rank=$rank device=$((DEVICE_BASE + rank)) port=$port pid=$(<"$LOG_DIR/node_${rank}.pid")"
done

deadline=$((SECONDS + STARTUP_TIMEOUT))
while (( SECONDS < deadline )); do
  failed=0
  for rank in $(seq 0 7); do
    pid="$(<"$LOG_DIR/node_${rank}.pid")"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "rank $rank exited during startup; see $LOG_DIR/node_${rank}.log" >&2
      failed=1
    fi
  done
  if (( failed )); then
    exit 1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$BASE_PORT/health" >/dev/null 2>&1; then
    echo "Service is healthy: http://127.0.0.1:$BASE_PORT/health"
    echo "Stop: $ROOT_DIR/scripts/dsv4/stop_python_cp2_feature_stack.sh LOG_DIR=$LOG_DIR"
    exit 0
  fi
  sleep 5
done

echo "Timed out waiting for service health: http://127.0.0.1:$BASE_PORT/health" >&2
exit 1
