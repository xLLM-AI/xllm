#!/usr/bin/env bash
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# Minimal Disaggregated PD smoke test for xLLM + YuanRong DataSystem.
# Baseline values follow the single-instance command that has already run
# successfully with this v0.10 engine baseline.
#
# This script starts:
#   1. etcd
#   2. optional DataSystem worker through dscli, if no worker is listening
#   3. xllm_master_serving
#   4. xLLM PREFILL instance
#   5. xLLM DECODE instance
# Then it sends one OpenAI-compatible chat request to xllm_master_serving.
#
# Typical usage on the 910 server:
#   cd ${ROOT}
#   bash tools/datasystem/smoke_pd.sh
#
# If your DataSystem worker is already running on another port:
#   Set DATASYSTEM_WORKER_PORT in tools/datasystem/config.env.
#
# If xllm-service is not in a common path:
#   Set XLLM_SERVICE_BIN in tools/datasystem/config.env.

ROOT="${ROOT}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/models/qwen2.5_7B_Instruct}"
MODEL_ID="${MODEL_ID:-qwen2.5_7B_Instruct}"

XLLM_BIN="${XLLM_BIN:-${ROOT}/build/xllm/core/server/xllm}"

KV_BACKEND="${KV_BACKEND:-DataSystem}"
case "${KV_BACKEND,,}" in
  datasystem|yuanrong|yuanrongdatasystem|yrdatasystem)
    KV_BACKEND="DataSystem"
    KV_TRANSFER_MODE="${KV_TRANSFER_MODE:-PULL}"
    ;;
  mooncake)
    KV_BACKEND="Mooncake"
    KV_TRANSFER_MODE="${KV_TRANSFER_MODE:-PUSH}"
    ;;
  *)
    echo "ERROR: KV_BACKEND must be Mooncake or DataSystem, got: ${KV_BACKEND}" >&2
    exit 2
    ;;
esac

if [[ "${KV_BACKEND}" == "DataSystem" && "${KV_TRANSFER_MODE}" != "PULL" ]]; then
  echo "ERROR: DataSystem requires KV_TRANSFER_MODE=PULL." >&2
  exit 2
fi

PREFILL_DEVICE="${PREFILL_DEVICE:-0}"
DECODE_DEVICE="${DECODE_DEVICE:-1}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${PREFILL_DEVICE},${DECODE_DEVICE}}"
HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-43432}"

MAX_MEMORY_UTILIZATION="${MAX_MEMORY_UTILIZATION:-0.86}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
COMMUNICATION_BACKEND="${COMMUNICATION_BACKEND:-hccl}"
ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-false}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
ENABLE_SCHEDULE_OVERLAP="${ENABLE_SCHEDULE_OVERLAP:-false}"
ENABLE_SHM="${ENABLE_SHM:-true}"

ETCD_HOST="${ETCD_HOST:-127.0.0.1}"
ETCD_CLIENT_PORT="${ETCD_CLIENT_PORT:-12389}"
ETCD_PEER_PORT="${ETCD_PEER_PORT:-12390}"
ETCD_ADDR="${ETCD_ADDR:-${ETCD_HOST}:${ETCD_CLIENT_PORT}}"

SERVICE_HTTP_PORT="${SERVICE_HTTP_PORT:-28888}"
SERVICE_RPC_PORT="${SERVICE_RPC_PORT:-28889}"
SERVICE_HOST="${SERVICE_HOST:-127.0.0.1}"

PREFILL_PORT="${PREFILL_PORT:-18010}"
DECODE_PORT="${DECODE_PORT:-18020}"
PREFILL_MASTER_NODE_ADDR="${PREFILL_MASTER_NODE_ADDR:-127.0.0.1:18888}"
DECODE_MASTER_NODE_ADDR="${DECODE_MASTER_NODE_ADDR:-127.0.0.1:18898}"
PREFILL_TRANSFER_PORT="${PREFILL_TRANSFER_PORT:-26000}"
DECODE_TRANSFER_PORT="${DECODE_TRANSFER_PORT:-26100}"
PREFILL_DISAGG_PD_PORT="${PREFILL_DISAGG_PD_PORT:-7777}"
DECODE_DISAGG_PD_PORT="${DECODE_DISAGG_PD_PORT:-7787}"

DATASYSTEM_WORKER_HOST="${DATASYSTEM_WORKER_HOST:-127.0.0.1}"
DATASYSTEM_WORKER_PORT="${DATASYSTEM_WORKER_PORT:-31512}"
DATASYSTEM_ETCD_HOST="${DATASYSTEM_ETCD_HOST:-127.0.0.1}"
DATASYSTEM_ETCD_PORT="${DATASYSTEM_ETCD_PORT:-32379}"
DATASYSTEM_ETCD_ADDR="${DATASYSTEM_ETCD_ADDR:-${DATASYSTEM_ETCD_HOST}:${DATASYSTEM_ETCD_PORT}}"
START_DATASYSTEM_WORKER="${START_DATASYSTEM_WORKER:-0}"
XLLM_YUANRONG_ENABLE_REMOTE_H2D="${XLLM_YUANRONG_ENABLE_REMOTE_H2D:-0}"

WAIT_SECONDS="${WAIT_SECONDS:-900}"
CURL_TIMEOUT_SECONDS="${CURL_TIMEOUT_SECONDS:-180}"
RESPONSE_MAX_TOKENS="${RESPONSE_MAX_TOKENS:-64}"
PROMPT="${PROMPT:-Introduce yourself in one sentence.}"
STOP_AFTER_TEST="${STOP_AFTER_TEST:-1}"

LOG_DIR="${LOG_DIR:-${ROOT}/log/disagg_pd_$(date +%Y%m%d_%H%M%S)}"
PIDS_FILE="${PIDS_FILE:-${LOG_DIR}/pids.txt}"

die() {
  echo "ERROR: $*" >&2
  echo >&2
  echo "Logs are under: ${LOG_DIR}" >&2
  for log in "${LOG_DIR}"/*.log; do
    [[ -f "${log}" ]] || continue
    echo >&2
    echo "===== tail -80 ${log} =====" >&2
    tail -n 80 "${log}" >&2 || true
  done
  exit 1
}

need_file() {
  local path="${1:-}"
  [[ -n "${path}" ]] || die "missing file path argument; an auto-detected path is empty"
  [[ -f "${path}" ]] || die "missing file: ${path}"
}

need_file_named() {
  local name="${1:-path}"
  local path="${2:-}"
  [[ -n "${path}" ]] || die "${name} is empty; auto-detect failed. Set ${name}=/path/to/file"
  [[ -f "${path}" ]] || die "${name} does not exist: ${path}"
}

need_dir() {
  local path="${1:-}"
  [[ -n "${path}" ]] || die "missing directory path argument; an auto-detected path is empty"
  [[ -d "${path}" ]] || die "missing directory: ${path}"
}

find_first_executable() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

find_in_root() {
  local root="$1"
  local name="$2"
  [[ -d "${root}" ]] || return 1
  find "${root}" -type f -name "${name}" -perm /111 2>/dev/null | head -n 1
}

find_shared_lib() {
  local name="$1"
  local candidate
  for candidate in \
    "${ROOT}/build/xllm/core/server/${name}" \
    "${ROOT}/build/xllm/core/framework/kv_cache_transfer/${name}" \
    "${ROOT}/build/cmake.linux-aarch64-cpython-311/xllm/core/server/${name}" \
    "${ROOT}/build/cmake.linux-aarch64-cpython-311/xllm/core/framework/kv_cache_transfer/${name}" \
    "${ROOT}/build/lib.linux-aarch64-cpython-311/xllm/core/server/${name}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  find "${ROOT}/build" -type f -name "${name}" 2>/dev/null | head -n 1
}

port_open() {
  local host="$1"
  local port="$2"
  timeout 1 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1
}

require_port_free() {
  local name="$1"
  local host="$2"
  local port="$3"
  if port_open "${host}" "${port}"; then
    die "${name} port ${host}:${port} is already listening. Stop the old PD process before this test."
  fi
}

record_pid() {
  local name="$1"
  local pid="$2"
  echo "${name} ${pid}" >> "${PIDS_FILE}"
}

process_alive() {
  local pid="$1"
  kill -0 "${pid}" >/dev/null 2>&1
}

wait_log_pattern() {
  local name="$1"
  local pid="$2"
  local log_file="$3"
  local pattern="$4"
  local timeout_seconds="$5"
  local deadline=$((SECONDS + timeout_seconds))

  echo "Waiting for ${name}..."
  while (( SECONDS < deadline )); do
    if ! process_alive "${pid}"; then
      die "${name} exited during startup"
    fi
    if grep -E -q "${pattern}" "${log_file}" 2>/dev/null; then
      echo "${name} startup detected."
      return 0
    fi
    sleep 2
  done

  die "${name} did not become ready within ${timeout_seconds}s"
}

wait_http_ready() {
  local name="$1"
  local url="$2"
  local timeout_seconds="$3"
  local deadline=$((SECONDS + timeout_seconds))

  echo "Waiting for ${name} HTTP endpoint: ${url}"
  while (( SECONDS < deadline )); do
    if curl -sS --connect-timeout 2 --max-time 5 "${url}" >/dev/null 2>&1; then
      echo "${name} HTTP endpoint responded."
      return 0
    fi
    sleep 2
  done

  echo "WARNING: ${name} HTTP endpoint did not respond to a probe; continuing to chat request loop."
}

cleanup_started_processes() {
  [[ -f "${PIDS_FILE}" ]] || return 0
  "$DS_PYTHON" "$ROOT/tools/datasystem/stop_installation.py" "$XLLM_BIN" "$XLLM_SERVICE_BIN" || true
  tac "${PIDS_FILE}" 2>/dev/null | while read -r name pid; do
    [[ -n "${pid:-}" ]] || continue
    if process_alive "${pid}"; then
      echo "Stopping ${name} pid=${pid}"
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

mkdir -p "${LOG_DIR}"
: > "${PIDS_FILE}"
trap '[[ "${STOP_AFTER_TEST}" == 1 ]] && cleanup_started_processes || true' EXIT

echo "xLLM root: ${ROOT}"
echo "Model path: ${MODEL_PATH}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"
echo "PREFILL device: npu:${PREFILL_DEVICE}"
echo "DECODE device: npu:${DECODE_DEVICE}"
echo "Service HTTP: http://${SERVICE_HOST}:${SERVICE_HTTP_PORT}"
echo "etcd: ${ETCD_ADDR}"
echo "DataSystem worker: ${DATASYSTEM_WORKER_HOST}:${DATASYSTEM_WORKER_PORT}"
echo "KV backend: ${KV_BACKEND}"
echo "KV transfer mode: ${KV_TRANSFER_MODE}"
echo "Log dir: ${LOG_DIR}"

need_dir "${ROOT}"
need_dir "${MODEL_PATH}"
need_file_named XLLM_BIN "${XLLM_BIN}"
chmod +x "${XLLM_BIN}" || true

source "$CANN_ENV_SH"
source "$ATB_ENV_SH"
unset LD_PRELOAD

export ASCEND_RT_VISIBLE_DEVICES
export HCCL_IF_BASE_PORT
export XLLM_KV_CACHE_BACKEND="${KV_BACKEND}"

if [[ "${KV_BACKEND}" == "DataSystem" ]]; then
  export XLLM_YUANRONG_WORKER_HOST="${XLLM_YUANRONG_WORKER_HOST:-${DATASYSTEM_WORKER_HOST}}"
  export XLLM_YUANRONG_WORKER_PORT="${XLLM_YUANRONG_WORKER_PORT:-${DATASYSTEM_WORKER_PORT}}"
  export XLLM_YUANRONG_TIMEOUT_MS="${XLLM_YUANRONG_TIMEOUT_MS:-60000}"
  export XLLM_YUANRONG_ENABLE_REMOTE_H2D
  # DataSystem's ACL plugin and torch_npu must use the same process-global
  # CANN runtime. The adapter still keeps a C ABI and RTLD_LOCAL boundary.
  export XLLM_YUANRONG_BRIDGE_BASE_NAMESPACE="${XLLM_YUANRONG_BRIDGE_BASE_NAMESPACE:-1}"
  if [[ -n "${DATASYSTEM_SDK:-}" ]]; then
    export LD_LIBRARY_PATH="${DATASYSTEM_SDK}/lib:${DATASYSTEM_SDK}/lib64:${LD_LIBRARY_PATH:-}"
  fi
  if [[ -z "${XLLM_YUANRONG_DATASYSTEM_BRIDGE:-}" ]]; then
    XLLM_YUANRONG_DATASYSTEM_BRIDGE="$(find_shared_lib libxllm_yuanrong_datasystem_bridge.so || true)"
  fi
  export XLLM_YUANRONG_DATASYSTEM_BRIDGE
  need_file_named XLLM_YUANRONG_DATASYSTEM_BRIDGE "${XLLM_YUANRONG_DATASYSTEM_BRIDGE}"
  export LD_LIBRARY_PATH="$(dirname "${XLLM_YUANRONG_DATASYSTEM_BRIDGE}"):${LD_LIBRARY_PATH:-}"
fi

VERIFY_SCRIPT="${ROOT}/tools/datasystem/verify_build.sh"
need_file_named VERIFY_SCRIPT "${VERIFY_SCRIPT}"
BUILD_WITH_MOONCAKE="${BUILD_WITH_MOONCAKE:-ON}" \
BUILD_WITH_DATASYSTEM="${BUILD_WITH_DATASYSTEM:-ON}" \
XLLM_BIN="${XLLM_BIN}" \
BRIDGE_SO="${XLLM_YUANRONG_DATASYSTEM_BRIDGE:-${BRIDGE_SO:-}}" \
  bash "${VERIFY_SCRIPT}"

ETCD_BIN="$(command -v "${ETCD_BIN:-etcd}")" || die "Set ETCD_BIN to an installed etcd executable"
[[ -n "$XLLM_SERVICE_BIN" && -x "$XLLM_SERVICE_BIN" ]] || die "Set XLLM_SERVICE_BIN in config.env"
DSCLI_BIN="$(command -v dscli || true)"

echo "xLLM binary: ${XLLM_BIN}"
echo "xLLM service: ${XLLM_SERVICE_BIN}"
echo "etcd binary: ${ETCD_BIN:-not found; will only work if ${ETCD_ADDR} is already listening}"
if [[ "${KV_BACKEND}" == "DataSystem" ]]; then
  echo "DataSystem bridge: ${XLLM_YUANRONG_DATASYSTEM_BRIDGE}"
  echo "  (YuanRong KV adapter is compiled into the xllm binary; no separate plugin .so)"
  echo "dscli: ${DSCLI_BIN:-not found}"
fi

ETCD_LOG="${LOG_DIR}/etcd.log"
SERVICE_LOG="${LOG_DIR}/xllm_service.log"
PREFILL_LOG="${LOG_DIR}/prefill.log"
DECODE_LOG="${LOG_DIR}/decode.log"
DS_LOG="${LOG_DIR}/datasystem_worker.log"

PREFILL_MASTER_HOST="${PREFILL_MASTER_NODE_ADDR%:*}"
PREFILL_MASTER_PORT="${PREFILL_MASTER_NODE_ADDR##*:}"
DECODE_MASTER_HOST="${DECODE_MASTER_NODE_ADDR%:*}"
DECODE_MASTER_PORT="${DECODE_MASTER_NODE_ADDR##*:}"

require_port_free "xllm-service HTTP" "127.0.0.1" "${SERVICE_HTTP_PORT}"
require_port_free "xllm-service RPC" "127.0.0.1" "${SERVICE_RPC_PORT}"
require_port_free "PREFILL HTTP" "127.0.0.1" "${PREFILL_PORT}"
require_port_free "PREFILL collective" "${PREFILL_MASTER_HOST}" "${PREFILL_MASTER_PORT}"
require_port_free "PREFILL disagg PD" "127.0.0.1" "${PREFILL_DISAGG_PD_PORT}"
require_port_free "DECODE HTTP" "127.0.0.1" "${DECODE_PORT}"
require_port_free "DECODE collective" "${DECODE_MASTER_HOST}" "${DECODE_MASTER_PORT}"
require_port_free "DECODE disagg PD" "127.0.0.1" "${DECODE_DISAGG_PD_PORT}"
if [[ "${KV_BACKEND}" == "Mooncake" ]]; then
  require_port_free "PREFILL KV transfer" "127.0.0.1" "${PREFILL_TRANSFER_PORT}"
  require_port_free "DECODE KV transfer" "127.0.0.1" "${DECODE_TRANSFER_PORT}"
fi

if port_open "${ETCD_HOST}" "${ETCD_CLIENT_PORT}"; then
  echo "etcd port ${ETCD_ADDR} is already listening; reusing it."
else
  [[ -n "${ETCD_BIN}" ]] || die "etcd ${ETCD_ADDR} is not listening and ETCD_BIN was not found. Start metastore/etcd first or set ETCD_BIN=/path/to/etcd"
  "${ETCD_BIN}" \
    --name "xllm-pd-smoke" \
    --data-dir "${LOG_DIR}/etcd-data" \
    --listen-client-urls "http://${ETCD_HOST}:${ETCD_CLIENT_PORT}" \
    --advertise-client-urls "http://${ETCD_HOST}:${ETCD_CLIENT_PORT}" \
    --listen-peer-urls "http://${ETCD_HOST}:${ETCD_PEER_PORT}" \
    --initial-advertise-peer-urls "http://${ETCD_HOST}:${ETCD_PEER_PORT}" \
    --initial-cluster "xllm-pd-smoke=http://${ETCD_HOST}:${ETCD_PEER_PORT}" \
    --initial-cluster-state "new" \
    > "${ETCD_LOG}" 2>&1 &
  ETCD_PID="$!"
  record_pid etcd "${ETCD_PID}"
  wait_log_pattern etcd "${ETCD_PID}" "${ETCD_LOG}" "ready to serve client requests|serving client traffic|started" 60
fi

if [[ "${KV_BACKEND}" != "DataSystem" ]]; then
  echo "Mooncake selected; DataSystem worker is not required."
elif port_open "${DATASYSTEM_WORKER_HOST}" "${DATASYSTEM_WORKER_PORT}"; then
  echo "DataSystem worker is already listening on ${DATASYSTEM_WORKER_HOST}:${DATASYSTEM_WORKER_PORT}; reusing it."
elif [[ "${START_DATASYSTEM_WORKER}" == "0" || "${START_DATASYSTEM_WORKER}" == "false" ]]; then
  die "DataSystem worker is not listening on ${DATASYSTEM_WORKER_HOST}:${DATASYSTEM_WORKER_PORT}, and START_DATASYSTEM_WORKER=${START_DATASYSTEM_WORKER}. Check the worker port or start DataSystem first."
elif [[ -n "${DSCLI_BIN}" ]]; then
  echo "Starting DataSystem worker through dscli..."
  "${DSCLI_BIN}" start -w \
    --worker_address "${DATASYSTEM_WORKER_HOST}:${DATASYSTEM_WORKER_PORT}" \
    --etcd_address "${DATASYSTEM_ETCD_ADDR}" \
    > "${DS_LOG}" 2>&1 &
  DSCLI_PID="$!"
  record_pid datasystem-dscli "${DSCLI_PID}"
  sleep 8
  if port_open "${DATASYSTEM_WORKER_HOST}" "${DATASYSTEM_WORKER_PORT}"; then
    echo "DataSystem worker startup detected."
  else
    echo "WARNING: dscli was started but worker port is still not open."
    echo "         Check ${DS_LOG}"
  fi
else
  echo "WARNING: dscli not found and DataSystem worker is not listening on ${DATASYSTEM_WORKER_HOST}:${DATASYSTEM_WORKER_PORT}."
  echo "         Start it manually, for example:"
  echo "         dscli start -w --worker_address \"${DATASYSTEM_WORKER_HOST}:${DATASYSTEM_WORKER_PORT}\" --etcd_address \"${DATASYSTEM_ETCD_ADDR}\""
fi

ENABLE_DECODE_RESPONSE_TO_SERVICE=true "${XLLM_SERVICE_BIN}" \
  --etcd_addr="${ETCD_ADDR}" \
  --http_server_port="${SERVICE_HTTP_PORT}" \
  --rpc_server_port="${SERVICE_RPC_PORT}" \
  --tokenizer_path="${MODEL_PATH}" \
  > "${SERVICE_LOG}" 2>&1 &
SERVICE_PID="$!"
record_pid xllm-service "${SERVICE_PID}"
sleep 5

"${XLLM_BIN}" \
  --model "${MODEL_PATH}" \
  --model_id "${MODEL_ID}" \
  --devices="npu:${PREFILL_DEVICE}" \
  --port "${PREFILL_PORT}" \
  --master_node_addr="${PREFILL_MASTER_NODE_ADDR}" \
  --nnodes=1 \
  --node_rank=0 \
  --max_memory_utilization="${MAX_MEMORY_UTILIZATION}" \
  --block_size="${BLOCK_SIZE}" \
  --communication_backend="${COMMUNICATION_BACKEND}" \
  --enable_prefix_cache="${ENABLE_PREFIX_CACHE}" \
  --enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  --enable_schedule_overlap="${ENABLE_SCHEDULE_OVERLAP}" \
  --enable_shm="${ENABLE_SHM}" \
  --enable_disagg_pd=true \
  --instance_role=PREFILL \
  --etcd_addr="${ETCD_ADDR}" \
  --transfer_listen_port="${PREFILL_TRANSFER_PORT}" \
  --disagg_pd_port="${PREFILL_DISAGG_PD_PORT}" \
  --kv_cache_transfer_type="${KV_BACKEND}" \
  --kv_cache_transfer_mode="${KV_TRANSFER_MODE}" \
  > "${PREFILL_LOG}" 2>&1 &
PREFILL_PID="$!"
record_pid prefill "${PREFILL_PID}"

"${XLLM_BIN}" \
  --model "${MODEL_PATH}" \
  --model_id "${MODEL_ID}" \
  --devices="npu:${DECODE_DEVICE}" \
  --port "${DECODE_PORT}" \
  --master_node_addr="${DECODE_MASTER_NODE_ADDR}" \
  --nnodes=1 \
  --node_rank=0 \
  --max_memory_utilization="${MAX_MEMORY_UTILIZATION}" \
  --block_size="${BLOCK_SIZE}" \
  --communication_backend="${COMMUNICATION_BACKEND}" \
  --enable_prefix_cache="${ENABLE_PREFIX_CACHE}" \
  --enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  --enable_schedule_overlap="${ENABLE_SCHEDULE_OVERLAP}" \
  --enable_shm="${ENABLE_SHM}" \
  --enable_disagg_pd=true \
  --instance_role=DECODE \
  --etcd_addr="${ETCD_ADDR}" \
  --transfer_listen_port="${DECODE_TRANSFER_PORT}" \
  --disagg_pd_port="${DECODE_DISAGG_PD_PORT}" \
  --kv_cache_transfer_type="${KV_BACKEND}" \
  --kv_cache_transfer_mode="${KV_TRANSFER_MODE}" \
  > "${DECODE_LOG}" 2>&1 &
DECODE_PID="$!"
record_pid decode "${DECODE_PID}"

wait_log_pattern prefill "${PREFILL_PID}" "${PREFILL_LOG}" "Started server process|Brpc Server started|Initialized YuanRong KV cache transfer|Create KVCacheTransfer" "${WAIT_SECONDS}"
wait_log_pattern decode "${DECODE_PID}" "${DECODE_LOG}" "Started server process|Brpc Server started|Initialized YuanRong KV cache transfer|Create KVCacheTransfer" "${WAIT_SECONDS}"
wait_http_ready xllm-service "http://${SERVICE_HOST}:${SERVICE_HTTP_PORT}/v1/models" 60

REQ_FILE="${LOG_DIR}/request.json"
RESP_FILE="${LOG_DIR}/response.json"
cat > "${REQ_FILE}" <<EOF
{
  "model": "${MODEL_ID}",
  "max_tokens": ${RESPONSE_MAX_TOKENS},
  "temperature": 0,
  "stream": false,
  "messages": [
    {"role": "user", "content": "${PROMPT}"}
  ]
}
EOF

echo "Sending PD smoke request to xllm-service..."
deadline=$((SECONDS + CURL_TIMEOUT_SECONDS))
HTTP_STATUS="000"
while (( SECONDS < deadline )); do
  for pid in "${SERVICE_PID}" "${PREFILL_PID}" "${DECODE_PID}"; do
    process_alive "${pid}" || die "one of the PD processes exited before request completed"
  done

  HTTP_STATUS="$(curl -sS \
    --connect-timeout 5 \
    --max-time 60 \
    -o "${RESP_FILE}" \
    -w "%{http_code}" \
    "http://${SERVICE_HOST}:${SERVICE_HTTP_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"${REQ_FILE}" || true)"

  if [[ "${HTTP_STATUS}" == "200" ]] && grep -q '"choices"' "${RESP_FILE}" 2>/dev/null; then
    break
  fi
  sleep 5
done

echo "HTTP status: ${HTTP_STATUS}"
cat "${RESP_FILE}" 2>/dev/null || true
echo

if [[ "${HTTP_STATUS}" != "200" ]]; then
  die "PD chat completion request failed with HTTP ${HTTP_STATUS}"
fi
if ! grep -q '"choices"' "${RESP_FILE}"; then
  die "PD response does not contain choices"
fi

if [[ "${KV_BACKEND}" == "DataSystem" ]]; then
  grep -q 'Selected KV cache backend: DataSystem' "${PREFILL_LOG}" ||
    die "PREFILL log does not prove DataSystem backend selection"
  grep -q 'Selected KV cache backend: DataSystem' "${DECODE_LOG}" ||
    die "DECODE log does not prove DataSystem backend selection"
  grep -q 'YuanRong MGetH2D pull' "${DECODE_LOG}" ||
    die "DECODE log contains no DataSystem MGetH2D transfer evidence"
else
  grep -q 'Create KVCacheTransfer for Mooncake' "${PREFILL_LOG}" ||
    die "PREFILL log does not prove Mooncake backend selection"
  grep -q 'Create KVCacheTransfer for Mooncake' "${DECODE_LOG}" ||
    die "DECODE log does not prove Mooncake backend selection"
fi

echo
echo "PD launcher request completed; test_pd.sh validates successful KV transfers."
echo "Service URL: http://${SERVICE_HOST}:${SERVICE_HTTP_PORT}"
echo "Logs: ${LOG_DIR}"
echo "PIDs file: ${PIDS_FILE}"

if [[ "${STOP_AFTER_TEST}" == "1" ]]; then
  cleanup_started_processes
else
  echo "Processes are still running. Stop them with:"
  echo "  while read -r name pid; do kill \"\$pid\" 2>/dev/null || true; done < ${PIDS_FILE}"
fi
