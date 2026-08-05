#!/bin/bash
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-43432}"  # HCCL communication base port

XLLM_BIN="${XLLM_BIN:-/opt/xllm/build/lib.linux-aarch64-cpython-311/xllm/xllm}"  # xLLM server binary
MODEL_PATH="${MODEL_PATH:-/path/to/model/Qwen3.5-4B}"   # Model path
MODEL_ID="${MODEL_ID:-Qwen3.5-4B}"                      # Served model name
PORT="${PORT:-18000}"                                   # Service port
DEVICE="${DEVICE:-0}"                                   # Logical device number
MASTER_NODE_ADDR="${MASTER_NODE_ADDR:-127.0.0.1:9748}"  # Master node address

$XLLM_BIN \
  --model "$MODEL_PATH" \
  --model_id="$MODEL_ID" \
  --port "$PORT" \
  --devices="npu:$DEVICE" \
  --master_node_addr="$MASTER_NODE_ADDR" \
  --nnodes=1 \
  --node_rank=0 \
  --max_memory_utilization=0.6 \
  --block_size=128 \
  --enable_prefix_cache="${ENABLE_PREFIX_CACHE:-false}" \
  --enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL:-false}" \
  --enable_schedule_overlap="${ENABLE_SCHEDULE_OVERLAP:-false}" \
  --enable_shm="${ENABLE_SHM:-false}" \
  ${EXTRA_ARGS:-}
