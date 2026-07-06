#!/bin/bash

# MiniMax-M3 BF16 single-node A3 launch example.
# Download weights before running:
#   huggingface-cli download MiniMaxAI/MiniMax-M3 --local-dir /models/MiniMax-M3

set -eo pipefail

# Launch config.
MODEL_PATH="/models/MiniMax-M3"
MODEL_ID="MiniMax-M3"
MASTER_NODE_ADDR="127.0.0.1:28994"
HOST="127.0.0.1"
START_PORT=18994
START_DEVICE=0
LOG_DIR="log_minimax_m3"
NNODES=16
DP_SIZE=1
EP_SIZE=1
EXPERT_PARALLEL_DEGREE=0
MAX_TOKENS_PER_BATCH=8192
MAX_SEQS_PER_BATCH=256
MAX_MEMORY_UTILIZATION=0.90

export GLOG_logtostderr=1
export GLOG_logbufsecs=0
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export OMP_NUM_THREADS=12
export HCCL_EXEC_TIMEOUT=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=2864
export HCCL_OP_EXPANSION_MODE="AIV"
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export ATB_CONVERT_NCHW_TO_AND=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export ATB_OPERATION_EXECUTE_ASYNC=2
export ATB_CONTEXT_WORKSPACE_SIZE=0
export INF_NAN_MODE_ENABLE=1
export ALLOW_INTERNAL_FORMAT=1
export XLLM_MINIMAX_NATIVE_DECODE_ATTN=1
export XLLM_MINIMAX_NATIVE_DECODE_MOE=1
export XLLM_MINIMAX_NATIVE_DECODE_MOE_MIN_BATCH=8
export XLLM_MINIMAX_WARMUP_DECODE_BUCKETS=1
export XLLM_MINIMAX_TILING_CACHE_SIZE=64
export XLLM_SCHED_IDLE_COLLECT_WINDOW_MS=20
export XLLM_GRAPH_WARMUP_PREFILL_TOKENS=1024
export XLLM_STREAM_HF_STATE_DICT_LOAD=1

mkdir -p "${LOG_DIR}"

for ((i = 0; i < NNODES; i++)); do
    DEVICE=$((START_DEVICE + i))
    PORT=$((START_PORT + i))
    LOG_FILE="${LOG_DIR}/node_${i}.log"
    xllm \
        --model "${MODEL_PATH}" \
        --model_id "${MODEL_ID}" \
        --backend=llm \
        --devices="npu:${DEVICE}" \
        --host="${HOST}" \
        --port="${PORT}" \
        --master_node_addr="${MASTER_NODE_ADDR}" \
        --nnodes="${NNODES}" \
        --node_rank="${i}" \
        --dp_size="${DP_SIZE}" \
        --ep_size="${EP_SIZE}" \
        --expert_parallel_degree="${EXPERT_PARALLEL_DEGREE}" \
        --max_memory_utilization="${MAX_MEMORY_UTILIZATION}" \
        --max_tokens_per_batch="${MAX_TOKENS_PER_BATCH}" \
        --max_seqs_per_batch="${MAX_SEQS_PER_BATCH}" \
        --communication_backend=hccl \
        --enable_chunked_prefill=true \
        --enable_prefix_cache=true \
        --enable_schedule_overlap=true \
        --enable_graph=true \
        --enable_atb_spec_kernel=false \
        > "${LOG_FILE}" 2>&1 &
done
