#!/bin/bash

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_xllm_math/bin/set_env.bash
set -u

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.9
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_CONTEXT_WORKSPACE_SIZE=0
export OMP_NUM_THREADS=12
export ALLOW_INTERNAL_FORMAT=1
export HCCL_IF_BASE_PORT=43432
export ACL_OP_INIT_MODE=1
export PYTHONHASHSEED=0
export XLLM_PYTHON_MODEL_PATH=/mnt/sfs_turbo/tongpan/xllm/build/lib.linux-aarch64-cpython-311

workspace=/mnt/sfs_turbo/tongpan/xllm
binary="$workspace/build/lib.linux-aarch64-cpython-311/xllm/xllm"
model=/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp
log_dir="${LOG_DIR:-$workspace/dsv4_py_aclgraph_fix1_logs}"
master_addr="${MASTER_ADDR:-127.0.0.1:10015}"
enable_graph="${ENABLE_GRAPH:-true}"
max_tokens_per_batch="${MAX_TOKENS_PER_BATCH:-2048}"
mkdir -p "$log_dir"
rm -f "$log_dir"/node_*.exit

for rank in $(seq 0 7); do
  port=$((18994 + rank))
  nohup bash -c 'ASCEND_RT_VISIBLE_DEVICES="$1" "$2" \
    --model "$5" \
    --model_id deepseek_v4 \
    --model_impl python \
    --host 127.0.0.1 \
    --port "$6" \
    --master_node_addr="$8" \
    --nnodes=8 \
    --node_rank="$7" \
    --max_memory_utilization=0.8 \
    --max_tokens_per_batch="${10}" \
    --max_seqs_per_batch=16 \
    --block_size=128 \
    --communication_backend=hccl \
    --tool_call_parser=deepseekv4 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph="$9" \
    --python_graph_backend=aclgraph \
    --enable_graph_double_buffer=false \
    --npu_kernel_backend=TORCH \
    --ep_size=8 \
    --dp_size=1 \
    > "$3" 2>&1
    echo $? > "$4"' _ \
    "$rank" "$binary" "$log_dir/node_$rank.log" \
    "$log_dir/node_$rank.exit" "$model" "$port" "$rank" "$master_addr" "$enable_graph" "$max_tokens_per_batch" \
    >/dev/null 2>&1 &
done
