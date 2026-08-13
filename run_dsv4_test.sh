#!/bin/bash
# 测试 C++ libtorch 组图模式的 DeepSeek-V4 推理服务
# model_impl=native(C++内置) + enable_graph=true(ACL graph组图) + npu_kernel_backend=TORCH(libtorch)

cd /mnt/sfs_turbo/tongpan/xllm

# --- 1. 环境变量(文档要求的三个 set_env + 性能变量) ---
# 第三方 set_env 脚本含未定义变量(如 ZSH_VERSION), source 时关闭 -u
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

XLLM_PATH=/mnt/sfs_turbo/tongpan/xllm/build/lib.linux-aarch64-cpython-311/xllm/xllm
MODEL_PATH=/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp
START_PORT=18994
COORD_PORT=10015
NNODES=8
MASTER="127.0.0.1:$COORD_PORT"
LOG_DIR=${LOG_DIR:-/mnt/sfs_turbo/tongpan/xllm/dsv4_logs}
ENABLE_GRAPH=${ENABLE_GRAPH:-true}
mkdir -p "$LOG_DIR"

# --- 2. 清理已有进程 ---
pkill -9 -x xllm 2>/dev/null || true
sleep 2

# --- 3. 启动 8 进程(每进程 1 卡, ep_size=8) ---
for i in $(seq 0 $((NNODES-1))); do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ASCEND_RT_VISIBLE_DEVICES=$i nohup "$XLLM_PATH" \
    --model "$MODEL_PATH" \
    --model_id deepseek_v4 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --master_node_addr="$MASTER" \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.8 \
    --max_tokens_per_batch=2048 \
    --max_seqs_per_batch=16 \
    --block_size=128 \
    --communication_backend=hccl \
    --tool_call_parser=deepseekv4 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph="$ENABLE_GRAPH" \
    --enable_graph_double_buffer=false \
    --npu_kernel_backend=TORCH \
    --ep_size=8 \
    --dp_size=1 \
    > "$LOG_FILE" 2>&1 &
  echo "[$(date +%H:%M:%S)] started node $i on port $PORT (device $i) -> $LOG_FILE"
done

echo "[$(date +%H:%M:%S)] 所有 $NNODES 个节点已启动, 等待加载 280GB 权重 + HCCL 建立 + graph 捕获..."
echo "监控: tail -f $LOG_DIR/node_0.log"
echo "成功标志: 日志出现 'Brpc Server Started'"
