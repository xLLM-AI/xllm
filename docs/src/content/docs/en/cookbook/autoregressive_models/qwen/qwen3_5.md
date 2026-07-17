---
title: "Qwen3.5"
sidebar:
  order: 1
---

+ Source code: https://github.com/xLLM-AI/xllm

+ Available in China: https://gitcode.com/xLLM-AI/xllm

+ Weight download: [modelscope-Qwen3.5-27B](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B)

## 1. Pull the Image Environment

First, download the image provided by xLLM:

```bash
# A3 arm (CANN 9)
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

Then create the corresponding container:

```bash
docker run -it -d \
    --ipc=host \
    -u 0 \
    --privileged \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --name xllm_qwen35 \
    --network=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /var/queue_schedule:/var/queue_schedule \
    -v /mnt/cfs/9n-das-admin/llm_models:/mnt/cfs/9n-das-admin/llm_models \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /export/home:/export/home \
    -v ~/.ssh:/root/.ssh \
    -v /home/:/home/ \
    -v /runtime/:/runtime/ \
    -w /home \
    quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

## 2. Pull the Source Code and Build

Download the official repository and module dependencies:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
pip install pre-commit
pre-commit install
git submodule update --init --recursive
```

Run the build to generate the executable under `build/`:

```bash
python setup.py build
```

Build artifact path: `build/xllm/core/server/xllm`

## 3. Start the Model

### Environment Variables

```bash
# 1. Configure dependency path environment variables
export ASDOPS_LOG_TO_STDOUT=0
export ASDOPS_LOG_LEVEL=3
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])' | tail -n 1)"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])' | tail -n 1)"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import site, os; print(os.path.join(site.getsitepackages()[0], "torch"))')"
export LIBTORCH_ROOT="$PYTORCH_INSTALL_PATH"
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH

# 2. Load environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_LEVEL=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.90
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=12
export HCCL_CONNECT_TIMEOUT=7200
export INF_NAN_MODE_ENABLE=0
export INF_NAN_MODE_FORCE_DISABLE=1

# 3. Clean up old logs
LOG_DIR="log"
mkdir -p $LOG_DIR
```

:::note
Qwen3.5 does not currently support TP=16.
:::

## Startup Command - Qwen3.5-27B (2 cards, TP=2, speculative decoding)

```bash
MODEL_PATH="/path/to/Qwen3.5-27B"
DRAFT_MODEL_PATH="/path/to/Qwen3.5-27B-mtp"

MASTER_NODE_ADDR="<master-host>:32764"
START_PORT=18076
NNODES=2

export ASCEND_RT_VISIBLE_DEVICES=14,15
export HCCL_IF_BASE_PORT=53433

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  ./xllm/build/xllm/core/server/xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.7 \
    --max_tokens_per_batch=32768 \
    --max_seqs_per_batch=8 \
    --block_size=128 \
    --communication_backend="lccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --node_rank=$i \
    --enable_shm=true \
    --task="generate" \
    --max_concurrent_requests=8 \
    --backend llm \
    --draft_model $DRAFT_MODEL_PATH \
    --num_speculative_tokens 3 \
    >> $LOG_FILE 2>&1 &
done
```
