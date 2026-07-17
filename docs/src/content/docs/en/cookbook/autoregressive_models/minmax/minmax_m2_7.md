---
title: "MiniMax-M2.7"
sidebar:
  order: 3
---

+ Source code: https://github.com/xLLM-AI/xllm

+ Available in China: https://gitcode.com/xLLM-AI/xllm

+ Weight download: [modelscope-MiniMax-M2.7](https://www.modelscope.cn/models/MiniMax/MiniMax-M2.7)
+ Offline dequantized weights: [modelscope-Minimax2.7-BF16-xLLM](https://modelscope.cn/models/Eco-Tech/Minimax2.7-BF16-xLLM)

## 0. Weight Preparation

The original MiniMax-M2.7 weights are in FP8 format. xLLM supports the following three loading methods:

### Method 1: Load FP8 weights directly (online dequantization)

Use the original FP8 weight path directly. xLLM will dequantize FP8 to BF16 during inference, so no additional preprocessing is required.

```bash
MODEL_PATH=/path/to/MiniMax-M2.7/
```

### Method 2: Offline dequantization

Use the tool script to convert FP8 weights to BF16 in advance to avoid the extra overhead of online dequantization:

```bash
python tools/dequant_minimax_fp8.py --input-dir /path/to/MiniMax-M2.7/ --output-dir /path/to/MiniMax-M2.7-bf16/
```

### Method 3: Download pre-converted BF16 weights

Download the dequantized BF16 weights directly:

```bash
git clone https://www.modelscope.cn/Eco-Tech/Minimax2.7-BF16-xLLM.git
```

## 1. Pull the Image Environment

First, download the image provided by xLLM:

```bash
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260429
```

Then create the corresponding container:

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name xllm_minimax --network=host \
 -v /var/queue_schedule:/var/queue_schedule \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
 -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
 -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
 -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
 -v /var/log/npu/slog/:/var/log/npu/slog \
 -v ~/.ssh:/root/.ssh  \
 -v /var/log/npu/profiling/:/var/log/npu/profiling \
 -v /var/log/npu/dump/:/var/log/npu/dump \
 -v /runtime/:/runtime/ -v /etc/hccn.conf:/etc/hccn.conf \
 -v /export/home:/export/home \
 -v /home/:/home/  \
 -w /export/home \
 quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260429
```

## 2. Pull the Source Code and Build

Download the official repository and module dependencies:

```bash
git clone https://github.com/xLLM-AI/xllm.git
cd xllm
git checkout preview/minimax-minimal
git submodule init
git submodule update
```

Download and install dependencies:

```bash
pip install --upgrade pre-commit
yum install numactl
```

Run the build to generate the executable under `build/`:

```bash
python setup.py build
```

Build artifact path: `build/xllm/core/server/xllm`

## 3. Start the Model

### If the service is being started for the first time after the machine has rebooted, initialize the devices first

If this is skipped and the NPU has not been initialized, the xLLM process may fail to start.

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### Environment Variables

```bash
##### 1. Configure dependency path environment variables
export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
export PYTORCH_NPU_INSTALL_PATH=/usr/local/libtorch_npu/
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LIBTORCH_ROOT="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/libtorch_npu/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

##### 2. Configure log-related environment variables
rm -rf /root/atb/log/
rm -rf /root/ascend/log/
rm -rf core.*
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_STDOUT=1
export ASDOPS_LOG_TO_FILE=1

##### 3. Configure performance and communication-related environment variables
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

export OMP_NUM_THREADS=12
export ALLOW_INTERNAL_FORMAT=1

export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export ATB_CONVERT_NCHW_TO_ND=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export ATB_OPERATION_EXECUTE_ASYNC=2
export ATB_CONTEXT_WORKSPACE_SIZE=0
export INF_NAN_MODE_ENABLE=1
export HCCL_EXEC_TIMEOUT=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_IF_BASE_PORT=2864
```

## Startup Command - MiniMax-M2.7 (single machine, 16 cards, TP=16)

```bash
BATCH_SIZE=256
# Maximum inference batch size
XLLM_PATH="build/xllm/core/server/xllm"
# Inference entry binary path, which is the build artifact from the previous step
MODEL_PATH=/path/to/MiniMax-M2.7/
# Model path

MASTER_NODE_ADDR="10.143.3.204:10015"
LOCAL_HOST="10.143.3.204"
# Service port
START_PORT=18994
LOG_DIR="logs"
NNODES=16

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup numactl -C $((i*40))-$((i*40+39)) $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.90 \
    --max_tokens_per_batch=8192 \
    --max_seqs_per_batch=$BATCH_SIZE \
    --communication_backend=hccl \
    --enable_chunked_prefill=false \
    --enable_prefix_cache=false \
    --enable_schedule_overlap=false \
    --enable_graph=false \
    --enable_atb_spec_kernel=false \
    > $LOG_FILE 2>&1 &
done
```
