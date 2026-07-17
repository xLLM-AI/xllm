---
title: "DeepSeek-V4"
description: "DeepSeek-V4 inference guide with xLLM on Ascend A3 devices"
---

# Inference with xLLM on Ascend A3 Devices

Source code: https://github.com/jd-opensource/xllm

China mirror: https://gitcode.com/xLLM-AI/xllm

Weight Download
Flash weights:
https://modelers.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp

Pro weights:
https://modelers.cn/models/Eco-Tech/DeepSeek-V4-Pro-w4a8-mtp


## 1. Pull the Docker Image

First, pull the xLLM-provided image:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-cann9-20260605
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-cann9-20260605
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

Then create the container:

```bash
sudo docker run -it --ipc=host -u 0 --privileged --name mydocker --network=host \
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
 quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605
```

## 2. Clone Source Code and Build

Clone the official repository and module dependencies:

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm 
git submodule update --init --recursive
```

Install dependencies:

```bash
pip install --upgrade pre-commit
```

Build the project; the executable `build/xllm/core/server/xllm` will be generated under `build/`:

```bash
python setup.py build --device npu
```

## 3. Launch the Model

### If restarting after a machine reboot, initialize the device first

> If not executed and the NPU is not initialized, the xllm process may fail to start

```bash
python -c "import torch_npu
for i in range(16):torch_npu.npu.set_device(i)"
```

### Export MTP weights

```bash
python tools/export_mtp.py --input-dir ${W4A8/W8A8 weights directory} --output-dir ${Exported MTP weights directory}
```

### Environment variables

```bash
##### 1. Configure dependency path environment variables

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source ${ASCEND_TOOLKIT_HOME}/opp/vendors/custom_xllm_math/bin/set_env.bash

##### 2. Configure logging environment variables
rm -rf /root/ascend/log/
rm -rf core.*

##### 3. Configure performance and communication environment variables
export HCCL_IF_BASE_PORT=43432
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_MEMORY_FRACTION=0.96
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export ATB_CONTEXT_WORKSPACE_SIZE=0
export OMP_NUM_THREADS=12
export ALLOW_INTERNAL_FORMAT=1

```

## Launch Command - Single-Node Example

```bash
BATCH_SIZE=256
# Maximum batch size for inference
XLLM_PATH="./myxllm/xllm/build/xllm/core/server/xllm"
# Path to the inference entry file (build artifact from the previous step)
MODEL_PATH=/path/to/dsv4
# Model path
DRAFT_MODEL_PATH=/path/to/dsv4_mtp
# Exported MTP weights path

MASTER_NODE_ADDR="11.87.49.110:10015"
LOCAL_HOST="11.87.49.110"
# Service Port
START_PORT=18994
LOG_DIR="logs"
NNODES=8

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup $XLLM_PATH -model-id ds \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    --max_memory_utilization=0.9 \
    --max_tokens_per_batch=2048 \
    --max_seqs_per_batch=32 \
    --block_size=128 \
    --communication_backend="hccl" \
    --tool_call_parser=deepseekv4 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --npu_kernel_backend=TORCH \
    --ep_size=8 \
    --dp_size=2 \
    > $LOG_FILE 2>&1 &
done

    # Variables required when MTP is enabled
    # --draft_model=$DRAFT_MODEL_PATH \
    # --num_speculative_tokens=1 \

# numactl -C xxxxx          NUMA core binding (query with: npu-smi info -t topo)
#--max_memory_utilization   Max memory usage ratio per NPU card
#--max_tokens_per_batch     Max tokens per batch (mainly limits prefill)
#--max_seqs_per_batch       Max sequences per batch (mainly limits decode)
#--communication_backend    Communication backend (hccl / lccl, hccl recommended here)
#--enable_schedule_overlap  Enable async scheduling
#--enable_prefix_cache      Enable prefix cache
#--enable_chunked_prefill   Enable chunked prefill
#--enable_graph             Enable aclgraph
#--draft_model              MTP - MTP weights path
#--num_speculative_tokens   MTP - Number of speculative tokens
```

A log message "Brpc Server Started" indicates the service has started successfully.

## Other Optional Environment Variables

```bash
#Enable deterministic computation
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0

# #Enable dynamic profiling mode
# export PROFILING_MODE=dynamic
# \rm -rf ~/dynamic_profiling_socket_*
```

## Launch Command - Dual-Node Example

### Node0 (master)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.110"
START_PORT=15890
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ )); do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$i \
    ......
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

#### Node1 (worker)

```bash
MASTER_NODE_ADDR="11.87.49.110:19990"
LOCAL_HOST="11.87.49.111"
START_PORT=15890
LOG_DIR="logs"
NNODES=32
LOCAL_NODES=16
export HCCL_IF_BASE_PORT=48439
unset HCCL_OP_EXPANSION_MODE

for (( i=0; i<$LOCAL_NODES; i++ )); do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  nohup  $XLLM_PATH \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --node_rank=$((i + LOCAL_NODES)) \
    ......
    --rank_tablefile=/yourPath/ranktable.json \
    > $LOG_FILE 2>&1 &
done
```

### ranktable reference

 [A3 ranktable configuration](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000066.html)

 [A2 ranktable configuration](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclug_000067.html)

 (Note the ranktable format differences between A3 and A2)
