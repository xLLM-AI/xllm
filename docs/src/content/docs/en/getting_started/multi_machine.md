---
title: "Multi-Node Deployment"
sidebar:
  order: 5
---
This example demonstrates how to launch a 32-GPU (NPU) deployment across 2 machines.
Launching Services on the First Machine:
```shell
bash start_deepseek_machine_1.sh
```

The start_deepseek_machine_1.sh script is as follows:
```bash
#!/bin/bash
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh 
export HCCL_IF_BASE_PORT=43432  # HCCL communication base port

# 4. Start distributed service
MODEL_PATH="/path/to/your/DeepSeek-R1"             # Model path
MASTER_NODE_ADDR="123.123.123.123:9748"            # Master node address (must be globally consistent)
LOCAL_HOST=123.123.123.123                         # Local IP for service launch
START_PORT=18000                                   # Service starting port
START_DEVICE=0                                     # Starting NPU logical device number
LOCAL_NODES=16                                     # Number of local processes (this script launches 16 processes)
LOG_DIR="log"                                      # Log directory
NNODES=32                                          # Total number of GPUs/NPUs (32 in this 2-machine example)

mkdir -p $LOG_DIR

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  /path/to/xllm \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=40000 \
    --max_seqs_per_batch=256 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --rank_tablefile=./ranktable_2s_32p.json \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

Launching Services on the Second Machine:
```shell
bash start_deepseek_machine_2.sh
```

The start_deepseek_machine_2.sh script is as follows:
```bash
#!/bin/bash
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_IF_BASE_PORT=43432  # HCCL communication base port

MODEL_PATH="/path/to/your/DeepSeek-R1"             # Model path
MASTER_NODE_ADDR="123.123.123.123:9748"            # Master node address (must be globally consistent)
LOCAL_HOST=456.456.456.456                         # Local IP for service launch
START_PORT=18000                                   # Service starting port
START_DEVICE=0                                     # Starting NPU logical device number
LOCAL_NODES=16                                     # Number of local processes (this script launches 16 processes)
LOG_DIR="log"                                      # Log directory
NNODES=32                                          # Total number of GPUs/NPUs (32 in this 2-machine example)

mkdir -p $LOG_DIR

for (( i=0; i<$LOCAL_NODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  /path/to/xllm \
    --model $MODEL_PATH \
    --host $LOCAL_HOST \
    --port $PORT \
    --devices="npu:$DEVICE" \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --max_tokens_per_batch=40000 \
    --max_seqs_per_batch=256 \
    --block_size=128 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --communication_backend="hccl" \
    --enable_schedule_overlap=true \
    --rank_tablefile=./ranktable_2s_32p.json \
    --node_rank=$((i + LOCAL_NODES))  \ > $LOG_FILE 2>&1 &
done
```
This example uses 2 machines. You can set the total number of GPUs/NPUs via `--nnodes`, where `--node_rank` specifies the global rank ID for each node.
The `--rank_tablefile=./ranktable_2s_32p.json`parameter points to the configuration file required for establishing the NPU communication domain. For instructions on generating this file, refer to [Ranktable Generation](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/README.md).

