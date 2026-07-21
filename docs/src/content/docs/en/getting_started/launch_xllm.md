---
title: "Launch xllm"
sidebar:
  order: 3
---
Taking Qwen3 as an example, the script for launching xllm is as follows. The provided script is suitable for both single-node single-device and single-node multi-device scenarios. When using multiple devices on a single node, you need to modify `NNODES` (one device represents one node), as well as environment variables such as `ASCEND_RT_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, or `MLU_VISIBLE_DEVICES`.

## NPU

```bash
#!/bin/bash
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432  # HCCL communication base port

MODEL_PATH="/path/to/model/Qwen3-8B"               # Model path
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master node address (must be globally consistent)
START_PORT=18000                                   # Service starting port
LOG_DIR="log"                                      # Log directory
NNODES=1                                           # Number of nodes (current script launches 1 process)

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --max_memory_utilization=0.86 \
    --block_size=128 \
    --communication_backend="hccl" \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_shm=true \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

## NVIDIA GPU

```bash
#!/bin/bash
set -e

rm -rf core.*

export CUDA_VISIBLE_DEVICES=0
# for debug
# export CUDA_LAUNCH_BLOCKING=1

MODEL_PATH="/path/to/model/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
START_PORT=18000
LOG_DIR="log"
NNODES=1

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --block_size=32 \
    --max_memory_utilization=0.8 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

## MLU

```bash
#!/bin/bash
set -e

rm -rf core.*

export MLU_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
START_PORT=18000
LOG_DIR="log"
NNODES=1

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --nnodes=$NNODES \
    --master_node_addr=$MASTER_NODE_ADDR \
    --block_size=16 \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

## Hygon DCU

```bash
#!/bin/bash
set -e

rm -rf core.*

export HIP_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
START_PORT=18000
LOG_DIR="log"
NNODES=1

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --nnodes=$NNODES \
    --master_node_addr=$MASTER_NODE_ADDR \
    --block_size=128 \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

## MetaX MACA

```bash
#!/bin/bash
set -e

rm -rf core.*

export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_OPS_PATH=/opt/conda/lib/python3.10/site-packages/flashinfer/data/aot/

MODEL_PATH="/path/to/model/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
START_PORT=18000
LOG_DIR="log"
NNODES=1

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --nnodes=$NNODES \
    --master_node_addr=$MASTER_NODE_ADDR \
    --block_size=128 \
    --max_memory_utilization=0.86 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=false \
    --enable_schedule_overlap=true \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

## Mthreads MUSA

```bash
#!/bin/bash
set -e

rm -rf core.*

export MUSA_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3.5-27B"
MASTER_NODE_ADDR="127.0.0.1:9748"
START_PORT=18000
LOG_DIR="log"
NNODES=1

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm \
    --model $MODEL_PATH \
    --port $PORT \
    --master_node_addr=$MASTER_NODE_ADDR \
    --nnodes=$NNODES \
    --block_size=64 \
    --max_memory_utilization=0.8 \
    --enable_prefix_cache=false \
    --enable_chunked_prefill=true \
    --enable_schedule_overlap=true \
    --enable_graph=true \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

See [Mthreads MUSA](/en/hardware/musa/) for details.
