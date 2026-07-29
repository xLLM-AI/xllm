---
title: "Mthreads MUSA"
description: "Run xLLM on Mthreads MUSA GPUs."
sidebar:
  order: 7
---

Use the MUSA device backend when running xLLM on Mthreads MUSA hardware; select the visible card(s) with `MUSA_VISIBLE_DEVICES`.

## Image and Container Startup

Pull the Mthreads xLLM image:

```bash
docker pull registry.mthreads.com/presale/devtech/xllm:0710
```

Start the container:

```bash
docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --shm-size=128g \
  --name xllm-musa \
  --device=/dev/mtgpu0 \
  --device=/dev/dri \
  --group-add video \
  --ulimit memlock=-1 \
  -v $HOME:$HOME \
  -w $HOME \
  registry.mthreads.com/presale/devtech/xllm:0710 \
  /bin/bash
```

Add `--device=/dev/mtgpuN` for additional GPUs. Select the physical GPU with `export MUSA_VISIBLE_DEVICES=0`.

## Server Startup Command

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

For a single-device run, the logical device id usually starts at `0`. For multi-GPU or multi-node deployments, keep device ids, `--node_rank`, `--nnodes`, and service ports aligned with your topology.

## Notes

- Image and container startup are also listed in [Quick Start](/en/getting_started/quick_start/).
- Launch flags and multi-node setup are covered in [Launch xllm](/en/getting_started/launch_xllm/).
