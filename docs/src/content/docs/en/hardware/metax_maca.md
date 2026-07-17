---
title: "MetaX MACA"
description: "Run xLLM on MetaX MACA devices with the MetaX MACA backend."
sidebar:
  order: 6
---

Use the MetaX MACA backend when running xLLM on MetaX MACA hardware.

## Image and Container Startup

Pull the MetaX MACA development image:

```bash
docker pull pub-registry1.metax-tech.com/dev-m01421/xllm-maca3.7.1.9:v1
```

Start the container:

```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm-maca \
--network=host \
--privileged=true \
--shm-size 100gb \
--device=/dev/mxcd \
--device=/dev/dri \
--device=/dev/infiniband \
--security-opt seccomp=unconfined \
--security-opt apparmor=unconfined \
--group-add video \
--ulimit memlock=-1 \
-v /opt/maca:/opt/maca \
-v $HOME:$HOME \
-w $HOME \
<docker_image_name> \
/bin/bash
```

## Server Startup Command

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

For a single-device run, `<device-id>` usually starts from `0`. For multi-worker deployments, keep device ids, `--node_rank`, `--nnodes`, and service ports aligned.

## Notes

- The current docs list a pre-built MetaX MACA development image in [Quick Start](/en/getting_started/quick_start/).
- The MetaX MACA container startup command requires device mounts such as `/dev/mxcd`, `/dev/dri`, and `/dev/infiniband`; the command above includes these mounts.
- Build xllm with MetaX MACA: python setup.py build --device maca
