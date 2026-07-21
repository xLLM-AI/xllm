---
title: "Hygon DCU"
description: "Run xLLM on Hygon DCU devices with the Hygon DCU backend."
sidebar:
  order: 5
---

Use the Hygon DCU backend when running xLLM on Hygon DCU hardware.

## Image and Container Startup

Pull the Hygon DCU development image:

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:xllm-dev-dcu-x86-20260617
```

Start the container:

```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm-dcu \
--privileged \
--network=host \
--shm-size 256g \
--device=/dev/kfd \
--device=/dev/dri \
--device=/dev/mkfd \
--security-opt seccomp=unconfined \
--group-add video \
-v /opt/hyhal:/opt/hyhal \
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

For a single-device run, `<device-id>` usually starts from `0`. For multi-worker deployments, keep device ids, `--node_rank`, `--nnodes`, and service ports aligned.

## Notes

- The current docs list a pre-built Hygon DCU development image in [Quick Start](/en/getting_started/quick_start/).
- The Hygon DCU container startup command requires device mounts such as `/dev/kfd`, `/dev/dri`, and `/dev/mkfd`; the command above includes these mounts.
- Check the [Model Support List](/en/supported_models/) before choosing a Hygon DCU deployment target.
