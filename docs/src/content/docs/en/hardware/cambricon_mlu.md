---
title: "Cambricon MLU"
description: "Run xLLM on Cambricon MLU devices with the MLU backend."
sidebar:
  order: 4
---

Use the MLU backend when running xLLM on Cambricon devices.

## Image and Container Startup

xLLM does not currently provide a public MLU image in the docs. If you already have the development image, start the container with:

```bash
sudo docker run -it \
--privileged \
--shm-size '128gb' \
--ipc=host \
--net=host \
--pid=host \
--name xllm-mlu \
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

export MLU_VISIBLE_DEVICES=0

MODEL_PATH="/path/to/model/Qwen3-8B"
MASTER_NODE_ADDR="127.0.0.1:9748"
START_PORT=18000
START_DEVICE=0
LOG_DIR="log"
NNODES=1

mkdir -p $LOG_DIR

for (( i=0; i<$NNODES; i++ ))
do
  PORT=$((START_PORT + i))
  DEVICE=$((START_DEVICE + i))
  LOG_FILE="$LOG_DIR/node_$i.log"
  xllm \
    --model $MODEL_PATH \
    --devices="mlu:$DEVICE" \
    --port $PORT \
    --nnodes=$NNODES \
    --master_node_addr=$MASTER_NODE_ADDR \
    --block_size=16 \
    --node_rank=$i \ > $LOG_FILE 2>&1 &
done
```

For a single-device run, `<device-id>` usually starts from `0`. For larger deployments, keep the selected device ids aligned with `--node_rank`, `--nnodes`, and per-worker ports.

## Notes

- xLLM does not currently provide a public MLU image in the docs. Use an available MLU development image with the container startup command above.
- The MLU launch example uses `--block_size=16` in the current docs.
- Check the [Model Support List](/en/supported_models/) before choosing an MLU deployment target.
