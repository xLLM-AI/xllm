---
title: "NVIDIA GPU"
description: "Run xLLM on NVIDIA GPUs with the CUDA backend."
sidebar:
  order: 2
---

Use the CUDA backend when running xLLM on NVIDIA GPUs.

## Image and Container Startup

Pull the CUDA development image:

```bash
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86
```

Start the container:

```bash
sudo docker run -it \
--privileged \
--shm-size '128gb' \
--ipc=host \
--net=host \
--pid=host \
--name=xllm-cuda \
-v $HOME:$HOME \
-w $HOME \
<docker_image_name> \
/bin/bash
```

## Server Startup Command

Set visible CUDA devices before launching the service:

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

For a single-device run, `<device-id>` usually starts from `0`. For multi-device or multi-node runs, keep the device ids, `--node_rank`, `--nnodes`, and service ports consistent with the launch topology.

## Notes

- The CUDA development image and Dockerfile references are maintained in [Quick Start](/en/getting_started/quick_start/).
- The default CUDA launch example uses `--block_size=32` and `--max_memory_utilization=0.8`; adjust these values according to the target model and available GPU memory.
- CUDA timeline collection is available through the profiling endpoints described in [Online Profiling](/en/dev_guide/online_profiling/).
