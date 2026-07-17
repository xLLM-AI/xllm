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

For a single-device run, `<device-id>` usually starts from `0`. For larger deployments, keep the selected device ids aligned with `--node_rank`, `--nnodes`, and per-worker ports.

## Context Parallelism

MLU Context Parallel (CP) is configured with `--cp_size`.

For MLU, the configured `cp_size` must equal the global world size, calculated
as `nnodes` multiplied by the number of devices used by each process. Current
MLU model-side CP has the following constraints:

- Supported model types are `deepseek_v32` and `glm_moe_dsa` (GLM-5).
- Only the `generate` task of text-generation models is supported.
- `dp_size` must be `1`, and `kv_split_size` must be `1`.
- `ep_size` must be either `1` or the global world size.
- Only the `DEFAULT` and `PREFILL` instance roles support CP.
- MTP- and Eagle3-based speculative decoding are not supported with MLU CP.
  Suffix speculative decoding is supported.

In a disaggregated Prefill/Decode deployment, configure `cp_size=N` on the
Prefill instance and `cp_size=1` on the Decode instance. The Decode instance
does not participate in MLU model-side CP.

## Notes

- xLLM does not currently provide a public MLU image in the docs. Use an available MLU development image with the container startup command above.
- The MLU launch example uses `--block_size=16` in the current docs.
- Check the [Model Support List](/en/supported_models/) before choosing an MLU deployment target.
