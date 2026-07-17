---
title: "Ascend NPU"
description: "Run xLLM on Ascend NPUs with the NPU backend."
sidebar:
  order: 3
---

Use the NPU backend when running xLLM on Ascend hardware.

## Image and Container Startup

Pull a pre-built Ascend development image:

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-20260306
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-20260306
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260306
```

Start the container:

```bash
docker run -it \
--ipc=host \
-u 0 \
--name xllm-npu \
--privileged \
--network=host \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
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

For distributed NPU serving, set `--communication_backend="hccl"`, keep `--master_node_addr`, `--node_rank`, and `--nnodes` consistent across all workers, and prepare the rank table when the deployment requires one.

## Notes

- Pre-built A2 and A3 development images are listed in [Quick Start](/en/getting_started/quick_start/).
- NPU launch examples usually enable HCCL communication and tune `--block_size`, `--max_memory_utilization`, chunked prefill, and schedule overlap for the target workload.
- Ascend kernel implementation details are documented separately in the TileLang kernel development guide.
