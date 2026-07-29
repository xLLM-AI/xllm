---
title: "NVIDIA GPU"
description: "使用 CUDA 后端在 NVIDIA GPU 上运行 xLLM。"
sidebar:
  order: 2
---

在 NVIDIA GPU 上部署 xLLM 时使用 CUDA 后端。

## 镜像和容器启动命令

拉取 CUDA 开发镜像：

```bash
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-cuda-x86
```

启动容器：

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

## 服务启动命令

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

单卡部署时 `<device-id>` 通常从 `0` 开始。多卡或多机场景中，需要让设备编号、`--node_rank`、`--nnodes` 和服务端口与部署拓扑保持一致。

## 注意事项

- CUDA 开发镜像和 Dockerfile 入口维护在 [快速开始](/zh/getting_started/quick_start/) 中。
- 当前 CUDA 启动示例使用 `--block_size=32` 和 `--max_memory_utilization=0.8`；实际部署时按模型规模和 GPU 显存调整。
- CUDA timeline 采集入口见 [在线性能采集](/zh/dev_guide/online_profiling/)。
