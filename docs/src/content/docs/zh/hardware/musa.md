---
title: "摩尔线程 MUSA"
description: "在摩尔线程 MUSA GPU 上使用 xLLM 进行大模型推理部署。"
sidebar:
  order: 7
---

在摩尔线程（Mthreads）MUSA GPU 上部署 xLLM 时使用 MUSA 设备后端；通过 `MUSA_VISIBLE_DEVICES` 选择可见的设备卡。

## 镜像和容器启动命令

拉取摩尔线程 xLLM 镜像：

```bash
docker pull registry.mthreads.com/presale/devtech/xllm:0710
```

启动容器：

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

按实际 GPU 数量增加 `--device=/dev/mtgpuN`；选择物理卡使用 `export MUSA_VISIBLE_DEVICES=0`。

## 服务启动命令

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

单卡部署时逻辑设备号通常从 `0` 开始。多卡或多机场景中，需要让设备编号、`--node_rank`、`--nnodes` 和服务端口与部署拓扑保持一致。

## 注意事项

- 镜像与容器启动说明亦见 [快速开始](/zh/getting_started/quick_start/)。
- 启动参数与多机部署见 [启动 xllm](/zh/getting_started/launch_xllm/)。
