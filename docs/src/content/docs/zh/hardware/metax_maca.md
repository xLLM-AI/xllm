---
title: "沐曦 MACA"
description: "使用沐曦 MACA 后端在沐曦 MACA 硬件上运行 xLLM。"
sidebar:
  order: 6
---

在沐曦 MACA 硬件上部署 xLLM 时使用沐曦 MACA 后端。

## 镜像和容器启动命令

拉取沐曦 MACA 开发镜像：

```bash
docker pull pub-registry1.metax-tech.com/dev-m01421/xllm-maca3.7.1.9:v1
```

启动容器：

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

## 服务启动命令

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

单卡部署时 `<device-id>` 通常从 `0` 开始。多 worker 部署中，需要让设备编号、`--node_rank`、`--nnodes` 和服务端口保持一致。

## 注意事项

- 当前文档在 [快速开始](/zh/getting_started/quick_start/) 中列出了沐曦 MACA 开发镜像。
- 沐曦MACA 容器启动需要挂载 `/dev/mxcd`、`/dev/dri`、`/dev/infiniband` 等设备；上面的命令已包含这些挂载。
- 在MetaX MACA容器中编译XLLM命令: python setup.py build --device maca
