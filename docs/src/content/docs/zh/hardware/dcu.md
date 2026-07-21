---
title: "海光 DCU"
description: "使用海光 DCU 后端在海光 DCU 硬件上运行 xLLM。"
sidebar:
  order: 5
---

在海光 DCU 硬件上部署 xLLM 时使用海光 DCU 后端。

## 镜像和容器启动命令

拉取海光 DCU 开发镜像：

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:xllm-dev-dcu-x86-20260617
```

启动容器：

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

## 服务启动命令

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

单卡部署时 `<device-id>` 通常从 `0` 开始。多 worker 部署中，需要让设备编号、`--node_rank`、`--nnodes` 和服务端口保持一致。

## 注意事项

- 当前文档在 [快速开始](/zh/getting_started/quick_start/) 中列出了海光 DCU 开发镜像。
- 海光 DCU 容器启动需要挂载 `/dev/kfd`、`/dev/dri`、`/dev/mkfd` 等设备；上面的命令已包含这些挂载。
- 选择海光 DCU 部署目标前，先在 [模型支持列表](/zh/supported_models/) 中确认模型覆盖情况。
