---
title: "寒武纪 MLU"
description: "使用 MLU 后端在寒武纪设备上运行 xLLM。"
sidebar:
  order: 4
---

在寒武纪设备上部署 xLLM 时使用 MLU 后端。

## 镜像和容器启动命令

当前文档不提供公开 MLU 镜像。如果您已经拥有了相应的开发镜像，可以根据下面的命令启动容器：

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

## 服务启动命令

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

单卡部署时 `<device-id>` 通常从 `0` 开始。更大规模部署中，需要让设备编号、`--node_rank`、`--nnodes` 和每个 worker 的端口保持一致。

## 注意事项

- 当前文档不提供公开 MLU 镜像，需要使用已有 MLU 开发镜像，并配合上面的容器启动命令。
- 当前 MLU 启动示例使用 `--block_size=16`。
- 选择 MLU 部署目标前，先在 [模型支持列表](/zh/supported_models/) 中确认模型覆盖情况。
