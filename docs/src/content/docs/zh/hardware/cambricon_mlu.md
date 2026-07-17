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

单卡部署时 `<device-id>` 通常从 `0` 开始。更大规模部署中，需要让设备编号、`--node_rank`、`--nnodes` 和每个 worker 的端口保持一致。

## Context Parallel

MLU Context Parallel（CP）通过 `--cp_size` 配置。

在 MLU 上，`cp_size` 必须等于全局 world size，即 `nnodes` 乘以每个进程
使用的设备数量。当前 MLU model-side CP 还有以下限制：

- 支持的模型类型为 `deepseek_v32` 和 `glm_moe_dsa`（GLM-5）。
- 仅支持文本生成模型的 `generate` 任务。
- `dp_size` 必须为 `1`，`kv_split_size` 必须为 `1`。
- `ep_size` 必须为 `1` 或全局 world size。
- 仅 `DEFAULT` 和 `PREFILL` 实例角色支持 CP。
- MLU CP 不支持基于 MTP 或 Eagle3 的投机解码，但支持 Suffix 投机解码。

在 Prefill/Decode 分离部署中，Prefill 实例配置 `cp_size=N`，Decode 实例
配置 `cp_size=1`。Decode 实例不参与 MLU model-side CP。

## 注意事项

- 当前文档不提供公开 MLU 镜像，需要使用已有 MLU 开发镜像，并配合上面的容器启动命令。
- 当前 MLU 启动示例使用 `--block_size=16`。
- 选择 MLU 部署目标前，先在 [模型支持列表](/zh/supported_models/) 中确认模型覆盖情况。
