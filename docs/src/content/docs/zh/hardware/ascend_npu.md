---
title: "昇腾 NPU"
description: "使用 NPU 后端在昇腾硬件上运行 xLLM。"
sidebar:
  order: 3
---

在昇腾硬件上部署 xLLM 时使用 NPU 后端。

## 镜像和容器启动命令

拉取预构建昇腾开发镜像：

```bash
# A2 x86
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-x86-20260306
# A2 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a2-arm-20260306
# A3 arm
docker pull quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-20260306
```

启动容器：

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

## 服务启动命令

```bash
#!/bin/bash
set -e

rm -rf core.*

source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432  # HCCL 通信基础端口

MODEL_PATH="/path/to/model/Qwen3-8B"               # 模型路径
MASTER_NODE_ADDR="127.0.0.1:9748"                  # Master 节点地址（需全局一致）
START_PORT=18000                                   # 服务起始端口
LOG_DIR="log"                                      # 日志目录
NNODES=1                                           # 节点数（当前脚本启动 1 个进程）

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

分布式 NPU 服务通常需要设置 `--communication_backend="hccl"`，并保证所有 worker 的 `--master_node_addr`、`--node_rank`、`--nnodes` 一致；需要 rank table 的部署还应提前准备对应配置。

## 注意事项

- A2 和 A3 开发镜像维护在 [快速开始](/zh/getting_started/quick_start/) 中。
- NPU 启动示例通常会启用 HCCL 通信，并按业务负载调整 `--block_size`、`--max_memory_utilization`、chunked prefill 和 schedule overlap 等参数。
- 昇腾 kernel 适配和开发细节见 TileLang kernel 开发指南。
