---
title: "启动xllm"
sidebar:
  order: 3
---


需要提前安装 xllm wheel 包或者使用 release 镜像：

```bash
# from source
python setup.py install
```

模型路径可以直接放在 `serve` 后，例如 `xllm serve /path/to/model`。原有的 `--model=/path/to/model` 和 `--model /path/to/model` 写法仍然支持；不要同时使用位置参数和 `--model` 指定模型。

单机部署时，将 `NNODES` 设置为所选设备的数量（一张卡对应一个 node）。以下脚本适用于单机单卡和单机多卡：

```bash
#!/bin/bash
set -e

MODEL_PATH="/path/to/model"
PORT=8010
MASTER_NODE_ADDR="127.0.0.1:9748"
NNODES=1

xllm serve "$MODEL_PATH" \
       --port=$PORT \
       --nnodes=$NNODES \
       --master_node_addr=$MASTER_NODE_ADDR \
       --block_size=128 \
       --max_memory_utilization=0.86 \
       --enable_prefix_cache=false \
       --enable_chunked_prefill=true \
       --enable_schedule_overlap=true \
       --enable_graph=true
```

多机部署时，需要添加 `--machine_rank` 参数，指定当前机器的编号。

## 验证并发送请求

示例使用 `8010` 端口。模型加载完成后，在另一个 shell 中检查 rank 0 的接口：

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

请求的 `model` 使用 `/v1/models` 返回的 ID。以 Qwen3-8B 示例为例：

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 64,
    "stream": false
  }'
```
