---
title: "OpenAI 接口：Embedding"
---

`/v1/embeddings` 接口返回文本的向量表示，使用前需要启动 Embedding 服务。仅改变客户端请求路径，不会把生成服务切换为池化服务。通用客户端约定请参阅 [OpenAI 兼容接口](/zh/getting_started/openai_api/)。

## 启动 Embedding 服务

对于支持池化输出的模型，设置 `--task=embed`。以下示例演示单张昇腾 NPU 上的 Qwen3 池化路径。如果需要用于检索的高质量向量，应选择经过 Embedding 训练的权重；仅对生成模型做池化并不保证检索质量。环境准备请参阅[启动说明](/zh/getting_started/launch_xllm/)。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_BASE_PORT=43432

xllm --model=/path/to/Qwen3-8B \
  --model_id=Qwen3-8B \
  --npu_kernel_backend=ATB \
  --enable_graph=false \
  --task=embed \
  --host=127.0.0.1 \
  --port=18002 \
  --master_node_addr=127.0.0.1:9748 \
  --nnodes=1 \
  --node_rank=0 \
  --block_size=128 \
  --max_memory_utilization=0.86 \
  --enable_prefix_cache=false \
  --enable_chunked_prefill=false \
  --enable_schedule_overlap=false \
  --enable_shm=false
```

请将此示例作为独立部署运行。并行运行多个服务时，需要分别分配设备、master 地址和通信端口。模型加载完成后，可通过 `/v1/models` 确认模型 ID。

该部署显式选择 ATB 实现。Embedding 请求只执行 prefill 和 pooling，没有自回归 decode 阶段。

## 使用 cURL

```bash
export HOST=127.0.0.1
export PORT=18002
export MODEL=Qwen3-8B

curl -sS "http://${HOST}:${PORT}/v1/embeddings" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"input\": [\"What is continuous batching?\", \"Batching improves throughput.\"],
    \"encoding_format\": \"float\"
  }"
```

## 使用 OpenAI Python 客户端

通过 `python -m pip install --upgrade openai` 安装 SDK，再运行：

```python
import os
from openai import OpenAI

host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "18002")
model = os.environ.get("MODEL", "Qwen3-8B")
client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")
response = client.embeddings.create(
    model=model,
    input=["What is continuous batching?", "Batching improves throughput."],
    encoding_format="float",
)
for item in response.data:
    print(item.index, len(item.embedding), item.embedding[:5])
print(response.usage)
```

`data[i].index` 对应输入位置，`data[i].embedding` 是该输入的向量，`usage` 给出输入 token 总量。编码单条文本时，将 `input` 设为字符串即可。

## Token ID 与编码格式

`input` 也支持单条非空 token ID 列表，以及批量的非空 token ID 列表。请使用所服务模型的 tokenizer，并通过 `input` 字段传入这些 ID。

`encoding_format="float"` 返回 JSON 数字数组；`encoding_format="base64"` 返回按小端 float32 编码的 Base64 数据。沿用前面的 Python 客户端，可按下面的方式解码：

```python
import base64
import struct

response = client.embeddings.create(
    model=model,
    input="Batching improves throughput.",
    encoding_format="base64",
)
raw = base64.b64decode(response.data[0].embedding)
embedding = struct.unpack(f"<{len(raw) // 4}f", raw)
print(len(embedding), embedding[:5])
```

向量维度由模型决定，xLLM 对 `dimensions` 参数返回 HTTP 400。该接口一次性返回完整结果，不提供流式输出。本页介绍文本 Embedding；xLLM 模型专用的多模态 Embedding 请求使用不同的输入格式。
