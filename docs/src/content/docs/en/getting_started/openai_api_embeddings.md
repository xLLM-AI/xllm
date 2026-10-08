---
title: "OpenAI APIs - Embeddings"
---

The `/v1/embeddings` endpoint returns vectors for text inputs. Start an embedding service before using it. A generation service does not become a pooling service merely because the client calls this endpoint. See [OpenAI-Compatible APIs](/en/getting_started/openai_api/) for the shared client conventions.

## Launch an embedding server

For a supported model with pooling output, set `--task=embed`. The example below demonstrates the Qwen3 pooling path on one Ascend NPU. Use weights trained for embeddings when you need retrieval-quality vectors; pooling a generation model alone does not guarantee that quality. Prepare the environment following the [launch guide](/en/getting_started/launch_xllm/).

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

Run this as a separate deployment. Concurrent servers need separate devices, master addresses, and communication ports. After model loading, query `/v1/models` to confirm the model ID.

This deployment explicitly selects the ATB implementation. Embedding requests only run prefill and pooling; they do not have an autoregressive decode phase.

## Using cURL

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

## Using the OpenAI Python client

Install the SDK with `python -m pip install --upgrade openai`, then run:

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

`data[i].index` corresponds to the input position, and `data[i].embedding` contains its vector. `usage` reports the total input-token count. To encode one text, pass a string instead of a list.

## Token IDs and encoding formats

`input` also accepts a non-empty list of token IDs for one input, or a list of non-empty token-ID lists for a batch. Use the served model's tokenizer and pass these IDs in `input`.

`encoding_format="float"` returns a JSON array of numbers. `encoding_format="base64"` returns Base64-encoded little-endian float32 data. Reuse the Python client above to decode it:

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

The vector length is determined by the model. xLLM rejects the `dimensions` parameter with HTTP 400. Embeddings are returned as a complete response; there is no streaming mode for this endpoint. This guide covers text embeddings; xLLM's model-specific multimodal embedding payloads are separate from this input format.
