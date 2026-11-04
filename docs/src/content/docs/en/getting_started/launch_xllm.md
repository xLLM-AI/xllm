---
title: "Launch xllm"
sidebar:
  order: 3
---


Install the xllm wheel package in advance, or use a release image:

```bash
# from source
python setup.py install
```

Place the model path directly after `serve`, for example `xllm serve /path/to/model`. The existing `--model=/path/to/model` and `--model /path/to/model` forms remain supported; specify the model using either the positional argument or `--model`.

For single-machine deployments, set `NNODES` to the number of selected devices (one device per node). The following script supports both single-device and multi-device launches:

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

When deploying on multiple machines, you need to add the `--machine_rank` parameter to specify the rank of the current machine.

## Verify and send a request

The example uses port `8010`. After model loading, check the rank-0 endpoints from another shell:

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

Use the ID returned by `/v1/models` as the request's `model`. For a Qwen3-8B deployment:

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
