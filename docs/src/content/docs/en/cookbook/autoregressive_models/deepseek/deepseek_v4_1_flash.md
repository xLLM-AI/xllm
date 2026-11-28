---
title: "DeepSeek-V4.1-Flash-1017"
description: "Serve the W8A8 checkpoint on a single Ascend A3 node with fixed-step DSpark and Python ACL Graph."
---

This recipe runs DeepSeek-V4.1-Flash-1017 W8A8 on one Ascend A3 host with 16 visible NPU devices (64 GB each). It uses DP4, TP4 and EP16, with asynchronous scheduling and five draft tokens per DSpark step.

## 1. Prepare the Ascend environment

Follow the [Ascend NPU setup guide](/en/hardware/ascend_npu/) to start an A3 ARM64 development container with CANN 9, all 16 devices and the checkpoint directory mounted. Run `npu-smi info` inside the container to check device visibility. With Engram host offload enabled, the local DP ranks share the INT8 tables through `/dev/shm`. The reference deployment provided a 256 GiB shared-memory mount. Check the available space with `df -h /dev/shm`.

Install an NPU wheel that includes V4.1-Flash-1017 support. For a source installation, run the following commands from the root of a checkout that includes the model.

```bash
git submodule update --init --recursive
python setup.py install --device npu
```

## 2. Prepare the checkpoint

Download the complete W8A8 checkpoint, including its tokenizer and embedded DSpark parameters. The target and draft both load from the same directory. [ModelScope weights](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8)

The reference deployment used a 32,768-token context limit. The commands below create a separate serving directory with that limit and link the original weight shards. If the checkpoint lacks `engram_hash_state.bin`, xLLM’s helper generates it in the serving directory from the matching tokenizer and config. Keep both directories available while the service runs.

```bash
export CHECKPOINT_DIR="/path/to/DeepSeek-V4.1-Flash-1017-w8a8"
export MODEL_DIR="$(mktemp -d)"

python - <<'PYTHON'
import json
import os
from pathlib import Path

checkpoint = Path(os.environ["CHECKPOINT_DIR"]).resolve(strict=True)
model = Path(os.environ["MODEL_DIR"])
config = json.loads((checkpoint / "config.json").read_text())
config["text_config"]["max_position_embeddings"] = 32768
(model / "config.json").write_text(json.dumps(config))

for entry in checkpoint.iterdir():
    if entry.name != "config.json" and not entry.name.startswith("."):
        (model / entry.name).symlink_to(entry)

if not (model / "engram_hash_state.bin").is_file():
    from xllm import xllm_export
    from xllm.python import initialize_runtime

    initialize_runtime("deepseek_v41")
    from xllm.python.models.deepseek_v41_engram import build_engram_hash_state

    build_engram_hash_state(str(model))
PYTHON
```

## 3. Start the service

Run the following in the same shell. The launcher starts 16 local ranks and assigns ports 8000–8015. DP4 gives each group four tensor-parallel ranks. EP spans all 16 ranks.

```bash
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=512
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

xllm serve "$MODEL_DIR" \
  --model_id=deepseek-v41-flash \
  --host=127.0.0.1 --port=8000 --nnodes=16 \
  --master_node_addr=127.0.0.1:19888 \
  --worker_memory_master_node_addr=127.0.0.1:19889 \
  --log-dir=./logs/deepseek-v41-flash \
  --dp_size=4 --ep_size=16 --cp_size=1 \
  --model_impl=python --npu_kernel_backend=TORCH --communication_backend=hccl \
  --enable_graph=true --python_graph_backend=aclgraph --enable_schedule_overlap=true \
  --speculative_algorithm=dspark --draft_model="$MODEL_DIR" --num_speculative_tokens=5 \
  --enable_fused_mc2=1 --enable_engram_host_offload=true \
  --kv_cache_dtype=auto --indexer_cache_dtype=int8 --block_size=128 \
  --max_memory_utilization=0.90 --max_cache_size=2147483648 \
  --max_tokens_per_batch=16384 --max_tokens_per_chunk_for_prefill=4096 \
  --max_seqs_per_batch=64 --max_concurrent_requests=32 \
  --enable_chunked_prefill=true --enable_mix_batch=false --enable_prefill_piecewise_graph=false \
  --enable_prefix_cache=true --enable_swa_bounded_replay=true
```

The 2 GiB KV cache cap and the 32K context limit are separate settings. The number of concurrent requests that can reach the context limit depends on their lengths and shared prefix hits. If the host has several network interfaces, set `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to the interface used for communication before launching.

## 4. Verify the service

From another shell in the container, check the rank-0 endpoint and send a chat request. The model list should include `deepseek-v41-flash`, and the chat response should contain a `choices` array. Each rank writes its startup log to `./logs/deepseek-v41-flash/node_<rank>.log`.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v41-flash",
    "messages": [{"role": "user", "content": "What is 12 times 8?"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```
