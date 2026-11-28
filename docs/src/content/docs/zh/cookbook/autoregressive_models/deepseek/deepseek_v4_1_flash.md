---
title: "DeepSeek-V4.1-Flash-1017"
description: "在单机 Ascend A3 上使用固定步数 DSpark 和 Python ACL Graph 部署 W8A8 权重。"
---

本文在一台 Ascend A3 服务器上部署 DeepSeek-V4.1-Flash-1017 W8A8，使用 16 个可见 NPU 设备，每个设备 64 GB。并行配置为 DP4、TP4、EP16，开启异步调度，每轮 DSpark 固定提出 5 个候选 token。

## 1. 准备 Ascend 环境

参照 [Ascend NPU 环境配置](/zh/hardware/ascend_npu/) 启动带 CANN 9 的 A3 ARM64 开发容器，挂载全部 16 个设备和权重目录。在容器内运行 `npu-smi info` 检查设备。开启 Engram 主存卸载后，同机 DP rank 通过 `/dev/shm` 共享 INT8 权重表。参考部署为共享内存提供了 256 GiB，可用 `df -h /dev/shm` 查看剩余空间。

安装包含 V4.1-Flash-1017 支持的 NPU wheel。如果从源码安装，请在包含该模型支持的仓库根目录运行以下命令。

```bash
git submodule update --init --recursive
python setup.py install --device npu
```

## 2. 准备模型权重

下载完整的 W8A8 权重，包含配套的 tokenizer 和内置 DSpark 参数。Target 和 draft 从同一个目录加载。[ModelScope 权重下载](https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8)

参考部署的上下文上限为 32,768 token。下面的命令新建服务目录，在其中设置该上限，并通过符号链接复用原始权重分片。如果权重中缺少 `engram_hash_state.bin`，脚本会用配套的 tokenizer 和配置在服务目录生成该文件。服务运行期间需保留这两个目录。

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

## 3. 启动服务

在同一个终端执行以下命令。启动器在本机拉起 16 个 rank，使用 8000–8015 端口。DP4 下每组有 4 个张量并行 rank，EP 跨全部 16 个 rank。

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

示例将 KV cache 上限设为 2 GiB，上下文长度上限单独设为 32K。能同时达到该长度的请求数取决于请求长度和共享前缀命中情况。如果主机有多块网卡，启动前将 `HCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 设为通信网卡名称。

## 4. 验证服务

在容器内另开终端，检查 rank 0 的服务并发送一条请求。模型列表应包含 `deepseek-v41-flash`，聊天接口应返回含有 `choices` 数组的应答。各 rank 的启动日志位于 `./logs/deepseek-v41-flash/node_<rank>.log`。

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
