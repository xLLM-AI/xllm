# Offline Inference

To facilitate users in quickly using xLLM for offline inference, we provide Python script examples for launching offline inference.

## LLM

LLM inference example: [:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate.py)

LLM Beam Search example: [:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_beam_search.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_beam_search.py)

Use `BeamSearchParams` with `beam_width` greater than `1`, then call `llm.beam_search(...)`:

```python
from xllm import BeamSearchParams, LLM

llm = LLM(model="/path/models/Qwen2-7B-Instruct", devices="npu:0")
params = BeamSearchParams(
    beam_width=2,
    top_logprobs=4,
    max_tokens=20,
)

outputs = llm.beam_search(
    [{"prompt": "Hello, my name is "}],
    params=params,
)
print(outputs[0].sequences[0].text)

llm.finish()
```

For LLM Beam Search, use `beam_width` as the switch. `top_logprobs` controls the top-k candidate count used for beam expansion at each decode step. If `top_logprobs` is left at its default value, xLLM uses `beam_width` as the top logprob count. Set `top_logprobs` to a value greater than `beam_width` when you want each beam to consider more candidate tokens. This beam-search top-k is different from the sampling cutoff parameter `top_k`. `best_of` is not the Beam Search switch, and this offline LLM guide does not use `num_return_sequences` to control the returned beams.

## Multi-Machine Offline Inference

When a single machine does not have enough devices to load the model, you can use multi-machine offline inference to distribute the model across multiple machines. All machines run the same script and are differentiated by the `nnodes`, `node_rank`, and `master_node_addr` parameters:

- The machine with `node_rank=0` is the **driver**, responsible for submitting inference requests and collecting results
- Machines with `node_rank>0` are **assistants**, which enter a wait loop after initialization and are scheduled by the driver

### Parameters

| Parameter | Description |
|-----------|-------------|
| `nnodes` | Total number of machines, defaults to 1 (single machine) |
| `node_rank` | Current machine index, range `[0, nnodes)`, driver is 0 |
| `master_node_addr` | Communication address of the driver machine in `host:port` format; must be reachable from all machines |
| `rank_tablefile` | Path to the HCCL collective communication topology file (required for multi-machine) |

### Example

Using 2 machines with 16 devices total, assuming the driver IP is `11.87.191.99`:

**Machine 0 (driver):**

```bash
python examples/generate.py \
    --model='/path/models/GLM-5-final-w8a8' \
    --devices='npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7' \
    --max_seqs_per_batch=300 \
    --max_memory_utilization=0.9 \
    --nnodes=2 \
    --node_rank=0 \
    --rank_tablefile=hccl_tools/hccl_2s_16p.json \
    --master_node_addr="11.87.191.99:19888"
```

**Machine 1 (assistant):**

```bash
python examples/generate.py \
    --model='/path/models/GLM-5-final-w8a8' \
    --devices='npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7' \
    --max_seqs_per_batch=300 \
    --max_memory_utilization=0.9 \
    --nnodes=2 \
    --node_rank=1 \
    --rank_tablefile=hccl_tools/hccl_2s_16p.json \
    --master_node_addr="11.87.191.99:19888"
```

### Notes

1. All machines must use the same model path, `devices` configuration, and `nnodes` value
2. `master_node_addr` must be the actual reachable IP of the driver; loopback addresses (`127.0.0.1`, `localhost`, etc.) are not allowed
3. `rank_tablefile` must be pre-generated via `hccl_tools` to describe the multi-machine collective communication topology
4. Assistant machines only need to construct the `LLM` object — there is no need to call `generate()` or define prompts; during `LLM` initialization, when `node_rank>0` is detected, the process automatically enters a wait loop and is shut down remotely when the driver calls `llm.finish()`
5. Machines can be started in any order; the framework waits internally for all machines to be ready

## Embedding

Generate embedding example: [:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py)

## VLM

VLM inference example: [:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py)
