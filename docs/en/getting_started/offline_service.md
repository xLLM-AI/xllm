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

## Embedding

Generate embedding example: [:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py)

## VLM

VLM inference example: [:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py)
