# 离线推理

为了方便用户快速使用xLLM进行离线推理，我们提供了启动离线推理的python脚本例子

## LLM

LLM推理示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate.py)

LLM Beam Search 示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_beam_search.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_beam_search.py)

使用 `BeamSearchParams` 设置大于 `1` 的 `beam_width`，然后调用 `llm.beam_search(...)`：

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

LLM Beam Search 使用 `beam_width` 作为开启参数。`top_logprobs` 控制每个解码步用于 beam 扩展的 top-k 候选数量。如果 `top_logprobs` 保持默认值，xLLM 会使用 `beam_width` 作为 top logprob 数量。如果希望每个 beam 考虑更多候选 token，可以将 `top_logprobs` 设置为大于 `beam_width` 的值。这里的 beam-search top-k 不同于采样截断参数 `top_k`。`best_of` 不是 Beam Search 开关，本文档也不使用 `num_return_sequences` 来控制 LLM 返回的 beam 数。

## Embedding

生成Embedding示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_embedding.py)

## VLM

VLM推理示例：[:simple-github: https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py](https://github.com/jd-opensource/xllm/blob/main/examples/generate_vlm.py)
