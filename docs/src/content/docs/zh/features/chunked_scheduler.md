---
title: "ChunkedPrefill调度器"
sidebar:
  order: 21
---
## 功能介绍
xLLM支持chunked prefill调度策略。Chunked prefill是一种优化大语言模型推理的技术，将长prompt分割成多个较小的chunk进行分批处理，而不是一次性处理整个prompt。
这种方法可以有效降低显存峰值使用量，提高Device利用率，并且能够更好地与decode阶段的请求进行调度和混合处理。

## 使用方式
上述策略已在xLLM实现，并向外暴露gflag参数，控制功能的开关。

- 开启chunked prefill，并设置chunked_size，如果不手动设置chunked size，则默认等于max_tokens_per_batch。
```bash
--enable_chunked_prefill=true
--max_tokens_per_chunk_for_prefill=20480 # optional
```

## 性能效果
开启chunked_prefill之后，在Qwen3-8B模型上，限制TPOT 50ms，TTFT时延 **下降46%**。

