---
title: "Zero Evict调度器"
sidebar:
  order: 22
---
## 功能介绍
xLLM支持zero_evict调度策略。zero_evict调度策略是一种尽可能减少请求淘汰率的调度算法，可以减少淘汰请求的prefill计算，减少TPOT。
这种调度算法通过模拟轮次，检测请求是否调度可以被调度且不导致其它请求被淘汰。

## 使用方式
上述策略已在xLLM实现，并向外暴露gflag参数，控制功能的开关。

- 开启zero_evict策略，并设置max_decode_token_per_sequence。
```
--use_zero_evict=true
--max_decode_token_per_sequence=256
```

## 性能效果
开启zero_evict之后，在Qwen3-8B模型上，限制E2E时延，TPOT时延 **下降27%**。

