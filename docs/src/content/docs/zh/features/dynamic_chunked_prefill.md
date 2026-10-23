---
title: 动态 Chunked Prefill
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

## 范围

动态 chunked prefill 根据启动 profiling 的结果和请求已缓存的前缀长度，
调整每轮 prefill 的 token 数量。默认关闭。

本实现迁移了 vLLM-Ascend Dynamic Chunked Pipeline Parallel 的启动测量、
二次延迟模型和动态 chunk 选择，并接入 xLLM 的 C++ 调度器。
该选项不会自动启用 PP；独立的[实验性 PP 路径](../pipeline_parallel)
新增 GLM 分层执行及连续 chunk 重叠，支持单条长请求的 DCPP。

参考实现：

- [vLLM-Ascend predictor](https://github.com/vllm-project/vllm-ascend/blob/8a2c3182c463a7391b4c9d21be3676fc12c3d14c/vllm_ascend/core/profiling_chunk_predictor.py)

## 使用

在原有 LLM 启动命令上添加：

```bash
--enable_chunked_prefill=true \
--enable_dynamic_chunking=true \
--max_tokens_per_chunk_for_prefill=4096 \
--dynamic_chunk_min_tokens=256 \
--dynamic_chunk_smooth_factor=1.0 \
--dynamic_chunk_profile_samples=16
```

也可以在配置 JSON 中设置同名字段。`--help` 会列出这些选项，配置导出会
保留非默认值。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `enable_dynamic_chunking` | `false` | 启用启动 profiling 和动态 chunk |
| `dynamic_chunk_min_tokens` | `256` | 对齐及资源约束前的最小预测长度，必须大于 0 且不超过基准 chunk |
| `dynamic_chunk_smooth_factor` | `1.0` | `(0, 1]`；越小越接近固定 chunk |
| `dynamic_chunk_profile_samples` | `16` | 测量长度数量，范围 `[8, 64]` |

基准 chunk 取 `max_tokens_per_chunk_for_prefill` 和 `max_tokens_per_batch`
的较小值。KV/CP 对齐粒度为 `lcm(64, block_size * kv_split_size_effective)`。
至少需要 8 个不同的对齐长度才能拟合；不足时记录警告并使用固定 chunk。
启动 profiling 需要容纳一个基准 chunk 的 KV 和模型上下文空间。

## 算法和调度

1. 复用 `ProfileManager::run_request` 执行真实的 synthetic prefill。
   每个长度先 warmup，再运行 3 次取延迟中位数。请求的 KV 会释放且不写入
   prefix cache。计时沿用 xLLM 的同步结果返回路径，包含执行及调度通信开销。
2. 用非负最小二乘拟合 `f(L) = a*L² + b*L + c`，长度先归一化以改善长序列
   下的数值条件。允许 `a=0` 的线性模型；数据退化、非有限值或无有效长度相关
   成本时使用固定 chunk，不捕获或隐藏模型执行错误。
3. 令目标增量延迟 `T = f(base) - f(0)`，对缓存历史 `H` 求解
   `a*x² + (2*a*H+b)*x = T`。使用稳定的正根公式，再做平滑、最小长度和对齐处理。
4. `PrefillFirstPolicy`、`DecodeFirstPolicy` 和 `UnifiedPolicy` 使用预测上限。
   prefix cache 匹配后按有效缓存长度预测；实际分配仍受全局/DP token 预算、
   KV 空间和既有 SLO 策略限制。混合调度的预算重分配不能突破动态上限。
   短尾块可以不对齐；无法容纳一个对齐块的剩余预算留给下一轮。

这个上限针对每个请求的 chunk，并不保证多请求整个 batch 的耗时固定。
现有延迟感知调度器仍负责其 batch 延迟预算。

## 支持边界

- 接入 `ContinuousScheduler` 及复用该调度路径的 `DisaggPDScheduler`。
  Decode-only 实例跳过动态 prefill profiling。
- `ZeroEvictionScheduler` 不支持此配置；会明确拒绝。
- 含线性 attention 层且启用 prefix cache 时，固定 chunk 边界用于状态检查点，
  暂不支持动态边界，会明确拒绝。
- 本次只在启动时拟合；未迁移 vLLM-Ascend 的在线 history-aware 再校准。
- 不保证长上下文外推的预测精度或吞吐提升。固定/动态模式需要同设备、同模型、
  同负载的 A/B 对比；没有 PP 时不能直接引用原 CPP 的流水线收益。

## 测试

相关测试：`dynamic_chunk_predictor_test`、`DynamicChunkConfigTest`，以及
`scheduler_test` 中的动态 chunk 调度回归测试。

后续：

1. 在空闲 NPU 上验证长输入、并发、prefix cache、CP 和 decode graph 的组合，
   比较输出、TTFT、吞吐及实际 chunk 序列。
2. 增加带历史长度的测量和有界在线再校准，明确样本归属和计时同步成本。
3. 在 PP 执行链上增加逐 stage profiling 和耗时均衡，
   验证真实的流水线空闲时间改善。

## PP 调度

启用 PP 时，一轮最多为每条请求规划 `pp_size` 个 chunk，预测历史随每个 chunk
递增，合计不超过 batch 预算；engine 消费这份计划。固定 chunk PP 也复用同一
调度和执行路径。参考 vLLM-Ascend 的启动预测与连续 prefill 在途执行机制，
目前未迁移在线校准或独立的 stage 耗时拟合。当前 checkout 的全模型验证应以
目标 NPU 实测为准，不沿用参考分支的历史性能结论。
