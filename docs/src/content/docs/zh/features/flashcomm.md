---
title: "FlashComm"
sidebar:
  order: 82
---

## 功能介绍

FlashComm 是 xLLM 在 NPU Tensor Parallel 推理场景下的 prefill 通信优化特性。它的目标是减少长输入 prefill 阶段中 row-parallel 线性层后的通信开销，并在合适场景下使用 Matmul + ReduceScatter 融合算子降低 kernel launch 和通信调度成本。

当前 FlashComm 主要包含两部分：

- **序列维度分片**：在 prefill 阶段将 token 序列按 TP rank 切分，让后续部分计算在本 rank 的 token shard 上执行。
- **MMRS 融合算子**：在 row-parallel 线性层中，将普通 `matmul + reduce_scatter` 替换为 torch_npu 的 `npu_mm_reduce_scatter_base`，即 Matmul + ReduceScatter 融合路径。

FlashComm 默认关闭。即使打开总开关，也只有满足运行条件时才会真正启用；不满足条件时会走原有执行路径。

## 设计说明

FlashComm 的执行流程如下：

1. 请求进入 prefill 阶段后，运行时根据 token 数、并行配置和开关构造 FlashComm 上下文。
2. 当上下文生效时，输入 hidden states 会按序列维度切分到不同 TP rank。
3. 在支持的 row-parallel 线性层中，优先尝试 MMRS 融合路径。
4. 如果 MMRS 不适用，例如 shape、dtype、bias 或通信上下文不满足要求，则回退到普通 matmul 和 reduce_scatter 路径。
5. 在需要完整 hidden states 的边界处，再通过 gather 恢复完整序列。

当前 MMRS 路径使用 torch_npu 提供的 `npu_mm_reduce_scatter_base`。xLLM 侧只保留薄封装，用于完成输入校验、HCCL group 获取、`comm_mode` 选择和日志记录，不重新实现 Matmul + ReduceScatter kernel。

## 适用场景

FlashComm 更适合以下场景：

- NPU 后端。
- 长输入 prefill，例如输入长度大于等于 8K tokens。
- TP 较大，当前默认建议 `TP >= 8`。
- `dp=1` 且 `cp=1`。
- prefill 占端到端时延比例较高，例如 8K/128、32K/1K 等长 prompt 场景。
- 使用 BF16/FP16 的非量化 row-parallel 线性层。

FlashComm 不适合或收益有限的场景：

- decode 阶段。FlashComm 只优化 prefill，不优化 decode，因此 TPOT 通常不会直接受益。
- 短输入，例如 2K 输入场景。通信占比不足时，切分、gather 和调度开销可能抵消收益。
- 高 decode 占比场景，例如长输出请求。整体 latency 可能主要由 decode 决定。
- `TP < 8`、`dp > 1` 或 `cp > 1` 的场景，当前默认不会启用。
- 量化 row-parallel 路径。当前 MMRS 只接入普通 BF16/FP16 matmul 路径。
- MoE 模型。当前 FC1 仅支持 Dense 模型；检测到 routed experts 时会自动关闭 FC1。

## 使用方式

FlashComm 和 MMRS 融合算子都默认关闭。推荐在长 prefill、TP=8 或更大 TP 的 NPU 服务中显式开启：

```bash
--enable_flashcomm1=true \
--enable_mmrs_fusion=true \
--flashcomm1_min_prefill_tokens=8192 \
--mmrs_comm_mode=aiv
```

推荐同时开启 Graph Mode，降低 Host 侧调度开销：

```bash
--enable_graph=true \
--enable_prefill_piecewise_graph=true
```

完整推荐配置示例：

```bash
--enable_graph=true \
--enable_prefill_piecewise_graph=true \
--enable_flashcomm1=true \
--enable_mmrs_fusion=true \
--flashcomm1_min_prefill_tokens=8192 \
--mmrs_comm_mode=aiv
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_flashcomm1` | `false` | FlashComm 总开关 |
| `enable_mmrs_fusion` | `false` | 是否启用 Matmul + ReduceScatter 融合算子 |
| `flashcomm1_min_prefill_tokens` | `8192` | prefill token 数达到该阈值后才允许启用 FlashComm |
| `mmrs_comm_mode` | `aiv` | torch_npu MMRS 通信模式，可选 `aiv`、`ai_cpu`、`none` |

通常建议保持 `mmrs_comm_mode=aiv`。如果某些 shape 在 AIV 路径出现 AICore 异常，可以临时切换为：

```bash
--mmrs_comm_mode=ai_cpu
```

使用 `mmrs_comm_mode=aiv` 时，需要加载包含 MMRS AIV 修复的 ops-transformers 9.1.0 或更高版本算子库。

## 最优配置建议

推荐从以下配置开始评估：

| 场景 | 建议 |
|------|------|
| 8K 输入 / 短输出 | 推荐开启 FlashComm 和 MMRS |
| 32K 输入 / 中短输出 | 推荐开启 FlashComm 和 MMRS |
| 2K 输入 / 长输出 | 不建议默认开启，收益通常不稳定 |
| TP=2 或 TP=4 | 不建议默认开启 |
| TP=8 | 当前最推荐评估的配置 |
| Chunked Prefill | 可以开启，但建议 chunk size 不小于 `flashcomm1_min_prefill_tokens`，否则单个 chunk 可能不会触发 FlashComm |

如果业务 workload 混合了短输入和长输入，建议保持默认关闭，并只在长输入服务、长上下文模型或独立部署的长 prompt workload 中开启。

## 性能与正确性注意事项

- FlashComm 只优化 prefill，因此观察收益时应重点关注 TTFT、prefill throughput 和 profiling 中 prefill 阶段的通信变化。
- TPOT、decode throughput 和长输出 latency 不一定改善；如果 decode 占比高，端到端 latency 可能看不到明显收益。
- 开启 MMRS 后应确认 profiling 中部分 row-parallel 后的 `allReduce` 或 `reduce_scatter` 开销被 Matmul + ReduceScatter 融合路径替代。
- 如果看到额外 gather、layout 转换或 Host 调度开销增加，可能会抵消 MMRS 的收益。
- 建议使用 warmup 后的多轮稳定请求评估，不要直接使用 profiling run 的绝对时延作为性能结论。

## 验证建议

上线或调参前，建议至少完成以下验证：

1. 对比 `enable_flashcomm1=false, enable_mmrs_fusion=false` 和 `enable_flashcomm1=true, enable_mmrs_fusion=true`。
2. 使用相同模型、相同 TP、相同输入输出长度和相同并发。
3. 记录 TTFT、TPOT、prompt throughput、decode throughput、request throughput 和 latency。
4. 对长输入场景额外采集 profiling，确认 MMRS 路径真正命中。
5. 做小规模数值一致性检查，确认开启和关闭 FlashComm 的输出一致。
