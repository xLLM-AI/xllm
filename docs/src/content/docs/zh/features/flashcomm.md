---
title: "FlashComm"
sidebar:
  order: 82
---

## 功能介绍

FlashComm 是 xLLM 在 NPU Tensor Parallel 推理场景下的 prefill 通信优化特性。它的目标是减少长输入 prefill 阶段中 row-parallel 线性层后的通信开销，并在支持的场景下使用 Matmul + ReduceScatter 融合算子降低 kernel launch 和通信调度成本。

FlashComm 包含两层能力：

- **序列维度分片（核心）**：在 prefill 阶段将 token 序列按 TP rank 切分，让后续部分计算在本 rank 的 token shard 上执行，并把 row-parallel 层后的 `all_reduce` 替换为 `reduce_scatter`，在需要完整序列的边界处再 gather 还原。这一层对所有 dtype（BF16 与各类量化）都生效，是收益的主要来源。
- **MMRS 融合算子（增量）**：在支持的 row-parallel 线性层中，将 `matmul + reduce_scatter` 替换为 torch_npu 的 `npu_mm_reduce_scatter_base`，进一步降低 kernel launch 和调度成本。当前融合内核覆盖 **BF16** 和 **w8a8_dynamic（int8 动态量化）** 两条路径。

FlashComm 默认关闭，由 `--enable_flashcomm1` 总开关控制。开关打开后，只有满足运行条件（prefill 阶段、token 数达标、`cp=1`）时才会真正启用；不满足条件时走原有执行路径。

## 设计说明

FlashComm 的执行流程如下：

1. 请求进入 prefill 阶段后，运行时根据 token 数、并行配置和开关构造 FlashComm 上下文。
2. 当上下文生效时，输入 hidden states 会按序列维度切分到不同 TP rank。
3. 在 row-parallel 线性层后，将通信从 `all_reduce` 改为 `reduce_scatter`；若该层 dtype 支持 MMRS，则优先尝试 `npu_mm_reduce_scatter_base` 融合路径。
4. 如果 MMRS 不适用（例如 dtype 未接入融合内核、shape/bias 不满足、或通信上下文缺失），则回退到普通 matmul + reduce_scatter，功能与数值保持一致，只是少了 kernel 融合。
5. 在需要完整 hidden states 的边界处（如 attention 的 q_a/kv 投影、MoE 输入），再通过 gather 恢复完整序列。

当前 MMRS 路径使用 torch_npu 提供的 `npu_mm_reduce_scatter_base`（BF16）和对应的 int8 融合入口（w8a8_dynamic）。xLLM 侧只保留薄封装，用于完成输入校验、HCCL group 获取、`comm_mode` 选择、量化 scale 传递和日志记录，不重新实现 kernel。

## 适用场景

FlashComm 更适合以下场景：

- NPU 后端。
- 长输入 prefill，例如输入长度达到 `flashcomm1_min_prefill_tokens`（默认 8192）及以上。
- prefill 占端到端时延比例较高，例如 8K/128、32K/1K 等长 prompt 场景。

收益有限的场景：

- decode 阶段。FlashComm 只优化 prefill，TPOT 通常不会直接受益。
- 短输入（低于阈值）。通信占比不足时，切分、gather 和调度开销可能抵消收益，且低于阈值时不会触发。
- 高 decode 占比场景（长输出）。整体 latency 可能主要由 decode 决定。
- `cp > 1`（context parallel）场景，当前不启用。

## 使用方式

FlashComm 的核心收益（序列分片 + reduce_scatter）通过 `--enable_flashcomm1=true` 一个开关即可获得：

```bash
--enable_flashcomm1=true
```

`enable_mmrs_fusion`（MMRS 融合算子）是在此基础上的**可选增量加速**，只在 `enable_flashcomm1=true` 时才被读取。它当前**默认关闭**，因为融合内核在部分 shape 上仍可能失败（问题修复后会考虑默认开启）。在已验证 shape 稳定的场景下，可显式开启获取额外收益：

```bash
--enable_flashcomm1=true \
--enable_mmrs_fusion=true
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_flashcomm1` | `false` | FlashComm 总开关 |
| `enable_mmrs_fusion` | `false` | 是否启用 Matmul + ReduceScatter 融合算子（仅在 `enable_flashcomm1=true` 时生效）。当前默认关闭，融合内核在部分 shape 上可能失败 |
| `flashcomm1_min_prefill_tokens` | `8192` | prefill token 数达到该阈值后才允许启用 FlashComm |
| `mmrs_comm_mode` | `aiv` | torch_npu MMRS 通信模式，可选 `aiv`、`ai_cpu`、`none` |

`flashcomm1_min_prefill_tokens` 的最优阈值与模型参数量相关，不同模型的 FC1 收益拐点不同，默认值 8192 是一个偏保守的通用起点。建议在各模型的部署文档中给出经过验证的推荐配置，而不是依赖单一默认值。

开启 MMRS 时，通常建议保持 `mmrs_comm_mode=aiv`。如果某些 shape 在 AIV 路径出现 AICore 异常，可以临时切换为 `--mmrs_comm_mode=ai_cpu`。使用 `aiv` 时需加载包含 MMRS AIV 修复的 ops-transformers 9.1.0 或更高版本算子库。

## 参考性能

DeepSeek-V4-Flash W8A8C16，A3，EP16 / dp4 / tp4，ais-bench gsm8k 8K 输入，并发 32：

| 配置 | TTFT (ms) | TPOT (ms) |
|------|-----------|-----------|
| baseline | 10211.2 | 49.6 |
| `+ enable_flashcomm1`（并显式开启 MMRS） | 8840.9 | 46.7 |

FlashComm 带来约 **−13.4% TTFT**，其中序列分片贡献主要部分（约 −11.8%），MMRS 融合在此基础上再贡献约 −1.8%。TPOT 的小幅改善来自 prefill 阶段更快带来的整体调度收益。

## 性能与正确性注意事项

- FlashComm 只优化 prefill，观察收益应重点关注 TTFT、prefill throughput 和 profiling 中 prefill 阶段的通信变化。
- TPOT、decode throughput 不一定明显改善；decode 占比高时端到端 latency 可能看不到大收益。
- MMRS 命中时，profiling 中部分 row-parallel 层后的 `reduce_scatter` 会被融合进 matmul；若成功路径无告警日志，说明融合已生效（回退才会打印 `FC1 MMRS skipped ...`）。
- 若看到额外 gather、layout 转换或 Host 调度开销增加，可能抵消 MMRS 收益。
- 建议使用 warmup 后的多轮稳定请求评估，不要用单次 profiling run 的绝对时延作为性能结论。

## 验证建议

上线或调参前，建议至少完成以下验证：

1. 对比 `enable_flashcomm1=false` 和 `enable_flashcomm1=true`；如需评估 MMRS 增量，再额外加上 `--enable_mmrs_fusion=true` 对比。
2. 使用相同模型、相同并行配置、相同输入输出长度和相同并发。
3. 记录 TTFT、TPOT、prompt throughput、decode throughput、request throughput 和 latency。
4. 对长输入场景额外采集 profiling，确认 MMRS 路径命中（无 `FC1 MMRS skipped` 告警）。
5. 做小规模数值一致性检查，确认开启和关闭 FlashComm 的输出一致。
