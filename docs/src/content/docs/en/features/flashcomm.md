---
title: "FlashComm"
sidebar:
  order: 82
---

## Feature Introduction

FlashComm is a prefill communication optimization for NPU Tensor Parallel inference in xLLM. It targets the communication overhead after row-parallel linear layers in long-prefill workloads, and uses a Matmul + ReduceScatter fused operator when the workload is suitable.

FlashComm currently contains two parts:

- **Sequence sharding**: during prefill, the token sequence is split across TP ranks so part of the following computation runs on local token shards.
- **MMRS fused operator**: in supported row-parallel linear layers, the normal `matmul + reduce_scatter` path is replaced by the torch_npu `npu_mm_reduce_scatter_base` operator, which fuses Matmul and ReduceScatter.

FlashComm is disabled by default. Even when the feature flag is enabled, it only becomes active when the runtime conditions are satisfied. Otherwise, xLLM falls back to the original execution path.

## Design

The FlashComm flow is:

1. When a request enters prefill, the runtime builds a FlashComm context from the token count, parallel configuration, and feature flags.
2. If the context is active, hidden states are sharded along the sequence dimension across TP ranks.
3. Supported row-parallel linear layers first try the MMRS fused path.
4. If MMRS is not applicable, for example because of unsupported shape, dtype, bias, or communication context, the caller falls back to the normal matmul and reduce_scatter path.
5. At boundaries that require full hidden states, xLLM gathers the sequence back.

The current MMRS path uses the torch_npu `npu_mm_reduce_scatter_base` operator. The xLLM wrapper is intentionally thin: it validates inputs, resolves the HCCL group, selects `comm_mode`, and records logs. It does not reimplement the Matmul + ReduceScatter kernel.

## Suitable Workloads

FlashComm is most suitable for:

- NPU backend.
- Long prefill, for example input length greater than or equal to 8K tokens.
- Large TP size. The current default activation condition recommends `TP >= 8`.
- `dp=1` and `cp=1`.
- Workloads where prefill is a large part of end-to-end latency, such as 8K/128 and 32K/1K.
- BF16/FP16 non-quantized row-parallel linear layers.

FlashComm is usually not suitable, or has limited benefit, for:

- Decode phase. FlashComm only optimizes prefill, so TPOT usually does not directly benefit.
- Short prompts, such as 2K input. Communication may not dominate enough to offset sharding, gather, and scheduling overhead.
- Long-output workloads where decode dominates total latency.
- `TP < 8`, `dp > 1`, or `cp > 1`, which are not enabled by the default activation rule.
- Quantized row-parallel paths. MMRS is currently wired only for the normal BF16/FP16 matmul path.
- MoE models. FC1 currently supports Dense models only and is automatically disabled when routed experts are present.

## Usage

Both FlashComm and the MMRS fused operator are disabled by default. For long-prefill NPU serving with TP=8 or larger, enable them explicitly:

```bash
--enable_flashcomm1=true \
--enable_mmrs_fusion=true \
--flashcomm1_min_prefill_tokens=8192 \
--mmrs_comm_mode=aiv
```

Graph Mode is also recommended to reduce Host-side scheduling overhead:

```bash
--enable_graph=true \
--enable_prefill_piecewise_graph=true
```

A complete recommended configuration is:

```bash
--enable_graph=true \
--enable_prefill_piecewise_graph=true \
--enable_flashcomm1=true \
--enable_mmrs_fusion=true \
--flashcomm1_min_prefill_tokens=8192 \
--mmrs_comm_mode=aiv
```

Flag reference:

| Flag | Default | Description |
|------|---------|-------------|
| `enable_flashcomm1` | `false` | Main FlashComm switch |
| `enable_mmrs_fusion` | `false` | Enables the Matmul + ReduceScatter fused operator |
| `flashcomm1_min_prefill_tokens` | `8192` | Minimum prefill token count required before FlashComm can become active |
| `mmrs_comm_mode` | `aiv` | torch_npu MMRS communication mode. Supported values: `aiv`, `ai_cpu`, `none` |

In most cases, keep `mmrs_comm_mode=aiv`. If some shapes hit AICore errors on the AIV path, switch temporarily to:

```bash
--mmrs_comm_mode=ai_cpu
```

When using `mmrs_comm_mode=aiv`, load an ops-transformers 9.1.0 or newer operator library that includes the MMRS AIV fix.

## Recommended Configuration

Use the following table as the initial tuning guidance:

| Scenario | Recommendation |
|----------|----------------|
| 8K input / short output | Enable FlashComm and MMRS |
| 32K input / medium or short output | Enable FlashComm and MMRS |
| 2K input / long output | Do not enable by default; benefits are usually unstable |
| TP=2 or TP=4 | Do not enable by default |
| TP=8 | Currently the best starting point for evaluation |
| Chunked Prefill | Can be enabled, but keep the chunk size no smaller than `flashcomm1_min_prefill_tokens`; otherwise a single chunk may not trigger FlashComm |

If your service mixes short and long prompts, keep FlashComm disabled by default and enable it only for long-input services, long-context models, or separately deployed long-prompt workloads.

## Performance and Correctness Notes

- FlashComm only optimizes prefill. Focus on TTFT, prompt throughput, and prefill communication in profiling.
- TPOT, decode throughput, and long-output latency may not improve. If decode dominates, end-to-end latency may show little benefit.
- With MMRS enabled, profiling should show that part of the row-parallel communication is replaced by the Matmul + ReduceScatter fused path.
- Extra gather, layout conversion, or Host scheduling overhead can offset the MMRS benefit.
- Use multiple warmup-completed stable rounds for performance conclusions. Do not use absolute latency from profiling runs as the final benchmark number.

## Validation Checklist

Before enabling FlashComm in production or changing thresholds, run at least:

1. Compare `enable_flashcomm1=false, enable_mmrs_fusion=false` against `enable_flashcomm1=true, enable_mmrs_fusion=true`.
2. Use the same model, TP size, input length, output length, and concurrency.
3. Record TTFT, TPOT, prompt throughput, decode throughput, request throughput, and latency.
4. For long-input workloads, collect profiling and verify that the MMRS path is actually hit.
5. Run a small numerical consistency check to confirm that outputs match with FlashComm on and off.
