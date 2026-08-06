---
title: "FlashComm"
sidebar:
  order: 82
---

## Overview

FlashComm is xLLM's prefill communication optimization for NPU Tensor Parallel inference. It reduces the communication cost after row-parallel linear layers during long-input prefill, and where supported, uses a Matmul + ReduceScatter fused operator to cut kernel-launch and communication-scheduling overhead.

FlashComm has two tiers:

- **Sequence-dimension sharding (core)**: during prefill, the token sequence is split across TP ranks so that later work runs on each rank's token shard, and the `all_reduce` after row-parallel layers is replaced with `reduce_scatter`, gathering the full sequence back only at the boundaries that need it. This tier applies to **all dtypes** (BF16 and every quantization path) and is the main source of the speedup.
- **MMRS fused operator (incremental)**: in supported row-parallel layers, `matmul + reduce_scatter` is replaced with torch_npu's `npu_mm_reduce_scatter_base`, further reducing launch and scheduling cost. The fused kernel currently covers the **BF16** and **w8a8_dynamic (int8 dynamic)** paths.

FlashComm is off by default, controlled by the `--enable_flashcomm1` master switch. Once enabled, it only activates when the runtime conditions are met (prefill stage, token count above threshold, `cp=1`); otherwise it falls back to the original execution path.

## Design

The FlashComm flow:

1. When a request enters prefill, the runtime builds a FlashComm context from the token count, parallel config, and switches.
2. When the context is active, the input hidden states are split along the sequence dimension across TP ranks.
3. After a row-parallel linear layer, communication switches from `all_reduce` to `reduce_scatter`; if the layer's dtype supports MMRS, the `npu_mm_reduce_scatter_base` fused path is tried first.
4. If MMRS does not apply (dtype without a fused kernel, unsupported shape/bias, or missing communication context), it falls back to plain matmul + reduce_scatter — functionally and numerically identical, just without the kernel fusion.
5. At boundaries needing the full hidden states (e.g. attention q_a/kv projections, MoE input), the full sequence is restored via gather.

The MMRS path uses torch_npu's `npu_mm_reduce_scatter_base` (BF16) and the corresponding int8 fused entry (w8a8_dynamic). xLLM keeps only a thin wrapper for input validation, HCCL group acquisition, `comm_mode` selection, quant-scale passing, and logging — it does not reimplement the kernel.

## When to use

FlashComm fits best when:

- Running on the NPU backend.
- Long-input prefill, e.g. input length at or above `flashcomm1_min_prefill_tokens` (default 8192).
- Prefill is a large share of end-to-end latency, e.g. 8K/128, 32K/1K long-prompt scenarios.

Limited benefit when:

- Decode stage. FlashComm only optimizes prefill, so TPOT usually does not benefit directly.
- Short inputs (below threshold). When communication is a small fraction, sharding/gather/scheduling overhead may cancel the gain, and below the threshold it won't trigger at all.
- Decode-heavy workloads (long outputs). End-to-end latency may be dominated by decode.
- `cp > 1` (context parallel), currently not enabled.

## Usage

The core benefit (sequence sharding + reduce_scatter) is delivered by the single `--enable_flashcomm1` master switch:

```bash
--enable_flashcomm1=true
```

`enable_mmrs_fusion` (the MMRS fused operator) is an **optional incremental** speedup on top, and is only read when `enable_flashcomm1=true`. It is **off by default**, because the fused kernel can still fail on some shapes (default-on will be reconsidered once that is fixed). On shapes verified to be stable, enable it explicitly for the extra gain:

```bash
--enable_flashcomm1=true \
--enable_mmrs_fusion=true
```

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_flashcomm1` | `false` | FlashComm master switch |
| `enable_mmrs_fusion` | `false` | Enables the Matmul + ReduceScatter fused operator (only read when `enable_flashcomm1=true`). Off by default because the fused kernel can fail on some shapes |
| `flashcomm1_min_prefill_tokens` | `8192` | Minimum prefill token count before FlashComm activates |
| `mmrs_comm_mode` | `aiv` | torch_npu MMRS communication mode: `aiv`, `ai_cpu`, or `none` |

The optimal `flashcomm1_min_prefill_tokens` threshold depends on model parameter count — the FC1 break-even point differs per model, so the default 8192 is a conservative general starting point. Prefer documenting a validated recommended config in each model's deployment doc rather than relying on a single default.

When MMRS is enabled, keep `mmrs_comm_mode=aiv` in general. If some shapes hit AICore errors on the AIV path, temporarily switch to `--mmrs_comm_mode=ai_cpu`. Using `aiv` requires an ops-transformers 9.1.0 or newer operator library that includes the MMRS AIV fix.

## Reference performance

DeepSeek-V4-Flash W8A8C16, A3, EP16 / dp4 / tp4, ais-bench gsm8k 8K input, concurrency 32:

| Config | TTFT (ms) | TPOT (ms) |
|--------|-----------|-----------|
| baseline | 10211.2 | 49.6 |
| `+ enable_flashcomm1` (with MMRS explicitly enabled) | 8840.9 | 46.7 |

FlashComm yields about **−13.4% TTFT**, with sequence sharding contributing the bulk (~−11.8%) and MMRS fusion adding roughly another −1.8% on top. The small TPOT improvement comes from faster prefill improving overall scheduling.

## Performance and correctness notes

- FlashComm only optimizes prefill, so focus on TTFT, prefill throughput, and the prefill-stage communication changes in profiling.
- TPOT and decode throughput may not improve noticeably; with a high decode share, end-to-end latency may show little gain.
- When MMRS hits, the `reduce_scatter` after some row-parallel layers is fused into the matmul in profiling; the success path is silent, so no warning means the fusion is active (only fallback prints `FC1 MMRS skipped ...`).
- Extra gather, layout conversion, or host-scheduling overhead can offset the MMRS gain.
- Evaluate with multiple stable rounds after warmup; don't use a single profiling run's absolute latency as a performance conclusion.

## Validation checklist

Before rollout or tuning, complete at least:

1. Compare `enable_flashcomm1=false` against `enable_flashcomm1=true`; to assess the MMRS increment, add `--enable_mmrs_fusion=true` as a further comparison.
2. Use the same model, same parallel config, same input/output lengths, and same concurrency.
3. Record TTFT, TPOT, prompt throughput, decode throughput, request throughput, and latency.
4. For long-input scenarios, collect profiling to confirm the MMRS path is hit (no `FC1 MMRS skipped` warnings).
5. Run a small numerical-consistency check to confirm outputs match with FlashComm on and off.
