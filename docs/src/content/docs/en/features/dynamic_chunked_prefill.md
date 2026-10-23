---
title: Dynamic Chunked Prefill
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

Dynamic chunked prefill uses startup latency measurements and each request's
cached history to choose its next prefill chunk. It is disabled by default.

This ports the startup profiling and quadratic chunk predictor from
[vLLM-Ascend Dynamic CPP](https://github.com/vllm-project/vllm-ascend/blob/8a2c3182c463a7391b4c9d21be3676fc12c3d14c/vllm_ascend/core/profiling_chunk_predictor.py),
with reference to SGLang's `DynamicChunkSizer`, into xLLM's C++ scheduler.
This option does not enable pipeline parallelism by itself. The separate
[experimental PP path](../pipeline_parallel) adds GLM stage execution and
consecutive chunk overlap, including a single long request.

## Configuration

Add these flags to an existing LLM command, or use the same keys in config JSON:

```bash
--enable_chunked_prefill=true \
--enable_dynamic_chunking=true \
--max_tokens_per_chunk_for_prefill=4096 \
--dynamic_chunk_min_tokens=256 \
--dynamic_chunk_smooth_factor=1.0 \
--dynamic_chunk_profile_samples=16
```

| Option | Default | Meaning |
| --- | --- | --- |
| `enable_dynamic_chunking` | `false` | Enable startup profiling and dynamic chunks |
| `dynamic_chunk_min_tokens` | `256` | Positive minimum before alignment and budget caps; no larger than the base chunk |
| `dynamic_chunk_smooth_factor` | `1.0` | Weight in `(0, 1]`; smaller values stay closer to the base chunk |
| `dynamic_chunk_profile_samples` | `16` | Number of measured lengths, between 8 and 64 |

The base chunk is the smaller of `max_tokens_per_chunk_for_prefill` and
`max_tokens_per_batch`. Alignment is
`lcm(64, block_size * kv_split_size_effective)`. Profiling requires at least
8 distinct aligned lengths and enough KV capacity and model context for the base
chunk. Insufficient distinct lengths or uninformative measurements produce a
warning and select static chunks. Model execution failures propagate.

## Execution

`ProfileManager::run_request` warms each length and measures three executions,
using their median. Synthetic KV is released without populating the prefix cache.
The existing synchronous output path supplies execution and orchestration timing.

A normalized, nonnegative least-squares fit estimates `f(L) = a*L² + b*L + c`.
Given history `H`, the predictor solves
`a*x² + (2*a*H+b)*x = f(base) - f(0)` using a stable positive-root formula.
A linear fit is allowed when curvature cannot be identified.

The result is smoothed, aligned, and capped by the base chunk, remaining prompt,
and available token budget. A final partial block can finish without alignment.
When no aligned block fits, the request remains queued. Prefix matching precedes
prediction; mixed-batch budget redistribution cannot enlarge a chunk past its
dynamic cap. Existing KV, DP token-budget, and SLO checks still apply.

The target applies to **one request's chunk**, not the aggregate time of a batch.
The existing latency-aware scheduler remains responsible for batch latency limits.

## Scope and validation

- Integrates with `PrefillFirstPolicy`, `DecodeFirstPolicy`, and `UnifiedPolicy`
  through `ContinuousScheduler` and the corresponding `DisaggPDScheduler` path.
  Decode-only instances skip dynamic prefill profiling.
- Rejects `ZeroEvictionScheduler` and linear-attention models
  using prefix-cache checkpoints tied to fixed chunk boundaries.
- Fitting is performed at startup only. The upstream online history-aware
  recalibration is not included in this phase.
- Tests cover predictor mathematics and limits, config JSON, and scheduler budget
  redistribution/queue retention. Device validation and performance comparisons
  must use the actual deployment model and workload.

With PP enabled, the scheduler plans up to `pp_size` chunks in one reservation,
advancing the predicted history after each chunk. The engine consumes the exact
plan. Fixed-chunk PP uses the same planning and execution path without fitting.
This follows vLLM-Ascend's profiling-based prediction and in-flight prefill
principle; its online calibration and stage-specific timing are not ported.
