---
title: GLM Pipeline Parallelism and DCPP
---

<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

The Python/NPU GLM path (`model_type=glm_moe_dsa`) supports eager pipeline
parallelism (PP) combined with tensor parallelism (TP), including GLM-5.3 W8A8.
`pp_size` defaults to `1`. On 16 workers, PP2 uses TP8 in each stage.

## Stage ownership

Stage `s` owns `[floor(n_layers*s/pp_size), floor(n_layers*(s+1)/pp_size))`.
Only the first stage loads embeddings; only the last loads the final norm and
LM head. Checkpoint names retain global layer indices, while KV/index caches use
local indices. The common block count respects the largest stage allocation,
including its actual number of full indexer layers.

Matching TP lanes exchange hidden states and residuals through an HCCL PP group.
When a boundary enters a shared indexer layer, the preceding stage also sends
int32 top-k indices. GLM-5.3's 78-layer PP2 layout starts stage 1 at shared layer
39, so this state is required. Sampling runs on the last stage's TP rank 0.
The GLM adapter shares the mainline DeepSeek/GLM decoder execution loop.

## Chunk execution

The scheduler reserves KV for up to `pp_size` consecutive prefill chunks per
request, bounded by the existing batch token budget. It records an ordered chunk
plan on the batch. Dynamic mode predicts each chunk using its own history length;
fixed mode uses the same pipeline with the configured fixed chunk size.

The engine interleaves chunks from admitted requests, including consecutive chunks
of one long request. Input preparation records each chunk's positions, prefix
length and sampling targets before dispatch. Each stage executes the same order
and retires all its TP workers before reusing buffers. Different stages may run
different chunks concurrently. Outputs are applied only after all stages succeed;
only the final prompt chunk samples the first generated token.

Pure decode batches are split at sequence boundaries into up to `pp_size`
nonempty microbatches. A single decode request cannot overlap dependent tokens.

## GLM-5.3 W8A8 configuration

Add these options to **every rank** of an existing 16-worker launch:

```bash
--model=/mnt/models/GLM-5.3-w8a8 \
--model_impl=python \
--pp_size=2 --dp_size=1 --ep_size=1 --cp_size=1 \
--enable_graph=false --enable_schedule_overlap=false \
--enable_task_pipeline=false --enable_shm=false \
--enable_in_batch_prefix_cache=false \
--enable_chunked_prefill=true --enable_dynamic_chunking=true \
--max_tokens_per_batch=16384 --max_tokens_per_chunk_for_prefill=4096 \
--dynamic_chunk_min_tokens=256 --dynamic_chunk_profile_samples=16
```

Set `python_model_path` to this checkout. `pp_size` must divide the worker count
and cannot exceed the decoder layer count. Allow more than one base chunk in the
batch budget to fill the pipeline with a single long prompt. Disabling
`enable_dynamic_chunking` selects fixed-chunk PP without startup profiling.

Current support requires online NPU Python GLM workers, eager execution,
DP=EP=CP=1, and no layerwise/KV split. Graphs, worker task pipeline, speculative
decoding, EPLB, PD, host-cache offload, external KV storage, XTensor, in-batch
prefix sharing, the multi-stream-parallel option and beam-search kernel mode are
rejected at startup. Beam-search requests are rejected before scheduling.

Startup profiling measures an isolated chunk across the entire pipeline. It does
not fit separate stage costs or perform online recalibration. Layer partitioning
is by layer count. See [dynamic chunk prediction](../dynamic_chunked_prefill).

## Verification

Regressions cover layer/weight ownership, residual and shared top-k equivalence,
four-process transport, stage ordering and overlap, common KV capacity, chunk
reservation limits, and sampling only at the final prompt chunk. Real HCCL,
full-model accuracy and performance must be validated on the target NPU system;
these regression tests alone do not establish an inference speedup.
