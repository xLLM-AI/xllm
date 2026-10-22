---
title: "OLMo2 CUDA validation preview"
sidebar:
  order: 1
---

<!-- Copyright 2026 The xLLM Authors. -->

This preview covers `allenai/OLMo-2-0425-1B-Instruct`, revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`, with the Python executor,
FlashInfer paged attention, FP16/BF16, eager execution and TP=1/2. Both compute
dtypes passed the frozen numerical matrices and real HTTP service checks
on the local CUDA setup. This contribution includes the OLMo2 C++
configuration loader, field validation, fixtures and tests originally
submitted in #2518 (Z25); it can be reviewed and merged as one implementation.
It is not a completed
platform support announcement. NPU, quantization, graph execution and
DP/CP/PP greater than one have not been accepted for this model.
This contribution targets NVIDIA CUDA. Ascend adaptation is a separate task.

The checkpoint has 16 layers, hidden size 2048, 16 MHA heads of size 128,
intermediate size 8192, vocabulary size 100352 and a 4096-token context.
Q/K RMSNorm spans the complete projection before head reshaping and RoPE.
TP=2 sums separate FP32 Q/K squared statistics across ranks, then applies
the sharded norm weights. Attention/MLP post-norm outputs are added to an
FP32 residual stream; projection weights and KV cache retain the selected
FP16/BF16 compute dtype. The embedding and LM head are independent weights.

On CUDA BF16, FP32 projection inputs are split into a BF16 rounded value and
a BF16 residual correction. Their two GEMMs produce FP32 outputs, summed
before normalization; row-parallel reduction and LM-head logits also remain
FP32. BF16 decode explicitly selects FlashInfer tensor cores. Weights and
KV caches remain BF16. This mixed-precision policy adds GEMM work and FP32
activation/communication memory; it is not pure BF16 arithmetic or a speedup
claim. FP16 retains its existing projection and decode policy.

## Prepare a separate FP16 configuration

Keep the downloaded checkpoint unchanged. Set these paths to the fixed
checkpoint and a new, nonexistent directory; this creates config-only
overrides and symlinks the original weights and tokenizer files.

```bash
export OLMO2_SOURCE=/path/to/OLMo-2-0425-1B-Instruct
export OLMO2_FP16=/path/to/OLMo-2-0425-1B-Instruct-fp16
python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["OLMO2_SOURCE"]).resolve()
target = Path(os.environ["OLMO2_FP16"])
target.mkdir()
for item in source.iterdir():
    if item.name != "config.json":
        (target / item.name).symlink_to(item)
config = json.loads((source / "config.json").read_text())
config.update(dtype="float16", torch_dtype="float16")
(target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

The source weights are BF16. Casting to FP16 changes some very small values;
this is an explicit compute configuration, not bitwise-preserved BF16 execution.
Use the checkpoint's serialized `tokenizer.json` for token-ID references.
Transformers 5.12.1 `AutoTokenizer` reconstructs a GPT2 tokenizer with a
different newline pre-tokenizer. HF template rendering followed by
`tokenizers.Tokenizer.from_file(...).encode(..., add_special_tokens=False)`
preserves the checkpoint tokenization used by the service.

## Reproduce service acceptance

Build the branch's CUDA executable and matching Python extension first.
Run from the repository root with its Python dependencies installed. Set
`XLLM_BINARY` to that executable; preserve the runtime library paths required
by your CUDA build. The local validation used Python 3.12.3, Torch 2.10.0+cu130,
Transformers 5.12.1, FlashInfer 0.6.14 and TVM-FFI 0.1.13.

```bash
export XLLM_BINARY=/path/to/built/xllm
python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" --dtype float16 \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-tp1

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" --dtype float16 \
  --tp 2 --devices 0,1 --port 18000 --output /tmp/olmo2-tp2
```

For BF16, use `--model "$OLMO2_SOURCE" --dtype bfloat16` instead of the
FP16 config path. `--dtype` verifies the loaded configuration; it does not
convert the checkpoint or override the server configuration.

Use new output directories for each run. The tool starts and stops its own
servers, saves commands, requests and SSE events, and fails on mismatches.
TP=2 uses two local server processes with `--nnodes=2 --node_rank=0/1` and a
shared rendezvous address. Exposing two GPUs to one server process alone
does not create TP=2 in this engine revision. It checks the actual loaded
TP topology and the requested compute dtype, template token count, stream equivalence,
EOS/ignore_eos, explicit stop tokens/strings, length, invalid inputs and
mixed-length concurrent isolation. These are service checks; they do not
replace numerical comparison of logits.
On a shared GPU, `--max-memory-utilization` and `--max-cache-size` expose
the server's existing memory controls; defaults remain 0.4 and 1073741824
bytes per rank. Check free memory before choosing them. Both effective values
are saved with the launch and performance records. A smaller cache can add
preemption/recomputation, so results with different capacities are not a
controlled performance comparison. These options never change the requests
or acceptance assertions.
Every stream must report a finish reason and end with `[DONE]`; requested
usage must also be present. Truncated chat streams and in-band errors fail
acceptance even when all expected text has already arrived.
The tool writes its final PASS result only after every server rank exits
with code zero. A crash or forced shutdown fails acceptance, and the rank
exit codes are retained in `shutdown.json`.

The tool preserves `FLASHINFER_CUDA_ARCH_LIST` when set by the caller;
otherwise FlashInfer detects the GPU architecture. It does not force the
RTX 4090's architecture on other NVIDIA devices. The effective override is
recorded in `launch.json`; validation on other GPU models is still required.

For this server's allocator/FFI environment, the successful runs additionally
set `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4` and placed the
matching Python TVM-FFI `lib` directory first in `LD_LIBRARY_PATH`.

Set `FLASHINFER_OPS_PATH` to the native FlashInfer kernel directory
(`/flashinfer_ops` in the CUDA image). The service acceptance tool also sends
unseeded requests with `temperature=0.7`, both regular and streaming, and
checks completion length and token usage. This exercises the CUDA random
sampler, which greedy-only checks do not load. Native kernel function lookup
uses the packed TVM-FFI interface to avoid a C++ `Optional` return-layout
mismatch when the embedded Python runtime loads a different TVM-FFI version.

## Tests and measurements

```bash
python -m pytest tests/python/test_olmo2.py tests/python/test_registry.py \
  tests/python/test_flashinfer_attention.py tests/python/test_validate_olmo2_cuda.py -q

python tools/validate_olmo2_cuda.py \
  --model "$OLMO2_FP16" --binary "$XLLM_BINARY" --dtype float16 \
  --tp 1 --devices 0 --port 18000 --output /tmp/olmo2-benchmark \
  --benchmark --build-label "record your compiler flags here"
```

The optional benchmark uses prompt lengths 64/512/2048, 32 output tokens,
concurrency 1/4, one warmup and three measured repetitions per group. Both
prefix caches and graphs are off; per-rank KV cache is capped at 1 GiB.
Raw SSE, memory snapshots and per-request timings are retained. TTFT is the
first nonempty text arrival; effective TPOT is `(request_end - TTFT)/(N-1)`
using reported completion tokens, including invisible special tokens. It is
a client metric, not an individual kernel or token-arrival measurement.
Record other GPU users and cache pressure before interpreting throughput.

The historical local native build used `XLLM_DEV_MODE=ON` (O1). Its measurements
are development baselines, not release performance or a speedup claim.
The October 20–21 revalidation used matching Release/O3 native and Python
artifacts after rebasing onto main `b6123035`. Its HTTP timing runs overlapped
CPU compilation of the repository tests, so they are diagnostic measurements,
not a controlled release benchmark.


The local development run used two RTX 4090 devices reporting 48537 MiB each.
The following historical FP16 results are aggregate throughput medians and ranges across three
repetitions; the raw requests are retained with the contribution evidence.
These are separate TP baselines, not an optimization comparison.

| TP | Prompt tokens | Concurrency | Output tokens/s, median (range) |
| --- | --- | --- | --- |
| 1 | 64 | 1 | 24.18 (22.93–24.38) |
| 1 | 64 | 4 | 90.64 (84.99–93.50) |
| 1 | 512 | 1 | 21.22 (21.12–22.27) |
| 1 | 512 | 4 | 59.63 (56.75–67.72) |
| 1 | 2048 | 1 | 17.06 (14.88–17.46) |
| 1 | 2048 | 4 | 34.03 (32.11–34.64) |
| 2 | 64 | 1 | 19.30 (19.01–19.56) |
| 2 | 64 | 4 | 67.22 (63.69–68.49) |
| 2 | 512 | 1 | 16.96 (16.80–17.43) |
| 2 | 512 | 4 | 51.36 (49.65–52.22) |
| 2 | 2048 | 1 | 14.76 (13.36–15.26) |
| 2 | 2048 | 4 | 33.96 (32.81–34.73) |

TP1 has 8192 cache slots in this configuration. The 2048 × 4 case
needs 8320 slots including output and therefore includes cache pressure.
TP2 has a different sharded cache capacity. These rows cannot establish
a release speedup or isolate collective overhead. Profiling separately
covered prefill512 and 16 same-history decode steps: 272 Q/K norm calls
and 544 separate Q/K sums. Numerous small Torch operations and conversions
make CPU dispatch a follow-up investigation; no kernel optimization or
post-optimization speedup is claimed.

## Numerical validation status

BF16 and FP16 each pass 591 TP1 checks and 1064 checks on each of two real
NCCL ranks, using unchanged frozen budgets and HF BF16 references. Each
dtype also passes seven HTTP functional groups and six benchmark conditions
with three repetitions for both TP configurations. Revalidation on main
`b6123035` with matching Release/O3 artifacts passed this complete matrix
using 80 physical cache pages for numerical tests. The final HTTP runs used
`--max-memory-utilization 0.4` for TP1 and `0.99` for TP2, with
`--max-cache-size 1073741824` for every rank. The first TP2 startup at 0.4
failed before readiness under shared-GPU load; that failure was retained.
Both GPUs were available when these numerical runs started. Earlier shared-GPU
runs needed 64 pages and a smaller HTTP cache cap; those resource failures and
successful retries remain in the contribution evidence. These results apply
to this local CUDA setup and do not establish a performance improvement.
The mixed-precision policy still needs maintainer review and independent
reproduction. The supported-model matrix remains unchanged.

## Reproduce full-checkpoint numerical acceptance

`tools/validate_olmo2_numerics.py` carries the original B06/B07 inputs and
unchanged v2 budgets in `tests/models/fixtures/olmo2/numerics.json`. It uses
the original BF16 checkpoint/config at revision
`48d788eca847d4d7548f375ad03d3c9312f6139e`; SHA256 checks reject a different
checkpoint. Pass `--dtype float16` explicitly to convert weights in memory
(normal FP16 rounding, including subnormal values); the source checkpoint is never rewritten. HF reference
generation requires Transformers 5.12.1 and compares against unmodified eager
BF16. FP32 references are diagnostics. No budget is fitted to xLLM output.

```bash
# Run from a built checkout with its matching xllm_export/runtime environment.
MODEL=/path/to/OLMo-2-0425-1B-Instruct
OUT=/path/to/new-validation
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py reference \
  --model "$MODEL" --output "$OUT/reference"
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py validate \
  --model "$MODEL" --reference "$OUT/reference" --output "$OUT/fp16-tp1" \
  --dtype float16
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tools/validate_olmo2_numerics.py validate --tp 2 --dtype float16 \
  --model "$MODEL" --reference "$OUT/reference" --baseline "$OUT/fp16-tp1" \
  --output "$OUT/fp16-tp2"
# This is a diagnostic control; a threshold failure deliberately exits 1.
CUDA_VISIBLE_DEVICES=0 python tools/validate_olmo2_numerics.py hf-control \
  --model "$MODEL" --reference "$OUT/reference" --output "$OUT/hf-control"
```

Use new output directories for every run. Keep the generated tensor files and
manifests outside Git. TP2 requires the same-dtype TP1 evidence and verifies
checkpoint, runtime source and tensor-file hashes. Every loaded weight/shard is checked exactly against the explicit dtype
conversion of its checkpoint tensor; conversion error is recorded separately.
Both ranks use real xLLM
NCCL collectives and paged FlashInfer execution, compare full vocabulary logits
against HF/TP1, and check rank equality. Cache starts with NaNs; nonmonotonic
pages, 127/128/129 boundaries, chunked prefill, mixed batches and total length
4096 are included. TP1 runs 591 comparisons; TP2 runs 1064 per rank, extending
the original B08 length coverage. Missing data, non-finite values, shape errors
and numeric failures cannot produce a successful exit code. Reports preserve
all failures; TP2 shares a failing status across ranks. Runtime exceptions
propagate and torchrun terminates the worker group.
The numerical runner defaults to 80 physical cache pages. `--cache-blocks 64`
removes unused preallocation while retaining the entire frozen matrix:
the longest cached and independent full-reference histories each need 32
disjoint pages. The tool computes the fixture's capacity requirement, rejects
smaller allocations and records the actual page count. Sequence lengths,
decode steps, mixed batches and numerical budgets remain unchanged.

Repeat both validation commands with `--dtype bfloat16` and distinct output
paths to validate the revised BF16 implementation. The `hf-control` command reproduces the
official eager BF16 cache/batch variation against its own full-prefill reference;
it does not waive xLLM failures or change the frozen criteria. Actual automated
reproduction is distinct from review/reproduction by another contributor.
