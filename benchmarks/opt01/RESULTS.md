# OPT-01 measured results

**Historical measurements, pending rebased NPU revalidation.** The PR source is
rebased onto `2339aba64eaf98b07b77b143875cb72385eee76f`; upstream's reference
RMSNorm now comes from `xllm_kernel.ops.normalization`. Every NPU result in this document
belongs to the original baseline below. No fresh NPU execution or full native
rebuild of the rebased submission is claimed. See `evidence/README.md`.

Submission checks: Ruff 0.14.0 lint passed repository-wide and all 554 Python
files passed its format check. Feature-file pre-commit checks passed. A full
`pre-commit run --all-files` also passed Ruff and codespell, but clang-format
reported a pre-existing difference in
`tests/core/framework/config/scheduler_config_input_test.cpp:315`; that unrelated
formatter edit was reverted and is not part of this change. No current-head
NPU result is inferred from these static checks.

Baseline: `8da1fc5a3937c4314206354bdc98ecd930921735`. Ascend910_9382,
Python 3.12.16, torch/torch_npu 2.10, CANN 9.2.1 and Triton 3.6.0.
The original local main at `22e1718` was retained; the development branch was
created at the exact runtime baseline after inspecting their API differences.

The production changes were run on NPU. The source checkout itself also passed
three pytest entrypoints and 18 real-model functional inputs, reusing the native
build artifact from the same commit. The environment needed a separately tracked
RTLD_LOCAL loader compatibility change and private ATB dependencies. No fresh
C++ compilation or portability to other CANN versions is claimed.

## Correctness

- Six eager configurations, six fake/meta configurations and eight graph
  replays. Both QKV and all three position axes changed at stable addresses.
- Twelve initialization/dispatch cases.
- Six deterministic forward cases, including position 0 and 262143, four local
  head configurations, legal alternate sections `[16,24,24]` and FP16 legacy
  preprocessing. Current source tests enforce `atol=rtol=0`.
- Single-image, multi-image, video and mixed-length functional cases at TP1/TP2.
- Direct comparisons of 2,052 real TP1/TP2 layer calls: Q/K/V and Attention output
  max absolute error zero. The actual output was preserved before the reference
  used the same backend workspace, metadata and KV slots.

Full-model TP4/TP8, arbitrary partial rotary, FP16 fusion and whole-model prefill
graph capture are not claimed. The tested service uses decode ACLGraph; the
three-axis fused operation was separately captured and replayed on NPU.

## Uninstrumented end-to-end performance

Actual 2,048 input and 2,048 output tokens, one warmup plus three measured waves,
same-device serial A/B/A. Prefix, encoder and processor caches disabled.

| Input / concurrency | A1 TTFT ms | B TTFT ms | A2 TTFT ms |
|---|---:|---:|---:|
| Text / 1 | 147.807 | 137.296 | 149.184 |
| Text / 32 | 2,183.723 | 2,067.405 | 2,247.996 |
| Single image / 1 | 176.297 | 161.835 | 177.878 |
| Single image / 32 | 2,487.828 | 2,303.123 | 2,427.137 |

TTFT was about 5–9% lower than both bracketing baselines in these cases. Long
decode throughput differences were much smaller; no equivalent decode speedup
is claimed.

The same warmed long-image profile had independent RMSNorm counts 73→1 and
mRoPE counts 36→0, replaced by 36 fused operations and one shared table prepare.
AddRmsNorm stayed at 72; InplaceCopy-associated Slice counts fell from 219 to 111.
Target norm/mRoPE device-task sum was about 13.98 ms before versus about 2.92 ms
for fused work, preparation and the remaining final norm. These are single-trace
task sums, not end-to-end durations.

## Additional prefill and memory measurements

Separate instrumented single-image 2,048/2,048 A/B/A, again three measured waves
at concurrency 1 and 32. NPU events bracket decoder forward, excluding vision,
LM head and queuing. At concurrency 32, the values below are total decoder event
time over each 32-request wave, not one request's latency. Instrumentation adds
synchronization, so this series is kept separate from the TTFT series above.

| Concurrency | A1 decoder prefill ms | B | A2 |
|---|---:|---:|---:|
| 1 | 140.518 | 129.504 | 139.976 |
| 32 | 4,062.943 | 3,731.242 | 4,057.503 |

Decoder prefill decreased about 7.5–8.2% against the bracketing baselines.

| Concurrency | A1 peak allocated MiB | B | A2 |
|---|---:|---:|---:|
| 1 | 55,416.537 | 55,417.038 | 55,416.537 |
| 32 | 56,283.333 | 56,285.438 | 56,283.333 |

The shared table costs approximately 0.501/2.104 MiB of peak allocation at these
concurrencies. Peak reserved memory was unchanged: 56,444/57,412 MiB. Exact Torch
peaks cover complete waves including vision and decode; additional device-HBM
samples at 50 ms intervals are lower-bound peak observations.

Instrumented output throughput A1/B/A2 was 62.421/61.437/61.401 token/s at C1 and
961.318/972.164/967.523 token/s at C32. C1 B is slower than A1 and nearly equal to
A2; no stable throughput improvement or statistical non-inferiority is claimed.

## Complete accuracy runs

EvalScope 1.8.1, no sampling limit, temperature 0, top-p 1, max output 2,048,
seed 42, concurrency 32. MMBench includes CN+EN. GSM8K uses the default 4-shot
format; dense Qwen3-8B disables thinking consistently in both variants.

| Model / dataset | Baseline correct | Candidate correct |
|---|---:|---:|
| Qwen3-VL-8B / MMBench | 7,480 / 8,658 | 7,499 / 8,658 |
| Qwen3-VL-8B / GSM8K | 1,250 / 1,319 | 1,249 / 1,319 |
| Qwen3-8B unchanged path / GSM8K | 1,233 / 1,319 | 1,224 / 1,319 |

Full responses and scores are not all identical. Same-device sequential A/B/A
on all 21 discrepant VL and 23 discrepant dense GSM inputs produced identical
full responses across those three runs. This does not replace the original full
scores or prove a particular cause for every batched-generation difference.

Full source fingerprints, raw profiler CSVs, per-request records, per-wave
min/max statistics, direct-tensor logs and complete accuracy evidence are
retained in the experiment delivery directory. Three waves on one machine do
not establish a cross-machine guarantee or statistical quality non-inferiority.
