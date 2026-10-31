# NPU sampling: implementation and validation

The cleanup measurements in this report describe the snapshot committed as
`b3c3b5429b0dabfb1debaf878a63cd049a4130ed`, originally based on xLLM
`9dee07ef4ab6d8130794ceb538e2c6c049d49ed2`. Subsequent integration with `main`
preserves its sampling controls and acceptance-count outputs; these timings
are not measurements of that integrated build. DeepInfer was inspected at
`553741557a57df44a46fb3ab72b6aafeaf945337`. Tests use Ascend 910B2, CANN 9.0.0,
PyTorch 2.9.0 and torch_npu 2.9.0.post2. Native tests use physical device 4;
serving measurements use devices 4–7.

Production selects supported optimizations automatically, with no CLI or JSON
switches. Only `BUILD_NPU_SAMPLING_ABLATION=ON` contains the mutable test mask,
native benchmark, and standalone probability-weight sampler used to isolate A.
That sampler has no Torch operator registration. Normal builds omit this
standalone reference and the mutable ablation control.

## Implementation and boundaries

| ID | Test bit | Optimization | Implementation |
| --- | ---: | --- | --- |
| A | 1 | Batched sampling | Batch seeded categorical draws on device; avoid copying seed/offset controls back to the host and constructing a generator per sampled request. |
| B | 2 | Random-row filtering | Compact random rows before temperature/min-p/top-k/top-p processing in supported ordinary mixed batches; scatter results back. Penalties and grammar still run for every row. |
| C | 4 | Fused logits sampling | Fuse normalization and categorical selection, avoiding a vocabulary-sized FP32 probability tensor. Requires A for ordinary sampling. Existing filtering stays separate. |
| D | 8 | Compact normalization | Keep the row maximum and shifted log-sum-exp separately for stable selected-score gathering and verification. Raw model logprobs use their own normalization, before penalties or filtering. Preserve full log-softmax for top-logprob requests. |
| E | 16 | Sparse DFlash2 | Preserve the DFlash2 selector's existing candidate support through proposal construction, transport, acceptance and exact residual recovery. |
| G | 32 | Verification sampling | Use the shared fused categorical kernel for bonus/recovery tokens in supported verifier batches. Acceptance/prefix handling stays in the existing verifier. |

The implementation follows the batching, on-chip normalization and compact
sampling-state ideas in DeepInfer's
[sampler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/sampler.py)
and [Ascend fused sampler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/fused_sample.py).
It uses the installed TileLang Ascend API and xLLM's existing sampling semantics.

Important boundaries:

- Keep xLLM's top-k-then-top-p order and native tie behavior. A probe replacing
  top-p-only filtering with a synthetic vocabulary-sized top-k changed one
  retained token on tied logits (`V=49152`, several p values). That candidate
  (F) is excluded. Full filtering fusion is not part of C.
- B applies to ordinary sampling with `B>=2`, `V>=32768`, at most half random
  rows, and no logprobs, returned logits or beam search. Verification retains its
  existing filter sequence.
- Ordinary D applies to selected logprobs only, BF16/FP16, `B>=128` and
  `V>=65536`; verification D requires at least 8 Mi logits elements.
  Subtracting LSE can round distinct logits into FP32 ties, so selecting raw-logit
  top-k would change reported token IDs. Top-logprob reporting keeps the
  original operation, including in the compact verifier.
- G requires at least eight verifier rows. Small-batch measurements did not
  justify this kernel there.
- Kernels support FP32/BF16/FP16 logits, unit inner stride, up to 4096 rows and
  262144 vocabulary entries, with validated address-span bounds. Unsupported
  shapes take the existing path before execution.
- E preserves existing, unique selector IDs and their exact q distribution.
  It does not truncate an arbitrary dense draft distribution. Sparse producer
  and consumer must be enabled together.
- A uses a stateless SplitMix64-derived uniform with request seed and offset.
  Seeded draws remain deterministic for fixed logits and independent of batch
  permutation/global RNG. The random sequence differs from the legacy
  multinomial generator. Unseeded draws use the normal NPU generator.
- Sampling metadata is propagated through device copies, reusable slots and
  packed inputs. The packed-input schema moves from 10 to 11; ranks must use the
  same version.
- A separate shutdown fix avoids synchronizing the TLS ACL top-k/top-p
  workspace after torch_npu has finalized the NPU runtime.

The verifier depends on a three-operation `DraftDistribution` interface:
normalization, in-place acceptance correction, and residual correction. Dense
and sparse implementations own their probability tensors. Point-mass drafts
need only metadata and perform no acceptance-correction device operation
because q is one at proposal positions. A move-only proposal validates the
pairing and owns its distribution until asynchronous slot retirement.
Construction checks establish representation invariants; compatibility checks
protect the producer/verifier boundary.

The production fused sampler and the A-only reference share TileLang macros
for seed hashing, chunk selection, and categorical selection within a tile.
The macros inline during AOT compilation. Their module is included in both
AOT dependency fingerprints. Numerical contracts remain on the production API;
there is no additional runtime dispatch layer.

## Cleanup validation before main integration

The cleanup isolates the A-only reference at build time, removes its Torch
registration, and extracts shared kernel logic. The generated weight sampler
and all three normalization kernels are unchanged. Generated fused kernels
for FP32/BF16/FP16 differ only in the order of two independent initializations.

Both builds pass: production has 348 cases (44 sampler, 31 CPU controls,
three filter-mask, 18 packed-input, 51 slot-buffer, 66 configuration, and
135 pipeline); the instrumented sampler has 93 cases. The prior production
count was 349: the weight-only endpoint test now belongs exclusively to the
instrumented build, while fused boundary and distribution coverage remains.
The production service and sampler contain neither the weight-sampler symbol
nor the mutable mask; the production kernel archive excludes its objects.
Clang-format 20.1.6, repository pre-commit checks, and diff checks pass.
These results cover the cleanup snapshot, not the subsequent main integration.
Results, commands, and source/binary identities are recorded under
`local/npu_sampling/review_revision/redundancy_cleanup/`.

Coverage includes fixed-seed repeatability, CPU FP64 CDF oracles, nonuniform
and masked support, vocabulary boundaries, padded rows, FP32/BF16/FP16,
unseeded RNG, greedy preservation, filter ties, stable selected logprobs,
dense/sparse recovery, point-mass acceptance, sentinels, proposal lifetime,
packed inputs, slot reuse, configuration, and task-pipeline transport.
The weight-only endpoint test belongs to the instrumented build; the
production fused sampler retains distribution and vocabulary-boundary tests.

## Main integration validation

The integration with `main` at `8c5b8e8c` preserves raw-model logprobs before
sampling transforms, draft-prefix penalties, and int32 accepted-token counts.
Seeded verification uses the ordinary sampler at each logical token offset,
so both paths select the same supported NPU implementation. Rejection gathers
clamp the full-acceptance count to a valid draft index and convert it to int64.
The integration also fixes an upstream calibration-loop variable rename that
otherwise prevents the task-pipeline runtime from compiling.

The `ca6272e9` checkpoint passes 309 cases: 46 sampler, 37 CPU controls,
150 speculative pipeline, 51 slot-buffer, 18 packed-input, and seven
speculative-metrics cases. Two new NPU regressions cover ordinary/verification
seed alignment in random and mixed batches, zero-length drafts, grammar
rejection, full acceptance counts, and raw logprobs under bias/top-k for both compact and
full top-logprob reporting. Clang-format 20.1.6 and repository pre-commit
checks pass. Commands, logs, and source/binary identities are recorded under
`local/npu_sampling/review_revision/main_integration/`.

The runtime's updated `dspark_greedy_verify_commit` TileLang kernel is
regenerated from the integrated source. Native performance, instrumented
ablation tests, and real-model E2E have not been rerun on this integration;
the timings elsewhere in this report retain their stated earlier versions.

Integration with the newer `main` at `ca9f8e63` adapts sampling controls to its
grouped property accessors. Host random-row counts belong to the selection
group; the unseeded-request flag belongs to the random group. Both retain
their device-copy, slot-reuse, and packed-input propagation. Main now uses
packed-input schema 10, so the additional sampling metadata uses schema 11
and rejects schema 10 inputs. This production integration passes 315 cases:
49 sampler, 37 CPU controls, 150 speculative pipeline, 54 slot-buffer,
18 packed-input, and seven speculative-metrics cases. The instrumented
benchmark, sampler, and conditional sampler-test sources also compile with
the ablation macro enabled; instrumented runtime tests and performance
measurements have not been rerun. Logs and source/binary identities are under
`local/npu_sampling/review_revision/property_integration/`.

## Native performance

The cleanup retake covers 33 cases and 165 measurements. Comparisons restore inputs
outside timing, warm up, synchronize the compute stream, and measure host wall
latency including launch and completion. Variants are randomly interleaved
within one process, with five repetitions. Memory is allocator peak above
preallocated inputs; it excludes raw ACL workspace and is not total HBM usage.
Ordinary workloads use BF16 and V=248320; verification uses FP32, S=7 and K=16.

| Native workload | Latency, baseline → optimized (us) | Extra allocator peak, baseline → optimized (MiB) |
| --- | ---: | ---: |
| Ordinary random, B=32, A+B+C | 4496.58 → 1163.70 | — |
| Ordinary mixed, B=32, A+B+C | 45033.23 → 992.78 | — |
| Selected logprobs, B=128, A+C+D | 15817.32 → 3552.39 | 363.75 → 76.63 |
| DFlash2 verification, B=32, D+E+G | 7283.92 → 2753.14 | 636.57 → 424.43 |

Joint controls: selected-logprob A+C takes 3583.39 us and 283.23 MiB;
adding D reduces peak to 76.63 MiB and takes 3552.39 us. D alone takes
15788.57 us, confirming its main benefit is memory. Verifier E, D+E, and
D+E+G take 5148.29, 4724.56, and 2753.14 us. At B=1, E takes 1413.22 us
versus 1327.06 us baseline (+6.5%), with peak 34.01 → 27.38 MiB (-19.5%).
D/G are inactive at that size. The retake therefore retains the documented
small-batch tradeoff rather than claiming a universal latency improvement.

The benchmark executable SHA-256 is
`544e9ffcc2679843611af9c798faf84a48628a9cff1f043bea882859ecb5554e`.
Full factorial results and all 165 iterations remain in `native.json` under
the current validation directory. These are comparisons of optimization
variants, not evidence that extracting shared macros itself improves latency.

The earlier point-mass refinement was separately compared with the original
verifier in one process: four small-batch shapes differed by -0.62% to +0.44%.
This supports baseline parity, not a new speedup. Mask 0 shares the current
interface, so mask comparisons alone cannot validate shared refactor costs.
The comparison and wider historical matrices remain in the
[validation appendix](npu_sampling_ablation_history.md).

## Serving performance

The following results belong to the measured instrumented service SHA-256
`b887d9e0310fa0903fd29f5f8511ad0d5676946f27aeae31e7427d83c8ea56ba`.
They precede the point-mass refinement and this cleanup; real-model E2E has
not been rerun for those changes. They must not be described as timings of
the current production executable.

Target: `/data/models/Qwen3.8-27B` (qwen3_5, V=248320). Draft:
`/home/lhp/models/z-lab/Qwen3.8-27B-DFlash2`, revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, K=16 and seven draft tokens.
The TORCH backend uses ACL graphs, chunked prefill, no prefix caching or
schedule overlap, 16 GiB KV cache, and linear-state capacity 128 (ordinary)
or 32 (DFlash2). Requests have a fixed 15-token prompt, temperature 0.7,
top-k 20, top-p 0.95, 64 completion tokens, and ignore-EOS. Mixed batches
contain 25% random requests. DFlash2 request seeds are unsupported.

Each comparison switches masks in the same resident four-rank process group.
The controller drains requests, stops its ranks, writes and reads back the
mask, resumes the ranks, and warms the next workload. Three repetitions
complete 2592 measured requests; all 63 switches preserve worker PIDs.
TPOT measures time between first and last nonempty chunks divided by tokens
minus one; multi-token draft chunks mean this is not an ITL distribution.
Throughput includes prefill and HTTP completion.

| Serving workload | Concurrency | TPOT, baseline → optimized (ms) | Throughput, baseline → optimized (tokens/s) |
| --- | ---: | ---: | ---: |
| Ordinary random | 32 | 35.29 → 31.29 | 836.58 → 936.22 |
| Ordinary mixed | 32 | 74.97 → 30.95 | 410.98 → 942.52 |
| Ordinary selected logprobs | 128 | 79.33 → 62.54 | 1501.81 → 1882.83 |
| Ordinary greedy control | 32 | 29.97 → 29.99 | 973.28 → 972.91 |
| DFlash2 random | 32 | 46.97 → 44.55 | 573.73 → 607.32 |
| DFlash2 mixed | 32 | 142.53 → 137.03 | 206.54 → 218.26 |
| DFlash2 greedy control | 8 | 26.48 → 25.98 | 226.20 → 225.94 |

These are medians, not medians of paired percentage changes. Ordinary
selected-logprob A+C without D measures 1876.06 tokens/s and 62.69 ms TPOT,
confirming that D adds little serving throughput. DFlash2 random D+E and
D+E+G measure 593.88 and 596.57 tokens/s; mixed measures 212.92 tokens/s
for both. All-on includes ordinary prefill sampling and measures 607.32 and
218.26 tokens/s. Acceptance and scheduling vary; no speedup is claimed for
greedy controls. Identical prompts already yield different greedy text within
baseline batches, so fixed-logit tests establish sampler correctness separately.
All 73728 returned selected-token logprobs are finite and nonpositive.

## Scope and limitations

- A/B/C primarily reduce ordinary sampling latency; D primarily reduces
  selected-logprob memory. Joint ablations are necessary to attribute benefits.
- Sparse verification trades some small-batch latency for memory. D/G dispatch
  only at supported sizes; no unconditional per-feature E2E gain is claimed.
- Fixed-logit tests establish sampler correctness. The model's repeated
  baseline responses/logprobs vary, so serving output identity does not prove
  sampler correctness or model-level bitwise determinism.
- Real-weight task-pipeline E2E is excluded from this 910B2 acceptance scope:
  the installed MegaGdnMtpDecodeVarlen implementation targets 910C. Unit tests
  cover transport, but no E2E performance claim is made for that mode.
- The [appendix](npu_sampling_ablation_history.md) retains earlier shape
  matrices, long-output/greedy controls, 36 exported timelines, artifact
  identities, and known counterexamples. Profiling was separate from timing.

## Reproduction and artifacts

Use a dedicated `BUILD_NPU_SAMPLING_ABLATION=ON` build for comparisons. AOT
source changes require regenerating kernels before the CMake rebuild. The
instrumented build needs both `seeded_sample` and `sampling_logits`; production
requires only `sampling_logits`. Production does not link the A-only wrapper
or its generated device object.

For example, after configuring the repository's NPU build:

```bash
python3 xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend --device a2 \
  --output-root build/cmake.linux-aarch64-cpython-311/xllm/compiler/tilelang \
  --kernels seeded_sample sampling_logits --jobs 4
cmake -S . -B build/cmake.linux-aarch64-cpython-311 \
  -DBUILD_NPU_SAMPLING_ABLATION=ON
cmake --build build/cmake.linux-aarch64-cpython-311 \
  --target xllm npu_sampling_benchmark sampler_test -j8

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
build/lib.linux-aarch64-cpython-311/xllm/npu_sampling_benchmark \
  --benchmark_filter='sparse_verification/B:32/S:7/.*logprobs:0/' \
  --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true \
  --benchmark_out=local/npu_sampling/verify_repro.json \
  --benchmark_out_format=json
```

Build directory names above describe this Python 3.11/aarch64 environment;
adapt them to the configured checkout. The benchmark's full Cartesian shape
grid is much larger than the selected acceptance subsets, so use a filter.

The serving tool and controller are checked in under `tools/`:

```bash
python3 tools/benchmark_npu_sampling.py ordinary \
  --target-model /data/models/Qwen3.8-27B --devices 4,5,6,7 \
  --artifacts local/npu_sampling/ordinary_repro

python3 tools/benchmark_npu_sampling.py dflash \
  --target-model /data/models/Qwen3.8-27B \
  --draft-model /home/lhp/models/z-lab/Qwen3.8-27B-DFlash2 \
  --devices 4,5,6,7 --profile \
  --artifacts local/npu_sampling/dflash_repro

python3 tools/benchmark_npu_sampling.py dflash \
  --target-model /data/models/Qwen3.8-27B \
  --draft-model /home/lhp/models/z-lab/Qwen3.8-27B-DFlash2 \
  --devices 4,5,6,7 --batch-sizes 1 --modes random \
  --tokens 512 --repeats 7 --masks 0 16 56 63 \
  --artifacts local/npu_sampling/dflash_long_repro
```

Run inside the matching CANN/PyTorch build environment. The controller must own
the service process tree and use an instrumented service plus matching
benchmark binary. The controller resolves only the test-specific mask symbol;
it refuses a production executable that lacks that symbol. Each run saves the actual command, executable
SHA, controller/source copies, PID readback, per-request usage, response hashes,
timing samples and speculative counters. `--task-pipeline` selects that mode;
`--profile-only` captures warmed windows without running the timing matrix.
The bit mask is only a test-build control. Reconfigure with
`-DBUILD_NPU_SAMPLING_ABLATION=OFF` and rebuild before deployment; normal
serving enables supported paths automatically.
