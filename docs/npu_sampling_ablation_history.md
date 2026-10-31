# NPU sampling: historical validation appendix

This appendix preserves earlier validation stages and their measured executable
identities. Counts and timings describe those stages, not the current working
version. See [the main report](npu_sampling_ablation.md) for the current design,
validation status, representative results, and reproduction commands.

## Correctness and regression checks

The original measured build passed the following checks:

- `sampler_test`: 65 tests pass, including NPU categorical support/distribution,
  vocabulary tails and padding, FP32/BF16/FP16, fixed-seed isolation, mixed
  unseeded RNG, greedy preservation, native filter ties, raw selected logprobs,
  and dense-versus-sparse rejection/recovery.
- `slot_buffer_test`: 51 tests pass; packed/shared input: 18 tests pass;
  ExecutionConfig: seven tests pass. These cover propagation/reset of the two
  new host-derived sampling controls and serialization schema 10.
- Existing block-worker/recurrent task-pipeline tests: 23 pass. These do not replace
  a real-weight task-pipeline E2E run.
- The eight rounded-top-logprob verifier cases were rerun at 192 x 65536
  logits elements, beyond D's activation threshold; all pass. This rerun is a
  subset of the 65 sampler tests, not eight additional unique tests.
- An additional fixed-logits allocation/repeatability probe completed 600
  calls over batch sizes 1/49/128, FP32/BF16 and V=248320 without a token-ID
  mismatch. This tests the sampler independently of serving/model variation.

Test logs are retained under `/tmp/npu_sampling_final_extra_tests.log`,
`/tmp/npu_sampling_acceptance_pipeline.log`, `/tmp/npu_sampling_shared_memory_test.log`,
`/tmp/npu_sampling_config_check.log`,
`/tmp/npu_sampling_final_tie_regression.log` and
`/tmp/npu_sampling_repeatability_probe.log`; the final manifest records their
identities. Copies are also saved under `local/npu_sampling/validation/`.
Source formatting uses clang-format 20.1.6 and Python lint uses
the repository's Ruff configuration.

The original submission review passed 164 regression cases: the original 161 plus
three additional block-worker option cases. Its combined log is
`local/npu_sampling/validation/npu_sampling_review_regression.log`.
It also checks the benchmark controller against a live owned child process
under normal Python and `python -O`.
That historical probe exercises real process stopping, all six memory writes/readbacks,
PID preservation and invalid-mask rejection without using an NPU. Controller
writes and measurement checks use explicit runtime validation, so Python's
optimization mode cannot remove them.

### Automatic-dispatch follow-up validation

The revised production build passes 160 targeted tests: sampler 35, sampling
controls 24, filter masks 3, slot buffers 51, packed/shared inputs 18,
execution config 6 and recurrent/block-worker pipeline 23. Production runs only
the automatic verifier policy; the test-only build retains the normalization
and verification factorial combinations. The old configuration round-trip
case is removed with the six options. Independent Torch references check
raw selected logprobs and rounded top-token ties.

The production executable has no ablation symbol, and its help has none of the
six removed options. Four-device Qwen3.8-27B smoke coverage passes random B=8,
mixed B=32, selected logprobs B=128, top-5 B=8 and greedy B=8: all 184 requests
return the requested 16 completion tokens. This is a functional smoke test,
not a new end-to-end performance acceptance dataset.

Logs, command lines and production hashes are retained under
`local/npu_sampling/automatic_validation/`. The controller also passes real
PIE executable symbol lookup and memory-write/readback tests under normal
Python and `python -O`; production executables reject that controller.

The instrumented build passes all 65 sampler cases, including the D/G
factorial combinations. A new three-repeat, interleaved native cross-check
confirms the revised test control selects the measured paths:

| Workload | Baseline us | Optimized us | Speedup |
| --- | ---: | ---: | ---: |
| Ordinary random, BF16, B=32, V=248320, A+B+C | 4435.83 | 1151.78 | 3.85x |
| Ordinary mixed 25% random, BF16, B=32, V=248320, A+B+C | 44659.83 | 986.73 | 45.26x |
| Verification, FP32, B=32, S=7, K=16, D+E+G | 7265.51 | 2684.52 | 2.71x |

The new test controller also passes a real-weight DFlash2 smoke in one
resident four-rank group: masks 63 and 0, random B=1, 32 completion tokens.
All four PID/readback checks pass. This verifies switching and sparse proposal
transport; these two short windows are not added to the 400-workload performance
acceptance dataset. The build is restored to production afterward; the sampler
binary matches its validated production SHA, and the relinked service is
checked with the same functional smoke.

The verification allocator peak remains 636.57 MiB versus 424.43 MiB
(-33.3%). Its complete eight-way D/E/G factorial and raw samples are in
local/npu_sampling/automatic_validation/instrument_native.json. These are
new measurements of the test build; the original tables below keep their
original build identities.

### Previous abstraction validation (3b70f45)

The previous revision encapsulated deterministic, dense and sparse q in a
value type. Its measurements below are retained as historical validation;
the current virtual interface and stable normalization are covered in the
next section. Unseeded verifier sampling uses the public categorical entry
point, retaining the NPU inverse-CDF dispatch and existing Gumbel primitive.

The refactor passes 92 production regression cases (37 sampling, 29 CPU
controls/distribution, three masks and 23 pipeline cases) and 67 instrumented
sampling cases. Production has no ablation symbol. A four-device production
model smoke passes all 184 requests across random, mixed, selected-logprob,
top-5 and greedy modes.

Five-repeat native checks preserve the optimized B=32 results: ordinary
random 1150.92 us, ordinary mixed 981.14 us, and D+E+G verification 2680.74 us.
Against the published implementation retested in the same session, these
change by -0.07%, +0.44% and +0.32%, respectively; allocator peaks are unchanged.
Each executable interleaves its ablation masks in one process. A separate
published/refactored/refactored/published B=1 check collects 14 samples per
version and mask: D+E changes by -0.45%, and D+E+G by -2.29%. It does not
reproduce the initial small-batch slowdown seen in separate-process medians.

The serving check switches masks in one resident four-rank group per model,
using physical devices 4–7, three repeats and 64 output tokens. Ordinary uses
masks 0/7; DFlash2 uses 0/24/56/63. All 576 ordinary and 864 DFlash2 requests
complete successfully. The table shows medians for baseline versus all
applicable optimizations, without replacing the historical acceptance matrix.

| Workload | Baseline tokens/s | Optimized tokens/s | TPOT ms, baseline → optimized |
| --- | ---: | ---: | ---: |
| Ordinary random, B=32 | 836.51 | 932.04 | 35.15 → 31.35 |
| Ordinary mixed, B=32 | 410.22 | 946.55 | 75.09 → 30.87 |
| Ordinary greedy, B=32 | 975.41 | 975.70 | 29.94 → 29.91 |
| DFlash2 random, B=32 | 570.62 | 587.18 | 45.68 → 43.81 |
| DFlash2 mixed, B=32 | 208.32 | 213.74 | 141.40 → 138.38 |
| DFlash2 greedy, B=8 | 228.70 | 227.65 | 26.71 → 25.55 |

DFlash2 combinations remain sensitive to acceptance and scheduling variation:
D+E random throughput is 614.03 tokens/s, exceeding all-on in this short run.
Greedy is a control, not a claimed speedup: its D+E+G-only TPOT is 31.22 ms
versus baseline 26.71 ms, although throughput changes by -1.19%; all-on TPOT
is 25.55 ms. These variable control results are retained in the raw records.

Build logs, binary hashes, native repeats, switching readbacks and all serving
results are under local/npu_sampling/review_followup/.

### Virtual distribution and stable normalization validation

The measurements and regression counts in this subsection precede the
point-mass acceptance refinement documented below.

The revision after 3b70f45 replaces the closed distribution variant with a
three-operation virtual interface: normalization, in-place acceptance
correction, and residual correction. Dense and sparse implementations own
their probability tensors; point-mass distributions need only metadata because
q is concentrated at the proposal tokens. A move-only proposal owns its
distribution and validates compatibility at construction. Producers fill buffers before publishing the
proposal; MTP passes grammar-invalid positions separately to verification.
All block-draft paths retain one proposal owner until slot retirement.

Compact normalization now returns the row maximum and the shifted
log-sum-exp separately. Selected scores use (logit - maximum) - log_sum,
avoiding cancellation when logits have a large common offset. For example,
uniform BF16 logits near 1e8 previously returned -8 instead of approximately
-11.090355 at V=65536. Regression tests cover ordinary and speculative
logprobs, large offsets, FP16/BF16 inputs, and top-logprob ties. Greedy
verification with top-logprobs skips the unused compact reduction.

Production regression coverage comprises 257 cases: 43 sampler, 30 CPU
controls/distribution, three filter-mask, 18 packed-input, 51 slot-buffer,
six execution-config, and 106 pipeline cases. The instrumented sampler
suite covers 85 cases. The final production smoke completes 184 requests
across random, mixed, selected-logprob, top-5, and greedy modes. Production
is restored with no ablation symbol. Added checks include a CPU FP64 CDF oracle over
15 dtype/vocabulary combinations, nonuniform and masked support, tail
and padded layouts, grammar rejection without proposal mutation, retained
asynchronous ownership, and completed DSpark Markov proposals.

Native measurements use five interleaved repetitions in one process,
V=248320, and the same input restoration/synchronization method as above.
All ordinary rows below use BF16; verification uses FP32 with S=7/K=16.

| Native workload | Latency, baseline → optimized (us) | Extra allocator peak, baseline → optimized (MiB) |
| --- | ---: | ---: |
| Ordinary random, B=32, A+B+C | 4369.84 → 1142.02 | — |
| Ordinary mixed, B=32, A+B+C | 45004.62 → 967.23 | — |
| Ordinary selected logprobs, B=128, A+C+D | 15335.44 → 3544.14 | 363.75 → 76.63 |
| DFlash2 verification, B=32, D+E+G | 7234.22 → 2695.30 | 636.57 → 424.43 |

Joint ablations remain necessary. For selected logprobs, A+C takes 3579.09 us
and peaks at 283.23 MiB; adding D takes 3544.14 us and peaks at 76.63 MiB.
D alone takes 15632.08 us, so its primary benefit is memory, not a standalone
latency gain. Verification E, D+E, and D+E+G take 5112.25, 4668.67, and
2695.30 us respectively. At B=1, E takes 1303.58 us versus 1237.90 us
baseline (+5.3%), while peak allocation drops 34.01 → 27.38 MiB (-19.5%).
The full 28-case native matrix and individual repetitions are retained.

Serving uses the same target/draft pair and four physical devices 4–7.
Every comparison switches masks within one resident four-rank group;
all 63 switches preserve worker PIDs and pass mask readback. Three
repetitions with 64 output tokens complete 2592 measured requests:
576 ordinary, 1152 selected-logprob, and 864 DFlash2 requests.

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

Artifacts are under local/npu_sampling/review_revision/. The measured
instrumented service SHA-256 is
b887d9e0310fa0903fd29f5f8511ad0d5676946f27aeae31e7427d83c8ea56ba.
The final guard cleanup affects only greedy verification with top-logprobs,
which is absent from these timing workloads; its regression tests are rerun
in both builds. Source and executable hashes distinguish timing artifacts
from the final production validation. The historical matrices below retain
their original executable identities.

### Point-mass acceptance refinement

The interface now applies the acceptance correction in place at proposal
positions. Point-mass drafts have q = 1 there, so their correction performs no
device operation and stores no duplicate token tensor. Dense and sparse
implementations retain their normalized correction. The verifier still has no
representation branch, and negative sentinel positions remain independently
rejected. The point-mass constructor validates its own integer token dtype.

The previous generic log-probability query introduced zero-fill, comparison,
masking and subtraction operations. A same-process comparison against the
original point-mass verifier exposed a 2.6–3.8% small-batch latency regression.
The repaired production implementation was compared again with that baseline
in one executable on physical NPU 4: FP32 fixed logits, alternating variants,
seven repetitions and 60 calls per variant per repetition. Input restoration
is outside timing; each timed call includes completion synchronization.
The table reports medians of per-repetition means.

| B / draft steps / vocabulary | Original baseline (us) | Repaired (us) | Change |
| --- | ---: | ---: | ---: |
| 1 / 1 / 49152 | 1082.28 | 1077.48 | -0.44% |
| 1 / 7 / 49152 | 1038.74 | 1043.29 | +0.44% |
| 1 / 7 / 248320 | 1146.86 | 1143.28 | -0.31% |
| 2 / 7 / 248320 | 1102.53 | 1095.67 | -0.62% |

These results show baseline parity in the tested cases; they do not establish
a new speedup. The native benchmark now includes point_mass_verification to
keep this shared interface cost visible. Its mask 0 also uses the current
interface, so structural refactors still require a revision comparison.

The instrumented retake includes 50 native measurements: ten cases with five
interleaved repetitions. Sparse verification at B=32/S=7/V=248320 remains
7239.41 us with mask 0 versus 2706.63 us with D+E+G; extra allocator peak
remains 636.57 versus 424.43 MiB. Small point-mass rows do not dispatch D/G
and their mask timings are controls, not independent optimization gains.

Final production validation passes 349 cases: 45 sampler, 31 CPU
controls/distribution, three filter-mask, 18 packed-input, 51 slot-buffer,
66 configuration, and 135 pipeline tests. Configuration and pipeline run
in full, beyond the prior 6/106-case subsets. The instrumented sampler
passes 93 cases, including point-mass versus one-hot dense verification on
CPU/NPU at batch sizes 1 and 8 with nonuniform target mass and sentinels.
Production is restored with BUILD_NPU_SAMPLING_ABLATION=OFF; neither the
service nor the production sampler test contains the mutable control symbol.

Artifacts, commands, sources and binary hashes for this refinement are under
local/npu_sampling/review_revision/pointmass_fix/. The isolated comparison
probe retains a token-only adapter for the original verifier implementation.
Real-model E2E was not rerun for this refinement; the serving results above
retain their original measured executable identities.

## Native ablations

Representative ordinary BF16 mixed batches, V=248320, 25% random rows:

| Rows | Baseline us | A us | B us | A+B us | A+B+C us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3073.2 | 2792.6 | 1169.3 | 879.3 | 876.3 |
| 8 | 8487.0 | — | 1318.3 | — | 894.6 |
| 32 | 45170.5 | 43930.5 | 2150.8 | 1012.9 | 984.4 |
| 128 | 135666.4 | — | 6305.0 | 2345.0 | 2077.4 |

These rows come from the matched ordinary and small-batch native runs.
The final normalization-source retake uses five interleaved repetitions:

| BF16 selected logprobs, B=128, V=248320 | Median us | Extra allocator peak MiB |
| --- | ---: | ---: |
| Baseline | 16408.8 | 363.8 |
| A | 3808.4 | 404.5 |
| D | 16186.4 | 303.1 |
| A+D | 3783.9 | 343.9 |
| A+C | 3577.6 | 283.2 |
| A+C+D | 3538.6 | 76.6 |

A alone can increase temporary memory by introducing a full probability tensor.
C removes that cost. D's main additional benefit is memory; its standalone
latency benefit is small. Top-5 requests retain the legacy normalization and
therefore do not get D's selected-logprob memory reduction.

Final verification retake, FP32 target logits, B=32, S=7, V=248320, K=16,
without reported logprobs:

| Variant | Median us | Extra allocator peak MiB |
| --- | ---: | ---: |
| Baseline | 7271.0 | 636.6 |
| D | 6861.1 | 636.6 |
| E | 5123.3 | 424.4 |
| D+E | 4648.8 | 424.4 |
| G | 5276.3 | 636.6 |
| D+G | 4836.5 | 636.6 |
| E+G | 3188.4 | 424.4 |
| D+E+G | 2680.6 | 424.4 |

The full factorial includes every single addition, pair, and leave-one-out from
D+E+G. With selected logprobs enabled, baseline is 7914.6 us and D+E+G is
2709.7 us. The sparse proposal removes approximately 212.2 MiB of dense q
storage at B=32, S=7. The benchmark includes proposal materialization, using
one dense scatter, conservatively cheaper than the real producer's per-step
scatters.

Sparse proposals also have a small-batch tradeoff. At B=1, S=7, V=248320,
the native E-only case takes 1335.0 us versus 1257.3 us for the baseline
(6.2% slower), while temporary allocator peak falls from 34.01 to 27.38 MiB
(19.5% lower). D and G are inactive at this size. The long-output serving
check below does not show a sustained regression, but this native latency
cost remains part of the feature's acceptance evidence; E is not claimed
to improve latency at every shape.

## Serving ablations

### Ordinary sampling

The final run completed 168 measured workloads (three paired repetitions)
using the same worker PIDs **184473, 184474, 184475, 184476** throughout.
All requests completed with the requested 64 output tokens. There was no
compilation or trace export during these timings. Each cell below is a
median across repetitions. Percentage changes are the median of the three
within-repeat ratios, rather than a ratio of separately aggregated medians.

| Workload | Concurrency | Mask | Baseline TPOT ms | Variant TPOT ms | Baseline tok/s | Variant tok/s | Paired TPOT change | Paired throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1 | 7 | 20.717 | 20.342 | 45.3 | 46.2 | -1.90% | +1.95% |
| random | 32 | 7 | 35.819 | 31.422 | 825.9 | 932.5 | -12.19% | +12.74% |
| random | 128 | 7 | 77.387 | 60.851 | 1553.6 | 1944.0 | -21.37% | +25.13% |
| mixed | 1 | 7 | 20.563 | 20.298 | 45.7 | 46.4 | -1.40% | +1.36% |
| mixed | 32 | 7 | 75.199 | 31.004 | 409.2 | 941.8 | -58.80% | +130.17% |
| mixed | 128 | 7 | 192.522 | 59.498 | 648.1 | 1981.4 | -69.13% | +205.07% |
| unseeded | 32 | 7 | 32.322 | 31.301 | 904.2 | 930.7 | -3.26% | +3.16% |
| logprobs | 128 | 13 | 77.866 | 61.202 | 1535.5 | 1921.1 | -21.40% | +25.11% |
| top5 | 128 | 13 | 78.856 | 62.763 | 1516.6 | 1873.2 | -20.43% | +23.52% |
| greedy | 32 | 7 | 30.082 | 30.162 | 962.5 | 955.4 | +0.21% | -0.74% |

Mixed concurrency 128 improves from 648.1 to 1981.4 tok/s (about 3.05x);
this large gain depends on skipping expensive greedy-row filtering. The
three paired TPOT ratios are 0.3084, 0.3087 and 0.3096. Random concurrency
128 improves throughput by about 25%. Greedy is a control: these features
are inactive, with +0.21% TPOT / -0.74% throughput in this run.

The concurrency-32 component comparison shows why joint measurements matter:

| Mask | Ordinary random TPOT ms | Ordinary mixed TPOT ms |
| --- | ---: | ---: |
| 0 | 35.819 | 75.199 |
| 1 | 31.379 | 73.711 |
| 2 | 35.822 | 32.542 |
| 3 | 31.513 | 31.124 |
| 5 | 31.451 | 73.645 |
| 7 | 31.422 | 31.004 |

A removes most seeded per-request overhead. B is inactive in an all-random
batch, but dominates the mixed-batch improvement. C contributes primarily
temporary-memory savings beyond A at these serving shapes. D alone on
selected logprobs has a paired TPOT ratio of 1.0013 and throughput ratio
of 0.9965; its additional acceptance benefit is the native memory reduction,
not a demonstrated standalone E2E latency gain. For top-5, D retains the
legacy normalization, as required for exact reported token IDs.

Full per-mask/per-repeat ratios, actual request usage and outputs remain in
`local/npu_sampling/same_process_ordinary_final.jsonl` and
`local/npu_sampling/e2e_summary.json`.

### DFlash2 ordinary execution path

The 64-token matrix completed 189 measured workloads, three paired repetitions
per variant, with worker PIDs **177095, 177096, 177097, 177098** unchanged.
It includes the full D/E/G factorial, A+B+C prefill controls (mask 7), and
all six options together (mask 63). Sparse producer and consumer are switched
as one E feature. Every request returned its requested output-token count.

| Workload | Concurrency | Baseline TPOT ms | Mask 63 TPOT ms | Baseline tok/s | Mask 63 tok/s | Paired TPOT change | Paired throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1 | 19.446 | 14.974 | 48.0 | 61.0 | -10.82% | +11.03% |
| random | 8 | 25.521 | 25.633 | 247.8 | 252.1 | +1.78% | +2.86% |
| random | 32 | 44.585 | 42.692 | 599.5 | 616.7 | -6.74% | +3.68% |
| mixed | 1 | 20.935 | 20.008 | 44.9 | 46.9 | -1.33% | +1.35% |
| mixed | 8 | 51.692 | 48.000 | 136.2 | 145.2 | -10.62% | +6.64% |
| mixed | 32 | 141.182 | 133.779 | 207.0 | 220.6 | -6.61% | +6.30% |

The concurrency-32 factorial gives the following serving view:

| Mask | Random TPOT ms | Random paired throughput change | Mixed TPOT ms | Mixed paired throughput change |
| --- | ---: | ---: | ---: | ---: |
| 0 | 44.585 | +0.00% | 141.182 | +0.00% |
| 8 | 44.857 | -2.71% | 136.539 | +4.18% |
| 16 | 44.114 | -0.58% | 138.332 | +5.71% |
| 24 | 42.505 | +3.06% | 135.328 | +4.49% |
| 32 | 45.945 | +0.54% | 134.790 | +4.94% |
| 40 | 43.746 | -2.23% | 140.363 | -0.82% |
| 48 | 44.426 | +0.70% | 137.918 | +5.51% |
| 56 | 44.301 | -3.21% | 137.464 | +1.04% |
| 7 | 44.384 | -2.61% | 140.623 | +2.76% |
| 63 | 42.692 | +3.68% | 133.779 | +6.30% |

The native factorial establishes reductions in verifier latency/memory with
fixed inputs. The short serving matrix supports the complete group at
concurrency 32 (roughly +3.7% random / +6.3% mixed throughput), but does not
establish a positive E2E latency benefit for every isolated feature. For
example, mask 56 has -3.2% random throughput in this short matrix despite a
lower median TPOT; it is not presented as an unconditional serving speedup.

DFlash2 request seeds are unsupported and acceptance lengths differ between
variants. At concurrency one, D and G are gated off, so masks 16 and 56
execute the same optimized path. Nevertheless, their short-request TPOT
medians are 18.979 and 21.726 ms: three short repetitions are too noisy to
infer their relative speed. The greedy short control also varies (mask 63
paired TPOT +9.1%, throughput -1.2%), with different committed-prefix lengths.
The long random and greedy controls below check these apparent regressions.

Raw data: `local/npu_sampling/same_process_dflash.jsonl`. The report does not
equate stochastic acceptance-length changes with kernel speedup.

### Long-output DFlash2 check

A separate resident four-rank run completed 28 workloads: concurrency one,
512 output tokens, seven paired repetitions of masks 0/16/56/63. Worker PIDs
**187506, 187507, 187508, 187509** remain unchanged. This extends the noisy
64-token single-request comparison; no CPU build/export overlaps the timing.

| Mask | TPOT ms | Tok/s | Paired TPOT change | Paired throughput change | Median committed tokens / draft | Paired TPOT ratio range |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 19.267 | 51.44 | +0.00% | +0.00% | 2.769 | 1.000–1.000 |
| 16 | 18.093 | 54.74 | -4.70% | +4.88% | 2.960 | 0.899–1.037 |
| 56 | 17.665 | 56.05 | -8.30% | +8.96% | 3.000 | 0.726–0.982 |
| 63 | 18.070 | 54.81 | -6.34% | +6.67% | 2.909 | 0.879–1.021 |

No sustained small-batch regression is observed in this retake. However,
acceptance-length variation remains visible: masks 16 and 56 have the same
active implementation at this shape, and their different aggregate times
must not be attributed to D/G, which are gated off. The per-feature memory
and fixed-input latency claims come from the native factorial. Mask 63
improves paired throughput by 6.67% here; this does not establish a guaranteed
speedup for other prompts or acceptance distributions.

Raw data: `local/npu_sampling/same_process_dflash_long.jsonl`.

### Long-output greedy control

The last control run uses concurrency eight, 512 output tokens, five paired
repetitions, and unchanged worker PIDs 190176, 190177, 190178, 190179.

| Mask | TPOT ms | Tok/s | Paired TPOT change | Paired throughput change | Paired TPOT ratio range |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 22.233 | 332.52 | +0.00% | +0.00% | 1.000–1.000 |
| 56 | 21.704 | 337.40 | -0.69% | +3.22% | 0.905–1.118 |
| 63 | 22.221 | 333.23 | +3.16% | +0.04% | 0.920–1.071 |

The mask-63 throughput ratio is 1.0004, with essentially unchanged separate
median TPOT (22.233 versus 22.221 ms). The median paired TPOT ratio is 1.0316,
and individual paired ratios range from 0.920 to 1.071. This control does not
establish a systematic throughput regression or a greedy latency improvement;
acceptance/scheduling variability remains larger than the sampling effect.
It is included rather than dropping the short greedy control that looked
worse. Raw data: `local/npu_sampling/same_process_dflash_greedy_long.jsonl`.

### Acceptance scope

- A+B+C passes the tested ordinary sampling matrix, including mixed/random,
  seeded/unseeded and greedy controls. C mainly removes the temporary
  probability tensor beyond the batched A path.
- D is useful for selected-logprob temporary memory and jointly with other
  verifier features; a standalone serving-latency gain is not established.
  Top-logprob requests intentionally retain the original normalization.
- D+E+G has the strongest fixed-input verifier result (63.1% less native
  latency, 33.3% less temporary allocator peak at the representative shape).
  The all-options serving group has modest gains for the tested target/draft
  pair, with stochastic uncertainty documented above. Isolated/group results
  are not all positive in the short E2E matrix; no blanket per-feature E2E
  speedup is claimed.
- F (synthetic full-vocabulary top-k in top-p-only filtering) is rejected
  because it changes tie behavior. Full filtering fusion is excluded.
- All six optimizations apply automatically behind the existing input guards;
  production exposes no feature switches. Real-weight task-pipeline E2E
  is outside this user-approved 910B2 acceptance scope and does not block
  completion. No performance claim is made for that mode.

The accepted serving datasets contain 400 measured workloads in total:
168 ordinary, 189 short DFlash2, 28 long random DFlash2, and 15 long greedy
controls. Earlier restart-per-variant runs and the overlapping-build pilot
are excluded. The implementation is submitted in [PR #2603](https://github.com/xLLM-AI/xllm/pull/2603).

## Device trace evidence

The existing online Ascend profiling API is used after timing, followed by
explicit `msprof --export=on` for **every** PROF directory. CANN 9 rejects
aclprof APIs when `PROFILING_MODE=dynamic`; that attach-mode variable is
removed for this online-API capture method. The initial failed capture is
retained separately.

Six ordinary capture windows cover four ranks/devices each: 24 complete
exported timelines. All contain timestamped events; paths, byte counts,
SHA-256 hashes, device IDs, task types and operator totals are in
`local/npu_sampling/serving_repeatability/profile_ordinary/trace_index.json`.
No browser/Perfetto UI inspection was performed.

Rank 0 / physical device 4, 32-token capture workloads:

| Window | Baseline | Optimized | Observation |
| --- | ---: | ---: | --- |
| Random, concurrency 32: per-request uniform kernels | 1024 | 0 | Replaced by 33 fused categorical invocations. |
| Random: synchronous memcpy API calls | 101 | 35 | Copy directions are not present in these timeline API events. |
| Random: asynchronous memcpy API calls | 1371 | 347 | Includes generator-state traffic; these are whole-window counts. |
| Mixed, concurrency 32: ApplyTopKTopP kernel total ms | 1355.34 | 1.19 | Random-row compaction avoids the expensive mixed control path. |
| Selected logprobs, concurrency 128: full log-softmax calls | 34 | 4 | The compact path handles 30 sufficiently large batches; small batches retain the existing path. |

Counts are observed within their own capture windows and include prefill.
They explain the implementation; user-visible speedups are determined from
separate measurements with profiling disabled. Summed kernel durations are
not wall-clock latency.

DFlash2 adds three capture windows (masks 0/56/63), again on all four
ranks: **12 additional complete timelines**, for 36 exported timelines
overall. The DFlash2 trace index is
`local/npu_sampling/accepted_dflash/profile_dflash/trace_index.json`.
On device 4, baseline versus all options:

- The native ScatterElementsV2 kernel count falls from 98 to 14. Sparse q
  construction removes per-draft-step full-vocabulary scatters; residual
  support correction still scatters once per verifier step.
- The optimized window contains 13 compact BF16 row-LSE calls and 13 FP32
  fused recovery draws. BF16 fused draws cover bonus/prefill selection.
- Native LogSumExp ReduceMax totals fall from 4956.6 to 182.5 us; the new
  compact target LSE contributes 5513.4 us itself. Remaining native LSE
  operations normalize the small candidate distribution. These totals
  describe changed work, rather than a standalone per-request speedup.
- Uniform kernel counts are both 58, but their total duration drops from
  17904.3 to 256.1 us as vocabulary-sized noise is replaced by compact
  uniform controls where supported. Acceptance and draft-selector draws
  still remain; this is not a claim that all RNG launches disappear.

The different windows execute different stochastic verification histories.
Raw counts/durations are retained, and no whole-window count is normalized
into a wall-clock speedup. Profiling is disabled for the serving timing
tables. No browser/Perfetto UI inspection was performed for either set.


## Original artifact identities

Native raw files are under `local/npu_sampling/`:

- `acceptance_native_joint.json`: final normalization/verification factorial retake.
- `final_native_small_and_joint.json`: final small-batch filter gate and joint controls.
- `final_native_ordinary.json`, `final_native_filter49k.json`,
  `final_native_logprobs.json`, `final_native_sparse.json`: wider shape/mode coverage.
- `acceptance_manifest.json`: source/native identities and experiment notes.

Earlier restart-per-variant results, incomplete-model startup attempts, and the
brief CPU-build-overlap ordinary pilot repeat are exploratory. They are not
used as the final serving acceptance data.

The final `acceptance_manifest.json` preserves per-run executable/source SHA-256,
worker PIDs, complete-request counts, test logs and trace-index identities.
Native-stage manifests remain separate: older wide-shape measurements are not
relabeled as the final serving executable. The matching native benchmark hash
for the final joint retake is `5f358b7e23fea423cc887c975a622441913d34686d453ae28769279f30d36a26`.
Final serving executable SHA-256: `22d7d6eece90aeef313a282caa5070ee4f5e37fae40c02b47b13ca464504014e`.
