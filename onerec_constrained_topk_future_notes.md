# OneRec Constrained Top-K Future Notes

## Why This Exists

This note records optimization ideas that were evaluated during the
`onerec-constrained-topk-perf` task but were not merged as production code for
the original `3B_git + beam256` scenario.

The point is not that these ideas are permanently bad. The point is that their
value depends on shape, beam width, constraint-table size, and end-to-end
critical path. Revisit them when those conditions change.

## Current Retained Baseline

For the original `3B_git` model and supported fused range `top_k <= 256`, the
retained custom-op path includes:

- wrapper guard: `top_k > 256` falls back to composite;
- external `AscendC::TPipe`;
- vector `ReduceSum` for logsumexp block accumulation;
- one-pass candidate scan;
- block-sort topK for large-degree rows;
- BF16 logit cache block size `64`.

The retained fused-op profile improvement was:

```text
RecConstrainedTopK avg: 562.733us -> 255.786us
improvement: 54.55%
```

This did not prove a stable large end-to-end QPS win. It is an opt-in
maxK<=256 fused path, not a general solution for larger beam widths.

## Why P1-P5 Were Low ROI For 3B_git + Beam256

| Item | Idea | Why it did not merge |
| --- | --- | --- |
| P1 candidate-plan overlap | Precompute sparse mask plan `{table_kind, begin, degree}` on a side stream | Profile did not prove candidate-plan lookup itself was material. Step0 range lookup is trivial, but step0 was still slow, pointing to gather/logsumexp/topK instead. |
| P2 step0 multi-core split | Split `rows=1, candidate=8023` across AIV cores | Kernel-local step0 improved, but perf-clean QPS/p99 regressed after `SyncAll`, workspace merge, and resource contention. |
| P3 compact merge | Compact block-local topK results, then do a second merge/sort | Correct but throughput regressed. Extra copy and second sort outweighed scalar-heap reduction. |
| P4 no-temperature fast path | Skip temperature read/divide when effective temperature is `1.0` | Correct but profile win was noise-level. It added a second temperature contract without performance payoff. |
| P5 candidate table layout | Reorder or prepack constraint tables for locality | For `3B_git`, step0/step1 candidate IDs were already local; step2 was sparse but low-degree. Fixing it would require token/code assignment or model-artifact changes. |

## What Changes With 3B_pyby + Beam384/512

New scenario:

```text
model: /export/home/maxiaolong/models/moe_model/3B_pyby
constraint file: beam_search_filter.bin
size: 8,814,311,028 bytes, about 8.21 GiB
business beam: 512
target beams to study: 384 and 512
reported throughput drop: about 18 QPS -> 1.5 QPS
```

This is a different performance regime. The old P1-P5 conclusions must be
revalidated because the constraint table is much larger and the requested beam
exceeds the current fused kernel contract.

Important file-format warning:

```text
8,814,311,028 % 20 = 8
```

The old parser assumed 20-byte records (`int64 item_id + 3 * int32 token`).
`3B_pyby` does not divide cleanly by that size, so future locality/degree tools
must first detect the real format or account for a header/trailer.

## New Priority Order

### 1. End-to-end profiler first

Do not start by editing the kernel. First prove where the new 18 -> 1.5 QPS
drop happens:

- model forward;
- constrained composite path;
- custom fused fallback boundary;
- host-side mask/table processing;
- H2D/D2H;
- response serialization for large output width;
- CPU thread contention.

Historically, beam384/512 fell back to composite because the fused contract was
`top_k <= 256`. That is no longer true after the P6 maxK512 work: the current
retained fused contract is `top_k <= 512`, and `top_k > 512` still falls back to
composite. Future profiler work must therefore distinguish the old composite
baseline from the new maxK512 fused baseline.

### 2. P6 true maxK=512 fused path

P6 has been implemented and retained. A real maxK=512 fused kernel covers
beam384 because `384 <= 512`.

This was not a wrapper-only change. It required:

- `kRecConstrainedTopKMaxK >= 512`;
- larger output token/logprob UB buffers;
- reviewed top value/token storage for `K=384/512`;
- revised block-sort/merge strategy;
- wrapper gating for `top_k <= 512`;
- beam384 and beam512 composite-vs-fused correctness tests;
- perf-clean A/B separate from old maxK<=256 numbers.

Validated result on `/export/home/maxiaolong/models/moe_model/3B_pyby`:

| Beam | Concurrency | QPS delta | p99 delta |
| --- | ---: | ---: | ---: |
| 384 | 1 | +27.86% | -23.45% |
| 384 | 2 | +41.95% | -30.19% |
| 512 | 1 | +33.81% | -24.52% |
| 512 | 2 | +51.59% | -32.29% |

Correctness compare had `failed_requests=0`; warnings were only equal-score
`tie-order` with matching logprobs.

### 3. Re-evaluate P1-P5 under new shapes

| Item | Re-evaluate when |
| --- | --- |
| P1 candidate-plan overlap | Range lookup or host/table lookup becomes a visible critical-path component in profiler. |
| P2 step0 multi-core | Step0 first-token degree is much larger than before and still dominates after P6. |
| P3 merge algorithm | `K=384/512` makes scalar heap or block merge material. This likely belongs inside P6 design, not the old compact-merge prototype. |
| P4 no-temperature | Only if profiler shows temperature division/read is material. This is unlikely to be first priority. |
| P5 layout/prepack | Candidate IDs are random or table access dominates; requires model artifact compatibility review. |

## Post-P6 Follow-up Status

The low-risk P3 block-size experiment has already been tried:

```text
kRecConstrainedTopKSortBlockElements: 2048 -> 4096
```

It passed full custom-op correctness (`26 passed`) but regressed beam512
perf-clean:

| Case | P6 2048 baseline | 4096 experiment | Decision |
| --- | --- | --- | --- |
| beam512 c1 | 19.562 QPS / p99 55.927ms | 15.435 QPS / p99 60.224ms | reject |
| beam512 c2 | 27.709 QPS / p99 79.025ms | 21.974 QPS / p99 84.786ms | reject |

Do not retry this as a production patch unless a new profiler shows the larger
sort block changes a different shape favorably.

## Post-P6 Follow-up Plan

The next useful optimization is not another constant tweak. The only
evidence-backed mainline is a real P3 merge redesign inside
`RecConstrainedTopK`.

### Priority 1: P3 Merge Redesign

Goal: reduce the scalar-heavy merge after block-local topK, especially for
`top_k=384/512`.

Why this is first:

- P6 already removed the old composite `SearchSorted + large GatherV3 + TopKV2`
  path.
- The remaining fused kernel still uses existing one-pass/block-sort/scalar
  merge logic.
- The `2048 -> 4096` block-size experiment regressed, so the next attempt must
  change the merge algorithm, not only the block size.

Required design:

- Keep block-local topK as the producer stage.
- Add a two-stage merge that keeps block topK candidates in UB when possible.
- Prefer vector compare/select merge over scalar heap maintenance.
- Use the current scalar heap as the fallback for unsupported degree/topK/UB
  shapes.
- Keep output token/logprob order and tie handling compatible with the
  composite oracle.

Before coding:

- Capture or reuse a P6 fused profile that shows `RecConstrainedTopK` scalar
  time is material for beam384/512.
- Identify exact row/degree/topK shapes that enter the merge path.
- Estimate UB usage for block topK buffers, merge temp buffers, output tokens,
  output logprobs, and candidate logits under `K=512`.
- Define a development-only guard for the prototype so the current P6 path
  remains the default until perf-clean passes.

Validation gate:

- Full custom-op correctness for `top_k=128/256/384/512`.
- Temperature coverage: no temperature plus scalar or per-row temperature.
- HTTP/debug composite-vs-fused compare for beam384 and beam512.
- Profiler shows lower `RecConstrainedTopK` total/avg and lower scalar
  contribution, without offsetting MTE/UB traffic.
- Perf-clean must improve QPS or p99 at beam512 c1/c2. If only profiler
  improves but end-to-end regresses, reject the patch.

### Deferred Experiments

These are not part of the next code patch. Each needs its own profiler trigger
and rollback plan.

| Direction | Why deferred | Reopen trigger |
| --- | --- | --- |
| Step0 multi-core split | Prior prototype improved local `rows=1` kernel time but regressed perf-clean because `SyncAll`, workspace merge, and resource contention erased the win. | Reopen only if post-P3 profile shows step0 is still a dominant p99 tail. |
| Candidate-plan side-stream overlap | Candidate range lookup is only a subset of fused work; step0 is slow even though range lookup is trivial. Side-stream work can steal AIV resource from forward. | Reopen only if stage attribution proves range lookup itself is material and a plan-only kernel is cheap. |
| Constraint-table layout/prepack | This changes model artifact format or preprocessing, not just kernel code. The old 3B_git locality report also showed many candidates are already sorted/local. | Reopen only after parsing the 3B_pyby extended 8.21 GiB filter format and proving locality/table access dominates. |
| `top_k > 512` support | This is a new maxK contract with UB/output/tiling changes, not an optimization of current P6. | Reopen only with a real business beam requirement above 512 and a fresh maxK capacity design. |

The rule is strict: do not combine these with P3 merge redesign. If P3 fails,
record that result and then pick exactly one deferred experiment with a clear
profile trigger.

## Measurement Gotchas

- `start_rec_git.sh` currently hardcodes `MODEL_PATH=.../3B_git`. Use a
  task-local run script or patch/override strategy that proves
  `/export/home/maxiaolong/models/moe_model/3B_pyby` is actually loaded.
- `benchmark_mxl_http_git.sh` hardcodes `--num-return-sequences 256` before
  passing extra args. Python `argparse` normally uses the last repeated value,
  but every beam384/512 run must include a dry-run proving
  `num_return_sequences=384` or `512`.
- Do not compare profiler/debug QPS with perf-clean QPS.
- Do not trust a beam384/512 "fused" run unless logs prove there is no
  `top_k exceeds fused kernel max` fallback.
- Record cold-start and steady-state separately. Loading an 8.21 GiB constraint
  table may dominate startup but not per-request latency.
