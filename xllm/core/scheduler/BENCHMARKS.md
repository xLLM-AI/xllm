# Scheduler CPU benchmarks

`continuous_scheduler_benchmark` uses the production scheduler and a real
`BlockManagerPool`, with a fake engine and tokenizer. It measures host scheduling
cost, not accelerator inference latency or serving throughput. Overlap is disabled
in this fixture; these results do not measure the overlap pipeline.

| Benchmark | Timed work | Parameters |
| --- | --- | --- |
| `BM_Scheduler_AddRequest` | Admission and enqueue | Request count |
| `BM_Scheduler_PrepareBatch_Prefill` | One prefill scheduling round | Request count |
| `BM_Scheduler_PrepareBatch_Decode` | One decode scheduling round | Request count |
| `BM_Scheduler_ProcessBatchOutput` | Metrics and response dispatch | Request count, streaming |
| `BM_Scheduler_DecodeWindow` | 16 consecutive decode scheduling rounds | Request count, sequences per request |
| `BM_Scheduler_CollectFinished` | Completed-request collection, KV release, response dispatch, and scheduling surviving decodes | Request count, completion percentage |

Request construction, admission setup, simulated token generation, checks, and
cleanup are excluded from the last two benchmarks. Completion dispatch is timed;
waiting for response callbacks is excluded. Background response work can still
contend with the benchmark thread. The tokenizer returns empty output, so this is
not a detokenization benchmark.

`DecodeWindow` expands all sequences before prefill, then warms prefill and one
decode round before measurement. Requests
finish after the final measured round. Checks enforce the sequence count and a
one-token budget for every sequence. The reported time is **per 16-round window**;
divide it by `decode_steps_per_iteration` for average time per round. `items/s`
counts scheduled sequences across all 16 rounds. The other benchmarks report time
per call, with items counted as requests.

The fixture uses 512-token prompts, one DP group, FCFS, and the default batch mode.
The new cases use up to 256 requests and 4 sequences per request. They cover a
stable decode batch and completion churn, but not memory-pressure preemption,
PD transfer, latency prediction, or real beam-search execution.

## Build and compare

Use the repository's normal configured accelerator build environment:

```bash
python setup.py test --test-name continuous_scheduler_benchmark
```

From the directory containing the built executable, collect repeated JSON results:

```bash
./continuous_scheduler_benchmark \
  --benchmark_filter='BM_Scheduler_(PrepareBatch|DecodeWindow|CollectFinished|ProcessBatchOutput)' \
  --benchmark_min_time=0.5s \
  --benchmark_repetitions=7 \
  --benchmark_out=after.json \
  --benchmark_out_format=json
```

Build the baseline revision with **the same benchmark source** and collect
`before.json` using the same command, hardware, compiler, allocator, build type,
and CPU affinity. Copy only the benchmark harness to the baseline; keep its
production scheduler unchanged. Compare the median `cpu_time` for each matching
case, accounting for the JSON time unit. Improvement percentage is
`100 * (before - after) / before`; compare repetition variability before claiming
a regression or improvement. Do not compare the old one-round benchmark directly
with the 16-round window.

## Optimization under test

Prefill and decode previously constructed two candidate vectors for every
request. Their storage now lives for one scheduling call and is cleared between
requests. For N requests with one sequence each, this changes those scratch
allocations from 2N to 2 per call. Beam decode also reuses its active-sequence
buffer. Varying sequence counts can grow capacity; storage is released at the end
of the call, so it is not retained indefinitely by the scheduler.

The request filtering, budget checks, KV allocation, queue order, and output
insertion paths are unchanged. Request scanning and bookkeeping helpers also
borrow `shared_ptr` references instead of creating temporary owners; ownership
copies into running/completed/response containers remain where needed.

`MixedSequenceCountsPreserveBatchesAcrossDecodeAndCancellation` in
`scheduler_test` checks both prefill-first and decode-first policies with 1, 4,
and 2 sequences per request, then cancels the middle request. It verifies sequence
identity, budgets, completion, and KV release across scheduling rounds.

No full scheduler timing result is recorded for this change: the local macOS
environment has no configured CMake build, and the build command currently exits
with `ModuleNotFoundError: No module named 'setuptools'`. Allocation counts above
describe the changed code path, not measured end-to-end speedups.
