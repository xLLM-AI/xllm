<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# OPT-01 retained measurement summaries

These are original summaries from the NPU experiment on baseline
`8da1fc5a3937c4314206354bdc98ecd930921735`. The measured candidate's exact
source fingerprints are in `measured_source_manifest.json`. The PR is rebased
onto `2339aba64eaf98b07b77b143875cb72385eee76f`; fresh NPU tests of its migrated
RMSNorm reference API remain pending. Do not label these records as current-head
test results.

| File | Evidence and interpretation |
|---|---|
| `performance_summary.json` | Uninstrumented same-device A/B/A; exact 2048/2048 token checks, C1/C32, TTFT, TPOT and output throughput. |
| `accuracy_summary.json` | Full EvalScope GSM8K and MMBench runs; preserve original scores and counts, including differences. |
| `extended_summary.json` | Direct Attention comparisons, per-wave decoder NPU-event times, exact allocator peaks and separate HBM samples. C32 prefill values are summed event times per 32-request wave. |
| `profile_summary.json` | A/B profiler operation counts and task-time summaries for the same warmed image workload. Task sums are not end-to-end durations. |
| `kernel_checks.json` | Eager/fake and input-mutating ACLGraph checks on the measured source checkout. |
| `boundary_checks.json` | Actual NPU boundary and supported fallback executions on the measured source checkout. |
| `measured_source_manifest.json` | Baseline, branch and SHA-256 hashes of the original 13 measured files. |

Complete raw evaluation responses, profiler exports and tensor logs are retained
in the experiment archive outside this repository. These small summaries are
included for reviewer traceability; they are not substitutes for all raw traces.
No raw benchmark images, model weights, cloud connection details or credentials
are included.

Three measured waves on one environment do not establish statistical precision
non-inferiority, a portability guarantee, or stable decode-throughput gains.
