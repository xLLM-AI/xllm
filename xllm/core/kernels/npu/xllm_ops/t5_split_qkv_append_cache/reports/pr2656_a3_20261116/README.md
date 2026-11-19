# PR #2656: A3 validation, 2026-11-16

The actual standalone xLLM source port at
[`cbe5ac665188298435b25ce1b36a3ece252ef769`](https://github.com/wangzhao-11a/xllm/commit/cbe5ac665188298435b25ce1b36a3ece252ef769)
passed its full functional suite and six fixed performance runs: the initial
run followed by five fresh processes. Each run passed all ten frozen targets.
The final pipeline and every prerequisite phase returned zero. Subsequent
documentation commits do not change the tested implementation.

## Changes and preserved controls

The original PR head `4649db26897d128389140138e1821d14aa5fb5e0` is retained in
history. Follow-up changes are:

- `38a01af5`: port the standalone independent load/store DMA alignment checks.
- `0c51960e`: pass project include paths to CANN opbuild and device compilation.
- `f990af10`: register direct-launcher includes through the official CANN helper,
  with a configure-time guard on the actual device include property.
- `cbe5ac66`: restore the three baseline `at::empty` output allocations.

Standalone baselines remain unchanged: `c9decf212667337d518e3a1d48dc3e07cd5fef0e`,
`f52671207769bf5f334972c997743eb6aa6f91b3`, and the alignment-fix control
`a2013d56db4ff8367160b157335d2ff670aad411`.

The intermediate f990 port passed all functional gates but failed the first
fixed performance run, scoring 8/10. Its five-process stability stage was not
started. The preserved a201 binaries were then measured once on the same host,
device, private Python and current official CANN environment. No baseline was
rebuilt or overwritten.

| Exact runtime | Fixed cases | 007 FP16 (us) | 017 BF16 (us) |
| --- | ---: | ---: | ---: |
| f990 intermediate port | 8/10 | 81.54460 | 84.78920 |
| a201 current-condition control | 10/10 | 74.76045 | 76.34305 |
| cbe5 final port, initial | 10/10 | 69.76450 | 70.12780 |

The native FP16/BF16 kernel objects and ACLNN API library are byte-identical
between a201, f990 and cbe5. The f990-to-cbe5 source delta changes only the three
allocation factories. The control supports retaining the baseline allocation
path, but does not establish a root cause for the externally reported eight
latency levels. No address, clock or profiler attribution is claimed.

## Environment and validation

The assigned environment was container `248a9b8f1bbd`, Ascend910_9382 (A3),
card 1/chip 1/physical device 3, runtime selector 1 and logical device 0.
CANN was 9.1.0; driver 26.1.1; openEuler 24.03 LTS-SP3.
The preserved private environment used Python 3.12.13, PyTorch 2.10.0+cpu,
torch_npu 2.10.0.post4 (`5dd8ef3f9b375b5ae4a83538d5785754148c3302`),
NumPy 2.2.6 and ml_dtypes 0.6.0. `TASK_QUEUE_ENABLE` and
`T5_ACLNN_EXECUTOR_REUSE` were unset. Each deployment used its sole private OPP.

| Gate | Result |
| --- | --- |
| Exact archive, frozen fixtures, script encoding and environment probes | Passed |
| Actual kernel-header CPU semantics | 776/776; CPU only |
| CANN ACLNN package, direct launcher, binding import, C++ example | Passed |
| C++ FP16/BF16 calls and invalid ACLNN metadata | Passed; zero workspace, zero bit mismatches |
| Repeated output smoke checks | 8 groups, 20 repeats each |
| Public accuracy | 35/35 |
| Extended accuracy | 150/150 |
| Engineering | 15/15 |
| Memory | 20/20 |
| Existing separate allocating baseline comparison and coverage check | Passed |
| Initial fixed performance plus five fresh-process runs | 10/10 in each run |

The separate allocating baseline comparison has a different protocol; its
measurements are not substituted for the fixed-target results.

## Fixed performance stability

Every process used the unchanged `reference/benchmark.py`, cases, targets and
checker: 50 warmups, 200 calls per round and five rounds. The primary metric is
the median of the five per-round mean wall times, including output allocation
and API submission through synchronized completion. Output caching, thresholds,
case selection and timing were not changed. The five repeats were declared
before starting and configured to stop on the first failure; no replacement
runs were made.

| Fresh process | Passed cases | 007 FP16 (us), limit 75 | 017 BF16 (us), limit 77 |
| --- | ---: | ---: | ---: |
| 1 | 10/10 | 69.34305 | 69.60120 |
| 2 | 10/10 | 69.71395 | 70.13220 |
| 3 | 10/10 | 70.30915 | 70.18205 |
| 4 | 10/10 | 69.40145 | 70.24015 |
| 5 | 10/10 | 69.68195 | 69.75690 |

The worst individual large-case round across the initial and repeated runs was
70.37705 us for 007 and 70.47830 us for 017. The formal limits apply to each
process's median, not its individual rounds. Three whole-board idle checks
preceded each measured process. Final port and control checks verified unchanged
source/binary hashes, unchanged listening ports and cleared device processes.

## Evidence and reproduction

[summary.json](summary.json) contains exact values and identities.
[evidence.tar.gz](evidence.tar.gz) preserves original UTF-8 text evidence with
an `EVIDENCE_MANIFEST.json` mapping every member to its original path, size and
SHA256. It includes final and failed reports, control reports, raw logs, binary
hash manifests, environment/process inventories, phase exit receipts, the
predeclared stability plan, execution wrappers and supplemental CPU harness.
Compiled binaries and generated CMake probe trees are excluded. Original full
archive hashes remain in that manifest. [SHA256SUMS.txt](SHA256SUMS.txt) checks
the published report files.

Historical failed wrapper receipts under `candidate/.../v3/` are retained.
The final run's receipt is
`candidate/t5_pr2656_validation_20261116/pipeline_cbe5ac66.exit` (zero),
with its component phase receipts under `artifacts/pr2656_validation_v3/`.

To rebuild, check out the pinned runtime commit, retain the repository layout,
load the recorded CANN/private Python environment and follow the component
[build and validation instructions](../../README.md). Use a fresh source/build
directory and private OPP, confirm the assigned device mapping and idle board,
then run the frozen suite. The archive's `candidate/.../v3` wrappers record the
exact external validation sequence; its absolute deployment paths describe this
run and must be adapted to an explicitly selected fresh checkout.

These are component-level results. The xLLM top-level build/model execution path
and task-requested driver `25.5.t6.b001` were not validated. The public external
five-run failure report was not located, so this report retains the newly
measured controls without claiming to reproduce that original artifact.
