<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# T5 QKV split and cache append

This directory contains a standalone Ascend C operator project targeting
Ascend910_9382 and CANN 9.1.0. It builds an ACLNN custom operator package, a direct
kernel launcher, a thin PyTorch binding, and an independent C++ example. It is
not yet connected to the xLLM model execution path or its top-level build.

## Contract

`solution.run(inputs, attributes)` accepts contiguous FP16 or BF16 NPU tensors:

| Tensor | Shape |
| --- | --- |
| Packed QKV | `[B, 1, 3 * H * D]` |
| Past key / value | `[B, H, L, D]` |
| Query output | `[B, H, 1, D]` |
| Key / value output | `[B, H, L + 1, D]` |

The packed input stores all query heads, then all key heads, then all value heads
within each batch. Optional Python attributes `headNum` and `headDim` must match
the input shapes. The supported ranges are `1 <= B <= 64`, `1 <= H <= 128`,
`0 <= L <= 8192`, `1 <= D <= 256`, and `B * H * D * (2 * L + 3) <= 2**28`.

Every call allocates three independent outputs and leaves the inputs unchanged.
Data is copied as 16-bit words, including NaNs, infinities, and signed zero.
The explicit ACLNN workspace is zero. Contiguous storage-offset views and empty
past caches are supported; blocked formats and noncontiguous inputs are rejected.
The Python extension uses exceptions at its untrusted-input and CANN boundaries
so invalid input can be reported without terminating the calling process.

`solution.run_direct` uses the same kernel through the direct launcher.
`solution.prepare` retains shape, dtype, device and attribute metadata; each
invocation still supplies the input tensors, resolves the current stream and
allocates new outputs.

## Implementation

The kernel partitions the outputs into 32,768-element tiles and uses up to 48
vector cores, with two 64 KiB DMA buffers. It gathers cache fragments into each
core's own output interval. Load and store alignment are checked independently:
an unaligned source can use a padded load while its aligned output uses ordinary
DMA. Large outputs use a bounded cached key suffix and
bypass L2 for the remaining key and value output. The policy is specific to the
910_9382 working set and is not a performance claim for other devices.

The ACLNN entry caches one repeatable executor per calling thread. Its cache
contains descriptors and executor metadata, not output tensors or workspace.
Every submission refreshes all nonempty tensor addresses. Shape, dtype, device,
stride or stream changes replace the plan. Set `T5_ACLNN_EXECUTOR_REUSE=0` before
importing the extension for the explicit ordinary-submission control.

## Build

Run the following from this directory on an aarch64 Linux host with the assigned
CANN 9.1.0 environment loaded. The accepted original runtime used Python 3.12.13,
PyTorch 2.10.0+cpu, torch_npu 2.10.0.post4 (git prefix `5dd8ef3`), NumPy 2.2.6 and
ml_dtypes 0.6.0. `msopgen`, CMake and a C++ compiler must be on `PATH`.

```bash
export T5_OPP_ROOT="$HOME/.local/t5_qkv_opp"
bash scripts/build.sh
export ASCEND_CUSTOM_OPP_PATH="$T5_OPP_ROOT/vendors/t5qkv${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

The build creates `artifacts/*.run`, a private OPP installation,
`build/direct/lib/libt5_direct.so`, `_t5_qkv*.so`, `build/examples/t5_example`,
and `artifacts/build_manifest.json`. Existing `build/aclnn` directories are
preserved: use a fresh checkout for a rebuild. No global CANN installation is
modified. The `tools/` scripts use the xLLM shared logger, so retain the repository
directory layout when building this port.

The generated ACLNN API is
`aclnnT5SplitQkvAppendCacheGetWorkspaceSize` followed by
`aclnnT5SplitQkvAppendCache`. See `examples/aclnn_example.cpp` for descriptor
ownership, allocation, submission and output checks.

## Validation

Static checks and the existing CPU address-model check need no device:

```bash
python tools/preflight.py --static
python tests/cpu_address_model.py
```

The CPU model exercises 35 public shapes and 120 additional random shapes;
it does not execute or compile the Ascend C kernel.

After reviewing device occupancy, worker processes and listening ports, select
the assigned free device and run the full device checks:

```bash
export ASCEND_RT_VISIBLE_DEVICES=<assigned_device>
export T5_DEVICE_CONFIRMED=1
# Set this only when the evaluation owner has accepted the actual driver.
export T5_ACCEPTED_DRIVER=26.1.1
bash scripts/validate.sh
```

The task's requested driver is `25.5.t6.b001`; leave `T5_ACCEPTED_DRIVER` unset
to require that version. Results from driver 26.1.1 do not validate the requested
driver. The script runs C++ smoke calls, public/extended correctness, stream and
repeated-output checks, memory checks, then the frozen performance protocol.
It preserves prior reports. Performance uses 50 warmups and five rounds of 200
calls, including fresh output allocation and synchronized ACLNN completion.

## Provenance and current review status

The implementation was originally imported from the standalone runtime
[`f526712`](https://github.com/wangzhao-11a/t5-qkv-append-cache/commit/f52671207769bf5f334972c997743eb6aa6f91b3).
The [original reports and binary manifest](https://github.com/wangzhao-11a/t5-qkv-append-cache/tree/874f2bbbb385bc032d004735ae524244401a29ce/reports/accepted_f526712)
remain separate from this PR. The later standalone alignment fix
[`a2013d5`](https://github.com/wangzhao-11a/t5-qkv-append-cache/commit/a2013d56db4ff8367160b157335d2ff670aad411)
is now ported here, with explicit CANN custom-compiler include registration and
the original `at::empty` output factories preserved.

The actual port at
[`cbe5ac66`](https://github.com/wangzhao-11a/xllm/commit/cbe5ac665188298435b25ce1b36a3ece252ef769)
was rebuilt and validated on A3 (Ascend910_9382, CANN 9.1.0, driver 26.1.1).
Public accuracy 35/35, extended accuracy 150/150, engineering 15/15 and memory
20/20 passed. The first fixed performance run and five additional fresh-process
runs each passed 10/10. Across the five repeats, the largest case medians were
70.30915 us for FP16 and 70.24015 us for BF16, against 75 / 77 us limits.

The [A3 report and original evidence](reports/pr2656_a3_20261116/README.md)
include the failed intermediate port and the preserved standalone control.
These results validate this standalone component in the recorded environment;
they do not validate the top-level xLLM build/model path or driver `25.5.t6.b001`.
The externally reported discrete latency levels have not been established as a
single root cause. See `source_provenance.json` for original per-file hashes,
preserved baselines and the exact tested port commit.
The earlier design discussion is [PR #2604](https://github.com/xLLM-AI/xllm/pull/2604).

Files in `reference/` are frozen task inputs, excluded from formatting. Their
exact imported bytes are checked against `reference_manifest.json`. The task's
own `SHA256SUMS.txt` had pre-existing filename/hash discrepancies, recorded as
`original_manifest_valid: false`; the imported-byte check must still pass.
