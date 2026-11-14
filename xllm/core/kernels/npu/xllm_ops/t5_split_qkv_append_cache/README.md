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
core's own output interval. Aligned fragments use ordinary DMA; other fragments
use the padded copy API. Large outputs use a bounded cached key suffix and
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

The implementation was imported from the accepted standalone runtime
[`f526712`](https://github.com/wangzhao-11a/t5-qkv-append-cache/commit/f52671207769bf5f334972c997743eb6aa6f91b3).
That original runtime passed all 10 fixed performance cases on Ascend910_9382
with driver 26.1.1. The large FP16/BF16 cases measured 70.22155 / 70.20120 us
against 75 / 77 us limits in the accepted full run. These measurements are not a
guarantee across processes, devices, drivers or this source port.

The [original reports and binary manifest](https://github.com/wangzhao-11a/t5-qkv-append-cache/tree/874f2bbbb385bc032d004735ae524244401a29ce/reports/accepted_f526712)
remain separate from this PR. This port changes naming, formatting, annotations,
include paths and validation-tool integration. **It has not yet been rebuilt with
CANN or validated on an NPU; it needs both before merge.** See
`source_provenance.json` for original per-file hashes and the upstream base.
The earlier design discussion is [PR #2604](https://github.com/xLLM-AI/xllm/pull/2604).

Files in `reference/` are frozen task inputs, excluded from formatting. Their
exact imported bytes are checked against `reference_manifest.json`. The task's
own `SHA256SUMS.txt` had pre-existing filename/hash discrepancies, recorded as
`original_manifest_valid: false`; the imported-byte check must still pass.
