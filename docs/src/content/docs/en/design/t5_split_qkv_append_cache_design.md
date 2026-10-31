---
title: "T5 QKV Split and Cache Append Operator Design"
---
<!--
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Overview

This document describes an Ascend NPU operator that splits QKV and appends K/V to their caches during single-token T5 decoding. The operator extracts Q from the fused current-token QKV input, appends the current K/V to their respective history caches, and produces three independent contiguous outputs in one kernel execution.

The design goals are to:

- Move FP16/BF16 values as raw 16-bit patterns without floating-point arithmetic or type conversion.
- Support empty history caches, contiguous storage-offset views, and row tails that are not 32-byte aligned.
- Share address calculation and partitioning logic across the two-stage ACLNN API, direct kernel launch, and the PyTorch entry point.
- Allocate independent outputs on every call, use no global workspace, and preserve input lifetime across streams.

This document is for design review. It does not replace the implementation, accuracy report, or performance acceptance report. Compliance with fixed performance targets must be established by reproducible measurements accompanying the subsequent code PR.

## 1. Interface and Semantics

Let `S=H×D`. The logical tensor shapes are:

| Name | Shape | Description |
| --- | --- | --- |
| `qkv` | `[B, 1, 3S]` | Current token laid out as all Q, then all K, then all V |
| `past_key` | `[B, H, L, D]` | Historical K cache |
| `past_value` | `[B, H, L, D]` | Historical V cache |
| `query` | `[B, H, 1, D]` | Independent Q output |
| `key` | `[B, H, L+1, D]` | Historical K followed by current K |
| `value` | `[B, H, L+1, D]` | Historical V followed by current V |

The native operator attributes are `headNum=H` and `headDim=D`. Framework entry points may infer them from tensor shapes, but explicitly supplied attributes must match those shapes. All inputs must be contiguous ND tensors on the same NPU device and have the same dtype. The host validates all dimension ranges and size products before launch and rejects invalid metadata before executing the kernel.

All three outputs are allocated independently and share no storage with the inputs or each other. The operator does not modify inputs in place, reuse outputs from previous calls, or delegate execution to the CPU.

## 2. Address Derivation

The following addresses are expressed in elements, with `b∈[0,B)`, `h∈[0,H)`, and `d∈[0,D)`.

The linear Q output address and its source are:

```text
q_out = (b×H + h)×D + d
q_src = b×3S + h×D + d
```

The linear K/V output address is:

```text
kv_out = ((b×H + h)×(L+1) + t)×D + d
```

For `t<L`, the source is the corresponding history cache:

```text
kv_src = ((b×H + h)×L + t)×D + d
```

For `t=L`, the source is the fused current QKV input. K and V use `s=1` and `s=2`, respectively:

```text
append_src = (b×3 + s)×S + h×D + d
```

This derivation maps every output position to exactly one input interval. The partitioned implementation must preserve this mapping; floating-point value comparison alone is not a sufficient proof of address correctness.

## 3. Bit Patterns and Boundary Handling

Both FP16 and BF16 are moved as `uint16_t`. NaN payloads, infinities, signed zeros, and subnormal values therefore pass through without numeric conversion or silent bit changes.

When `L=0`, K/V are generated only from the current QKV input, and the empty history tensors are never read. A contiguous storage-offset view is described by its data pointer, which already includes `storage_offset`; the ACL tensor offset remains zero so that the offset is not applied twice.

The regular DMA path requires fragment lengths and physical source/destination addresses to satisfy 32-byte alignment. Alignment checks must include the GM base address rather than relying only on logical indices. Unaligned fragments use the bounded copy path, which neither reads beyond the input nor writes padding into neighboring output regions.

## 4. Core Partitioning and On-Chip Buffers

### 4.1 Work Partitioning

Q, K, and V are each divided into contiguous tiles of 32768 elements (64 KiB), with work IDs ordered as Q, K, and V. At most 48 AIVs acquire work using `blockIdx + n×blockNum`; small inputs launch only the number of cores that have useful work.

Each tile is further divided at batch, history-row, and appended-token boundaries. For large aligned caches, the implementation may partition each `[B,H]` row independently to avoid extra cross-row fragments caused by the output shift from the appended token. The final work item for a row should handle both the history tail and the appended token rather than creating a separate, load-imbalancing tiny work item for every token.

Exactly one work item owns each output position. Even when a logical row crosses work items, ownership must be divided at complete 32-byte blocks so that separate cores never race within the same aligned block.

### 4.2 UB and Workspace

Each AIV uses `TQueBind<VECIN, VECOUT, 2>` to manage two 64 KiB UB buffers, forming a 128 KiB double-buffered queue. GM→UB and UB→GM transfers are synchronized through the same queue, while scalar state retains only a small amount of shape, tile, and offset information.

The operator explicitly sets `workspace=0`. After calling `GetWorkspaceSize`, the ACLNN caller checks the result again and reports an error instead of implicitly allocating global memory if it is nonzero.

## 5. Host, Framework, and Stream Design

The ACLNN entry follows the standard two-stage sequence: create six tensor descriptors, call `GetWorkspaceSize`, and submit execution on the caller's current stream. The direct kernel-launch entry reuses the same plan and device kernel so that it can cross-check the ACLNN path.

The PyTorch entry point performs these steps:

1. Validate inputs, attributes, and shapes, and apply the input device guard.
2. Allocate the three outputs with independent `at::empty` calls.
3. Obtain the current NPU stream on the input device and establish ordering with preceding operations according to torch_npu submission semantics.
4. Call `recordStream` for inputs that may be used on another stream, then submit the ACLNN or direct kernel.

Outputs are allocated on the current stream, and allocation-stream ordering protects their reuse; no redundant reclamation events are added for same-stream outputs. Cross-stream input registration remains in place to cover inputs released by the caller immediately after submission.

`prepare` stores only metadata such as shape, dtype, device, and attributes. It retains neither input tensors nor a fixed stream. Every subsequent call revalidates the metadata, obtains the current stream, and allocates fresh outputs.

## 6. Resource Budget and Integer Safety

The host uses 64-bit integers for shape products and validates their ranges before kernel launch. The device uses 32-bit indices only after all linear indices and products have been proven bounded, so the reduced-width path cannot introduce overflow.

The operator allocates no global result cache or temporary workspace beyond its three outputs. Their total size in bytes is:

```text
2×B×H×D×(2L+3)
```

This covers `BHD` elements for Q and `2×BHD×(L+1)` elements for K/V, including the two-byte element size. Memory validation must account for framework allocator overhead separately rather than including it in the operator workspace.

## 7. Validation Design

The implementation PR should provide reproducible validation beyond ordinary random inputs:

- Bitwise ACLNN and direct-launch comparisons for every public shape, plus FP16/BF16 special-bit-pattern checks.
- Empty caches, odd dimensions, unaligned row tails, contiguous storage-offset views, and two consecutive cache appends.
- Rejection of invalid attributes, dtypes, devices, ranks, shapes, contiguity, and internal formats.
- Dual-stream calls, early input release, retained outputs, and restoration of active memory after repeated releases.
- A CPU address model that checks coverage, bounds, and source addresses for every tile and fragment. This model validates only the address algorithm and does not replace NPU execution.
- The fixed performance protocol of 50 warmup calls and five 200-call rounds, reporting every round's API-to-completion wall time and its median while preserving raw logs.

Only checks that were actually executed and passed may be marked as passing. Environment differences, unexecuted checks, and missed targets must be disclosed explicitly.
