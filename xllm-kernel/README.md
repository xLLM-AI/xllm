<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# xLLM Kernel

`xllm-kernel` contains explicit operator contracts, NPU backend adapters,
preparation, and independent numerical and performance tools. Runtime imports
are lazy: package and suite discovery do not initialize Torch or a device.
The current operator inventory follows the GLM-5.3 PyTorch graph implementation
in `glm5_2.py`, its shared `deepseek_v32.py` path, layers, and NPU wrappers.

This remains a **draft**. The current solution reclassification has not been
compiled or tested. The earlier `5c93a0f6` revision built the optional xlite bridge
and wheel; that does not validate the renamed schemas or loaders here. The full
host build remains paused. Existing model calls are unchanged; only the earlier
RMSNorm integration is automatically selected by host startup.

## Install and prepare

From the repository root, in the matching Torch/torch_npu environment:

```bash
python -m pip install --no-deps -e ./xllm-kernel
```

Implementations with a `torch.ops.xllm_ops` binding require the matching xLLM
native extension; TileLang also requires its compiled AOT variants. Build the
host separately when needed with `python setup.py build`. No CMake/CTest target
runs these package tests. The `torch_npu` binding requires its matching CANN
runtime; `torch` compositions include temporary tensors and layout conversions.

Optional xlite bridges require installed GVirt revision
`4a6dc3102d5778928e4d21116bc4d4694d51e584`:

```bash
cd xllm-kernel
XLLM_KERNEL_BUILD_XLITE=1 MAX_JOBS=4 python setup.py build_ext --inplace
```

The bridge uses PyTorch's current NPU stream and torch_npu's submission queue.
It does not own a runtime, synchronize per call, or compile during preparation.
The build manifest pins Torch/torch_npu, GVirt, and the common/BF16/FP16 launcher
libraries. The package also supports an optional compiled wheel through
`XLLM_KERNEL_BUILD_XLITE=1 python -m pip wheel --no-build-isolation --no-deps ./xllm-kernel`.
Rebuild after source or schema changes; the previous wheel cannot validate this revision.

Prepare each operation before graph capture and retain its callable:

```python
from xllm_kernel.ops.silu_and_mul import prepare

plan = prepare("npu", "aclgraph", "npu.aclnn.silu_and_mul.torch_npu")
output = plan.function(gate_up)
```

Every implementation name is `npu.<solution>.<operation>[.<variant>]`. Selection is explicit;
missing dependencies and unsupported modes fail without an execution fallback.
Each operation owns its signature, domain and mutation contract. `_out` and
in-place operations have distinct contracts and reset their mutated inputs
between validation and timing calls.

The existing package-level `initialize`, `prepare_rms_norm`, and
`XLLM_KERNEL_RMS_NORM_IMPL` remain the host's RMSNorm integration. New operation
modules expose their own `prepare`; they do not change model policy globally.

## Solutions and bindings

Following TokenSpeed's separation of semantics and implementations, each operation
owns its contract and one module per solution. `KernelSpec` records the solution,
variant and binding independently; numerical and performance reports preserve all
three fields (report schema version 2).

| Solution | Actual implementation | Binding / variants |
| --- | --- | --- |
| `aclnn` | Standard or custom operators exposed through the ACLNN API | `xllm` via `torch.ops.xllm_ops`; `torch_npu` via torch_npu |
| `atb` | ATB operations | Existing `torch.ops.xllm_ops` schemas |
| `hccl` | HCCL collectives | `current_stream` via xLLM; `process_group` via torch.distributed |
| `xlite` | GVirt device launchers | Package-owned `xllm_kernel._xlite` bridge |
| `tilelang` | Compiled TileLang AOT kernels | Existing `torch.ops.xllm_ops` schema |
| `torch` | PyTorch NPU compositions | Torch tensor operations |

The catalog contains 37 contracts and 70 implementation variants, not 70 distinct
kernel libraries. For example, `npu.aclnn.rms_norm.xllm` and
`npu.aclnn.rms_norm.torch_npu` use different bindings of the ACLNN solution.
Host wrappers may dispatch among several ACLNN kernels, including custom kernels;
a variant identifies the wrapper rather than promising a fixed binary.
The former `npu.xllm_native.*` and `npu.cann.*` selectors are replaced by these
solution names; update CLI and environment overrides when upgrading this draft.

The operation `einsum` (and its `_out` contract) replaces `atb_matmul_ein_sum`:
its fixed signature is `[T,H,D] x [H,D,O] -> [T,H,O]`. ATB, Torch and xlite
implementations share that operation name. Only the ATB loader retains the host's
old schema name. The package-owned xlite schema is now `xlite_einsum`.

## Layout and responsibilities

```text
python/xllm_kernel/
  registry.py, operator.py       # preparation and immutable contracts
  ops/<operation>/<solution>.py  # operation-owned implementation loaders
  backends/                     # shared dependency and communicator loading
  fake/<operation>.py           # metadata for package-owned Torch schemas
  numerics/<operation>.py       # inputs, independent CPU reference, restrictions
  numerics/                     # common suite, comparison, reset and report lifecycle
  benchmark/                    # timing, statistics and reports
csrc/xlite/                     # optional current-stream launcher bridges
test/                          # CPU/package tests
  npu/                         # independent device accuracy and graph checks
  distributed/                 # explicit multi-rank tests under torchrun
```

The suite catalog lists modules only. Common runners and reports contain no
operator-name branches or operator-specific shapes, references or tolerances.
Numerical domain helpers are explicitly named; output and alias semantics
remain in each suite. There is no installed `testing/` package.

## Operator and solution coverage

| Operation | Solution and optional variant |
| --- | --- |
| `add` | `torch`, `xlite` |
| `all_gather` | `hccl.current_stream`, `hccl.process_group` |
| `all_reduce` | `hccl.current_stream`, `hccl.process_group` |
| `all_to_all_single` | `hccl.process_group` |
| `dequant_swiglu_quant` | `aclnn.xllm`, `aclnn.torch_npu` |
| `dynamic_quant` | `aclnn.xllm`, `aclnn.torch_npu`, `xlite` |
| `einsum` | `atb`, `torch`, `xlite` |
| `einsum_out` | `atb` |
| `embedding` | `torch`, `xlite` |
| `fused_add_rms_norm` | `aclnn.xllm`, `aclnn.torch_npu`, `xlite` |
| `fused_add_rms_norm_dynamic_quant` | `aclnn.torch_npu` |
| `grouped_matmul` | `aclnn.torch_npu` |
| `grouped_matmul_out` | `aclnn.xllm` |
| `inplace_partial_rotary_mul` | `aclnn.xllm`, `aclnn.torch_npu` |
| `interleaved_rotary_embedding` | `aclnn.torch_npu`, `xlite` |
| `layer_norm` | `torch`, `xlite` |
| `lightning_indexer` | `xlite`, `aclnn.xllm`, `torch` |
| `lightning_indexer_out` | `xlite`, `aclnn.xllm` |
| `linear` | `torch`, `xlite` |
| `moe_finalize_routing` | `aclnn.torch_npu` |
| `moe_gate_routing` | `xlite`, `aclnn.torch_npu` |
| `moe_gmm1` | `aclnn.torch_npu`, `aclnn.xllm` |
| `moe_token_dispatch` | `aclnn.torch_npu`, `aclnn.xllm` |
| `moe_token_unpermute` | `aclnn.torch_npu` |
| `quant_matmul` | `aclnn.xllm`, `aclnn.torch_npu` |
| `quant_matmul_out` | `aclnn.xllm` |
| `quantize_per_tensor` | `aclnn.xllm` |
| `reduce_scatter` | `hccl.current_stream`, `hccl.process_group` |
| `reshape_paged_cache` | `atb`, `torch` |
| `rms_norm` | `aclnn.xllm`, `aclnn.torch_npu`, `xlite` |
| `rms_norm_dynamic_quant` | `aclnn.xllm`, `aclnn.torch_npu` |
| `router_matmul` | `torch` |
| `scatter_nd_update` | `aclnn.xllm`, `torch` |
| `sfa_dcp_remap` | `tilelang`, `torch` |
| `silu_and_mul` | `aclnn.xllm`, `aclnn.torch_npu`, `xlite` |
| `sparse_flash_attention` | `aclnn.xllm`, `torch` |
| `sparse_flash_attention_out` | `xlite`, `aclnn.xllm` |

The inventory is for BF16/FP16 and W8A8 INT8 GLM paths. APIs expose the domains
used by their suites, not every overload of the underlying CANN library.
`all_reduce`, `all_gather`, and `reduce_scatter` offer HCCL `process_group`
(eager) and `current_stream` (eager/ACLGraph) variants. `all_to_all_single`
is eager only. Other declared implementations expose eager and ACLGraph entry
points; actual device support still requires validation.

Key restrictions and differences:

- ATB `reshape_paged_cache` is restricted to Ascend 910. Preparation rejects
  other device families because the host selects ACLNN on Ascend 950.

- xlite dynamic quantization is BF16 only; zero-row repair is part of execution.
  Norm, add, activation, embedding, matmul and RoPE have explicit dtype, alignment
  and dimension limits in their bridges, fake implementations and suites.
- xlite router selection supports FP32 logits with 256 experts, top-k <=32,
  one group or eight groups retaining four. Bitmap conversion is timed.
- RMSNorm dynamic quantization uses the ACLNN host binding or torch_npu's
  add-norm-quant with a zero residual; zero construction is timed. The xlite
  norm followed by BF16 quantization is not registered as the fused FP32 contract.
- MoE dispatch and GMM1 convert V2/V3 group metadata explicitly. Private NZ
  weight conversion happens during case preparation, before measurement.
- Paged indexer/attention suites cover one sequence, decode and multi-token
  causal queries, shuffled block tables, and attention index padding. The
  native profile uses index head 128/top-k 2048 and MLA latent width 512.
  Indexer accuracy checks bounds, causality, uniqueness, and selected CPU
  scores within 1.5%; it does not claim exact index equality at numerical ties.
- TileLang DCP remap uses the existing AOT specialization: <=256 rows,
  top-k 2048. The existing half-layout TileLang RoPE is not registered under
  the interleaved GLM contract.
- xlite's quantized matmul/grouped matmul FP16 intermediate and bitmap routing
  layouts are not substituted for GLM's BF16/INT32/NZ interfaces. Fused MLA
  preparation remains host-owned. xlite indexer adapters own and reset their
  scratch tensors, pack weights and include that work in timing. xlite sparse
  attention is available through `_out` for one-query BF16 decode only: sparse
  index compaction, KV gather and workspace allocation are timed; no LSE is produced.
- Compound MLA preprocessing and grouped MoE schedules remain host-owned and
  are covered here through their component operators. Optional MegaMoE EP,
  FP8 KV-cache attention, quantized index caches, MLAPO v2 fixed shapes, and
  MTP scheduler metadata are outside this standalone BF16/W8A8 operator inventory.
  Ordinary ATen view/split/concat/copy and request scheduling remain in PyTorch/xLLM.

## Independent accuracy and performance tools

Both commands discover the same operation-owned suites. Use
`--operation NAME --help` to inspect dimensions without a device SDK. Inputs
are seeded on CPU; references are independent CPU arithmetic. Validation
checks shapes, dtypes, finite values, read-only inputs, declared mutations and
aliasing. Integer metadata and cache writes are exact; quantization uses
explicit code/scale tolerances. XLite unit tests also compare fake metadata.

```bash
python -m xllm_kernel.numerics --operation silu_and_mul --device npu --mode eager \
  --implementation npu.aclnn.silu_and_mul.torch_npu --rows 1 16 128 --widths 4096 \
  --revision "$(git rev-parse HEAD)" --output silu-aclnn-numerics.json
python -m xllm_kernel.benchmark --operation silu_and_mul --device npu --mode aclgraph \
  --implementation npu.xlite.silu_and_mul --rows 1 16 128 --widths 4096 \
  --iterations 100 --repeats 11 --revision "$(git rev-parse HEAD)" --output silu-xlite.json
```

Run each backend in a fresh process on idle devices, with identical cases,
precision and timer settings. Accuracy and graph-output validation must pass
before timing. Reports include exclusions and failures; zero applicable cases
cannot pass. Preparation, warmup and capture are recorded separately. Pure
operators use 64 calls per captured graph; mutable operators use one call and
restore buffers outside the device/host intervals. Wall timing includes resets.
The cache policy is repeated hot inputs. No performance numbers are supplied
by this source-only change.

For multi-rank collectives, initialize explicitly through the CLI:

```bash
HCCL_OP_EXPANSION_MODE=AIV torchrun --nproc-per-node=2 --master-port=29653 \
  -m xllm_kernel.benchmark --distributed --operation all_reduce --device npu \
  --mode aclgraph --implementation npu.hccl.all_reduce.current_stream \
  --iterations 100 --revision "$(git rev-parse HEAD)" --output all-reduce.json
```

Each rank writes its own report (`all-reduce.rank0.json`, etc.). Reports retain
local samples and per-repeat maxima across ranks. Every rank agrees on
validation success before entering timing. HCCL groups are created before
preparation and destroyed after report writing, never inside a captured call.

## Tests

Run package tests independently, without host CMake wiring:

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu \
  --ignore=xllm-kernel/test/distributed
python -m pytest -q xllm-kernel/test/npu/test_silu_and_mul_backends.py
HCCL_OP_EXPANSION_MODE=AIV torchrun --nproc-per-node=2 --master-port=29654 \
  -m pytest -q xllm-kernel/test/distributed
```

The NPU fixture loads the matching host for integration checks. Optional
providers must be installed to run their cases; missing implementations fail
instead of being silently treated as a successful comparison. The legacy
RMSNorm integration tests still require separate processes and their explicit
`XLLM_KERNEL_RMS_NORM_IMPL` selection. These tests do not establish model-level
correctness or end-to-end performance.

## Development

Add the operation contract and lazy loaders, its independent numerical suite,
fake metadata for new schemas, and tests under the package. Keep implementation
selection outside model execution. Add a suite catalog entry and document
input restrictions and conversions. Validate the actual model integration
separately when changing model call sites.

See the development guide ([English](../docs/src/content/docs/en/dev_guide/xllm_kernel_dev.md) |
[简体中文](../docs/src/content/docs/zh/dev_guide/xllm_kernel_dev.md)).
