---
title: "xllm-kernel Operator Development Guide"
sidebar:
  order: 6
---

`xllm-kernel/` provides operator interfaces, implementation registration and
preparation, and independent numerical validation and benchmarking for Python
models. This guide covers adding an operator, adding an implementation of an
existing operator, and integrating it into a model. See the
[package README](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/README.md)
for installation and public APIs.

## 1. Responsibilities and directories

| Location | Responsibility | When adding an operator |
| --- | --- | --- |
| `python/xllm_kernel/ops/<operation>/` | Public operator semantics, solution implementations and registration, runtime binding | Define the interface or add an implementation |
| `python/xllm_kernel/registry.py` | `OpContract`, `KernelSpec`, `PreparedKernel`, and preparation-time selection | Reuse without branches on operator names |
| `python/xllm_kernel/fake/` | Fake implementations for package-owned Torch operators | Extend when needed |
| `csrc/<solution>/` | Optional native extensions, Torch bridges, and device launcher adapters | Extend when Python or existing Torch operators are insufficient |
| `python/xllm_kernel/numerics/<operation>.py` | Operator arguments, inputs, reference, tolerances, supported domain, and `OperationSuite` | Provide a suite for each operator |
| `python/xllm_kernel/numerics/catalog.py` | Explicit discovery of built-in suites | Add the module entry, without operator logic |
| `python/xllm_kernel/numerics/{arguments,suite,verify,report}.py` | Common CLI arguments, case interfaces, validation lifecycle, and numerical reports | Reuse through the interfaces |
| `python/xllm_kernel/benchmark/` | Common benchmark CLI, NPU timing, and reports | Reuse prepared cases without interpreting operator tensor arguments |
| `test/`, `test/npu/` | Common infrastructure tests, independent device tests, and fixtures | Add the corresponding pytest cases |

All paths above are relative to `xllm-kernel/`. Model structure, batch metadata,
KV cache management, and request scheduling remain in the xLLM runtime.
Operators consume prepared inputs; a solution must not introduce its own
request scheduler or cache lifecycle.

Shared files must not hard-code operator report titles, input roles, shapes,
tolerances, references, or implementation-specific domains. Public API
re-exports, the suite catalog's module list, and extension source lists in build
scripts are explicit integration points and may name operators. Computation
semantics and conditional logic belong in the corresponding operator module.

## 2. Define the contract before implementing the kernel

Define the following before development, and keep the code, tests, and README
consistent:

- Input/output roles, shapes, dtypes, devices, layouts, strides, allowed empty
  dimensions, and scalar parameter ranges.
- Input mutation, output aliasing, and any residual, output-buffer, or
  quantization semantics.
- Numerical reference, accumulation and output precision, tolerances, and
  handling of nonfinite values.
- Supported execution modes, including eager and ACLGraph, plus workspace,
  stream, device context, and lifetime requirements.
- Required Torch/torch_npu versions, device SDKs, third-party libraries, and
  build artifacts.

[`OpContract`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/registry.py)
currently describes `operation`, `version`, `mutates`, `aliases_inputs`, and
`numerical_mode`. `KernelSpec` also declares the implementation name, `device`,
`execution_modes`, `input_domain`, `solution`, `variant`, `binding`, and `priority`. `input_domain` is
currently a descriptive string: the registry does not use it to validate shapes
or layouts automatically. The implementation must check arguments before launch.

Only implementations with matching semantics may share an `OpContract` when
adding a solution for an existing operator. Changes to mutation, aliasing,
quantization, or output semantics require a distinct contract; an implementation
name alone does not express that difference. Numerical equivalence may use
declared tolerances and should not imply bitwise equality by default.

## 3. Implementation, registration, and model integration

### 3.1 Implementation and registration

Follow TokenSpeed's separation of operation semantics and implementation solution.
Keep `ops/<operation>/__init__.py` independent of the implementation library, and
put each solution in `ops/<operation>/<solution>.py`. The current solutions are
`aclnn`, `atb`, `hccl`, `xlite`, `tilelang`, and `torch`.

- `solution` names the implementation library or composition: ACLNN API calls,
  ATB operations, HCCL communication, GVirt launchers, TileLang AOT, or PyTorch.
- `variant` distinguishes implementations within a solution. ACLNN uses `xllm`
  and `torch_npu`; HCCL uses `current_stream` and `process_group`.
- `binding` records the entry point, such as `torch.ops.xllm_ops`, `torch_npu`,
  `torch.distributed`, or `xllm_kernel._xlite`. A binding is not another solution.

A solution file exports lazy `load()` or `load_<variant>()` functions returning
callables. Declare their paths, domains, modes and metadata with `Backend` in
`OPERATOR`, and export `prepare = OPERATOR.prepare`. The shared `Operator`
registers and freezes only the selected implementation. See the
[ACLNN adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/aclnn.py).

Names have the form `npu.<solution>.<operation>[.<variant>]`, for example
`npu.aclnn.rms_norm.xllm` and `npu.aclnn.rms_norm.torch_npu`. Both belong to ACLNN;
a comparison between their bindings must not be labeled as two kernel libraries.
The former `npu.xllm_native.*` and `npu.cann.*` selectors have been replaced;
update CLI arguments and environment overrides when upgrading this draft.

When reusing a host-owned Torch operator, preserve ownership of its schema,
implementation, and fake registration. For a new package-owned Torch operator,
use the package namespace and provide both a real device bridge and its fake
implementation. Check build artifacts and versions during preparation. Do not
download dependencies, compile, or reselect implementations during request
execution or graph replay.

NPU implementations use the current PyTorch stream and follow torch_npu's
submission queue and Tensor lifetime rules. Do not create private streams or
synchronize the device inside an operator for implementation convenience.
Cross-stream execution, device switching, and asynchronous resource release
require corresponding tests. FakeTensor checks logical shape, dtype, and device
semantics where possible; real device implementations still check device
conditions such as physical formats.

### 3.2 Preparation and execution

```text
Load and check dependencies
    -> register(KernelSpec, callable)
    -> registry.freeze()
    -> registry.prepare(contract, device, execution_mode, implementation)
    -> PreparedKernel
    -> model retains the callable
    -> eager calls or graph capture/replay
```

The registry accepts no new implementations after freezing. `prepare` filters by
contract, device, execution mode, and explicit implementation name, then selects
by priority and stable name order. Execution uses the bound callable directly.
Preparation and execution errors propagate without automatic fallback. Changing
the implementation policy requires recreating the worker and graphs.

The package-level `initialize` currently re-exports RMSNorm initialization.
The host binds RMSNorm through `XLLM_KERNEL_RMS_NORM_IMPL`; this is not yet a
multi-operator initialization manager. A new operator needs an explicit
preparation entry point for the operation and host integration before model import
or capture. A new solution for an existing operator also needs changes to that
operation's initialization selection, offline suite, and host configuration entry
point.

Model integration replaces the intended call sites. Preserve the semantics of
the public API, input layouts, and cache views, then validate the actual model
path. Passing operator-level accuracy checks does not replace model integration
tests.

## 4. Provide an operator-specific offline suite

The common interfaces are defined in
[`numerics/suite.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/suite.py).
The numerical and benchmark CLIs share the same suite so they do not maintain
separate shapes, inputs, or references.

### 4.1 `OperationSuite`

Each `numerics/<operation>.py` exports a `SUITE`:

| Field | Contract |
| --- | --- |
| `operation` | Matches the prepared implementation's `OpContract.operation` |
| `implementations` | Full implementation names supported by the suite; unique across the catalog |
| `add_arguments(parser)` | Registers operator-specific CLI arguments |
| `cases(args)` | Validates operator-specific arguments and returns `KernelCase` instances without allocating device inputs |
| `prepare(device, mode, implementation)` | Loads the selected implementation and returns a `PreparedKernel` |

Add the module entry to `_BUILTINS` in `numerics/catalog.py`. Importing a suite
through the catalog must require only Python metadata. Defer Torch, device SDK,
and native extension imports until preparation or execution so that `--help`
works without a device environment.

The shared CLI handles device, execution mode, explicit implementation name,
warmup, revision, and output; the benchmark CLI also handles iterations and
repeats. `--operation` can be explicit or inferred from `--implementation`.
Operator arguments such as shape and dtype combinations belong in
`add_arguments` and `cases`, without operator-specific branches in the shared CLI.

### 4.2 `KernelCase` and `PreparedCase`

Each logical case implements these interfaces:

| Interface | Result and responsibility |
| --- | --- |
| `case_id` | Reproducible identifier that distinguishes case parameters and seed |
| `parameters` | JSON-serializable parameter dictionary |
| `unsupported_reason(plan)` | Returns a reason when inapplicable, otherwise `None`; do not infer support from execution failures |
| `prepare(plan)` | Creates inputs and returns a `PreparedCase` |

`PreparedCase` has four fields:

- `function()`: a zero-argument callable binding the inputs, scalar parameters,
  and prepared implementation. It may return a tensor, tuple, or another
  structure defined by the operator.
- `validate(function)`: invokes the supplied zero-argument callable and returns
  a result containing `status`, using the operator's reference and tolerances.
  It checks the output structure, numerical values, mutation, aliasing, and
  other contract requirements.
- `metadata`: serializable operator-specific details such as input strides and
  formats.
- `reset()`: optional callback restoring mutated inputs outside timed event pairs.

The validator must check the supplied callable's result. It must not ignore
that callable and invoke the original implementation again. ACLGraph validation
supplies a function that reads captured outputs; executing the eager operator
again would hide incorrect graph outputs. Validation results and metadata must
also avoid overwriting common identity fields such as case ID, operation, and
implementation.

By default, generate inputs on the CPU with a fixed seed and move them to the
device as required, so candidates receive the same logical data. The reference
must be independent of the implementation under test. Domain filtering must
match the implementation's argument checks. Direct device tests still cover
boundaries; marking every case `not_applicable` is not validation.

[`numerics/rms_norm.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/rms_norm.py)
provides a complete example. Shared CLI tests also use a test operator with
size/scale arguments and dictionary outputs to verify that runners do not depend
on RMSNorm's input/weight/eps structure. That operator tests the shared workflow;
it does not represent additional device operator support.

## 5. Shared validation, timing, and reporting

```text
Parse common arguments + suite-specific arguments
    -> suite.prepare
    -> check case support
    -> case.prepare
    -> validate(function)
    -> warmup
    -> optional graph capture/replay + validate(captured_output)
    -> benchmark timing
    -> JSON / Markdown reports
```

`numerics/verify.py` manages preparation, status, and the reporting lifecycle. It
does not interpret tensor roles or choose an operator's reference.
`numerics/report.py` and `benchmark/report.py` use generic titles and record the
operation per row, without inferring operator types from names or reading an
operator's shape fields. Report schema version 2 includes `solution`, `variant`,
and `binding` alongside the full implementation name in JSON and Markdown.

Failures must remain visible: record reasons for inapplicable cases; do not time
numerical failures; report and raise capture/replay failures. A run with no
passing cases cannot produce a successful report. Device execution errors must
not trigger eager or solution fallback followed by accepted performance results.

The current timer supports NPU eager and ACLGraph execution. Pure cases use 64
calls per benchmark capture; cases with `PreparedCase.reset` use one call per
capture and restore their state before replay, outside timed event pairs.
New stateful operators must define reset behavior and timing boundaries in their
suite; do not reuse mutated inputs without restoring them.

Preparation, warmup, and capture are excluded from steady-state timing. Keep
inputs, cases, devices, execution modes, and timing settings consistent. Report
raw samples and dispersion, and record resource occupancy. Eager device-event
intervals include host submission gaps and do not directly measure the pure
device execution time of one kernel. Distinguish operator-level performance
from end-to-end results.

## 6. Tests and validation

Package tests run independently through pytest. They are not wired into the
host's `tests/CMakeLists.txt`, CTest, or `all_tests`. Run the following commands
from the repository root after installing the package and dependencies as
described in its README.

```bash
# CPU tests for shared infrastructure, numerical reports, and CLIs
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu --ignore=xllm-kernel/test/distributed

# Device integration tests: build the matching host/bridge first, then run
# sequentially on idle devices
XLLM_KERNEL_RMS_NORM_IMPL=npu.aclnn.rms_norm.xllm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_rms_norm_npu.py
XLLM_KERNEL_RMS_NORM_IMPL=npu.xlite.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_xlite_rms_norm_npu.py
```

The device fixture in `test/npu/conftest.py` loads the host and initializes
Python kernels; pytest does not build them. Implementation bindings cannot be
switched within one process, so ACLNN host-binding and xlite tests use separate processes.
Direct pytest runs do not acquire CTest device resource locks; ensure devices
are idle separately. Device-switching tests require two visible NPUs.

The offline tool examples use a currently supported operator. After integrating
a new operator, replace the operation, implementation, and specific arguments:

```bash
python -m xllm_kernel.numerics --operation rms_norm --device npu --mode eager \
  --implementation npu.aclnn.rms_norm.xllm --rows 1 16 --widths 6144 \
  --revision "$(git rev-parse HEAD)" --output numerics.json
python -m xllm_kernel.benchmark --operation rms_norm --device npu --mode aclgraph \
  --implementation npu.aclnn.rms_norm.xllm --rows 1 16 --widths 6144 \
  --iterations 50 --revision "$(git rev-parse HEAD)" --output benchmark.json
```

Validate a new operator in this order, with coverage determined by its contract:

1. SDK-free imports, registration/preparation, independent installation, and
   CLI arguments.
2. CPU references, tolerances, incorrect outputs, nonfinite values, and
   mutation/alias detection.
3. Real-device dtypes, shapes, layouts, strides, empty inputs, boundaries, and
   invalid arguments.
4. Non-default streams, device guards, asynchronous object lifetimes, repeated
   calls, and memory stability.
5. Eager/ACLGraph output agreement and interleaved replay after input changes.
6. Operator performance comparisons on exclusive devices, followed by
   integration validation for the intended model paths.

CPU test doubles and FakeTensor establish only the corresponding logical
behavior. They are not evidence of NPU accuracy or performance.

## 7. Delivery checklist

- Keep the contract, real argument checks, fake checks, suite domain, and
  README consistent.
- Do not add branches for the new operator's name, shapes, input roles, or
  tolerances to shared runners, reports, or the registry.
- Keep suite imports free of device dependencies. Reject unknown
  implementations and mismatched operations before execution preparation.
- Connect a new solution to host selection, the offline suite, and device tests.
- Keep unit tests under the independent package's `test/`; do not modify host
  test targets to run them automatically.
- Check formatting, wheel contents, CLI examples, and reports. Distinguish
  source checks, CPU tests, builds, NPU tests, and model evidence in the change
  description.
- Before adding JIT, state reset, multiple devices, or a new platform, define
  explicit interfaces and lifecycles, then add implementations.

## 8. GLM-5.3 operator inventory and explicit adapters

The initial expanded inventory follows `xllm/python/models/glm5_2.py`, its
`deepseek_v32.py` parent path, `layers/embedding.py`, `layers/linear.py`, and
`kernels_npu/{normalization,activation,quantization,rotary_embedding,moe,mla,sparse_attention}.py`.
This is the full GLM-5.3 DSA model path, not the GLM-5.3 Flash model. The base
is `feat/glm53-pytorch-kernel-package` at `26c4d550`.

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

The inventory contains 37 operation contracts and 70 implementation variants
across six solutions. `aclnn` includes both standard CANN operators and custom
operators exposed through the ACLNN API. An xLLM binding can dispatch to multiple
ACLNN kernels; the variant name identifies the wrapper, not a fixed binary.
`atb` wraps actual ATB execution. `hccl` identifies the communication library.
`torch` is an explicit PyTorch NPU composition, and `tilelang` resolves an existing
compiled AOT operator. xlite uses pinned GVirt launchers on the current stream.
Only the selected solution and binding are loaded.

The public `einsum` / `einsum_out` names describe the fixed contraction
`[T,H,D] x [H,D,O] -> [T,H,O]`. They replace `atb_matmul_ein_sum` / `_out` so the
operation name does not imply ATB for its Torch or xlite implementations. The
ATB loader preserves the host's original schema names. `reshape_paged_cache`
is labeled ATB only on Ascend 910; its loader rejects other device families
because the host switches to ACLNN on Ascend 950.

New operation modules declare an `OPERATOR` and export `prepare`:

```python
from xllm_kernel.ops.dynamic_quant import prepare
plan = prepare("npu", "aclgraph", "npu.aclnn.dynamic_quant.torch_npu")
quantized, scale = plan.function(hidden)
```

Loaders return a callable before capture; `Operator` performs registration,
freezing and preparation. Adding an adapter does not change
the model's backend policy. Existing model calls, including compound MLA and
MoE schedules, remain host-owned.

For new numerical suites, `tensor_suite` builds operation-owned shape matrices.
`Example` binds a tensor tree, a CPU reference, declared mutations, comparison
tolerances and optional semantic canonicalizers. `TensorCase` carries metadata
without loading a device SDK. `PreparedCase.reset` restores mutable inputs
outside the timed event pairs; graph capture uses one operation for mutable
cases. Per-operator test files share the lifecycle in `test/npu/operator_checks.py`.
No package test is added to `tests/CMakeLists.txt`.

Source audit boundaries:

- Model defaults include hidden size 6144, vocabulary 154880, 256 experts and
  top-8 routing. Embedding uses hidden-dimension TP shards (384 at TP=16).
  Cases use decode/prefill row counts; local head and expert counts are explicit
  CLI dimensions, not an inferred full deployment configuration.
- xlite fused RMSNorm then BF16 quantization, FP16 intermediate quantized GEMMs,
  and bitmap dispatch do not match the fused FP32/NZ contracts directly.
  No duplicate name is registered to hide an extra rounding or layout change.
- xlite indexer adapters pack the head weights and own/reset scratch tensors.
  The sparse-attention `_out` adapter supports one-query BF16 decode without
  LSE; compaction of padding indices, KV gather and all workspace costs are
  timed. Fused MLA preparation remains host-owned. The existing TileLang
  half-layout RoPE is not interleaved.
- Native sparse attention keeps its distinct RoPE ACLNN entry point; the
  legacy same-name CANN ABI is not used as a substitute. The independent CPU
  attention reference checks the output and log-sum-exp, including causal
  masks, shuffled pages and -1 sparse-index padding.
- Lightning Indexer compares unique causal selections by their CPU scores
  (1.5% tolerance), allowing numerical ties; exact index equality is not claimed.
- MegaMoE, optional FP8 KV/INT8 index caches, fixed-shape MLAPO v2, and MTP
  scheduler metadata need separate state/layout contracts. Ordinary ATen
  tensor manipulation remains in the host. The present inventory is BF16/W8A8.

Communication uses the generic `--distributed` CLI lifecycle under `torchrun`.
It owns the default HCCL group, produces rank-specific reports and records
per-repeat rank maxima. HCCL `process_group` variants are eager-only; the HCCL
`current_stream` all-reduce/gather/reduce-scatter providers expose ACLGraph.
Set `HCCL_OP_EXPANSION_MODE=AIV` before their graph tests. All-to-all is equal
split and eager-only. See the package README for commands and exact domains.

**Delivery status:** this solution reclassification has only source and formatting
checks; no build or tests have run on this revision. The earlier `5c93a0f6`
revision built the xlite bridge and package wheel, which does not validate the
renamed schemas or loaders here. The full host build remains paused by request.
The new cases are validation specifications, not passing device results.

Routing and SwiGLU semantics were cross-checked against the official
[MoeInitRoutingV2 API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md)
and [DequantSwigluQuant API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_dequant_swiglu_quant.md).
The installed runtime must provide those requested interfaces; preparation or
execution fails visibly when it does not. Documentation is not binary ABI proof.
