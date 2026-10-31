---
title: "xLLM-Kernel Development Guide"
---

`xllm-kernel/` provides operator interfaces, implementation registration and
preparation, and independent numerical validation and benchmarking for Python
models. This guide covers adding an operator, adding an implementation of an
existing operator, and integrating it into a model. See the
[package README](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/README.md)
for installation. Architecture, public APIs, and the development and validation
workflow are documented here.

## 1. Responsibilities and directories

The component organizes kernels by operation, keeping model-facing APIs
independent of the selected solution. Contracts make numerical behavior,
mutation, aliasing, supported devices, and input domains explicit. Implementations
can reuse vendor operators or provide package-owned extensions without changing
the model API.

Registration, dependency checks, and binding happen before model execution and
graph capture. Models retain the prepared callable, keeping selection and
compilation off the serving path. Numerical validation and performance measurement
run independently of an xLLM server. New operator families and platforms extend
these same boundaries, with tests and measurements for each implementation.

The following files separate operation definitions, shared infrastructure, and
backend dependencies:

| Location | Responsibility | When adding an operator |
| --- | --- | --- |
| `pyproject.toml`, `python/xllm_kernel/__init__.py` | Standalone package metadata, initialization and public re-exports; ordinary import stays SDK-free | Export the intended public API |
| `python/xllm_kernel/catalog.py` | `Operation` / `BackendModule` definitions and discovery of runtime operations | Add the operation module to `_BUILTINS` |
| `python/xllm_kernel/selection.py` | Unified overrides, compatibility filtering and priority selection | Reuse the selector and declare implementation metadata |
| `python/xllm_kernel/runtime.py` | Load candidates for all operations, freeze once, and publish fixed bindings together | Reuse shared initialization |
| `python/xllm_kernel/ops/<operation>/` | Public operator semantics, solution implementations and registration, runtime binding | Define the interface or add an implementation |
| `python/xllm_kernel/registry.py` | `OpContract`, `KernelSpec`, `PreparedKernel`, and preparation-time selection | Reuse without branches on operator names |
| `python/xllm_kernel/custom_ops.py` | Torch registration with optional `fake_impl`; implementations keep fake functions beside their kernels | Reuse for new Python ops or existing C++ schemas |
| `csrc/<solution>/` | Optional native extensions, Torch bridges, and device launcher adapters; currently `xlite/` | Add the implementation and device registration |
| `setup.py` | Extension sources, library dependencies, and independent build options | Wire the required extension while keeping default installation pure Python |
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
re-exports, runtime and suite catalog module lists, and extension source lists in
build scripts are explicit integration points and may name operators. Computation
semantics and conditional logic belong in the corresponding operator module.

## 2. Define the contract before implementing the kernel

Define the following before development, and keep the code, tests, and operator
documentation consistent:

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

Declare the callable at module scope in `ops/<operation>/<solution>.py` with
`@register_kernel(...)`. Importing the module resolves dependencies and runs the
decorator, which constructs `KernelSpec`, registers it in `KernelRegistry.get()`,
and returns the same callable without a forward wrapper. No backend-specific
`register(registry)` entry point or explicit registry argument is needed.

```python
import torch
import torch_npu
from xllm_kernel import register_kernel
from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT

_native_rms_norm = torch_npu.npu_rms_norm


@register_kernel(
    name="npu.aclnn.rms_norm.torch_npu",
    contract=RMS_NORM_CONTRACT,
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="torch_npu.npu_rms_norm tensor domain",
    solution="aclnn",
    variant="torch_npu",
    binding="torch_npu",
    priority=20,
)
def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _native_rms_norm(input, weight, epsilon=eps)[0]
```

For an existing Torch op or callable instance, use
`register_kernel(...)(existing_callable)`. xlite uses this form after checking
and loading its bridge and registering the fake implementation. The implementation
name, contract, device, execution modes, input domain and solution are required;
priority defaults to 0. Duplicate names and registration after `freeze()` are errors.

The package and decorator import without Torch or a device SDK. Backend modules
load their dependencies when imported. Each operation exports an `OPERATION`
with its contract and `BackendModule` declarations; the shared catalog discovers
these SDK-free definitions before importing available backend candidates.

`register_kernel` handles implementation selection metadata.
`custom_ops.register_custom_op` separately handles Torch schemas, device kernels
and optional fake implementations. See the existing
[native adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/aclnn_torch_npu.py)
and
[xlite adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/xlite.py).

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

### 3.2 Automatic selection and preparation

```text
initialize(device, execution_mode, operations, overrides)
    -> discover all OPERATION definitions from catalog._BUILTINS
    -> resolve XLLM_KERNEL_OVERRIDE_<OPERATION> and explicit overrides
    -> import available backend modules; decorators register callables
    -> freeze the shared registry once
    -> select every operation by contract/device/mode, priority and stable name
    -> publish all bindings together as a read-only mapping
    -> model retains the callable for eager or graph execution
```

No override is required for normal startup. Selection uses declared metadata and
priority; it does not inspect input tensors, benchmark candidates, specialize on
shape, or reselect during forward/replay. The selected implementation still
validates its own input domain. Current RMSNorm choices and their supported
domains are described in [section 3.4](#34-public-api-and-current-coverage).

The host calls `xllm_kernel.initialize(device="npu", operations=("rms_norm",))` after loading native
operators and before model import or graph capture, then logs the resulting
operation/name pairs. Direct callers receive `Mapping[str, PreparedKernel]`:

```python
from xllm_kernel import initialize

bindings = initialize(device="npu", operations=("rms_norm",), overrides={"rms_norm": "xlite"})
plan = bindings["rms_norm"]
```

Use `XLLM_KERNEL_OVERRIDE_RMS_NORM=xlite` for the same choice at worker startup.
The generic rule is `XLLM_KERNEL_OVERRIDE_<OPERATION>` with an uppercase operation
name. Each value is a solution or exact implementation name; exact names take
precedence over solution names. Environment values override the explicit mapping.
Unknown operation keys, unknown targets and incompatible explicit choices fail
before bindings are published. Overrides preserve contract/device/mode checks.

Missing declared backend dependencies exclude that module and are reported in
`PreparedKernel.excluded`. Errors while importing an installed backend, including
ABI or launcher-hash mismatches, propagate. They never trigger another backend.
All operations must prepare successfully before any binding is published.
Repeating the same effective configuration returns the same mapping; changing
it requires a new worker and graphs. Numerical and benchmark CLIs use this same
initializer and reject a policy that changes their requested implementation.

To add an operation, export an `OPERATION = Operation(contract, backends)` from
its public module and add that module to `catalog._BUILTINS`. Each `BackendModule`
declares its import path, device and required modules. Add solutions to that
operation's backend list and use `@register_kernel` for their metadata/priority.
The host needs no new operator-specific environment-variable handling. Extend
the operation's offline suite and model call sites separately.

The catalog contains 1 operations and 3 implementation variants; see section 8.
The model host currently binds RMSNorm only. Registration does not replace model
call sites, and model integration tests remain necessary for new consumers.


`operations` declares the complete set required by this process; omitting it
prepares the entire catalog. The model host currently requests `("rms_norm",)` so
unused collectives do not require a process group. Each standalone CLI prepares
only its target. Multi-operation consumers declare all operations in one call.
Changing this set, device, mode or effective overrides requires a new worker.

`BackendModule.implementations` provides SDK-free CLI choices; decorators own
complete selection metadata. Binding variants use separate modules such as
`aclnn_torch_npu.py` and `aclnn_xllm.py`. Host adapters require `xllm_export`, loaded
by the model before initialization or by the standalone CLI when explicitly
requested. Independent torch_npu/xlite preparation remains available without the
host. Import or ABI errors in installed backends still propagate.

### 3.3 Vendor operators, custom registration, and extension builds

RMSNorm currently provides these implementation entry points:

| Implementation | Execution entry | Torch schema owner | Package build option |
| --- | --- | --- | --- |
| `npu.aclnn.rms_norm.torch_npu` | `torch_npu.npu_rms_norm(input, weight, epsilon=eps)[0]` | torch_npu | None |
| `npu.xlite.rms_norm` | `csrc/xlite/rms_norm.cpp` | `xllm_kernel::xlite_rms_norm` | `XLLM_KERNEL_BUILD_XLITE=1` |

Native calls torch_npu directly and returns only its normalized output, matching
the former C++ wrapper's tuple selection. It requires neither an xLLM host nor
a package-owned native extension. Its selector is `npu.aclnn.rms_norm.torch_npu`. The vendor owns its schema and FakeTensor implementation.

`custom_ops.register_custom_op` is the shared preparation-time Torch registration
helper. For a new Python kernel, provide its callable, explicit `mutates_args`
and `dispatch_key`, and an optional `fake_impl`. For a schema/device kernel
already registered by C++, omit the callable, mutation metadata and dispatch key;
the helper resolves that existing operator and registers the optional fake.
It retains its Torch library and propagates registration errors. Repeated backend
initialization must reuse the prepared state rather than redefine schemas or fake
implementations. Fake functions live in the corresponding operation module,
not a separate `fake/` directory.

```python
from xllm_kernel.custom_ops import register_custom_op

# New Python kernel: the operation module supplies my_kernel and my_fake.
register_custom_op(
    "my_op", my_kernel, mutates_args=(),
    dispatch_key="PrivateUse1", fake_impl=my_fake,
)
# Existing C++ schema: load the xlite bridge before adding its fake.
register_custom_op("xlite_rms_norm", fake_impl=_fake_rms_norm)
```

Native support needs no package-owned extension. For xlite bridge development,
use a matching Torch/torch_npu and CANN environment with GVirt revision
`4a6dc3102d5778928e4d21116bc4d4694d51e584`, as required by the package's `setup.py`.
The bridge records dependency versions and launcher hashes in its build manifest.
Follow the package README for installation; after editing the C++ bridge, rebuild
it in place before starting a new worker:

```bash
(cd xllm-kernel && XLLM_KERNEL_BUILD_XLITE=1 MAX_JOBS=4 python setup.py build_ext --inplace)
```

Extensions load through `torch.ops.load_library` during preparation, not ordinary
package import. Existing host `xllm_ops` schemas and C++ model callers retain their
ownership and behavior. FakeTensor support describes metadata for tracing and
compilation; it is not required by eager/device-graph execution itself and is not
a numerical reference. Validate these paths separately.

### 3.4 Public API and current coverage

Public operators are exposed through `xllm_kernel.ops`. After package
initialization, model code uses the public function or retains its prepared
callable for the lifetime of the graph:

```python
from xllm_kernel import prepare_rms_norm
from xllm_kernel.ops import rms_norm

plan = prepare_rms_norm()
output = rms_norm(input, weight, eps)
# A model may retain plan.function directly for repeated calls or graph capture.
```

| Operator | Implementation | Execution modes | Priority and selection |
| --- | --- | --- | --- |
| `rms_norm` | `npu.aclnn.rms_norm.torch_npu` | Eager, ACLGraph | Priority 20; default for this operation |
| `rms_norm` | `npu.aclnn.rms_norm.xllm` | Eager, ACLGraph | Priority 10; requires the loaded host extension |
| `rms_norm` | `npu.xlite.rms_norm` | Eager, ACLGraph | Priority 0; selectable through the unified override |

Native preserves torch_npu's tensor domain. Its higher priority retains that
broader domain as the default; it does not imply that native is faster for every
shape. xlite accepts FP16/BF16 inputs and weights on the same NPU with the same
dtype, ND layout, widths divisible by 64 and at most 8192, at most `2**32 - 1`
input elements, and finite nonnegative FP32 epsilon. Strided ND inputs and empty
rows are supported. Neither implementation mutates or aliases its inputs;
numerical equivalence is tolerance-based.

The current host integration covers plain RMSNorm in shared layers and GLM-5.3
MLA preprocessing. Residual, quantized, and Gemma normalization keep their
existing paths. Model integration for other operators, CUDA adapters, shape specialization,
profiling/replay, compiler monitoring, and tuning remain future extensions.
Adding a declaration or test double does not establish device or model support.

## 4. Provide an operator-specific offline suite

The common interfaces are defined in
[`numerics/suite.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/suite.py).
The numerical and benchmark CLIs share the same suite so they do not maintain
separate shapes, inputs, or references. `numerics/` owns input generation,
independent references, tolerances, contract validation, and numerical reports;
`benchmark/` reuses prepared cases and adds timing. The package's `test/` tree
owns pytest cases and fixtures and is excluded from the runtime wheel; there is
no intermediate `testing/` package.

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

The validator must check the supplied callable's result. It must not ignore
that callable and invoke the original implementation again. ACLGraph validation
supplies a function that reads captured outputs; executing the eager operator
again would hide incorrect graph outputs. Validation results and metadata must
also avoid overwriting common identity fields such as case ID, operation, and
implementation.

By default, generate inputs on the CPU with a fixed seed and move them to the
device as required, so candidates receive the same logical data. The reference
must be independent of the implementation under test. Numerical references are
not serving implementations or execution-error fallbacks. Domain filtering must
match the implementation's argument checks. Direct device tests still cover
boundaries; marking every case `not_applicable` is not validation.

[`numerics/rms_norm.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/rms_norm.py)
provides a complete example. Shared CLI tests also use a test operator with
size/scale arguments and dictionary outputs to verify that runners do not depend
on RMSNorm's input/weight/eps structure. That operator tests the shared workflow;
it does not represent additional device operator support.

`PreparedCase.reset` optionally restores mutable inputs before validation, warmup,
capture/replay, and each timed call.

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
operator's shape fields. JSON and Markdown reports include implementation,
environment, coverage, validation results, and failure reasons.

Failures must remain visible: record reasons for inapplicable cases; do not time
numerical failures; report and raise capture/replay failures. A run with no
passing cases cannot produce a successful report. Device execution errors must
not trigger eager or solution fallback followed by accepted performance results.

The NPU timer supports eager and ACLGraph execution, reporting host enqueue,
device-event intervals and synchronized wall time. Without reset, each capture
contains 64 independent calls. Mutable cases capture one call and reset before
capture, replay and each measurement. Reset is outside device-event and host
enqueue intervals; the mutating path's wall time includes reset and reports this
boundary explicitly.

Preparation, warmup, and capture are excluded from steady-state timing. Keep
inputs, cases, devices, execution modes, and timing settings consistent. Report
raw samples and dispersion, and record resource occupancy. Eager device-event
intervals include host submission gaps and do not directly measure the pure
device execution time of one kernel. Inputs are reused with a hot-cache policy;
monitor other work throughout timing and use repeated baseline/candidate/baseline
runs to expose drift. Distinguish operator-level performance from end-to-end
results.

## 6. Tests and validation

Run these commands from the repository root after installing the package and
the dependencies of the implementation being tested.

### 6.1 Standalone numerical validation and benchmarks

The native suite calls torch_npu directly, and the xlite suite loads its optional
bridge; neither imports the xLLM host or requires a server. Both CLIs provide
`--help` without loading Torch or a device SDK. Use `--operation NAME --help` for
suite-specific options. For example:

```bash
python -m xllm_kernel.numerics --operation rms_norm --device npu --mode eager \
  --implementation npu.aclnn.rms_norm.torch_npu --rows 1 16 --widths 6144 \
  --revision "$(git rev-parse HEAD)" --output numerics.json
python -m xllm_kernel.benchmark --operation rms_norm --device npu --mode aclgraph \
  --implementation npu.aclnn.rms_norm.torch_npu --rows 1 16 --widths 6144 \
  --iterations 50 --revision "$(git rev-parse HEAD)" --output benchmark.json
```

Use `npu.xlite.rms_norm` for the xlite bridge. Omitting the shape and dtype
options uses the RMSNorm suite's default matrix: BF16/FP16, rows 1/16/128/4096,
and widths 512/2048/6144. Inputs are generated deterministically on the CPU and
checked against an FP32 CPU reference. Additional operators supply their own
parameters and reference through the same interfaces.

Only numerically passing cases are timed; ACLGraph timings additionally require
captured-output validation. Zero passing cases or failed validation cannot yield
a successful report. Follow section 5's timing and resource controls when
comparing implementations.

### 6.2 Package and host-integration tests

Package tests run independently through pytest. They are not wired into the
host's `tests/CMakeLists.txt`, CTest, or `all_tests`. Common tests require the
package, PyTorch, pytest, and setuptools, but no NPU or host extension:

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu --ignore=xllm-kernel/test/distributed
```

NPU integration tests require matching host and backend build artifacts. Run
the backends sequentially in separate processes on idle devices:

```bash
XLLM_KERNEL_OVERRIDE_RMS_NORM=npu.aclnn.rms_norm.torch_npu \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_rms_norm_npu.py
XLLM_KERNEL_OVERRIDE_RMS_NORM=npu.xlite.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_xlite_rms_norm_npu.py
```

The device fixture in `test/npu/conftest.py` loads the host and initializes
Python kernels; pytest does not build them. Native tests compare the torch_npu
adapter against the host and use a subprocess to check execution and FakeTensor
behavior without a host or native package extension. A process keeps its original
binding, so native and xlite use separate processes. Direct pytest does not
acquire CTest resource locks; ensure devices are idle separately. Device-guard
tests require two visible NPUs. Use the standalone CLIs above when host artifacts
are not part of the validation scope.

### 6.3 Validation order and evidence

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

For a new operator, define its contract and public API first, implement and
register its backends, connect the `OPERATION` and runtime catalog, then provide
its offline suite and tests before integrating model call sites. For another
solution of the same contract, reuse the public API and operation definition;
extend its backend list, suite implementation list, supported cases, and device
tests. Shared initialization handles the selection without a new host branch.

- Keep the contract, real argument checks, fake checks, suite domain, and
  operator documentation consistent.
- Do not add branches for the new operator's name, shapes, input roles, or
  tolerances to shared runners, reports, or the registry.
- Keep suite imports free of device dependencies. Reject unknown
  implementations and mismatched operations before execution preparation.
- Connect a new solution through its operation's backend declarations, offline
  suite, and device tests; preserve current-stream and resource-lifetime rules.
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
This is the full GLM-5.3 DSA model path. The code follows main's catalog, runtime,
selection and import-driven registration architecture.

| Operation | Solution and optional variant |
| --- | --- |
| `rms_norm` | `aclnn.xllm`, `aclnn.torch_npu`, `xlite` |

The inventory contains 1 operation contracts and 3 implementation variants
across the listed solutions. `aclnn` includes both standard CANN operators and custom
operators exposed through the ACLNN API. An xLLM binding can dispatch to multiple
ACLNN kernels; the variant name identifies the wrapper, not a fixed binary.
`atb` wraps actual ATB execution. `hccl` identifies the communication library.
`torch` is an explicit PyTorch NPU composition, and `tilelang` resolves an existing
compiled AOT operator. xlite uses pinned GVirt launchers on the current stream.
Initialization imports available candidates for the requested operations, then freezes and selects once.

For new numerical suites, `tensor_suite` builds operation-owned shape matrices.
`Example` binds a tensor tree, a CPU reference, declared mutations, comparison
tolerances and optional semantic canonicalizers. `TensorCase` carries metadata
without loading a device SDK. `PreparedCase.reset` restores mutable inputs
outside the timed event pairs; graph capture uses one operation for mutable
cases. Per-operator test files share the lifecycle in `test/npu/operator_checks.py`.
No package test is added to `tests/CMakeLists.txt`.

Catalog entries and CPU tests do not establish NPU or full-model correctness.
Device accuracy, performance and end-to-end evidence must record the actual
revision, inputs, occupancy and exit status; pre-refactor results do not validate
this implementation.

Routing and SwiGLU semantics were cross-checked against the official
[MoeInitRoutingV2 API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md)
and [DequantSwigluQuant API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_dequant_swiglu_quant.md).
The installed runtime must provide those requested interfaces; preparation or
execution fails visibly when it does not. Documentation is not binary ABI proof.
