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
| `csrc/<solution>/` | Optional native extensions, Torch bridges, and device launcher adapters; currently `npu/` and `xlite/` | Add the implementation and device registration |
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
`execution_modes`, `input_domain`, `solution`, and `priority`. `input_domain` is
currently a descriptive string: the registry does not use it to validate shapes
or layouts automatically. The implementation must check arguments before launch.

Only implementations with matching semantics may share an `OpContract` when
adding a solution for an existing operator. Changes to mutation, aliasing,
quantization, or output semantics require a distinct contract; an implementation
name alone does not express that difference. Numerical equivalence may use
declared tolerances and should not imply bitwise equality by default.

## 3. Implementation, registration, and model integration

### 3.1 Implementation and registration

Provide `register(registry)` in `ops/<operation>/<solution>.py`. Resolve dependencies
and check runtime conditions before registering the `KernelSpec` and callable.
See the existing
[native adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/npu.py)
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

### 3.3 Package-owned extensions and builds

RMSNorm currently provides these implementation entry points:

| Implementation | C++ source | Torch operator | Build option |
| --- | --- | --- | --- |
| `npu.xllm_native.rms_norm` | `csrc/npu/rms_norm.cpp` | `xllm_kernel::rms_norm` | `XLLM_KERNEL_BUILD_NPU=1` |
| `npu.xlite.rms_norm` | `csrc/xlite/rms_norm.cpp` | `xllm_kernel::xlite_rms_norm` | `XLLM_KERNEL_BUILD_XLITE=1` |

The native extension calls torch_npu's RMSNorm directly without linking the
xLLM host. During preparation, `ops/rms_norm/npu.py` checks for
`xllm_kernel._npu`: when present, it loads and binds the package operator;
otherwise, it requires the host to have registered `xllm_ops::rms_norm`.
Both paths use the same implementation name and contract. A failure to load
an installed extension is an error, not a reason to switch to the host.
This is preparation-time dependency selection, not execution-error fallback.

Build from the repository root in a matching Torch/torch_npu environment:

```bash
python -m pip install --no-deps -e ./xllm-kernel
(cd xllm-kernel && XLLM_KERNEL_BUILD_NPU=1 MAX_JOBS=4 python setup.py build_ext --inplace)

# Build an independently installable native wheel
XLLM_KERNEL_BUILD_NPU=1 MAX_JOBS=4 \
  python -m pip wheel --no-build-isolation --no-deps ./xllm-kernel
```

Set both `XLLM_KERNEL_BUILD_NPU=1` and `XLLM_KERNEL_BUILD_XLITE=1` to include
both extensions in one wheel. The xlite extension also requires the GVirt
revision specified in the README and a matching CANN environment. With neither
option set, installation stays pure Python. Build directories are isolated by
option; generated `.so` files do not belong in the source repository.

Extensions load through `torch.ops.load_library` during preparation, not during
ordinary package import. Extensions sharing the `xllm_kernel` namespace use
`TORCH_LIBRARY_FRAGMENT` to add schemas without redefining the entire namespace;
device implementations register for `PrivateUse1`. Register the corresponding
fake implementation after loading its schema, and avoid duplicate registration
on repeated preparation. Tests should cover execution without the host,
coexistence with host operators, and different extension loading orders.

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

`PreparedCase` has three fields:

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
operator's shape fields.

Failures must remain visible: record reasons for inapplicable cases; do not time
numerical failures; report and raise capture/replay failures. A run with no
passing cases cannot produce a successful report. Device execution errors must
not trigger eager or solution fallback followed by accepted performance results.

The current timer supports NPU eager and ACLGraph execution. Each graph benchmark
capture contains 64 independent calls. Cases must therefore be safe to invoke
repeatedly, with reusable input state during timing. For new operators that
mutate inputs, consume state, or depend on randomness, first design explicit
reset behavior and timing boundaries, then extend the common case interface.
Do not apply the current loop unchanged.

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
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu

# Device integration tests: build the matching host/bridge first, then run
# sequentially on idle devices
XLLM_KERNEL_RMS_NORM_IMPL=npu.xllm_native.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_rms_norm_npu.py
XLLM_KERNEL_RMS_NORM_IMPL=npu.xlite.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_xlite_rms_norm_npu.py
```

The device fixture in `test/npu/conftest.py` loads the host and initializes
Python kernels; pytest does not build them. With the package native extension
built, native tests compare the package operator against the host and use a
separate subprocess to check execution and FakeTensor behavior without importing
the xLLM host. Without that extension, tests exercise the host path and skip the
standalone extension case. Implementation bindings cannot be switched within
one process, so native and xlite tests use separate processes.
Direct pytest runs do not acquire CTest device resource locks; ensure devices
are idle separately. Device-switching tests require two visible NPUs.

The offline tool examples use a currently supported operator. With the package
native extension installed, the native suite does not load the xLLM host;
otherwise it loads `xllm_export`. Use these CLIs to validate the standalone
extension without the host-dependent pytest fixture. After integrating a new
operator, replace the operation, implementation, and specific arguments:

```bash
python -m xllm_kernel.numerics --operation rms_norm --device npu --mode eager \
  --implementation npu.xllm_native.rms_norm --rows 1 16 --widths 6144 \
  --revision "$(git rev-parse HEAD)" --output numerics.json
python -m xllm_kernel.benchmark --operation rms_norm --device npu --mode aclgraph \
  --implementation npu.xllm_native.rms_norm --rows 1 16 --widths 6144 \
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
