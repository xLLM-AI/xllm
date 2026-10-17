<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# xLLM Kernel

xLLM Kernel is the operator component for Python-defined models in xLLM. It
provides public operator APIs, implementation registration and preparation,
and standalone numerical validation and benchmarking. The package starts
with Ascend NPU support and is organized to accommodate more operator
families and backend implementations.

It features:

- Operator contracts that define numerical, mutation and aliasing semantics.
- A preparation-only registry that separates implementation selection from
  model execution.
- Adapters for existing native operators and optional package-owned extensions.
- Shared numerical validation, independent performance tools and package tests.
- A standalone Python package whose import does not load Torch, a device SDK
  or native libraries, or initialize a device.

The current implementation provides NPU RMSNorm with native and optional xlite
solutions. The architecture below describes the component as a whole; current
operator coverage is listed under [Public API](#public-api).

## Installation

Python 3.10 or later is required. From the xLLM repository root:

```bash
python -m pip install --no-deps -e ./xllm-kernel
```

This installs the Python package. Device execution requires the dependencies
of the selected implementation; the package does not install a generic Torch
wheel over an existing vendor environment. The xLLM host wheel also includes
`xllm_kernel`.

For standalone native execution, build the package-owned NPU extension in a
matching PyTorch/torch_npu environment (no xLLM host build is needed):

```bash
cd xllm-kernel
XLLM_KERNEL_BUILD_NPU=1 MAX_JOBS=4 python setup.py build_ext --inplace
cd ..
```

A standalone native wheel can be built with:

```bash
XLLM_KERNEL_BUILD_NPU=1 python -m pip wheel --no-build-isolation --no-deps ./xllm-kernel
```

Without the package extension, native execution still uses the matching xLLM
host built in its configured PyTorch, torch_npu and CANN environment:

```bash
python setup.py build
```

Optional package-owned extensions are built ahead of worker startup. The
current xlite bridge requires installed GVirt revision
`4a6dc3102d5778928e4d21116bc4d4694d51e584` and matching Torch/torch_npu and CANN:

```bash
cd xllm-kernel
XLLM_KERNEL_BUILD_XLITE=1 MAX_JOBS=4 python setup.py build_ext --inplace
cd ..
python setup.py build
```

A standalone compiled wheel can be built with
`XLLM_KERNEL_BUILD_XLITE=1 python -m pip wheel --no-build-isolation --no-deps ./xllm-kernel`.
It contains the bridge and build manifest; GVirt launcher libraries and the
NPU software stack remain external dependencies. Preparation checks the
recorded GVirt revision, Torch/torch_npu versions and launcher-library hashes
before loading the extension. Ordinary pure-Python installation does not
build or load it. Set both `XLLM_KERNEL_BUILD_NPU=1` and
`XLLM_KERNEL_BUILD_XLITE=1` to include both extensions in one build.

## Design Goals

- Organize kernels by operation, with public APIs independent of the
  selected implementation.
- Make supported devices, execution modes, input domains and numerical
  contracts explicit.
- Reuse existing native implementations while allowing new solutions to be
  developed and packaged within this component.
- Resolve dependencies and bind callables before model execution and graph
  capture, keeping selection and compilation off the serving path.
- Verify correctness and measure performance without starting an xLLM server.
- Extend operator and platform coverage incrementally, with tests and
  measurements for each supported implementation.

## Overall Design

### Layered system

```text
Preparation: load dependencies -> register implementations -> freeze registry
                                                               |
                                                    prepare(contract, device,
                                                      mode, implementation)
                                                               |
                                                        PreparedKernel
                                                               |
Execution: model -> public operator API / retained callable -> implementation
                                                               |
                                               native Torch op / local bridge
```

- **Contracts** — `OpContract` identifies an operation and its version,
  numerical mode, mutation behavior and output aliasing rules.
- **Registration** — each `KernelSpec` describes a named implementation,
  its contract, device, execution modes, input domain, solution and priority.
- **Preparation** — `KernelRegistry.prepare` filters registered implementations
  by contract, device, execution mode and optional explicit name, then selects
  by priority and stable name ordering. It returns an immutable
  `PreparedKernel` with the callable and selection diagnostics. Tensor-domain
  validation belongs to the selected implementation; preparation does not
  currently specialize plans by shape.
- **Execution** — models use a bound callable for the lifetime of their graphs.
  The registry does not run on the forward path. Missing dependencies and
  unsupported selections fail during preparation; execution errors propagate
  without switching implementations.

The xLLM host initializes the package after loading native operators and
before model import or graph capture. Repeating the same initialization is
safe; changing the device or implementation policy requires a fresh worker
and fresh graphs.

### Directory structure

```text
xllm-kernel/
├── pyproject.toml                 # Standalone Python package metadata
├── setup.py                       # Optional ahead-of-time extension build
├── python/xllm_kernel/
│   ├── __init__.py                # Initialization and public re-exports
│   ├── registry.py                # Contracts, implementation specs, prepared callables
│   ├── ops/                       # Public APIs and implementations by operation
│   │   └── rms_norm/              # Current native and xlite adapters
│   ├── fake/                      # Fake implementations for package-owned Torch ops
│   ├── numerics/                  # Inputs, reference, tolerance, validation and CLI
│   └── benchmark/                 # Timing, statistics, performance reports and CLI
├── csrc/npu/                      # Native NPU operators and Torch registration
├── csrc/xlite/                    # Optional xlite bridge sources
└── test/
    ├── test_*.py                  # Package, registry, numerical-report and CLI tests
    └── npu/                       # NPU fixtures and integration tests
```

Implementations live under `ops/<operation>/`. A solution may wrap an existing
Torch operator or use a package-owned extension. Both expose a callable with
an explicit contract and register through `KernelRegistry`.

### Solution choices

- **Native adapters** expose operators provided by device libraries or the
  xLLM host through the package's contracts and preparation interface.
- **Package-owned extensions** own their Torch namespace, device bridge and
  fake implementation. The current xlite bridge submits work on the current
  PyTorch NPU stream through the torch_npu queue, without a private stream or
  operator-level synchronization.
- **Numerical references** live under `numerics/` and provide independent
  correctness checks. They are not serving implementations or error fallbacks.

## Independent validation tools

The tools separate numerical validation, performance measurement, and tests:

- `python -m xllm_kernel.numerics` owns input generation, reference evaluation,
  tolerances, mutation/alias checks and numerical reports.
- `python -m xllm_kernel.benchmark` reuses numerical validation and owns warmup,
  eager/ACLGraph timing, statistics and performance reports.
- `test/` owns pytest cases, fixtures and assertions. It is outside the
  installed runtime package; there is no intermediate `testing/` package.

Each operation supplies an `OperationSuite` from `numerics/<operation>.py`.
It owns CLI options, case generation and implementation preparation; the
explicit catalog in `numerics/catalog.py` discovers these suites without
loading a device SDK. A `KernelCase` checks applicability and produces a
`PreparedCase` containing a callable, validator and metadata. Shared runners
and reports consume these interfaces without knowing tensor argument names,
output structures or operator-specific tolerances and limits.

Both CLIs provide `--help` without loading Torch or a device SDK. They currently
run RMSNorm cases with deterministic CPU-generated inputs and an FP32 CPU
reference. `--operation` selects a suite, or is inferred from the explicit
`--implementation`. Use `--operation NAME --help` for its options. For example,
with a built package NPU extension or native host:

```bash
python -m xllm_kernel.numerics --operation rms_norm --device npu --mode eager \
  --implementation npu.xllm_native.rms_norm --revision "$(git rev-parse HEAD)" \
  --output numerics.json
python -m xllm_kernel.benchmark --operation rms_norm --device npu --mode aclgraph \
  --implementation npu.xllm_native.rms_norm --revision "$(git rev-parse HEAD)" \
  --iterations 50 --output benchmark.json
```

Use `npu.xlite.rms_norm` to validate the optional compiled package without
loading the xLLM host. Neither CLI requires an xLLM server. JSON and Markdown
reports record the implementation, environment, coverage and failure reasons.
Zero applicable cases or failed validation cannot produce a passing report.
Cases must pass numerical checks before timing; in ACLGraph mode, graph-output
checks must also pass.

Benchmarks exclude preparation, warmup and capture from steady-state samples.
Eager measurements include host submission, device-event intervals and
synchronized wall time. ACLGraph timing amortizes launch overhead over 64
independent operations and reports per-operation samples and statistics.
Inputs are reused with a hot-cache policy. Run on idle devices, monitor other
work throughout measurement, and compare identical case and timer settings;
use repeated baseline/candidate/baseline runs to expose drift.

### Tests

Package tests run independently with pytest and are not part of the host's
CMake/CTest targets or `all_tests`. Run the following commands from the xLLM
repository root.

With the package, PyTorch, pytest and setuptools installed, run common tests
without an NPU or host extension:

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu
```

NPU integration tests require the matching host and implementation dependencies.
Run native and xlite tests sequentially in separate pytest processes with
explicit implementation selection:

```bash
XLLM_KERNEL_RMS_NORM_IMPL=npu.xllm_native.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_rms_norm_npu.py
XLLM_KERNEL_RMS_NORM_IMPL=npu.xlite.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_xlite_rms_norm_npu.py
```

The local `test/npu/conftest.py` loads the matching host extension and initializes
Python kernels before collection. These commands do not build the extension
or acquire a CTest resource lock; run them on idle devices. The device-guard
tests require two visible NPUs. Package import and fake-tensor checks do not
establish device correctness or whole-model performance.

## Public API

Public operators are exposed through `xllm_kernel.ops`. The current API can be
used inside an initialized xLLM host as follows:

```python
from xllm_kernel import prepare_rms_norm
from xllm_kernel.ops import rms_norm

plan = prepare_rms_norm()
output = rms_norm(input, weight, eps)
# A model may retain plan.function directly for its graph's entire lifetime.
```

| Operator | Implementation | Execution modes | Selection |
| --- | --- | --- | --- |
| `rms_norm` | `npu.xllm_native.rms_norm` | Eager, ACLGraph | Default |
| `rms_norm` | `npu.xlite.rms_norm` | Eager, ACLGraph | Explicit opt-in |

The host reads `XLLM_KERNEL_RMS_NORM_IMPL` at startup; set it to
`npu.xlite.rms_norm` before starting workers to select xlite. Direct package
users can call `xllm_kernel.initialize(device="npu", implementation=...)`
after providing the selected implementation's dependencies. Native execution
requires either the package NPU extension or preloaded host Torch operators.

The native adapter preserves its existing tensor domain. xlite accepts
FP16/BF16 inputs and weights on the same NPU, ND layout, widths divisible by
64 and at most 8192, at most `2**32 - 1` input elements, and finite nonnegative
FP32 epsilon. Strided ND inputs and empty rows are supported. Neither
implementation mutates or aliases its inputs; equivalence is tolerance-based.

The current host integration covers plain RMSNorm in shared layers and
GLM-5.3 MLA preprocessing. Residual, quantized and Gemma RMSNorm and other
operator families remain on their existing paths. CUDA adapters, shape-based
specialization, profiling/replay, compiler monitoring and tuning are future
extensions, not current package capabilities.

## Adding an operator or implementation

1. Define the operation contract and public API under `ops/<operation>/`, including
   input domains, mutation/aliasing behavior and numerical expectations.
2. Register each implementation with a `KernelSpec`. Resolve dependencies at
   preparation time and validate unsupported arguments before launching work.
3. Bind the callable before model import/capture. Preserve current-stream
   ordering and keep selection, compilation and synchronization out of the
   operator's steady-state path.
4. Add fake support for new package-owned Torch operators. Define its inputs,
   reference, tolerances and `OperationSuite` in `numerics/<operation>.py`,
   then add the suite to the catalog. The common benchmark runner reuses its
   prepared cases; keep operator logic out of shared runners and reports.
5. Add package and device tests under `test/`, covering correctness, unsupported
   domains, streams and graph execution. Document the supported API and
   dependencies here, and validate the intended model integration separately.

See the operator development guide
([English](../docs/src/content/docs/en/dev_guide/xllm_kernel_dev.md) |
[简体中文](../docs/src/content/docs/zh/dev_guide/xllm_kernel_dev.md))
for the implementation workflow and validation requirements.

## Acknowledgment

This project was made possible thanks to the following open-source projects:

- [TokenSpeed](https://github.com/lightseekorg/tokenspeed) - xLLM Kernel draws inspiration from TokenSpeed's kernel architecture design.
