---
title: "xLLM-Kernel Development Guide"
---

`xllm-kernel/` provides operator APIs, implementation registration and preparation,
and independent numerical validation and benchmarking for Python models.
This guide covers the common workflow for adding operators, extending implementations,
and integrating and validating model call sites. RMSNorm is only an example;
other operators use the same interfaces. See the
[package README](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/README.md)
for installation.

## 1. Responsibilities and directories

The component organizes implementations by operator, keeping model-facing APIs
independent of the selected solution. Contracts describe numerical behavior,
mutation, aliasing, supported devices, and input domains. Implementations can
reuse vendor operators or provide package-owned extensions without changing the
model API.

Registration, dependency checks, and binding happen before model execution and
graph capture. Models retain prepared callables, keeping selection and compilation
off the serving path. Numerical validation and benchmarking run independently of
an xLLM server. Additional operator families and platforms follow these boundaries,
with tests and measurements for each implementation.

Directories are organized by responsibility:

| Location | Responsibility and integration point |
| --- | --- |
| `python/xllm_kernel/ops/<operation>/` | Operator contracts, public APIs, implementations, and registration |
| `python/xllm_kernel/registry.py` | `OpContract`, `KernelSpec`, `PreparedKernel`, and the registry |
| `python/xllm_kernel/{catalog,runtime,selection}.py` | Discover operators, load dependencies, select implementations, and publish bindings |
| `python/xllm_kernel/custom_ops.py` | Register package-owned Torch schemas, device implementations, and fakes |
| `csrc/<solution>/` | Directory convention for native extension sources and device launcher bridges |
| `pyproject.toml`, `setup.py`, `MANIFEST.in` | Package metadata, extension builds, and distribution artifacts |
| `python/xllm_kernel/numerics/<operation>.py` | Operator-specific arguments, inputs, references, tolerances, and suite |
| `python/xllm_kernel/numerics/catalog.py` | Register suite modules in `_BUILTINS` |
| `python/xllm_kernel/numerics/{suite,factory,inputs,tensor_case}.py` | Case interfaces, input construction, tensor-tree validation, and reset |
| `python/xllm_kernel/numerics/{arguments,verify,report,distributed}.py` | Shared CLI, validation, reporting, and distributed execution |
| `python/xllm_kernel/benchmark/` | Reuse suites for timing and performance reports |
| `test/`, `test/npu/` | Shared interface tests and device implementation tests |

All paths above are relative to `xllm-kernel/`. Model structure, batch metadata,
KV cache management, and request scheduling belong to the xLLM runtime. Operators
consume prepared inputs; a solution must not introduce its own request scheduler
or cache lifecycle.

Each operator module defines its inputs, shapes, reference, tolerances, and supported
domain. The shared registry, runners, and reports reuse these through interfaces.
Public API exports, catalog module lists, and extension source lists are the
integration points for new operators.

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
describes operator semantics: `operation`, `version`, `mutates`, `aliases_inputs`,
and `numerical_mode`. `KernelSpec` describes an implementation: its name, device,
execution modes, input domain, solution, variant, binding, and priority.
`input_domain` is a descriptive string; the underlying implementation or vendor
operator validates actual arguments.

Implementations of the same operator can share an `OpContract` only when their
semantics match. Changes to mutation, aliasing, quantization, or output semantics
require a distinct contract, rather than just a different implementation name.
Numerical equivalence may use declared tolerances and should not imply bitwise
equality by default.

## 3. Implementation, registration, and model integration

### 3.1 Implementation and registration

Place implementations in `ops/<operation>/`, usually in `<solution>.py`; separate
modules may distinguish bindings or variants. Declare the callable at module scope
with `@register_kernel(...)`. Importing the module registers it in the shared
registry and returns the callable unchanged, without a forward wrapper.

Use this naming convention for `name`:

```text
<device>.<solution>.<operation>[.<variant>]
```

- `device`: the target device, matching the `device` field, such as `npu`.
- `solution`: the execution solution, matching the `solution` field, such as `torch`.
- `operation`: the operator name from `OpContract.operation`, such as `rms_norm`.
- `variant`: an optional suffix distinguishing implementations of the same
  operator within a device and solution.

Square brackets denote an optional segment and are not part of the actual name.
When omitting `variant`, omit its leading dot as well; for example, RMSNorm uses
`npu.torch.rms_norm`. Names must be unique across the process registry.
This format is a naming convention; the registry does not parse it to populate
metadata, so declare the corresponding fields separately.
The following example registers RMSNorm through a vendor API:

```python
import torch

from xllm_kernel import PlatformEnum, current_platform, register_kernel
from xllm_kernel.ops.normalization import RMS_NORM_CONTRACT

if current_platform.is_npu():
    import torch_npu

    _native_rms_norm = torch_npu.npu_rms_norm


    @register_kernel(
        name="npu.torch.rms_norm",
        contract=RMS_NORM_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="torch_npu.npu_rms_norm tensor domain",
        solution="torch",
        binding="torch_npu",
        priority=20,
    )
    def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return _native_rms_norm(input, weight, epsilon=eps)[0]
```

### 3.2 Automatic selection and preparation

Call `initialize()` before model import and graph capture to obtain read-only
operator bindings:

```python
from xllm_kernel import initialize

bindings = initialize(device="npu", operations=("rms_norm",))
plan = bindings["rms_norm"]
```

```text
Discover definitions → Import/register required implementations → Freeze the registry
                     → Select implementations → Publish bindings
```

- Declare the process's complete required operator set in one `operations` argument;
  omitting it prepares the entire catalog. Implementations are filtered by contract,
  device, and execution mode, then selected by descending priority and lexicographic
  name order.
- Use `overrides` or `XLLM_KERNEL_OVERRIDE_<OPERATION>` to select a full implementation
  name or solution; environment variables take precedence.
- The same configuration reuses its bindings, without selection during forward/replay.
  Configuration changes require a new worker.
- Missing dependencies exclude an implementation; errors loading an installed
  implementation propagate. Bindings are published only after all required operators
  prepare successfully.

### 3.3 Vendor operators and custom registration

Reuse vendor-owned Torch schemas directly. For new package-owned Torch operators,
use `register_custom_op()`:

```python
from xllm_kernel.custom_ops import register_custom_op

# Python implementation: declare mutation and the dispatch key explicitly.
register_custom_op(
    "my_op", my_kernel, mutates_args=(),
    dispatch_key="PrivateUse1", fake_impl=my_fake,
)

# Existing C++ schema: load its extension first, then add only the fake.
register_custom_op("my_native_op", fake_impl=my_native_fake)
```

- Place native extensions in `csrc/<solution>/` and update `setup.py`, source
  manifests, and wheel artifacts together.
- Load extensions and register operators during preparation, then reuse that state.
- Fakes infer metadata such as shape, dtype, and device; they do not replace
  numerical validation.

### 3.4 Public API and model integration

Keep public definition modules SDK-free; implementation modules load device
dependencies. To add an operator:

1. Declare its contract and `OPERATION` in `ops/<group>/__init__.py`.
   Use `BackendModule` to list implementation modules, dependencies, and names.
   Export an `OPERATIONS` tuple, even when the group contains only one operator:

   ```python
   OPERATIONS = (OPERATION,)
   ```

   Groups with several operators export every declaration in that tuple, as
   `ops/normalization/__init__.py` does for the normalization operators and fused-QKV helpers.
2. Add the group module to `catalog._BUILTINS` and export its public API
   from `ops/__init__.py`.
3. Have the public API obtain its fixed binding through `OPERATION.get_prepared()`;
   include the operation name, rather than the group name, in `operations` during
   model initialization.
4. Connect the model or layer call sites. Add an offline suite when a custom kernel
   needs numerical or performance validation; native Torch adapters do not require
   a duplicate numerical-comparison suite.

Example public API call after initialization:

```python
from xllm_kernel.ops import rms_norm

output = rms_norm(input, weight, eps)
```

Models may also retain `bindings[operation].function` for the graph's entire lifetime.

## 4. Provide an operator-specific offline suite

Numerical validation and benchmarking share an `OperationSuite`. Each operator
defines its inputs, reference, and tolerances.

### 4.1 `OperationSuite`

There are currently no built-in numerical suites, including RMSNorm. The shared
numerical and benchmark tools remain available for custom kernels. Before using
operation-specific CLI options, implement a suite in `numerics/<operation>.py`,
export `SUITE`, and add its full module path to `_BUILTINS` in
`numerics/catalog.py`.

The following is a template for a custom operation. Define all three callbacks
in that module and use the operation and implementation names from its catalog:

```python
from xllm_kernel.numerics.suite import OperationSuite

SUITE = OperationSuite(
    operation="my_operation",
    implementations=("npu.my_solution.my_operation",),
    add_arguments=_add_arguments,  # Register CLI arguments
    cases=_cases,                # Generate cases
    prepare=_prepare,            # Prepare the implementation
)
```

Suite discovery loads only metadata. Import Torch, device SDKs, and extensions
during preparation or execution so that `--help` works independently.

### 4.2 `KernelCase` and `PreparedCase`

| Object | Interfaces and fields |
| --- | --- |
| `KernelCase` | `case_id`, `parameters`, `unsupported_reason(plan)`, `prepare(plan)` |
| `PreparedCase` | `function`, `validate`, `metadata`, optional `reset` |

`prepare(plan)` creates inputs and returns a `PreparedCase`. `validate(function)`
uses an independent reference and tolerances to check the supplied callable's
output, mutation, and aliasing; under ACLGraph it checks captured outputs.
Restore mutable inputs through `reset`.
`unsupported_reason(plan)` returns a reason for unsupported cases, otherwise `None`.

### 4.3 Reusable tensor-tree cases

For tensor-tree outputs, reuse `tensor_suite()` in `numerics/factory.py` to generate
a suite from positive dimension matrices; the builder returns an `Example`.
`TensorCase` binds inputs during preparation, and `numerics/inputs.py` provides
random inputs with a fixed CPU seed. Use a custom `KernelCase` for empty dimensions.
See the generic helpers in
[`numerics/factory.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/factory.py)
and the `Example` / `TensorCase` interfaces in
[`numerics/tensor_case.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/tensor_case.py).
The synthetic suites in
[`test/test_kernel_cli.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/test/test_kernel_cli.py)
illustrate the CLI protocol using CPU test doubles, rather than device kernels.

## 5. Shared validation, timing, and reporting

```text
Prepare implementation → Check support and create inputs → Validate → Warmup
                      → Optional graph capture/replay and output validation
                      → Benchmark timing → JSON / Markdown
```

Time only cases that pass validation. Record reasons for inapplicable cases and
propagate execution or graph errors. Zero passing cases cannot produce a successful
report.

| Mode | Execution and timing |
| --- | --- |
| Eager | Call the implementation repeatedly; report host enqueue time, device-event intervals, and synchronized wall time |
| ACLGraph | Capture and replay, validating graph output first; benchmarks capture 64 calls per graph by default, or 1 with `reset` |

- Preparation, warmup, and capture are excluded from steady-state timing;
  the numerical CLI captures only one call per graph.
- Run `reset` outside device-event and host enqueue intervals. The mutating path's
  wall time includes its cost.
- Reports record implementation, environment, coverage, and validation results;
  performance reports also include raw timing samples. Record `solution`, `variant`,
  and `binding` separately.
- Launch `--distributed` with torchrun and at least two NPU ranks. Write reports per
  rank and aggregate validation consensus and maximum timing samples across ranks.

Keep inputs and configuration consistent for performance comparisons, record device
occupancy, and use baseline/candidate/baseline runs to check drift.
Eager event intervals include host submission gaps. Report operator-level and
end-to-end performance separately.

## 6. Tests and validation

Run the following commands from the repository root after installing the package.
Device execution also requires the dependencies of the implementation being tested
and an idle NPU.

### 6.1 Standalone numerical validation and benchmarks

```bash
python -m xllm_kernel.numerics --help
python -m xllm_kernel.benchmark --help
```

The suite catalog is empty by default, so operation-specific commands are not yet
available. First register a custom suite as described in section 4. The following
templates assume it declares `my_operation` and `npu.my_solution.my_operation`;
replace those names and add any arguments required by your suite:

```bash
python -m xllm_kernel.numerics --operation my_operation --help
python -m xllm_kernel.benchmark --operation my_operation --help

python -m xllm_kernel.numerics --operation my_operation --device npu --mode eager \
  --implementation npu.my_solution.my_operation \
  --revision "$(git rev-parse HEAD)" --output numerics.json
python -m xllm_kernel.benchmark --operation my_operation --device npu --mode aclgraph \
  --implementation npu.my_solution.my_operation \
  --iterations 50 --revision "$(git rev-parse HEAD)" --output benchmark.json
```

Each registered suite defines its operator's arguments and defaults. Use
`--operation NAME --help` only after registering that suite.

### 6.2 Package tests

Package tests run independently through pytest. They are not wired into the host's
`tests/CMakeLists.txt`, CTest, or `all_tests`. Common tests require the package,
PyTorch, pytest, and setuptools:

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu
```
