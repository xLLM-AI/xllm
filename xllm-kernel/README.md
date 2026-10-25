<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# xLLM Kernel

xLLM Kernel is the operator component for Python-defined models in xLLM. It
provides public operator APIs, implementation registration and preparation,
and standalone numerical validation and benchmarking. The package starts
with Ascend NPU support and is organized to accommodate more operator
families and backend implementations.


It features:

- Shared lazy platform detection and platform-specific backend discovery.
- Operator contracts that define numerical, mutation and aliasing semantics.
- A preparation-only registry that separates implementation selection from
  model execution.
- Adapters for existing native operators and optional package-owned extensions.
- Shared numerical validation, independent performance tools and package tests.
- A standalone Python package whose import does not load Torch, a device SDK
  or native libraries, or initialize a device.

See the development guide
([English](../docs/src/content/docs/en/dev_guide/xllm_kernel_dev.md) |
[简体中文](../docs/src/content/docs/zh/dev_guide/xllm_kernel_dev.md))
for architecture, public APIs, operator development, and validation.

GEMM APIs live in `xllm_kernel.ops.gemm` and are also exported by
`xllm_kernel.ops`. Like normalization and activation, the group publishes
`OPERATIONS`; backend modules register only on their matching platform.

| Operation | Input and output | Implementations |
| --- | --- | --- |
| `einsum` | BF16 `[T,H,D] x [H,D,O] -> [T,H,O]` | `npu.atb.einsum` |
| `einsum_out` | Same contraction into caller-owned contiguous BF16 output | `npu.atb.einsum_out` |

Only `einsum_out` mutates an argument and returns its output buffer; that
buffer must not alias either operand. `einsum` allocates a fresh output.
ATB bindings require the matching xLLM native library to be loaded before
`initialize()`. Selection stays fixed before graph capture; overrides
cannot bypass platform or contract compatibility.

The native implementation lives in `csrc/atb/atb_matmul_ein_sum.cpp`.
The xLLM NPU build compiles this source and registers its existing
`torch.ops.xllm_ops` entry points; the bindings use the matching xLLM native library.

## Installation

### Install the standalone package

Run the following from the **xLLM repository root**:

```bash
python -m pip install --no-deps -e ./xllm-kernel
```

### Use xllm-kernel with the xLLM wheel

The xLLM wheel includes the kernel package. Build and install it with the normal
[xLLM build prerequisites](../docs/src/content/docs/en/getting_started/quick_start.md).
`xllm-kernel` is used by the Python model executor, so launch with
`--model_impl=python` and a supported Python model.

From the repository root:

```bash
# 1. build the xLLM wheel and then install it
python setup.py bdist_wheel
python -m pip install --force-reinstall --no-deps ./dist/xllm_npu*.whl
# or
# 2. install xLLM wheel directly for development
python setup.py install
```

## Acknowledgment

- [TokenSpeed](https://github.com/lightseekorg/tokenspeed) - xLLM Kernel draws inspiration from TokenSpeed's kernel architecture design.
