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

See the development guide
([English](../docs/src/content/docs/en/dev_guide/xllm_kernel_dev.md) |
[简体中文](../docs/src/content/docs/zh/dev_guide/xllm_kernel_dev.md))
for architecture, public APIs, operator development, and validation.

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
