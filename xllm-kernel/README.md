<!-- Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. -->

# xLLM Kernel

xLLM Kernel is the operator component for Python-defined models in xLLM. It
provides public operator APIs, implementation registration and preparation,
and standalone numerical validation and benchmarking. The package starts
with Ascend NPU support and is organized to accommodate more operator
families and backend implementations.

The catalog provides 1 operation contracts and 3 implementation variants across
ACLNN, ATB, HCCL, Torch, TileLang and xlite. Model integration currently uses
RMSNorm; registering a backend does not replace other model call sites.

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

### Install xllm-kernel with the xlite backend

Run the following from the **xLLM repository root**:

```bash
# Adjust this path to your CANN installation.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install "setuptools>=68" wheel

# Install the Python package and compile its xlite backend in place.
XLLM_KERNEL_BUILD_XLITE=1 MAX_JOBS=4 python -m pip install \
  --no-build-isolation --no-deps -e ./xllm-kernel
```

Check that the xlite backend can be loaded and prepared:

```bash
python - <<'PY'
from xllm_kernel import initialize

prepared = initialize(device="npu", operations=("rms_norm",), overrides={"rms_norm": "xlite"})["rms_norm"]
print(prepared.spec.name)  # npu.xlite.rms_norm
PY
```


### Use xllm-kernel with the xLLM wheel

Complete the kernel installation above first, then build and install the xLLM
wheel with the normal
[xLLM build prerequisites](../docs/src/content/docs/en/getting_started/quick_start.md).
`xllm-kernel` is used by the Python model executor, so launch with
`--model_impl=python` and a supported Python model.

From the same repository root, after installing xllm-kernel :

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
