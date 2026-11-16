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

## Native runtime (migration PR01)

The independent native target is `xllm_kernel::native`. It owns the ACLNN
launcher, current-stream lookup, workspace allocation and resource location;
it does not include or link xLLM core utilities. Operator implementations and
schemas stay with their existing owner until their scheduled migration.
`core/kernels/npu/aclnn/pytorch_npu_helper.hpp` is a forwarding include for
unmigrated families, with removal due in PR21.

Build a standalone NPU wheel in the matching Torch/CANN environment:

```bash
cd xllm-kernel
XLLM_KERNEL_BUILD_NATIVE=1 XLLM_KERNEL_USE_NPU=ON python setup.py bdist_wheel
```

Without these options the package remains Python-only. An SDK-free native
runtime build is also available with `XLLM_KERNEL_BUILD_NATIVE=1` alone. The
xLLM NPU build uses the same CMake target and installs its shared library,
headers and exported CMake target into the sibling `xllm_kernel` package.
Native clients set `xllm_kernel_DIR` to the installed `cmake/` directory,
call `find_package(xllm_kernel CONFIG REQUIRED)`, and link
`xllm_kernel::native`. Headers live under `include/xllm_kernel/native/`. The C ABI version entrypoint is
`xllm_kernel_native_abi_version()`; operator-specific APIs follow in later PRs.

Startup order and ownership:

1. Ordinary package, catalog and DSL imports load no Torch, device SDK or
   native library.
2. Python workers explicitly call `xllm_kernel.native.load_native()` before
   backend imports/preparation. This registers package OPP roots before Torch
   loading and loads the package `libxllm_kernel.so` once. Native clients link that
   same library directly; they call `native::configure_resources()` before device initialization (the
   xLLM native startup does this before `aclInit`). PR01 ships no custom OPP components.
3. The runtime owns device initialization, graph objects, caches and borrowed
   communication handles. `native::current_stream()` borrows the caller's
   stream; workspace tensors use Torch's allocator and stay alive through
   enqueue, including the allocator's graph-pool lifetime.
4. `xllm.python.kernel_requirements.NPU_WORKER_OPERATIONS` declares the full
   migrated worker operation set. A model-specific subset can be supplied to
   `initialize_runtime(operations=...)`; it is prepared once before model
   construction/capture. A different operation set requires a fresh worker.
   Every family migration updates this declaration alongside its callers.
5. Preparation freezes implementation selection. Forward/capture/replay call
   the retained binding and do not select, benchmark or retry another backend.

The [migration ledger](../docs/design/xllm_ops_migration.json) assigns components,
all 68 NPU schemas, native entrypoints, DSLs and resources to their owning PRs.
Target names are proposals until that PR defines its contract. PR02 must prove
the custom-source/resource/provider chain with GammaAddRmsNorm; the existing
Torch NPU normalization adapter validates preparation only.
