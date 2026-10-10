# xLLM Python package

[English](README.md) | [简体中文](README_zh.md)

The Python executor separates model semantics from hardware execution. Keep the
dependency direction one-way:

```text
models  ->  layers  ->  kernels
        model_executor coordinates execution
        distributed owns parallel collectives
```

## Package responsibilities

### `models/`

Model architectures, configuration interpretation, weight loading, and model
forward composition. Models describe the network in terms of layers and logical
operators.

### `layers/`

Reusable neural-network layers and parameter ownership. Simple layers call the
active `kernels` package directly. A complex model may instead own distinct
`layers/<device>/<model>/` lowerings when devices require different operator
boundaries, parameter layouts, persistent workspaces, or graph lifecycle.

Qwen3.5 is the reference layout for this exception: its public model shell is
shared, while CUDA and NPU select separate decoder-layer implementations once
during construction. Backend selection must not occur inside the forward hot
path, and peer layer packages must not import one another.

### `kernels_<device>/`

Hardware kernels, one peer package per platform, mirroring the C++ split under
`xllm/core/kernels/`:

```text
kernels_cuda/
kernels_npu/
```

The peers share no code and never import each other. Runtime initialization
publishes the active package as `xllm.python.kernels`; the entry points are
described under [Import and initialization](#import-and-initialization).

Reusable layers and backend-owned layers may write
`from xllm.python import kernels` and reach the package selected during runtime
initialization. A backend-owned layer is imported only after its device has
been selected, so its API may follow that backend's native fusion boundary.
`setup.py` ships only the kernel package matching `--device`. xLLM builds for
more devices than the executor covers, so a device without a peer package ships
the rest of `xllm.python` and runtime initialization raises for its platform.

`model_executor/executor.py` separately selects the attention backend and graph
runner. Attention backends hold state across steps (wrappers, workspaces,
cached plans) and are wired into the executor's lifecycle, whereas a kernel
package exports stateless functions. Keeping their
selection in the executor is deliberate.

Use relative imports for sibling modules and `xllm`-rooted absolute imports
across packages.

A platform package owns everything specific to its hardware:

- Triton, FlashInfer and vendor-library launchers, under a per-framework
  subdirectory (`triton/`, `flashinfer/`);
- `torch.library` `custom_op` registration and FakeTensor contracts for the
  operators it implements in Python;
- FakeTensor contracts for its C++ operators, collected in `_custom_op.py`;
- mutation declarations and `torch.compile` graph boundaries;
- the weight layouts its kernels require.

Heavy launchers (TileLang, Triton, FlashInfer) stay lazily imported inside the
semantic operator body unless a build tool explicitly imports a leaf DSL
module.

The NPU package contains two independent leaf implementation libraries:

```text
kernels_npu/
├── tilelang/
└── triton/
```

They depend only on their DSL/framework packages and their own helpers. They do
not import the NPU semantic API, `_custom_op.py`, `torch.ops.xllm_ops`, the AOT
compiler, or the native xLLM extension. This lets C++ build tooling import the
same Python DSL implementation before the xLLM binary exists.

### Python DSL ownership and AOT reuse

`kernels_npu/tilelang/` and `kernels_npu/triton/` own the Python DSL source.
A leaf module contains the program builder, JIT launcher, implementation-local
validation and reference/debug helpers. Runtime semantic modules may call these
leaf modules, but the leaf modules never depend back on the semantic package.

Ascend TileLang AOT adapters live under
`xllm/compiler/tilelang/targets/ascend/aot/`. They contain only build metadata
and lowering concerns such as kernel-family registration, dispatch schemas,
specializations, exported ABI and source generation. An AOT adapter imports the
program builder from `kernels_npu/tilelang/`; it must not own or copy the DSL
kernel body.

The allowed dependency directions are:

```text
kernels_npu semantic API  ->  kernels_npu/{tilelang,triton}
Ascend TileLang compiler  ->  kernels_npu/tilelang
kernels_npu/{tilelang,triton}  ->  DSL/framework dependencies only
```

Build tooling imports a concrete leaf DSL module directly. It neither imports
the `kernels_npu` semantic API nor calls `initialize_runtime()`, and build-time
versus runtime behavior is not selected with an environment variable.

### Platform-owned kernel APIs

Each platform package owns its public API and declares that API in its own
`__all__`. Peer packages do not need to export the same names: models and layers
reuse the stable `xllm.python.kernels` binding, while the active peer supplies
the operations needed by the models supported on that platform.

An existing unsupported stub may remain when its explicit
`NotImplementedError` is useful, but new operators do not require matching
stubs in every peer. The model/platform support matrix in
`model_platform_support.py` records the coarse combinations expected to work.
The registry rejects an unsupported combination before importing the model
implementation.

Semantically similar operators on different platforms may use different public
functions, Torch schemas, fusion boundaries, parameters, and model-layer
composition. Each platform's tests define its own kernel contract; do not add
a peer facade solely to make a shared complex layer compile.

### `distributed/`

Parallel process groups, topology, and collectives. Shared orchestration
selects platform-specific collective implementations in `cuda/` and `npu/`.

### `model_executor/`

Execution orchestration: eager or graph runners, forward context, attention
backend setup, cache binding, and lifecycle.

## Import and initialization

`import xllm` and `import xllm.python` do not load the native extension or
initialize kernels. Native operator registration, Python runtime
initialization, and model construction are separate steps:

| Entry point | When native code becomes available | When Python kernels initialize |
| --- | --- | --- |
| `setup.py` / AOT compilation | No extension is loaded during compilation | Not initialized |
| xLLM server with embedded Python | Native operators are supplied by the host before `ensure_python_interpreter()` initializes Python | `ensure_python_interpreter()` calls `initialize_runtime()` |
| `tests/python` via pytest | Shared initialization in `conftest.py` loads `xllm_export` before test collection | `conftest.py` then calls `initialize_runtime()` |
| Python offline inference API (`xllm.LLM`) | `_load_public_api()` loads `xllm_export` while preparing the API, before engine construction | The worker calls `initialize_runtime()` if the Python model executor is selected |

Extension loading establishes access from a Python process to xLLM's C++
engine and operators. The offline API and test setup own this step. The server
already starts with native code and embeds Python afterwards, while the build
imports DSL sources to produce that native code. Neither needs to load a
separate extension for those imports.

Loading the extension makes native operators available. `initialize_runtime()`
then initializes the platform's Python kernels; it does not load the native
library. Each entry point owns the ordering described below.

### Shared runtime initialization

`initialize_runtime()` requires native operators to be registered already; it
does not load `xllm_export`. It selects the platform package, calls that
package's `_initialize_runtime()` hook, and publishes the result both as the
`xllm.python.kernels` attribute and in `sys.modules`. There is no `kernels/`
directory. These import forms resolve to the same selected package.

NPU initialization imports `_custom_op.py` to register native FakeTensor
implementations, then uses `_EXPORTS` to import semantic modules and publish
their functions. CUDA imports its Python operator wrappers when its package
is imported, but defers native FakeTensor registration to its runtime hook.
FakeTensor implementations describe operators for graph tracing.

Successful initialization is reused within each interpreter. Model creation,
weight loading, and process-group setup happen separately; each worker or
pytest process has its own initialization state.

### Build time: `setup.py` and AOT compilation

On NPU, `setup.py build` runs TileLang AOT compilation before the C++ build.
AOT adapters import DSL modules such as `kernels_npu.tilelang.rope`. Python
executes the parent packages first, but none initializes the runtime on this
path. The build then compiles the engine, stages Python packages and platform
resources, and builds `export_module`. `bdist_wheel` packages the staged files.

The NPU package must defer semantic imports here: modules such as `activation`
bind `torch.ops.xllm_ops` functions at import time. Importing them eagerly would
require native operators before the build has produced them. `_EXPORTS`
currently implements that deferred binding. AOT compilation does not run
pytest or load its `conftest.py`.

### xLLM server with embedded Python

When xLLM server selects the Python model implementation, `PyCausalLM` calls
`ensure_python_interpreter()`. That function keeps native operator registration
linked, creates an interpreter if needed, sets the package search path, imports
`xllm.python`, and calls `initialize_runtime()` before model construction.
An existing interpreter is reused. On NPU, a newly created interpreter also
runs the existing `_npu_bootstrap` adaptation before runtime initialization.

The worker supplies native operators from its linked code, so this path does
not require a separate `xllm_export` extension. `--python_model_path`, or
`XLLM_PYTHON_MODEL_PATH` when the flag is empty, selects the directory containing
the `xllm` package. With neither set, normal Python package lookup applies.

### Python tests

After building, run tests directly from the checkout root:

```bash
python -m pytest tests/python/test_model_executor.py
```

Before collecting test modules, pytest automatically loads
`tests/python/conftest.py`. It loads `xllm_export` first, then calls
`initialize_runtime()`, so tests can import model, layer, and kernel modules
normally. Test files do not need to repeat these steps. Direct pytest commands
do not build the extension.

Run the NPU Python suite with `python setup.py test --test-name python_tests`;
its per-file targets depend on `xllm_export`. CTest runs each file in a separate
pytest process, using the same `conftest.py`.

The independent kernel package owns its tests in `xllm-kernel/test/`.
Run its separate CTest group with
`python setup.py test --test-name python_kernel_tests`; see
[`xllm-kernel/README.md`](../../xllm-kernel/README.md#independent-validation-tools)
for common tests and the native/xlite NPU test commands.

### Python offline inference API

This mode uses public Python APIs such as `xllm.LLM` to call the C++ engine:

```text
from xllm import LLM
  -> load xllm_export -> import the Python API wrapper
LLM(...)
  -> construct Options -> call C++ LLMMaster / VLMMaster -> create model workers
```

During API initialization, `_load_public_api()` loads the extension before
importing the Python API wrappers. This happens before engine or model
construction and does not initialize the Python model runtime. The C++ engine
may use either a native model or a Python model. Selecting a Python model leads
to the embedded initialization path above; API callers do not need to call
`initialize_runtime()` themselves.
Worker processes may create their own interpreters rather than reuse the API
caller's interpreter.

### Source checkouts and installed wheels

When loading `xllm_export`, source checkouts use their own standard `build/`
output matching the current Python interpreter and platform. Missing output
raises an error; no copy or link into the source package is needed. Installed
wheels load the extension inside their package. Run outside the checkout,
without placing its root ahead of the installation on `sys.path`, to use a
wheel.

Once loaded, the extension is reused in that process. Rebuilding it on disk
does not replace the loaded code; start a new process to use the new build.

With `--python_model_path` pointing to the checkout root, a service restart
picks up Python model and layer edits. Changes to C++ or Python DSL kernels
compiled into native AOT artifacts require rebuilding those artifacts.

NPU source checkouts also use the independent `xllm_kernel` Python package.
Install the matching source revision with
`python -m pip install --no-deps -e ./xllm-kernel` before starting workers or
running Python tests. The regular xLLM wheel includes this package. Its first
adapter routes plain RMSNorm through the existing native operator; fused and
quantized normalization retain their current paths.

## Adding an operator

1. Add the implementation under the matching platform package's framework
   subdirectory. Keep a Python DSL implementation independent when it is shared
   by JIT runtime and native AOT builds.
2. When the native build consumes a TileLang implementation, add a thin AOT
   adapter under `xllm/compiler/tilelang/targets/ascend/aot/`; do not duplicate
   the program body there.
3. Bind the name in that platform's family module and package exports
   (`_EXPORTS` for NPU, explicit imports for CUDA), and update `__all__`.
   Register a `custom_op` and its FakeTensor implementation when
   the computation needs a stable graph node, FakeTensor propagation, or
   mutation tracking. FakeTensor contracts for C++ operators go in the
   platform's `_custom_op.py`.
4. Add launcher-level numerical tests and graph/FakeTensor tests for that
   platform. Do not add peer stubs solely to keep export lists aligned.
5. Update `model_platform_support.py` when the operator changes the supported
   model set.
6. Verify that dependencies still flow from models through layers to kernels,
   never in the reverse direction.

## Adding a platform

1. Create `kernels_<device>/` beside the existing peers and implement the API
   needed by the first models targeted for the platform.
2. Keep the package independent: it owns its exports and does not import a
   peer.
3. Add `_custom_op.py` with the FakeTensor implementations of the platform's
   C++ operators. Import it from the package's `_initialize_runtime()` hook
   after native operators have been registered.
4. Teach `Platform` in `xllm/python/platform.py` to report `<device>` (extend
   `PlatformEnum` and `_torch_device_type()`) and add the matching
   branch to `xllm.python.initialize_runtime()`.
5. `python setup.py build --device <device>` then stages it. Before that, a build
   for the device logs the packages that do exist and ships none of them.
6. Mark a model supported in `model_platform_support.py` only after its
   platform path passes functional tests.
