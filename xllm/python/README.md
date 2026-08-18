# xLLM Python package layout

The Python executor separates model semantics from hardware execution. Keep the
dependency direction one-way:

```text
models  ->  layers  ->  kernels
        model_executor coordinates execution
        distributed owns tensor-parallel collectives
```

## Package responsibilities

### `models/`

Model architectures, configuration interpretation, weight loading, and model
forward composition. Models describe the network in terms of layers and logical
operators.

### `layers/`

Reusable neural-network layers and parameter ownership. A layer calls
`kernels`, which is already bound to the active platform, so a layer contains no
hardware branch.

### `kernels_<device>/`

Hardware kernels, one peer package per platform, mirroring the C++ split under
`xllm/core/kernels/`:

```text
kernels_cuda/
kernels_npu/
```

The peers share no code and never import each other. The embedded C++ bootstrap
calls `xllm.python.initialize_runtime()` after registering native operators and
before importing a model module. That function selects the active package and
publishes it as `xllm.python.kernels`:

```python
xllm.python.initialize_runtime()
from xllm.python import kernels
```

Layers and models write `from xllm.python import kernels` and reach a fixed
name, so they carry no hardware branch. `setup.py` ships only the package
matching `--device`. xLLM builds for more devices than the executor covers, so
a device without a peer package ships the rest of `xllm.python` and runtime
initialization raises for its platform.

This is one of two places where the executor branches on hardware. The other is
`model_executor/executor.py`, which selects the attention backend and the graph
runner. Attention backends are not kernels: they hold state across steps
(wrappers, workspaces, cached plans) and are wired into the executor's
lifecycle, whereas a kernel package exports stateless functions. Keeping their
selection in the executor is deliberate.

There is no `kernels` directory on disk, so runtime initialization also
publishes the bound package in `sys.modules` under that name. After
initialization, every import form resolves -- `import xllm.python.kernels` and
`from xllm.python.kernels import rms_norm` as well as the attribute form -- and
all of them reach the same module object, so no kernel package is imported
twice.

Imports inside a platform package are relative (`from .triton.l2_norm import
...`), so the package does not name itself and a peer can be created by copying
it. Absolute imports rooted at `xllm` are for code outside the package.

A platform package owns everything specific to its hardware:

- Triton, FlashInfer and vendor-library launchers, under a per-framework
  subdirectory (`triton/`, `flashinfer/`);
- `torch.library` `custom_op` registration and FakeTensor contracts for the
  operators it implements in Python;
- FakeTensor contracts for its C++ operators, collected in `_custom_op.py`;
- mutation declarations and `torch.compile` graph boundaries;
- the weight layouts its kernels require.

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
functions, Torch schemas, fusion boundaries, and parameters. Each platform's
tests define its own kernel contract.

### `distributed/`

Tensor-parallel process groups and collectives. The collectives are
hardware-neutral: only the ProcessGroup backend differs (NCCL on CUDA, HCCL on
NPU) and torch selects it from the device.

### `model_executor/`

Execution orchestration: eager or graph runners, forward context, attention
backend setup, cache binding, and lifecycle.

## Import safety

`import xllm.python` is runtime-neutral: it does not detect a platform or import
a kernel package. `import xllm.python.kernels_npu` is also build-safe and does
not register fake operators or import semantic modules. The C++ worker
registers `torch.ops.xllm_ops` first and then calls
`xllm.python.initialize_runtime()`.

Heavy launchers (TileLang, Triton, FlashInfer) stay lazily imported inside the
semantic operator body unless a build tool explicitly imports a leaf DSL
module. A platform package must never import a peer package.

Editing Python needs no rebuild. `--python_model_path` puts the source tree on
`sys.path`, and every peer package is present there, so a service restart picks
up the change.

## Adding an operator

1. Add the implementation under the matching platform package's framework
   subdirectory. Keep a Python DSL implementation independent when it is shared
   by JIT runtime and native AOT builds.
2. When the native build consumes a TileLang implementation, add a thin AOT
   adapter under `xllm/compiler/tilelang/targets/ascend/aot/`; do not duplicate
   the program body there.
3. Bind the name in that platform's family module and add it to that package's
   `__all__`, registering a `custom_op` and its FakeTensor implementation when
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
   C++ operators.
4. Teach `Platform` in `xllm/python/platform.py` to report `<device>` (extend
   `PlatformEnum` and the detection in `Platform.enum`) and add the matching
   branch to `xllm.python.initialize_runtime()`.
5. `setup.py --device <device>` then ships it. Before that, a build for the
   device logs the packages that do exist and ships none of them.
6. Mark a model supported in `model_platform_support.py` only after its
   platform path passes functional tests.
