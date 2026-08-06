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

The peers share no code and never import each other. `xllm/python/__init__.py`
binds the one matching the active platform:

```python
if platform.is_gpu():
    from xllm.python import kernels_cuda as kernels
elif platform.is_npu():
    from xllm.python import kernels_npu as kernels
```

Layers and models write `from xllm.python import kernels` and reach a fixed
name, so they carry no hardware branch. `setup.py` ships only the package
matching `--device`. xLLM builds for more devices than the executor covers, so
a device without a peer package ships the rest of `xllm.python` and the import
above raises for its platform.

This is one of two places where the executor branches on hardware. The other is
`model_executor/executor.py`, which selects the attention backend and the graph
runner. Attention backends are not kernels: they hold state across steps
(wrappers, workspaces, cached plans) and are wired into the executor's
lifecycle, whereas a kernel package exports stateless functions. Keeping their
selection in the executor is deliberate.

There is no `kernels` directory on disk, so `xllm/python/__init__.py` also
publishes the bound package in `sys.modules` under that name. Every import form
then resolves -- `import xllm.python.kernels` and `from xllm.python.kernels
import rms_norm` as well as the attribute form -- and all of them reach the same
module object, so no kernel package is imported twice.

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

### The `__all__` contract

Every platform package exports the same set of names, declared in its own
`__all__`. Where a platform has no kernel for a name, it still exports the name,
bound to a function that carries the intended signature and raises
`NotImplementedError` explaining what is missing and where the reference
implementation lives.

That makes the stub set the work list for a new platform: a kernel author fills
in stubs against a peer's `__all__` without reading that peer's kernel sources.
Nothing enforces the contract at runtime -- keeping the packages independent
means keeping the check in review.

Semantically identical operators on different platforms may use different Torch
schemas, different fusion boundaries, and different parameters. The shared
contract is the Python signature behind the exported name, not the Torch
operator name.

### `distributed/`

Tensor-parallel process groups and collectives. The collectives are
hardware-neutral: only the ProcessGroup backend differs (NCCL on CUDA, HCCL on
NPU) and torch selects it from the device.

### `model_executor/`

Execution orchestration: eager or graph runners, forward context, attention
backend setup, cache binding, and lifecycle.

## Import safety

Only one platform package is ever imported, so it may import its own hardware
dependencies at module level. It must never import a peer package. Heavy
launchers (Triton modules, FlashInfer) stay lazily imported inside the operator
body so that importing the package does not compile kernels.

`import xllm.python` pulls in the platform's kernel package, so it needs the
platform's runtime and `torch.ops.xllm_ops`. The C++ worker registers those
before importing this package.

Editing Python needs no rebuild. `--python_model_path` puts the source tree on
`sys.path`, and every peer package is present there, so a service restart picks
up the change.

## Adding an operator

1. Add the launcher under the matching platform package's framework
   subdirectory.
2. Bind the name in that platform's family module and add it to the package
   `__all__`, registering a `custom_op` and its FakeTensor implementation when
   the computation needs a stable graph node, FakeTensor propagation, or
   mutation tracking. FakeTensor contracts for C++ operators go in the
   platform's `_custom_op.py`.
3. Add the name to every peer package, as a real kernel or as a stub carrying
   the signature and the reason it is missing.
4. Add launcher-level numerical tests and graph/FakeTensor tests.
5. Verify that dependencies still flow from models through layers to kernels,
   never in the reverse direction.

## Adding a platform

1. Create `kernels_<device>/` beside the existing peers.
2. Copy a peer's `__all__` and implement or stub every name. A stub carries the
   full signature and a `NotImplementedError` naming the reference
   implementation.
3. Add `_custom_op.py` with the FakeTensor implementations of the platform's
   C++ operators.
4. Teach `platform.current_platform()` to report `<device>` and add the matching
   branch to the import in `xllm/python/__init__.py`.
5. `setup.py --device <device>` then ships it. Before that, a build for the
   device logs the packages that do exist and ships none of them.
