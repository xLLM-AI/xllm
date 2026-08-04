# xLLM Python package layout

The Python executor separates model semantics from hardware execution. Keep the
dependency direction one-way:

```text
models / layers  ->  ops  ->  kernels
        model_executor coordinates execution
```

## Package responsibilities

### `models/`

Model architectures, configuration interpretation, weight loading, and model
forward composition. Models should describe the network in terms of layers and
logical operators, not import a hardware kernel directly.

### `layers/`

Reusable neural-network layers and parameter ownership. A layer may select a
logical operator, but hardware dispatch and graph registration belong to
`ops/`.

### `ops/`

The graph-facing operator layer. It owns:

- `torch.library` schemas and `custom_op` registration;
- FakeTensor shape and dtype contracts;
- mutation declarations;
- `torch.compile` graph boundaries;
- dispatch between compatible hardware implementations.

An op wrapper should be thin. It may validate the semantic contract and lazily
import a launcher from `kernels/`, but it should not contain `@triton.jit`,
autotune configurations, or hardware launch parameters.

### `kernels/`

Hardware execution implementations. It owns:

- Triton, FlashInfer, and other backend launchers;
- kernel layout, stride, dtype, and shape constraints;
- autotune configurations and launch parameters;
- backend-specific utilities.

Kernel modules must not register Torch operators or import `xllm.python.ops`.
A launcher can allocate outputs and validate execution constraints before
starting a kernel.

Triton implementations are classified by the hardware backend required by the
implementation:

```text
kernels/triton/cuda/
kernels/triton/npu/
```

This classification describes the implementation, not the mathematical
operator. For example, split-QKV plus RMSNorm and RoPE is hardware-neutral as
an operation, while an implementation using CANN extensions, `torch.npu`, and
VectorCore properties belongs under `kernels/triton/npu/`.

A Torch namespace such as `xllm_triton` is an operator registration identity.
It does not determine the source directory of the implementation.

### `model_executor/`

Execution orchestration: eager or graph runners, forward context, attention
backend setup, cache binding, and lifecycle. It should coordinate models and
operators rather than implement mathematical kernels.

## Import safety

Hardware dependencies must be imported lazily from the owning op or launcher.
Importing public `xllm.python.ops` must not require Triton CUDA kernels, CANN
extensions, FlashInfer, or device-specific native schemas that are unrelated
to the selected backend. Package `__init__.py` files must not eagerly import
both CUDA and NPU implementations.

## Adding an operator

1. Add the hardware launcher under the matching `kernels/` backend.
2. Add a semantic wrapper under `ops/` when the computation needs a stable
   graph node, FakeTensor propagation, mutation tracking, or backend dispatch.
3. Register a FakeTensor implementation with the same shape and dtype contract.
4. Add launcher-level numerical tests and op-level graph/FakeTensor tests.
5. Add an import-isolation test when the implementation introduces an optional
   backend dependency.
6. Verify that dependencies still flow from models and layers through ops to
   kernels, never in the reverse direction.
