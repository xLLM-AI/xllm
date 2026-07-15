---
title: "xLLM Ascend TileLang Kernel Development Guide"
sidebar:
  order: 2
---
This document explains how to add or modify an Ascend TileLang kernel in xLLM. The examples use the current `rope` kernel throughout.

Relevant directories:

- Python kernel definitions: `xllm/xllm/compiler/tilelang/targets/ascend/kernels`
- NPU runtime wrappers: `xllm/xllm/core/kernels/npu/tilelang`

Builds and tests should be run inside the NPU container.

## 1. First Decide What Kind of Change You Are Making

- Add a `specialization`
  - Add one more compiled parameter combination to an existing kernel
  - Reuse the same wrapper, the same runtime dispatch fields, and the same C ABI
  - Typical changes are updates to `DISPATCH_SCHEMA` or `SPECIALIZATIONS`
- Add a `kernel`
  - Add a new logical operator
  - Typical changes are a new Python kernel file, a new wrapper C++ file, and one CMake registration

For `rope`:

- Adding one more item like `{"variant_key": "...", "head_dim": ..., "rope_dim": ..., "dtype": ...}` to `SPECIALIZATIONS` means adding a new `specialization`
- Adding a new external interface such as `xxx_wrapper.cpp` means adding a new `kernel`

## 2. Development Order

The recommended order is:

1. Implement the TileLang kernel in a Python file such as `rope.py`
2. Implement `generate_source(...)` to lower the kernel into Ascend-C source
3. Declare `DISPATCH_SCHEMA` and `SPECIALIZATIONS`
4. Generate `registry.inc` once and inspect it
5. Then write or update the runtime specialization construction logic in the wrapper
6. Wire it into CMake and run tests

The key idea behind this order is:

- implement the kernel itself first
- then fix the runtime dispatch schema
- then write the wrapper against the generated `registry.inc`

## 3. Write the Python Kernel

Using `rope.py` as the example, the Python side can be understood in three layers:

- `build_rope_kernel(...)`: kernel implementation
- `generate_source(...)`: AOT export
- `RopeKernel`: kernel registration plus dispatch schema and compiled instance declaration

### 3.1 Implement `build_rope_kernel(...)`

`build_rope_kernel(...)` is the actual TileLang kernel implementation. This is where you write:

- `@T.prim_func`
- input and output tensor shapes
- parallel task organization under `with T.Kernel(...)`
- UB allocation and the actual compute logic

The simplified structure in `rope.py` looks like this:

```python
def build_rope_kernel(
    head_dim: int,
    rope_dim: int,
    vec_core_num: int,
    ub_buffer_bytes: int,
):
    task_num = vec_core_num
    m_num = vec_core_num // 2

    @T.prim_func
    def rope_in_place_kernel(...):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            task_id = cid * 2 + vid
            ...

    return rope_in_place_kernel
```

Here, `head_dim` and `rope_dim` are the compile-time parameters that this implementation actually depends on.

Vector kernels such as `rope` must also follow the fixed-task convention used by the current AOT path. In the current AOT flow, the kernel launch `block_num` is fixed at compile time, which means:

- runtime input shapes do not change the kernel launch `block_num`
- runtime input shapes only change workload splitting across the fixed tasks

The convention in the current `rope.py` is:

```python
task_num = vec_core_num
m_num = vec_core_num // 2

with T.Kernel(m_num, is_npu=True) as (cid, vid):
    task_id = cid * 2 + vid
```

This means:

- `cid` ranges over `[0, vec_core_num // 2)`
- `vid` ranges over `[0, 2)`
- the total task count is fixed as `task_num = vec_core_num`

As a result, `rope.py` also derives the compile-time token count for one specialization using the fixed task count:

```python
max_rows_num_in_ub = _derive_max_rows_num_in_ub(...)
compile_num_tokens = task_num * max_rows_num_in_ub
```

### 3.2 Implement `generate_source(...)`

`generate_source(...)` lowers the TileLang kernel above into the final source code. The export layer takes one specialization and turns it into compilable Ascend-C source.

For `rope`, the core logic is:

```python
@staticmethod
def generate_source(head_dim: int, rope_dim: int, dtype: str) -> str:
    vec_core_num = detect_vec_core_num()
    tilelang_kernel = build_rope_kernel(
        head_dim=head_dim,
        rope_dim=rope_dim,
        vec_core_num=vec_core_num,
        ub_buffer_bytes=FIXED_UB_BUFFER_BYTES,
    )
    with tilelang.tvm.transform.PassContext(...):
        kernel = tilelang.engine.lower(tilelang_kernel)
    return kernel.kernel_source
```

The rules here are:

- the inputs to `generate_source(...)` come from the current `SPECIALIZATIONS` entry
- `generate_source(...)` calls `build_rope_kernel(...)`
- the return value is the lowered source string

### 3.3 Declare `DISPATCH_SCHEMA` and `SPECIALIZATIONS`

After the kernel implementation and export layer are done, use an `@register_kernel` class to attach the kernel to the framework.

The current minimal template in `rope.py` is:

```python
from ....common.spec import DispatchField, TilelangKernel, register_kernel

@register_kernel
class RopeKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("head_dim", "int32"),
        DispatchField("rope_dim", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "hd128_rd128_bf16",
            "head_dim": 128,
            "rope_dim": 128,
            "dtype": "bf16",
        },
        {
            "variant_key": "hd576_rd64_bf16",
            "head_dim": 576,
            "rope_dim": 64,
            "dtype": "bf16",
        },
    ]

    @staticmethod
    def generate_source(head_dim: int, rope_dim: int, dtype: str) -> str:
        ...
```

There are two concepts to distinguish here:

- `DISPATCH_SCHEMA`
  - defines the field names, order, and types of the runtime specialization
  - is the single source of truth for the C++ specialization struct, builder, and lookup interface
- `SPECIALIZATIONS`
  - represents the set of instances that will actually be compiled
  - each item corresponds to one variant

The rules are:

- every field in `DISPATCH_SCHEMA` must appear in every `SPECIALIZATIONS` item
- `SPECIALIZATIONS` may contain extra fields; those fields are passed into `generate_source(...)`, but do not enter the runtime dispatch schema
- `variant_key` is the unique identifier for that specialization
- `DISPATCH_SCHEMA` and `SPECIALIZATIONS` must match the runtime specialization one-to-one

For `rope`, the runtime dispatch dimensions are:

- `head_dim`
- `rope_dim`
- `dtype`

So these three fields must appear in both:

- `DISPATCH_SCHEMA`
- every `SPECIALIZATIONS` item

At build time, Ascend build resolves the actual `bisheng_arch` from the `--device a2|a3` value passed by the main build path.

### 3.4 Inspect the Generated Ascend-C Source

When debugging the implementation details in `build_rope_kernel(...)`, or comparing how different kernel styles affect the final code generation, use the common `compile-kernels` entry to regenerate artifacts and inspect the Ascend-C source for the specialization you care about.

For `rope`, you can fix:

- `head_dim=576`
- `rope_dim=64`
- `dtype=bf16`

Then regenerate the `rope` artifacts:

```bash
python xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend \
  --device a3 \
  --output-root /tmp/tilelang_debug \
  --kernels rope \
  --force
```

It is recommended to keep `--force` so the source and object files are regenerated from the current code instead of reusing an old cache hit.

This command uses an isolated debug output directory, `/tmp/tilelang_debug`, so only the debug artifacts for `rope` are generated there and they do not get mixed with artifacts from other kernels in the main build directory.

After that, you can directly inspect the generated source for the specialization, including the entry function, UB allocation, and vector compute logic:

```bash
sed -n '1,200p' \
  /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp

rg -n 'extern "C"|__global__|alloc_ub|alloc_shared|g_tilingKey' \
  /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp
```

To compare two kernel implementations, keep the specialization fixed, run `compile-kernels --force` before and after the change, then diff the generated `.cpp` file:

```bash
cp /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp \
  /tmp/rope_before.cpp

diff -u /tmp/rope_before.cpp \
  /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp
```

This helps isolate specialization changes from kernel implementation changes.

After generation, the main files to inspect are:

- `/tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp`
- `/tmp/tilelang_debug/targets/ascend/rope/registry.inc`
- `/tmp/tilelang_debug/targets/ascend/rope/manifest.json`

These correspond to:

- the final Ascend-C source for one specialization
- the runtime dispatch interface directly included by the wrapper
- the full compiled artifact record for the current kernel

The recommended debugging sequence is:

1. run `compile-kernels --force` to regenerate the current kernel artifacts
2. inspect the `.cpp` for the specialization and analyze the code generation result
3. inspect `registry.inc` and `manifest.json` to confirm they match expectations
4. finally run `rope_wrapper_test` to check end-to-end behavior and performance

## 4. Update the Wrapper

When adding a new `kernel`, you need a new wrapper. When adding a new `specialization`, the wrapper only needs an update if the runtime specialization semantics change.

For `rope_wrapper.cpp`, the manually written parts should remain:

- tensor shape, dtype, and layout validation
- reshaping inputs into `x_rows / sin_rows / cos_rows`
- constructing the runtime specialization from tensors
- assembling launch arguments and calling `entry->fn(...)`

### 4.1 What `registry.inc` Generates Automatically

`registry.inc` is generated automatically from the Python-side `DISPATCH_SCHEMA`, `SPECIALIZATIONS`, and the exported Ascend-C ABI.

For `rope`, the generated content includes:

- `RopeSpecialization`
- `RopeHeadDim`
- `RopeRopeDim`
- `RopeDType`
- `RopeKernelFn`
- `make_rope_specialization(...)`
- `find_rope_kernel_entry(...)`
- `available_rope_variant_keys()`

For `rope_wrapper.cpp`, `registry.inc` directly provides dispatch-related definitions such as `RopeSpecialization`, `operator==(...)`, and `RopeKernelFn`. Dtype conversion uses the shared helper `to_tilelang_dtype(...)`.

### 4.2 What the Wrapper Actually Needs to Write

The most important handwritten logic in `rope_wrapper.cpp` is constructing the runtime specialization from the tensors. The current code looks like this:

```cpp
RopeSpecialization build_runtime_specialization(const torch::Tensor& x_rows) {
  return make_rope_specialization(
      RopeHeadDim{static_cast<int32_t>(x_rows.stride(0))},
      RopeRopeDim{static_cast<int32_t>(x_rows.size(1))},
      RopeDType{to_tilelang_dtype(x_rows.scalar_type())});
}
```

For `rope`:

- `head_dim` maps to `x_rows.stride(0)`, which is the `x_stride` used by the kernel
- `rope_dim` maps to `x_rows.size(1)`
- `dtype` maps to `x_rows.scalar_type()`

The runtime path is:

1. the wrapper reshapes the inputs into `x_rows / sin_rows / cos_rows`
2. `build_runtime_specialization(...)` constructs a specialization from `x_rows`
3. `find_rope_kernel_entry(...)` performs an exact match in the static registry
4. after a match, `entry->fn(...)` calls the actual compiled symbol

The current lookup strategy is a linear scan with exact matching. If any of `head_dim`, `rope_dim`, or `dtype` differs, the lookup will miss.

So when you add a new `specialization`, the main things to cross-check are:

- the field semantics in Python-side `DISPATCH_SCHEMA`
- the field values in Python-side `SPECIALIZATIONS`
- the field values constructed by `build_runtime_specialization(...)` in the wrapper

All three must match exactly.

### 4.3 Generate and Inspect `registry.inc` First

Before writing or modifying wrapper code, generate `registry.inc` once and inspect it. Focus on:

- whether the generated field order in `RopeSpecialization` matches expectations
- whether the generated wrapped field type names match expectations
- the parameter order of `make_rope_specialization(...)`
- the generated entry symbol names

`registry.inc` is the direct contract for the wrapper. Inspect it first, then write the wrapper against it.

## 5. Update CMake

When adding a new `kernel`, register it in `xllm/xllm/core/kernels/npu/tilelang/CMakeLists.txt`.

CMake registration is unified through the high-level helper:

- `tilelang_register_runtime_kernel(NAME <kernel> WRAPPER_SRCS <srcs...>)`

Using `rope` as the example, the minimal template is:

```cmake
tilelang_register_runtime_kernel(
  NAME rope
  WRAPPER_SRCS rope_wrapper.cpp
)
```

This helper will:

- derive the manifest path as `TILELANG_GENERATED_ROOT/targets/ascend/<kernel>/manifest.json`
- import the manifest
- add the wrapper source and compiled objects into `tilelang_kernels`
- append the `XLLM_TL_<KERNEL>_REGISTRY_INC=...` compile definition automatically

So when adding a new runtime kernel, the CMake-side work mainly consists of two things:

1. make sure the Python side can already generate the manifest for that kernel
2. add one `tilelang_register_runtime_kernel(...)` entry in the TileLang CMakeLists

For day-to-day kernel additions, add one `tilelang_register_runtime_kernel(...)` line directly in CMake. `tilelang_import_kernel_manifest(...)` stays underneath as the implementation base for that higher-level helper.

## 6. Validate

The recommended validation order is:

1. compile the TileLang kernel and inspect the generated `registry.inc`
2. then run the full wrapper test

Common commands:

```bash
python xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend \
  --device a3 \
  --output-root build/cmake.linux-aarch64-cpython-311/xllm/compiler/tilelang \
  --kernels rope

python setup.py test --test-name rope_wrapper_test --device a3
```

The first command generates `manifest.json`, `registry.inc`, and the object files. The second command validates the full integration path.

