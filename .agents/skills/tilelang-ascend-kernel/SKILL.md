---
name: tilelang-ascend-kernel
description: Use when the user wants to add, modify, debug, or review an xLLM TileLang Ascend kernel or specialization, including Python kernel definitions, generated Ascend-C source, runtime wrapper dispatch, TileLang CMake wiring, and NPU tests.
---

# TileLang Ascend Kernel

## When to use

Use this skill when the task involves any of the following in the xLLM repo:

- `xllm/compiler/tilelang/targets/ascend/kernels/*.py`
- `xllm/core/kernels/npu/tilelang/*_wrapper.cpp`
- `xllm/core/kernels/npu/tilelang/CMakeLists.txt`
- generated TileLang artifacts such as `manifest.json`, `registry.inc`, or specialization `.cpp`

Run build and test commands inside the NPU container.

Run TileLang commands from the xLLM repo root, not from an installed-package environment.

## Entry points and TL_ROOT

- Prefer `python xllm/compiler/tilelang_launcher.py ...` for end-to-end TileLang compile flows.
- From the xLLM repo root, use `export TL_ROOT=$PWD/third_party/tilelang-ascend` for xLLM TileLang tooling and verify `test -f "$TL_ROOT/tilelang/__init__.py"`.
- Before any raw script does `import tilelang`, run `export TL_ROOT=$PWD/third_party/tilelang-ascend && source third_party/tilelang-ascend/set_env.sh`, then execute the script.
- Do not run kernel files directly with `python rope.py`; use module execution because these files rely on relative imports.
- For direct kernel-script debugging, run them as modules and pass required CLI args:

```bash
cd xllm
python -m compiler.tilelang.targets.ascend.kernels.rope \
  --output ../.tmp/rope.cpp
# Expected: [INFO] RoPE output matches torch reference
```

- The same module-style rule applies to other kernel files under `xllm/compiler/tilelang/targets/ascend/kernels/`.

## Primary Reference And Mode Preference

Primary reference:

- `third_party/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md`

Use `third_party/tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-tile-ops.md` when the task depends on `T.tile.xxx` semantics such as `compare`, `select`, `cast`, or other vector intrinsics.

Default to Expert mode for xLLM Ascend kernels:

- prefer `T.tile.xxx`, explicit UB/shared allocation, and explicit `T.copy`
- prefer explicit `T.serial` control for row/block traversal
- do not introduce Developer mode `T.Parallel` unless the kernel is a clearly tile-local element-wise expression and the change does not reduce control over UB usage, temporary buffers, or exact runtime semantics
- when translating Triton kernels, preserve the Triton runtime semantics first, then choose the smallest Expert-mode lowering that matches them

## Common Triton To TileLang-Ascend Semantics

Use this table as the quick semantic mapping when translating Triton kernels:

| Triton pattern | TileLang-Ascend pattern | Notes |
| --- | --- | --- |
| `x + y`, `x - y`, `x * y`, `x / y` | `T.tile.add/sub/mul/div` | Prefer tile ops in Expert-style vector code instead of hand-written scalar loops. |
| `tl.exp(x)`, `tl.log(x)`, `tl.abs(x)` | `T.tile.exp`, `T.tile.ln`, `T.tile.abs` | TileLang uses `ln`, not `log`. |
| `x.to(tl.float32)` or `tl.cast(...)` | `T.tile.cast(dst, src, "CAST_NONE", count)` | Pick a non-default cast mode only when rounding semantics are required. |
| `x <= y`, `x < y`, `x >= y`, `x == y` | `T.tile.compare(mask, x, y, "LE"/"LT"/"GE"/"EQ")` | `T.tile.compare` produces a bit mask, not a float tensor. |
| `tl.where(cond, a, b)` | `T.tile.select(dst, selMask, a, b, selMode)` | API-level match. If `cond` is a comparison expression such as `x <= y`, materialize `selMask` with `T.tile.compare(...)` first; use the matching `VSEL_*` mode for tensor-tensor or tensor-scalar selection. |
| `tl.full(shape, value, dtype)` | allocate buffer + `T.tile.fill(dst, value)` | Separate allocation from initialization. |
| `tl.arange(0, N)` | `T.tile.createvecindex(dst, 0)` or explicit loop indices | Prefer `createvecindex` only when the kernel truly needs a vector index tensor. |

Rules for semantic-preserving lowering:

- Preserve Triton control-flow, masking, and parameter semantics. Do not substitute a numerically similar formula unless the runtime-visible behavior is unchanged for the supported input domain.
- Keep the kernel ABI aligned with the lowering. Every runtime parameter must either participate in the TileLang implementation or be removed from the interface.
- Add targeted tests for branch, mask, and boundary behavior. Do not rely only on random inputs if some paths are hit only under specific values.
- Choose the correct `VSEL_*` mode based on the source operands. `VSEL_CMPMASK_SPR` is the natural match for a mask produced by `T.tile.compare`; `VSEL_TENSOR_SCALAR_MODE` and `VSEL_TENSOR_TENSOR_MODE` are for explicit tensor/scalar or tensor/tensor selection modes.

## New kernel

Follow this order:

1. Implement `build_<kernel>_kernel(...)`
2. Implement `generate_source(...)`
3. Declare `DISPATCH_SCHEMA` and `SPECIALIZATIONS`
4. Run TileLang compilation once and inspect `registry.inc`
5. Add or update `<kernel>_wrapper.cpp`
6. Register the kernel in `xllm/core/kernels/npu/tilelang/CMakeLists.txt` with:
   - `tilelang_register_runtime_kernel(NAME <kernel> WRAPPER_SRCS <srcs...>)`

For wrapper work:

- do kernel precision alignment on the Python side first (`build_<kernel>_kernel(...)` / `generate_source(...)`), not in the C++ wrapper
- handwrite tensor checks, layout transforms, and `build_runtime_specialization(...)`
- use generated `make_<kernel>_specialization(...)` and `find_<kernel>_kernel_entry(...)`
- do not handwrite kernel-specific specialization structs or kernel fn typedefs
- **never use `permute`, `contiguous`, `transpose`, `reshape` (when non-trivial), `clone`, or any operation that triggers a device memory copy in the wrapper**. The wrapper must only pass pointers to existing tensors. If the kernel needs a different layout, handle it inside the kernel itself or require the caller to provide the correct layout.

## New specialization

Use this path when the kernel logic and wrapper ABI stay the same.

1. Update the existing kernel's `SPECIALIZATIONS`
2. Confirm every runtime dispatch field still matches `DISPATCH_SCHEMA`
3. Re-run TileLang compilation
4. Check that `registry.inc` contains the new entry
5. Check that the wrapper's `build_runtime_specialization(...)` still constructs matching values

## Debug generated Ascend-C

When the task is to inspect codegen or compare two kernel implementations, use an isolated output root:

```bash
python xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend \
  --device a3 \
  --output-root .tmp/tilelang_debug \
  --kernels <kernel> \
  --force
```

Then inspect:

- `.tmp/tilelang_debug/targets/ascend/<kernel>/<variant_key>/<kernel>_<variant_key>_kernel.cpp`
- `.tmp/tilelang_debug/targets/ascend/<kernel>/registry.inc`
- `.tmp/tilelang_debug/targets/ascend/<kernel>/manifest.json`

Use this path before changing wrapper code when you need to understand generated symbols, field order, or ABI.

## Validate

Prefer the narrowest command first:

- `python -m py_compile xllm/compiler/tilelang/targets/ascend/kernels/<kernel>.py`
- `cd xllm && python -m compiler.tilelang.targets.ascend.kernels.<kernel> --output ../.tmp/<kernel>.cpp`
- `python xllm/compiler/tilelang_launcher.py prepare-ascend`
- `python setup.py test --test-name <wrapper_test_target> --device npu`

## References

Read [tilelang_ascend_kernel_dev](https://docs.xllm-ai.com/en/dev_guide/tilelang_ascend_kernel_dev/) for mechanism details.
Use `rope` as the concrete template:
- `xllm/compiler/tilelang/targets/ascend/kernels/rope.py`
- `xllm/core/kernels/npu/tilelang/rope_wrapper.cpp`
- `xllm/core/kernels/npu/tilelang/CMakeLists.txt`
