---
title: "xLLM Ascend TileLang Kernel 开发指南"
sidebar:
  order: 2
---
本文说明在 xLLM 中新增或修改 Ascend TileLang kernel 的开发方式。示例全程使用当前的 `rope` kernel。

相关目录：

- Python kernel 定义：`xllm/xllm/compiler/tilelang/targets/ascend/kernels`
- NPU runtime wrapper：`xllm/xllm/core/kernels/npu/tilelang`

构建和测试应在 NPU 容器中执行。

## 1. 先判断修改类型

- 新增 `specialization`
  - 给现有 kernel 增加一组新的编译参数组合
  - 仍复用同一个 wrapper、同一套 runtime dispatch 字段和同一套 C ABI
  - 典型动作是修改 `DISPATCH_SCHEMA` 或 `SPECIALIZATIONS`
- 新增 `kernel`
  - 新增一个新的逻辑算子
  - 典型动作是新增 Python kernel 文件、wrapper C++ 文件和一条 CMake 接线

对 `rope` 来说：

- 给 `SPECIALIZATIONS` 增加一项 `{"variant_key": "...", "head_dim": ..., "rope_dim": ..., "dtype": ...}`，这是新增 `specialization`
- 新增一个新的 `xxx_wrapper.cpp` 对外接口，这是新增 `kernel`

## 2. 开发顺序

推荐按下面顺序开发：

1. 在 `rope.py` 这类 Python 文件里先写 TileLang kernel 实现
2. 实现 `generate_source(...)`，把 kernel lower 成 Ascend-C 源码
3. 声明 `DISPATCH_SCHEMA` 和 `SPECIALIZATIONS`
4. 先生成一次 `registry.inc` 并查看内容
5. 再写或修改 wrapper 里的 runtime specialization 构造逻辑
6. 接入 CMake 并运行测试

这个顺序的重点是：

- 先把 kernel 本身实现出来
- 再把 runtime dispatch schema 固定下来
- 最后根据生成出来的 `registry.inc` 写 wrapper

## 3. 编写 Python Kernel

以 `rope.py` 为例，Python 侧可以按三层理解：

- `build_rope_kernel(...)`：kernel 实现
- `generate_source(...)`：AOT 导出
- `RopeKernel`：注册 kernel，并声明 dispatch schema 与编译实例

### 3.1 实现 `build_rope_kernel(...)`

`build_rope_kernel(...)` 才是 TileLang kernel 的实现主体。这里负责写：

- `@T.prim_func`
- 输入输出张量 shape
- `with T.Kernel(...)` 下的并行任务组织
- UB 分配和实际计算逻辑

`rope.py` 的精简骨架如下：

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

这里的 `head_dim`、`rope_dim` 是这一组实现真正依赖的编译参数。

`rope` 这类 vector kernel 还要遵守当前 AOT 使用方式下的固定任务约定。当前路径是 AOT 编译，kernel launch 的 `block_num` 会在编译时固定下来，因此：

- 运行时输入 shape 不影响 kernel launch 的 `block_num`
- 运行时输入 shape 只影响固定任务之间的 workload 切分

当前 `rope.py` 的约定是：

```python
task_num = vec_core_num
m_num = vec_core_num // 2

with T.Kernel(m_num, is_npu=True) as (cid, vid):
    task_id = cid * 2 + vid
```

这表示：

- `cid` 范围是 `[0, vec_core_num // 2)`
- `vid` 范围是 `[0, 2)`
- 总任务数固定为 `task_num = vec_core_num`

因此，`rope.py` 在推导单个 specialization 的编译 token 数时，也按固定任务数计算：

```python
max_rows_num_in_ub = _derive_max_rows_num_in_ub(...)
compile_num_tokens = task_num * max_rows_num_in_ub
```

### 3.2 实现 `generate_source(...)`

`generate_source(...)` 负责把上面的 TileLang kernel lower 成最终源码。导出层的职责，是把一组 specialization 参数转换成可编译的 Ascend-C 源码。

对 `rope` 来说，核心逻辑如下：

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

这里的规则是：

- `generate_source(...)` 的输入来自当前这组 `SPECIALIZATIONS`
- `generate_source(...)` 内部调用 `build_rope_kernel(...)`
- 返回值是 lower 后的源码字符串

### 3.3 声明 `DISPATCH_SCHEMA` 与 `SPECIALIZATIONS`

当 kernel 实现和导出层写完后，再通过 `@register_kernel` 类把它接入框架。

`rope.py` 当前的最小模板如下：

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

这里要分清楚两个概念：

- `DISPATCH_SCHEMA`
  - 定义 runtime specialization 的字段名、顺序和类型
  - 是 C++ 侧 specialization struct、builder 和查表接口的单一真相源
- `SPECIALIZATIONS`
  - 表示要实际编译出的实例集合
  - 每一项都对应一个 variant

规则如下：

- `DISPATCH_SCHEMA` 中每个字段都必须出现在每一项 `SPECIALIZATIONS` 里
- `SPECIALIZATIONS` 中可以有额外字段，这些字段会传给 `generate_source(...)`，但不会进入 runtime dispatch schema
- `variant_key` 是这一组 specialization 的唯一标识
- `DISPATCH_SCHEMA` 和 `SPECIALIZATIONS` 必须与 runtime specialization 一一对应

对 `rope` 来说，runtime dispatch 维度是：

- `head_dim`
- `rope_dim`
- `dtype`

所以这三个字段必须同时出现在：

- `DISPATCH_SCHEMA`
- 每一项 `SPECIALIZATIONS`

构建时，Ascend build 会根据主构建路径传入的 `--device a2|a3` 解析实际使用的 `bisheng_arch`。

### 3.4 查看生成的 Ascend-C 源码

在调试 `build_rope_kernel(...)` 的实现细节，或者比较不同 kernel 写法对最终代码生成的影响时，建议通过公共入口 `compile-kernels` 重新生成产物，再查看对应 specialization 的 Ascend-C 源码。

对 `rope` 来说，可以先固定：

- `head_dim=576`
- `rope_dim=64`
- `dtype=bf16`

然后重新生成 `rope` 的编译产物：

```bash
python xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend \
  --device a3 \
  --output-root /tmp/tilelang_debug \
  --kernels rope \
  --force
```

这里建议带上 `--force`，保证源码和 object 会按当前修改重新生成，不直接命中旧 cache。

这里把 `--output-root` 指到独立的调试目录 `/tmp/tilelang_debug`，这样当前命令只会在这个目录下生成 `rope` 的调试产物，不会和主构建目录里的其他 kernel 产物混在一起。

执行后，可以直接查看对应 specialization 的源码中的入口函数、UB 分配和向量计算逻辑：

```bash
sed -n '1,200p' \
  /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp

rg -n 'extern "C"|__global__|alloc_ub|alloc_shared|g_tilingKey' \
  /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp
```

如果要比较两种 kernel 写法的差异，做法是保持 specialization 不变，在修改前后各执行一次 `compile-kernels --force`，再对生成的 `.cpp` 做 diff：

```bash
cp /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp \
  /tmp/rope_before.cpp

diff -u /tmp/rope_before.cpp \
  /tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp
```

这样可以把“specialization 变化”和“kernel 实现变化”分开看。

执行后，可以重点查看这些文件：

- `/tmp/tilelang_debug/targets/ascend/rope/hd576_rd64_bf16/rope_hd576_rd64_bf16_kernel.cpp`
- `/tmp/tilelang_debug/targets/ascend/rope/registry.inc`
- `/tmp/tilelang_debug/targets/ascend/rope/manifest.json`

这三类文件分别对应：

- 某个 specialization 的最终 Ascend-C 源码
- wrapper 会直接包含的 runtime dispatch 接口
- 当前 kernel 的全部编译产物记录

调试顺序建议是：

1. 先执行 `compile-kernels --force` 重新生成当前 kernel 的产物
2. 查看对应 specialization 的 `.cpp`，分析代码生成结果
3. 再看 `registry.inc` 和 `manifest.json` 是否符合预期
4. 最后通过 `rope_wrapper_test` 看完整接入后的结果和性能

## 4. 修改 Wrapper

新增 `kernel` 时，需要新增 wrapper。新增 `specialization` 时，只有 runtime specialization 语义变化，才需要同步修改 wrapper。

对 `rope_wrapper.cpp` 来说，人工需要保留的内容是：

- tensor shape、dtype、layout 校验
- 把输入整理成 `x_rows / sin_rows / cos_rows`
- 从 tensor 构造 runtime specialization
- 组装 launch 参数并调用 `entry->fn(...)`

### 4.1 `registry.inc` 会自动生成什么

`registry.inc` 由 Python 侧的 `DISPATCH_SCHEMA`、`SPECIALIZATIONS` 和导出出来的 Ascend-C ABI 自动生成。

对 `rope` 来说，生成内容包括：

- `RopeSpecialization`
- `RopeHeadDim`
- `RopeRopeDim`
- `RopeDType`
- `RopeKernelFn`
- `make_rope_specialization(...)`
- `find_rope_kernel_entry(...)`
- `available_rope_variant_keys()`

对 `rope_wrapper.cpp` 来说，`registry.inc` 会直接提供 `RopeSpecialization`、`operator==(...)`、`RopeKernelFn` 等 dispatch 相关定义。`dtype` 转换统一使用公共函数 `to_tilelang_dtype(...)`。

### 4.2 wrapper 里真正要写的东西

`rope_wrapper.cpp` 中最关键的人工逻辑，是从 tensor 构造 runtime specialization。当前写法如下：

```cpp
RopeSpecialization build_runtime_specialization(const torch::Tensor& x_rows) {
  return make_rope_specialization(
      RopeHeadDim{static_cast<int32_t>(x_rows.stride(0))},
      RopeRopeDim{static_cast<int32_t>(x_rows.size(1))},
      RopeDType{to_tilelang_dtype(x_rows.scalar_type())});
}
```

对 `rope` 而言：

- `head_dim` 对应 `x_rows.stride(0)`，也就是 kernel 使用的 `x_stride`
- `rope_dim` 对应 `x_rows.size(1)`
- `dtype` 对应 `x_rows.scalar_type()`

运行时路径是：

1. wrapper 把输入整理成 `x_rows / sin_rows / cos_rows`
2. `build_runtime_specialization(...)` 从 `x_rows` 构造 specialization
3. `find_rope_kernel_entry(...)` 在静态 registry 中精确匹配
4. 命中后通过 `entry->fn(...)` 调用实际编译出来的符号

当前查找策略是线性扫描加精确匹配。只要 `head_dim`、`rope_dim`、`dtype` 有一个不一致，就不会命中。

所以新增 `specialization` 时，要重点核对的是：

- Python 侧 `DISPATCH_SCHEMA` 的字段语义
- Python 侧 `SPECIALIZATIONS` 的字段值
- wrapper 里 `build_runtime_specialization(...)` 构造出的字段值

这三者必须完全对齐。

### 4.3 先生成并查看 `registry.inc`

在写或修改 wrapper 之前，先生成一次 `registry.inc` 并查看内容。重点看：

- 生成出来的 `RopeSpecialization` 字段顺序是否符合预期
- 生成出来的字段包装类型名是否符合预期
- `make_rope_specialization(...)` 的参数顺序是什么
- 生成出来的 entry symbol 名称是什么

`registry.inc` 是 wrapper 的直接契约，先看它，再写 wrapper。

## 5. 修改 CMake

新增 `kernel` 时，在 `xllm/xllm/core/kernels/npu/tilelang/CMakeLists.txt` 中接入这个 kernel。

CMake 接入统一通过高层 helper 完成：

- `tilelang_register_runtime_kernel(NAME <kernel> WRAPPER_SRCS <srcs...>)`

以 `rope` 为例，最小模板如下：

```cmake
tilelang_register_runtime_kernel(
  NAME rope
  WRAPPER_SRCS rope_wrapper.cpp
)
```

这条 helper 会完成：

- 按 `TILELANG_GENERATED_ROOT/targets/ascend/<kernel>/manifest.json` 推导 manifest 路径
- 导入 manifest
- 把该 kernel 的 wrapper source 和 compiled objects 加入 `tilelang_kernels`
- 自动追加 `XLLM_TL_<KERNEL>_REGISTRY_INC=...` compile definition

因此，新增一个 runtime kernel 时，CMake 侧主要就是两件事：

1. 保证 Python 侧已经能生成该 kernel 的 manifest
2. 在 `tilelang` 的 CMakeLists 里新增一条 `tilelang_register_runtime_kernel(...)`

日常新增 kernel 时，直接在 CMake 中增加一条 `tilelang_register_runtime_kernel(...)`。`tilelang_import_kernel_manifest(...)` 作为这条高层 helper 的实现基础，保留在底层。

## 6. 验证

推荐按下面顺序验证：

1. 先编译 TileLang kernel，并查看生成的 `registry.inc`
2. 再跑完整的 wrapper 测试

常用命令：

```bash
python xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend \
  --device a3 \
  --output-root build/cmake.linux-aarch64-cpython-311/xllm/compiler/tilelang \
  --kernels rope

python setup.py test --test-name rope_wrapper_test --device a3
```

第一条命令用于生成 `manifest.json`、`registry.inc` 和 object；第二条命令用于验证完整接入路径。

