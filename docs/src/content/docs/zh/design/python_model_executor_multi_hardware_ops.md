# Python 模型执行器 · 多硬件平台 xllm_ops 算子注册方案

> 适用范围:`--model_impl=python` 的嵌入式 CPython 模型执行器(`xllm/python/`
> 包 + `PyCausalLM` C++ 桥)。本文回答两个问题:
>
> 1. `xllm_ops` 如何做成一套跨硬件(CUDA / NPU / …)的算子注册方案;
> 2. 同一份 Python 构图能否在多硬件间复用。
>
> 结论先行:**沿用 PyTorch dispatcher 的"设备无关 schema + 按 DispatchKey 绑定
> 后端实现"范式即可;Python 构图与 `register_fake` 100% 复用,新增硬件只需在
> C++ 侧补一个 `TORCH_LIBRARY_IMPL(xllm_ops, <DispatchKey>, m)` 并做少量 device
> 字符串 plumbing。** 这与 vLLM / SGLang 的做法完全一致。

---

## 1. 当前实现(CUDA 已落地)

Python 构图从不写任何 `#ifdef`/设备分支,每个算子只调用**一个** symbol
`torch.ops.xllm_ops.<op>`:

- 派发层 `xllm/python/ops/dispatch.py` —— `rms_norm` / `fused_add_rms_norm` /
  `silu_and_mul` / `fused_qk_norm_rope` / `attention` / `all_reduce` /
  `all_gather` 全部直接 `return torch.ops.xllm_ops.<op>(...)`。
- Meta(抽象)实现 `xllm/python/ops/fake_impls.py` —— `torch.library.register_fake`
  提供 shape 推导,`torch._dynamo.disallow_in_graph` 让 attention 逐层 graph-break。
  仅在开启 torch.compile 后端时才导入(`model_runner/graph_runner.py::maybe_compile`)。

C++ 侧分两处注册,schema 均**设备无关**声明一次,实现按 DispatchKey 绑定:

| 位置 | 声明 | CUDA 实现 | 覆盖算子 |
| --- | --- | --- | --- |
| `xllm/core/kernels/cuda/xllm_ops_library.cpp:125` | `TORCH_LIBRARY(xllm_ops, m)` | `TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m)`(:139) | 无状态 fused kernel:`rms_norm` / `fused_add_rms_norm` / `silu_and_mul` / `fused_qk_norm_rope` |
| `xllm/models/py_model_bridge.cpp:161` | `TORCH_LIBRARY_FRAGMENT(xllm_ops, m)` | `TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m)`(:167) | 有状态 op(读线程局部 forward context):`attention` / `all_reduce` / `all_gather` |

> 一个 library 只能有一个 `TORCH_LIBRARY`,其余追加用 `TORCH_LIBRARY_FRAGMENT`——
> 这正是把"无状态算子"和"依赖 forward context 的有状态算子"分文件注册的原因。

设备信息通过配置字典传给 Python 侧:`PyCausalLM::build_config_dict`
(`xllm/models/py_causal_lm.cpp:146`)写入 `d["device"] = c10::str(device_)`,
Python 模型据此在正确设备上建权重/中间张量,进而让张量的 device 决定 dispatch。

---

## 2. 业界对照(基于本地 vLLM / SGLang 源码)

### 2.1 vLLM —— 平台抽象 + `current_platform.dispatch_key`

`vllm/utils/torch_utils.py:935 direct_register_custom_op`:

```python
my_lib.define(op_name + schema_str, tags=tags)          # schema 一次,设备无关
my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)# 按平台 DispatchKey 绑定
if fake_impl is not None:
    my_lib._register_fake(op_name, fake_impl)           # Meta 一次,设备无关
```

`dispatch_key` 默认取 `current_platform.dispatch_key`。各平台在
`vllm/platforms/*.py` 声明:CUDA=`"CUDA"`(`cuda.py:209`)、CPU=`"CPU"`、
XPU=`"XPU"`、ROCm 复用 `"CUDA"`(`rocm.py:448`)。**同一 op 名 + 同一 fake_impl,
换平台只换 impl 的 DispatchKey。**

### 2.2 SGLang —— Ascend NPU 用 `PrivateUse1`

`sglang/docs_new/.../ascend_npu_operator_development.mdx`(SGL-Kernel-NPU 指南)
Step 5 的 `csrc/pytorch_extensions.cpp`:

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {              // schema(设备无关)
    m.def("helloworld(Tensor x, Tensor y) -> Tensor");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {     // NPU 实现绑定到 PrivateUse1
    m.impl("helloworld", TORCH_FN(sglang::npu_kernel::helloworld));
}
```

- 命名空间固定 `npu`,调用方写 `torch.ops.npu.<op>()`;
- **NPU 后端的 DispatchKey 固定为 `PrivateUse1`**(PyTorch 对 NPU 的标识);
- Ascend C kernel 编成 `libsgl_kernel_npu.so`;Triton 算子则从
  `sgl_kernel_npu` 直接 import(与我们的 `kernels/triton_ops.py` 对应)。

**xLLM 现状与二者逐一对齐**:单命名空间(`xllm_ops` ↔ vLLM 的 `_C` / SGLang 的
`npu`)、schema 声明一次、按 DispatchKey 绑定实现、`register_fake` 设备无关共享。
差别仅是 xLLM 目前只绑了 `CUDA` 一个 key。

---

## 3. 多硬件注册方案

保持"一套 schema + 一份 register_fake + 每硬件一份 impl"三分层,新增硬件是**加法**
而非改造:

```
                     torch.ops.xllm_ops.<op>          ← Python 构图唯一入口(不变)
                              │  (PyTorch dispatcher 按 tensor.device 路由)
        ┌─────────────────────┼──────────────────────────┬─────────────────┐
        ▼                     ▼                          ▼                 ▼
  DispatchKey=CUDA     DispatchKey=PrivateUse1     DispatchKey=Meta   (未来: CPU/XPU…)
  cuda 融合 kernel      Ascend NPU kernel           register_fake
  (已实现)             (新增,本方案)              (设备无关,已实现,复用)
```

落地要点:

1. **schema 不动**:`xllm_ops` 的 `.def(...)` 已是设备无关(见 §1 表),NPU 直接复用。
2. **新增 NPU 实现**:仿 CUDA,新增
   `TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m)`,`m.impl` 到 Ascend 版
   `rms_norm` / `fused_add_rms_norm` / `silu_and_mul` / `fused_qk_norm_rope`。
   放在 NPU 专属编译单元(参照现有 `xllm/core/kernels/npu/xllm_ops/` 目录),
   仅在 `--device npu` 构建时编入,与 CUDA impl 互不侵入。
3. **有状态 op 的 NPU 实现**:`attention` / `all_reduce` / `all_gather` 依赖
   `PyForwardContext`,需要在 NPU 分支上提供对应实现(attention 走 NPU 版
   `layer::Attention`,collective 走 NPU 的 `parallel_state`),同样以
   `TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m)` 绑定。
4. **`register_fake` 复用**:`fake_impls.py` 是纯 shape/dtype 推导,与设备无关,
   NPU 上 torch.compile / 图捕获直接复用,无需新增。
5. **device 字符串 plumbing**:`build_config_dict` 的 `d["device"]` 已透传
   (如 `"npu:0"`),Python 侧据此建张量;确保 `torch_npu` 已 import 使
   `PrivateUse1` 后端与 `npu:0` 设备可用。
6. **Triton 灰度**:`dispatch.py` 的 `XLLM_USE_TRITON` 开关把 `silu_and_mul`
   路由到 `kernels/triton_ops.py`;NPU 上可挂 Ascend 适配的 Triton kernel
   (对应 SGLang 从 `sgl_kernel_npu` import triton 算子的做法),与 vendor kernel
   并存、按开关切换。

---

## 4. 能否复用 Python 构图 —— 可以,完全复用

论证(三条充分条件都已满足):

1. **构图不含设备分支**:`xllm/python/models/qwen3.py` 与 `layers/*` 只通过
   `xllm/python/ops/dispatch.py` 调用 `torch.ops.xllm_ops.<op>`,没有任何
   `if cuda/npu` 或 `.cuda()` 硬编码;设备由传入张量携带。
2. **dispatcher 按 device 路由**:同一 op 名在运行时由 PyTorch 按
   `tensor.device` 选择 `CUDA` 或 `PrivateUse1` 实现——这正是 vLLM
   (`dispatch_key`)与 SGLang(`PrivateUse1`)复用同一上层代码的机制。
3. **图捕获前提设备无关**:`register_fake`(Meta)只做 shape 推导,
   `disallow_in_graph` 只标记 graph-break 点,均与硬件无关;NPU 上无论走 eager
   还是图模式都复用同一份 `fake_impls.py`。

**因此,`models/` + `layers/` + `ops/dispatch.py` + `ops/fake_impls.py` +
`model_runner/` 全部跨硬件复用,零改动。** 不可复用的只有两类,且都在 Python
构图之外:

- **C++ 后端实现绑定**(`TORCH_LIBRARY_IMPL(..., PrivateUse1, ...)` + Ascend
  kernel)——本就应按硬件分别实现;
- **少量运行环境 plumbing**:import `torch_npu` 激活 `PrivateUse1`、device
  字符串取 `npu:x`、图模式后端(NPU 侧对应 SGLang 的 `NPUPiecewiseBackend`,
  可后续在 `model_runner/` 内按设备选择)。

---

## 5. NPU(Ascend)落地清单

按依赖顺序,均为**新增**,不改动既有 CUDA 路径与 Python 构图:

1. C++:新增 NPU 编译单元,`TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m)` 绑定
   4 个无状态 fused op 的 Ascend 实现(复用 `xllm/core/kernels/npu/` 既有 kernel)。
2. C++:在 NPU 分支为 `attention` / `all_reduce` / `all_gather` 提供
   `PrivateUse1` 实现(attention → NPU `layer::Attention`;collective → NPU
   `parallel_state`),同样经 forward context 取状态。
3. 运行时:确保进程 import `torch_npu`(或等价)使 `PrivateUse1`/`npu:x` 可用;
   `PyCausalLM` 构造在 NPU 线程上完成 attention 模块初始化。
4. 图模式(可选,后续):在 `model_runner/graph_runner.py` 内按设备选择捕获后端
   (CUDA 走 torch.compile `cudagraphs`;NPU 参照 SGLang `NPUPiecewiseBackend`),
   `maybe_compile` 增加一层设备判定即可,`models/` 无感。
5. 校验:先 eager 数值对齐(与 NPU 原生 C++ 路径逐 token 对比),再开图模式。

---

## 6. 边界与风险

- **`PrivateUse1` 的一次性绑定**:PyTorch 全局只有一个 `PrivateUse1` 槽位;若
  同进程已被 `torch_npu` 占用即为 NPU,无冲突;不要在同一进程混挂两种
  PrivateUse1 后端。
- **有状态 op 的图捕获**:`all_reduce` / `all_gather` 已按 out-of-place 语义
  实现以便被图捕获(见 `py_model_bridge.cpp` 注释);NPU 实现需保持同样的
  out-of-place 约定,否则图模式下会捕获失败。
- **in-place op 的 schema 契约**:`fused_add_rms_norm` / `fused_qk_norm_rope`
  是 `(a!)` + void 返回;NPU 实现必须保持相同的 mutating 语义与返回签名,
  否则 functionalization 会拒绝或数值错乱。
- **数值对齐口径**:跨硬件 kernel 不保证 bit 级一致(浮点归约顺序不同),
  对齐标准应为"贪心解码 token 序列一致 / 相对误差阈值内",而非 byte-identical。
