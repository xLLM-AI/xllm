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
> 字符串 plumbing。**

---

## 1. 当前实现(CUDA 已落地)

### 1.1 Python ops 层结构

```
xllm/python/ops/
├── __init__.py        # 统一导出 (ops.rms_norm, ops.batch_decode, ...)
├── compute.py         # 计算 op 绑定 + register_fake
├── attention.py       # attention op 绑定 + disallow_in_graph
└── collectives.py     # TP 通信 op (torch.library.custom_op 注册)
```

设计原则:
- **model / layer 只调 `ops.*`**,不出现 `torch.ops.xllm_ops.*` 硬编码。
- 每个 op 的绑定、`register_fake`、`disallow_in_graph` **在同一文件同一位置**。
- 无环境变量依赖;无 Python 层 dispatch 逻辑(PyTorch C++ dispatcher 按 tensor device 路由)。

### 1.2 三类 op

| 类别 | 文件 | op 列表 | compile 语义 |
| --- | --- | --- | --- |
| 计算 op | `compute.py` | `rms_norm` / `fused_add_rms_norm` / `silu_and_mul` / `fused_qk_norm_rope` | `register_fake` → 可入图 |
| Attention op | `attention.py` | `reshape_paged_cache` / `batch_prefill` / `batch_decode` / `batch_chunked_prefill` / `update_*_plan` | `reshape_paged_cache` 可入图;`batch_*` 为 `disallow_in_graph`(依赖 volatile plan/workspace) |
| 通信 op | `collectives.py` | `all_reduce` / `all_gather` | `torch.library.custom_op` 注册,Dynamo 作为 opaque 节点捕获 |

### 1.3 C++ 侧注册

Schema 均**设备无关**声明一次,实现按 DispatchKey 绑定:

| 位置 | 声明 | CUDA 实现 | 覆盖算子 |
| --- | --- | --- | --- |
| `core/kernels/cuda/xllm_ops_library.cpp` | `TORCH_LIBRARY(xllm_ops, m)` | `TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m)` | 无状态 fused kernel: `rms_norm` / `fused_add_rms_norm` / `silu_and_mul` / `fused_qk_norm_rope` |
| `core/kernels/cuda/xllm_attention_ops.cpp` | `TORCH_LIBRARY_FRAGMENT(xllm_ops, m)` | `TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m)` | Attention kernel: `reshape_paged_cache` / `batch_prefill` / `batch_decode` / `batch_chunked_prefill` / `update_*_plan` |

通信 op (`all_reduce` / `all_gather`) 直接在 Python 侧通过 `torch.library.custom_op` 注册(调用 `torch.distributed`),不经过 C++。

设备信息通过配置字典传给 Python 侧:`PyCausalLM::build_config_dict` 写入
`d["device"] = c10::str(device_)`,Python 模型据此在正确设备上建权重/中间张量,
进而让张量的 device 决定 dispatch。

---

## 2. 业界对照(基于本地 vLLM / SGLang 源码)

### 2.1 vLLM — IR op + priority dispatch

`vllm/ir/op.py` 定义 op 时提供 native PyTorch 参考实现,通过
`@op.register_impl("provider", supported=...)` 注册设备特定实现,运行时按
priority config 选择。底层仍走 `torch.library` 的 `CompositeExplicitAutograd`
dispatch key。

### 2.2 SGLang — `direct_register_custom_op` + platform dispatch

`sglang/srt/utils/common.py:direct_register_custom_op`:

```python
my_lib.define(op_name + schema_str)
if is_npu():
    my_lib.impl(op_name, op_func, "PrivateUse1")
elif is_xpu():
    my_lib.impl(op_name, op_func, "XPU")
elif is_musa():
    my_lib.impl(op_name, op_func, "MUSA")
else:
    my_lib.impl(op_name, op_func, "CUDA")
if fake_impl is not None:
    my_lib._register_fake(op_name, fake_impl)
```

**xLLM 现状与二者对齐**:单命名空间(`xllm_ops`)、schema 声明一次、按
DispatchKey 绑定实现、`register_fake` 设备无关共享。差别仅是 xLLM 目前只绑了
`CUDA` 一个 key。

---

## 3. 多硬件注册方案

保持"一套 schema + 一份 register_fake + 每硬件一份 impl"三分层,新增硬件是**加法**
而非改造:

```
                     ops.batch_decode(q, ...)              ← Python 构图唯一入口(不变)
                              │
                     torch.ops.xllm_ops.batch_decode      ← PyTorch dispatcher 按 tensor.device 路由
                              │
        ┌─────────────────────┼──────────────────────────┬─────────────────┐
        ▼                     ▼                          ▼                 ▼
  DispatchKey=CUDA     DispatchKey=PrivateUse1     DispatchKey=Meta   (未来: XPU…)
  cuda fused kernel      Ascend NPU kernel           register_fake
  (已实现)             (新增,本方案)              (设备无关,已实现,复用)
```

落地要点:

1. **schema 不动**:`xllm_ops` 的 `.def(...)` 已是设备无关,NPU 直接复用。
2. **新增 NPU 实现**:新增
   `TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m)`,`m.impl` 到 Ascend 版
   kernel。放在 NPU 专属编译单元(`xllm/core/kernels/npu/`),
   仅在 `--device npu` 构建时编入,与 CUDA impl 互不侵入。
3. **Attention op 的 NPU 实现**:`reshape_paged_cache` / `batch_prefill` /
   `batch_decode` / `batch_chunked_prefill` / `update_*_plan` 需提供对应 NPU
   实现,同样以 `TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m)` 绑定。
4. **`register_fake` 复用**:Python 侧 `compute.py` / `attention.py` 中的
   `register_fake` 是纯 shape/dtype 推导,与设备无关,NPU 上直接复用。
5. **通信 op**:`collectives.py` 中的 `all_reduce` / `all_gather` 使用
   `torch.distributed` API,本身设备无关(NCCL/HCCL 由 backend 决定)。
6. **device 字符串 plumbing**:`build_config_dict` 的 `d["device"]` 已透传
   (如 `"npu:0"`);确保 `torch_npu` 已 import 使 `PrivateUse1` 后端可用。

---

## 4. 能否复用 Python 构图 — 可以,完全复用

论证(三条充分条件都已满足):

1. **构图不含设备分支**:`models/qwen3.py` 与 `layers/*` 只通过 `ops.*` 调用算子,
   没有任何 `if cuda/npu` 或 `.cuda()` 硬编码;设备由传入张量携带。
2. **dispatcher 按 device 路由**:同一 op 名在运行时由 PyTorch 按
   `tensor.device` 选择 `CUDA` 或 `PrivateUse1` 实现。
3. **compile 语义设备无关**:`register_fake`(Meta)只做 shape 推导,
   `disallow_in_graph` 只标记 graph-break 点,均与硬件无关。

**因此,`models/` + `layers/` + `ops/` + `model_runner/` 全部跨硬件复用,零改动。**
不可复用的只有 C++ 后端实现绑定(`TORCH_LIBRARY_IMPL` + 设备 kernel)——
本就应按硬件分别实现。

---

## 5. NPU(Ascend)落地清单

按依赖顺序,均为**新增**,不改动既有 CUDA 路径与 Python 构图:

1. C++:新增 NPU 编译单元,`TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m)` 绑定
   4 个无状态 fused op 的 Ascend 实现(复用 `core/kernels/npu/` 既有 kernel)。
2. C++:在 NPU 分支为 attention op 提供 `PrivateUse1` 实现。
3. 运行时:确保进程 import `torch_npu` 使 `PrivateUse1`/`npu:x` 可用;
   `PyCausalLM` 构造在 NPU 线程上完成初始化。
4. 图模式(可选):在 `model_runner/graph_runner.py` 内按设备选择捕获后端
   (CUDA 走手动 CUDA graph;NPU 参照 SGLang `NPUPiecewiseBackend`)。
5. 校验:先 eager 数值对齐(与 NPU 原生 C++ 路径逐 token 对比),再开图模式。

---

## 6. 边界与风险

- **`PrivateUse1` 的一次性绑定**:PyTorch 全局只有一个 `PrivateUse1` 槽位;若
  同进程已被 `torch_npu` 占用即为 NPU,无冲突;不要在同一进程混挂两种
  PrivateUse1 后端。
- **通信 op 的图捕获**:`all_reduce` / `all_gather` 已注册为
  `torch.library.custom_op`,Dynamo 将其捕获为 opaque 节点;NPU 上
  `torch.distributed` backend 切换为 HCCL 即可,Python 代码不变。
- **in-place op 的 schema 契约**:`fused_add_rms_norm` / `fused_qk_norm_rope`
  是 `(a!)` mutating;NPU 实现必须保持相同的 mutating 语义与返回签名。
- **数值对齐口径**:跨硬件 kernel 不保证 bit 级一致(浮点归约顺序不同),
  对齐标准应为"贪心解码 token 序列一致 / 相对误差阈值内",而非 byte-identical。
