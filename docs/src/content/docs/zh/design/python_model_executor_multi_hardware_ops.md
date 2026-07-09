# Python 模型执行器 · 多硬件算子注册

## 当前架构

Python 模型执行器的算子分三类，各有不同的多硬件扩展方式：

| 类别 | 注册位置 | 当前实现 | 新增硬件需做 |
|------|----------|----------|-------------|
| 计算 op (`rms_norm` / `fused_add_rms_norm` / `silu_and_mul` / `fused_qk_norm_rope` / `reshape_paged_cache`) | C++ `TORCH_LIBRARY(xllm_ops)` + `TORCH_LIBRARY_IMPL(CUDA)` | `core/kernels/cuda/xllm_ops_library.cpp` | 新增 `TORCH_LIBRARY_IMPL(xllm_ops, <DispatchKey>, m)` |
| 通信 op (`all_reduce` / `all_gather`) | Python `torch.library.custom_op` 调 `torch.distributed` | `python/ops/collectives.py` | 无需改动（backend 由 init_process_group 决定） |
| Attention (`batch_prefill` / `batch_decode`) | flashinfer Python API 直接调用 | `python/layers/attention.py` | 替换为目标平台的 attention 实现 |

关键设计点：
- Schema 声明一次（设备无关），`register_fake` 设备无关共享，按 `DispatchKey` 路由到硬件实现。
- `models/` 和 `layers/` 只调 `ops.*` 接口，不含设备分支，构图代码跨硬件复用。

## 风险

- `PrivateUse1` 全局唯一——同进程不能混挂两种 PrivateUse1 后端。
- in-place op（`fused_add_rms_norm` / `fused_qk_norm_rope`）的 NPU 实现必须保持 `(a!)` mutating 语义。
- 跨硬件不保证 bit-identical，对齐标准为贪心解码 token 序列一致。
