# Python 模型执行器 · 算子与交互架构

## C++ ↔ Python 交互

xLLM 进程通过 pybind11 嵌入式解释器调用 Python 模型。交互分三个阶段：

### 1. 初始化（`PyCausalLM` 构造）

```
C++ ModelArgs/ParallelArgs → PROPERTY 反射 → PyDictVisitor → py::dict config
```

传给 Python 模型的 `config` dict 包含：
- 所有 `ModelArgs` 字段（`hidden_size`、`n_heads`、`n_kv_heads`、`n_layers`、`head_dim` 等）
- 所有 `ParallelArgs` 字段（`tp_size`、`tp_rank`）
- `dtype`（字符串：`"bfloat16"` / `"float16"`）
- `device`（字符串：`"cuda:0"` / `"npu:0"`）
- `python_graph_backend`、`max_seqs_per_batch`（decode graph 捕获上限）

Python 模型类通过 `registry.get_model_class(model_type)` 查找，用 `config` dict 构造。

### 2. 权重加载（`load_model`）

C++ `StateDict`（safetensors mmap）通过 `PYBIND11_EMBEDDED_MODULE(xllm_weight_loader)` 暴露为 Python 对象，接口：
- `state_dict.keys() → list[str]`
- `state_dict.has(name) → bool`
- `state_dict.get_tensor(name) → torch.Tensor`（零拷贝，直接引用 mmap 内存）

Python 模型的 `load_weights(state_dicts, tp_rank, tp_size)` 负责按 TP 切分加载。

### 3. 前向推理（每步调用）

C++ 每步构造参数并调用 `py_model_.forward(tokens, positions, attention_metadata, kv_caches)`：

| 参数 | 类型 | 说明 |
|------|------|------|
| `tokens` | `Tensor[int32]` | 当前步 token ids |
| `positions` | `Tensor[int32]` | 位置编码 |
| `attention_metadata` | `dict` | attention 元数据（见下） |
| `kv_caches` | `list[tuple[Tensor, Tensor]]` | 每层 (k_cache, v_cache) |

`attention_metadata` dict 字段：

| 字段 | 含义 |
|------|------|
| `slot_mapping` | 新 token 写入 cache 的 slot 位置 |
| `paged_kv_indptr` | paged KV 的 CSR indptr |
| `paged_kv_indices` | paged KV 的物理 block indices |
| `paged_kv_last_page_len` | 每个序列最后一页的有效 token 数 |
| `is_prefill` | 是否为 prefill 阶段 |
| `is_chunked_prefill` | 是否为 chunked prefill |
| `enable_cuda_graph` | 是否在 graph capture/replay 中 |
| `q_cu_seq_lens` | query 累积序列长度（prefill 时） |
| `kv_cu_seq_lens` | KV 累积序列长度（prefill 时） |
| `qo_indptr` | chunked prefill 的 query-output indptr |

返回值：`torch.Tensor`（hidden_states），C++ 侧再调 `compute_logits` 得到 logits。

## 算子分类

| 类别 | 注册方式 | 位置 |
|------|----------|------|
| 计算 op (`rms_norm` / `fused_add_rms_norm` / `silu_and_mul` / `fused_qk_norm_rope` / `reshape_paged_cache`) | C++ `TORCH_LIBRARY(xllm_ops)` schema + `TORCH_LIBRARY_IMPL(CUDA)` | `core/kernels/cuda/xllm_ops_library.cpp` |
| 通信 op (`all_reduce` / `all_gather`) | Python `torch.library.custom_op` 调 `torch.distributed` | `python/ops/collectives.py` |
| Attention (`batch_prefill` / `batch_decode`) | flashinfer Python API 直接调用 | `python/layers/attention.py` |

设计约束：
- `models/` 和 `layers/` 只调 `ops.*` 接口，不含设备分支。
- 计算 op 的 `register_fake` 设备无关，torch.compile 图中复用。
- Attention 不走 `torch.ops`，由 Python 层直接管理 plan/workspace 生命周期。

## 风险

- in-place op（`fused_add_rms_norm` / `fused_qk_norm_rope`）的非 CUDA 实现必须保持 `(a!)` mutating 语义。
- 跨硬件不保证 bit-identical，对齐标准为贪心解码 token 序列一致。
