# Python 模型执行器：算子与交互架构

## 当前支持范围

Python 模型执行器通过 `--model_impl=python`（`py` 会在配置初始化时规范化为
`python`）启用。目前只在 CUDA 构建中注册，内置 Python 模型仅包含 Qwen3
dense。本文描述的是当前实现，不表示 NPU、MUSA 或其他模型已经可用。

`model_impl`、`python_model_path` 和 `python_graph_backend` 都可以通过命令行或
JSON 配置设置，非默认生效值会写入启动配置转储。`python_model_path` 是包含
`python` package 的目录；未设置时回退到 `XLLM_PYTHON_MODEL_PATH`，再由嵌入式
解释器按原有 `sys.path` 查找。

## C++ 与 Python 交互

xLLM 服务进程通过 pybind11 嵌入 CPython，并在同一进程中调用 Python 模型。
交互分为初始化、权重加载和逐步前向三个阶段；下面将初始化阶段中的 TP 通信组
创建单独列出。

### 1. 初始化（`PyCausalLM` 构造）

```text
C++ ModelArgs/ParallelArgs -> PROPERTY 反射 -> PyDictVisitor -> py::dict config
```

`ensure_python_interpreter()` 先保证 `torch.ops.xllm_ops.*` 的 C++ 静态注册不会被
链接器裁剪，再初始化解释器并导入 `python` package。传给 Python 模型的
`config` dict 包含：

- 可反射的 `ModelArgs` 和 `ParallelArgs` 字段；
- 根据实际 TP group 得到的 `tp_size`、`tp_rank`；
- `dtype` 和当前 CUDA `device`；
- `python_graph_backend`；
- 调度器的 `max_seqs_per_batch`，作为 decode cudagraph 的最大 batch 上限。

Python 侧通过 `registry.get_model_class(model_type)` 查找模型类并用该 dict 构造。
当前 registry 只注册 `Qwen3ForCausalLM`/`qwen3`。

### 2. TP 通信组初始化

当 `tp_size > 1` 时，Python executor 不复用或包装 C++ `ProcessGroup`。C++ 只把
当前 TP 子组对应的 rendezvous host/port、rank 和 world size 传给
`python.ops.init_tp_group()`；Python 使用 `torch.distributed.TCPStore` 和
`ProcessGroupNCCL` 创建独立的 PyTorch TP group。

通信组按 CUDA device 缓存。TP 层通过 `torch.ops.xllm_ops.all_reduce` 和
`torch.ops.xllm_ops.all_gather` 调用 `torch.distributed`；若 TP>1 时尚未初始化
group，算子直接报错，不会静默返回局部张量。TP=1 时各层不调用通信算子。

### 3. 权重加载（`load_model`）

C++ `StateDict` 通过 `PYBIND11_EMBEDDED_MODULE(xllm_weight_loader)` 暴露为
Python 对象，接口为：

- `state_dict.keys() -> list[str]`
- `state_dict.has(name) -> bool`
- `state_dict.get_tensor(name) -> torch.Tensor`

`get_tensor()` 直接返回 C++ `StateDict` 持有的 tensor，不在 pybind bridge 中
额外复制。Python 模型的 `load_weights(state_dicts, tp_rank, tp_size)` 负责权重
变换、TP 切分和复制到目标参数。

### 4. 前向推理（每步调用）

C++ 每步调用
`py_model_.forward(tokens, positions, attention_metadata_dict, kv_caches)`：

| 参数 | 类型 | 说明 |
|------|------|------|
| `tokens` | `Tensor[int32]` | 当前步 token ids |
| `positions` | `Tensor[int32]` | 位置编码 |
| `attention_metadata_dict` | `dict` | attention 元数据 |
| `kv_caches` | `list[tuple[Tensor, Tensor]]` | 每层 `(k_cache, v_cache)` |

`attention_metadata_dict` 包含以下字段：

| 字段 | 含义 |
|------|------|
| `slot_mapping` | 新 token 写入 cache 的 slot 位置 |
| `paged_kv_indptr` | paged KV 的 CSR indptr |
| `paged_kv_indices` | paged KV 的物理 block indices |
| `paged_kv_last_page_len` | 每个序列最后一页的有效 token 数 |
| `is_prefill` | 是否为普通 prefill |
| `is_chunked_prefill` | 是否为 chunked prefill |
| `enable_cuda_graph` | C++ attention metadata 中的 graph 状态 |
| `use_tensor_core` | 当前固定为 `false` |
| `q_cu_seq_lens` | 普通 prefill 的 query 累积长度，可选 |
| `kv_cu_seq_lens` | 普通 prefill 的 KV 累积长度，可选 |
| `qo_indptr` | chunked prefill 的 query-output indptr，可选 |

Python runner 负责 attention plan、forward context 和 graph 调度。模型返回
`hidden_states`，C++ 随后调用 Python `compute_logits()`；`selected_idxes` 存在时
先选择需要计算 logits 的 token。

## Graph 执行模式

`python_graph_backend` 的行为如下：

| 配置值 | 行为 |
|--------|------|
| `off`、空字符串、`none`、`0` | 全部 eager |
| `cudagraphs` | prefill/chunked prefill eager；decode 使用完整 CUDA graph |
| 其他值 | 作为 `torch.compile(model, backend=...)` 的 backend |

`cudagraphs` 按 batch bucket 捕获 decode graph：`1/2/4/8`，之后按 16 向上取整，
不超过 `max_seqs_per_batch`。该上限来自调度配置，不再有独立的
`python_graph_max_batch` 参数。

每个 bucket 使用固定地址的 input、position 和 attention metadata buffer。
padding 序列使用 block manager 预留的 block 0；`paged_kv_indices` 静态 buffer
在真实 KV block 数之外额外预留 `batch_pad` 个 index，避免 KV cache 接近满载时
dummy metadata 越界。FlashInfer plan 在 capture/replay 外执行，并与 graph 使用
同一条专用 CUDA stream。

## 算子分类

| 类别 | 注册/调用方式 | 位置 |
|------|---------------|------|
| CUDA 计算 op（`rms_norm`、`fused_add_rms_norm`、`silu_and_mul`、`fused_qk_norm_rope`、`reshape_paged_cache`） | C++ `TORCH_LIBRARY(xllm_ops)` schema + `TORCH_LIBRARY_IMPL(CUDA)` | `core/kernels/cuda/cuda_ops_library.cpp` |
| TP 通信 op（`all_reduce`、`all_gather`） | Python `torch.library.custom_op` + `torch.distributed` NCCL group | `python/ops/collectives.py` |
| Attention（prefill/decode） | 直接调用 FlashInfer Python API | `python/layers/attention.py` |

约束如下：

- `models/` 和 `layers/` 通过 `python.ops` 调用计算和通信算子；设备实现由当前
  CUDA 注册或 Python collective group 决定。
- C++ 注册的计算 op 在 Python 中补充 `register_fake`，供 `torch.compile` 使用。
- Attention 不注册为 `torch.ops`，由 Python 层管理 FlashInfer wrapper、plan 和
  workspace 生命周期。
- in-place op（`fused_add_rms_norm`、`fused_qk_norm_rope`）必须保持 schema 中的
  mutating 语义。
- TP=1/TP=2 的正确性验收以相同请求的贪心 token 序列与 native executor 一致为
  基准，不要求不同硬件间 bit-identical。
