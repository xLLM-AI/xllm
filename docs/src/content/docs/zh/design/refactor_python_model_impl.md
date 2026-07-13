<!--
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Python Model Execution 重构方案

## 1. 目标

本文描述 PR #1891 基础上的 Python model execution 重构路径。上层架构及 C++/Python
职责边界以
[`cpp_framework_python_model_architecture.md`](cpp_framework_python_model_architecture.md)
为准。

本次重构只完成以下内容：

1. 增加 `PyExecutorImpl`，并注入 `PyCausalLM` 已创建的同一个 Python model；
2. 在 `PyExecutorImpl::run()` 第一次执行时 lazy bind KV cache；
3. 用最小只读 `AttentionMetadataView` 替代每步构造的 metadata `dict`；
4. Python model execution 直接返回 hidden states tensor；
5. 将 FlashInfer wrapper、workspace 和 plan 移入 `FlashInferBackend`；
6. 使用 scoped `ForwardContext` 让 attention layer 调用当前 backend；
7. FlashInfer 每个 step plan 一次，所有 Qwen3 attention layer 复用该 plan；
8. 保持现有 decode full CUDA graph 的 stream 顺序和 padding capacity；
9. TP=1 和 TP=2 使用相同 case 回归。

其他能力不在本次重构中设计或实现。

## 2. 当前问题

PR #1891 当前执行链路为：

```text
LLMWorkerImpl
  -> BaseExecutorImpl
  -> PyCausalLM::forward()
  -> Python PyModelBase.forward()
  -> model-owned runner
  -> Qwen3Model.forward()
```

该路径已经能够运行 eager、decode full CUDA graph 和 TP=1/TP=2，但有以下结构问题：

1. Python model execution 经过 `PyCausalLM::forward()`，没有对应的 Python executor；
2. Python model 创建并持有 runner；
3. model/layer 持有 FlashInfer wrapper、workspace 和 plan；
4. C++ 每个 step 创建 metadata `dict`；
5. C++ 每个 step 重新创建 KV cache 的 Python list/tuple wrapper。

重构只修正这些问题，不改变现有功能行为。

## 3. 目标结构

```text
LLMWorkerImpl
  |-- model_: CausalLM
  |     `-- PyCausalLM
  |           `-- py_model_ -----------------------.
  |                                                 |
  `-- model_executor_: Executor                     |
        `-- PyExecutorImpl                          |
              `-- Python ModelExecutor(model=py_model_)
                    |-- AttentionBackend
                    |-- EagerRunner
                    |-- DecodeCudaGraphRunner
                    `-- InductorRunner (optional)
```

### 3.1 PyCausalLM

`PyCausalLM` 负责：

- 初始化 embedded CPython 和 Python import path；
- 通过 model registry 创建一次 Python model；
- 将权重加载和 logits 计算转发给这个 model；
- 向 `PyExecutorImpl` 提供同一个 Python model object。

`PyCausalLM` 不再创建或调用 runner。Python mode 的 model forward 必须经过
`PyExecutorImpl`，不能保留绕过 Python `ModelExecutor` 的生产路径。

### 3.2 PyExecutorImpl

`PyExecutorImpl` 负责：

- 检查传入的 `CausalLM*` 是 `PyCausalLM`；
- 使用 `PyCausalLM` 已创建的 model 构造 Python `ModelExecutor`；
- 第一次 `run()` 时绑定 KV cache；
- 每个 step 创建 `AttentionMetadataView`；
- 每个 step 调用一次 `ModelExecutor.execute()`；
- 使用 Python 返回的 tensor 构造 C++ `ModelOutput`。

`ExecutorImplFactory` 已经把 `CausalLM*` 传给 executor creator，因此不需要增加新的 model
创建入口。`PyExecutorImpl` 取得已有 model：

```cpp
PyExecutorImpl::PyExecutorImpl(CausalLM* model, /* config */) {
  py_causal_lm_ = dynamic_cast<PyCausalLM*>(model);
  CHECK(py_causal_lm_ != nullptr)
      << "PyExecutorImpl requires PyCausalLM";

  pybind11::gil_scoped_acquire gil;
  py_executor_ = create_model_executor(py_causal_lm_->python_model(), config);
}
```

`PyExecutorImpl` 对 `PyCausalLM` 持有 non-owning pointer，Python `ModelExecutor` 对
`py_model_` 持有强引用。worker 中 model 的生命周期长于 executor。

### 3.3 Executor 选择

executor 选择先判断 model implementation：

```text
model_impl=python -> PyExecutorImpl
model_impl=native -> existing native executor selection
```

`model_impl=py` 归一化为 `python`。Python model 不能因 native `enable_graph` 配置进入
`CudaGraphExecutorImpl`；Python graph 仍由 `python_graph_backend` 选择。

## 4. 每步 Bridge

### 4.1 KV Cache Lazy Bind

不修改公共 `ExecutorImpl` 接口。`ExecutorImpl::run()` 已经接收
`std::vector<KVCache>&`，因此 `PyExecutorImpl::run()` 在第一次调用时完成转换和绑定：

```text
first PyExecutorImpl::run(kv_caches)
  -> create one Python tuple[(k_cache, v_cache), ...]
  -> ModelExecutor.bind_kv_caches(tuple)
  -> FlashInferBackend.bind_kv_caches(tuple)
```

后续 step 复用同一个 tuple 和 per-layer tensor wrapper，不再遍历并重建 Python 对象。
实现记录已经绑定的 cache 数量；后续 `run()` 发现数量变化时直接报错。

### 4.2 AttentionMetadataView

当前 C++ bridge 把 `layer::AttentionMetadata` 转成 Python `dict`。重构后通过 pybind 暴露
只读 `AttentionMetadataView`。view 持有本 step 的
`std::shared_ptr<layer::AttentionMetadata>`，属性直接返回原有 tensor handle 或标量，不
复制 tensor storage。

本次只暴露当前 FlashInfer Python 路径实际读取的字段：

| 字段 | 用途 |
|---|---|
| `slot_mapping` | KV cache write |
| `paged_kv_indptr` | paged prefill/decode plan |
| `paged_kv_indices` | paged prefill/decode plan |
| `paged_kv_last_page_len` | paged prefill/decode plan |
| `qo_indptr` | paged prefill plan |
| `q_cu_seq_lens` | ragged prefill plan |
| `kv_cu_seq_lens` | ragged prefill plan |
| `kv_seq_lens_host` | decode CUDA graph 的 host planner input，避免 plan 中 D2H |
| `is_prefill` | ragged prefill 路由 |
| `is_chunked_prefill` | paged prefill/mixed 路由 |

`PyExecutorImpl::run()` 优先复用 `ModelInputParams::attn_metadata`。如果当前调用链尚未构造
metadata，则沿用现有 `AttentionMetadataBuilder` 构造一次。view 持有所得 `shared_ptr`，
保证 Python execute 期间对象有效。

本次不修改 `AttentionMetadataBuilder`，也不增加新的 batch 状态。C++ metadata 已保存 scheduler
构造的 host cumulative KV lengths；view 只为这份现有 vector 提供零拷贝 tensor 视图。decode
CUDA graph runner 用它更新每个 bucket 的固定 host planner buffer，避免 FlashInfer 通用
`plan()` 每个 token 将 device indptr 同步回 CPU。

### 4.3 调用与返回值

C++ 到 Python 的执行接口保持最小化：

```python
def execute(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    metadata: AttentionMetadataView,
) -> torch.Tensor:
    ...
```

返回值是 hidden states tensor。`PyExecutorImpl` 直接用它构造当前 `ModelOutput`，不增加
Python output dataclass。

## 5. Python ModelExecutor

`ModelExecutor` 接收已有 model，持有 attention backend，并根据配置创建 execution-mode
runner：

```python
class ModelExecutor:
    def __init__(self, model, config):
        self.model = model
        self.attention_backend = create_flashinfer_backend(config)
        self.eager_runner = EagerRunner(model, self.attention_backend)
        self.decode_cuda_graph_runner = maybe_create_decode_runner(...)
        self.inductor_runner = maybe_create_inductor_runner(...)

    def bind_kv_caches(self, kv_caches):
        self.attention_backend.bind_kv_caches(kv_caches)

    def execute(self, input_ids, positions, metadata):
        runner = self.select_runner(metadata)
        return runner.execute(input_ids, positions, metadata)
```

`ModelExecutor` 不访问 model registry，不创建第二个 model。构造 `FlashInferBackend` 时只传入
FlashInfer 当前实际需要的 Qwen3 attention 参数、device 和 dtype，不让 backend 持有或
检查整个 model。

Python model 保持当前 forward 调用方式：

```python
hidden_states = model(input_ids, positions)
```

从 `PyModelBase` 和 Qwen3 model 中移除 `_init_runner()`、`_runner`、
`plan_attention()` 以及 `PagedAttention` 的创建和逐层注入。权重加载、model forward 和
logits 行为保持不变。

## 6. AttentionBackend 与 FlashInferBackend

runner、`ForwardContext` 和 attention layer 只依赖最小 `AttentionBackend` 接口：

```python
class AttentionBackend(ABC):
    def bind_kv_caches(self, kv_caches): ...
    def prepare(self, metadata, *, graph_mode=False): ...
    def execute(self, q, k, v, layer): ...

    @property
    def num_kv_blocks(self): ...
```

`FlashInferBackend` 是当前实现，不放在 model executor/runner 目录中。

`FlashInferBackend` 持有：

- decode、ragged prefill 和 paged prefill wrapper；
- wrapper workspace；
- 已绑定的 per-layer KV cache；
- 当前 step 的 plan state；
- KV cache write 和 attention execute。

当前 Qwen3 的 attention layer 共享同一份 plan：

```text
one scheduler step
  -> FlashInferBackend.plan(...) once
  -> FlashInferBackend.execute(...) num_layers times
```

attention layer 只保留模型计算所需的静态属性和 `layer_id`，forward 时调用当前 backend：

```python
class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        scale,
        sliding_window,
        layer_id,
    ):
        ...

    def forward(self, q, k, v):
        backend = get_forward_context().attention_backend
        return backend.execute(q, k, v, self)
```

model/layer 不 import FlashInfer，不读取 KV cache list，也不调用 plan。

## 7. Scoped ForwardContext

`ForwardContext` 只发布当前 attention backend：

```python
@dataclass(frozen=True, slots=True)
class ForwardContext:
    attention_backend: AttentionBackend
```

context 使用 `ContextVar` 并按作用域恢复：

```python
@contextmanager
def forward_context(context):
    token = _current_context.set(context)
    try:
        yield
    finally:
        _current_context.reset(token)
```

runner 在 model forward 外建立 context，attention layer 从 context 取得 backend。异常或连续
step 结束后不会保留上一次 context；未设置 context 时 attention layer 直接报错。

metadata 和 KV cache 由 backend 在 plan/bind 时取得，不通过 context 再发布一份 raw
metadata 或 KV cache list。

## 8. Runner 与 Decode Graph

`EagerRunner`、`DecodeCudaGraphRunner` 和 `InductorRunner` 都继承最小 `BaseRunner`，并由
`ModelExecutor` 持有和选择。`ModelExecutor` 本身不保留另一套 model forward 实现。
当前配置语义下三者是 execution mode：`off` 选择 `EagerRunner`，`cudagraphs` 选择
`DecodeCudaGraphRunner`，其他 `torch.compile` backend 名称选择 `InductorRunner`。
本次重构不改变该语义，也不组合 Inductor 与手工 decode CUDA graph。

eager step：

```text
EagerRunner
  -> AttentionBackend.prepare(live_metadata)
  -> scoped ForwardContext(AttentionBackend)
  -> model(input_ids, positions)
```

decode full graph step：

```text
DecodeCudaGraphRunner
  -> select decode bucket
  -> copy input and metadata into persistent buffers
  -> AttentionBackend.prepare(static_metadata, graph_mode=True)
  -> scoped ForwardContext(AttentionBackend)
  -> capture or replay model
  -> return the live output slice
```

FlashInfer plan 位于 model forward 和 captured region 之外。decode graph 保持当前 stream
顺序：

1. graph stream 等待调用方 current stream；
2. graph stream 更新 persistent input 和 metadata；
3. 同一 graph stream 执行 FlashInfer plan；
4. capture 或 replay model；
5. 调用方 current stream 等待 graph stream。

每个 decode bucket 使用独立的 FlashInfer wrapper，并在构造时设置 `use_cuda_graph=True`，将
wrapper 绑定到该 bucket 的 persistent device metadata。首次 capture 使用完整 `plan()` 完成
module 初始化，后续 step 使用 FlashInfer `fast_decode_plan()` 和固定 host indptr 更新 plan。
不能用普通 wrapper capture 后再逐 step 调用通用 `plan()`：通用 plan 的 D2H 会形成固定 host
bubble，而且重新 plan 不能改变已经 capture 的 kernel launch topology。

graph padding 保持当前容量约束：`paged_kv_indices` 除真实 KV index capacity 外，还必须为
bucket padding 保留足够容量。`indptr[-1]` 不能超过 indices capacity。

## 9. 实施顺序

### Phase 1：PyExecutorImpl 与最小 Bridge

- 新增并注册 `PyExecutorImpl`；
- Python model 选择 `PyExecutorImpl`；
- 将 `PyCausalLM.py_model_` 注入 `ModelExecutor`；
- 将现有 runner 从 model 移到 `ModelExecutor`；
- 在 `PyExecutorImpl::run()` 中 lazy bind KV cache；
- 增加最小 `AttentionMetadataView`；
- Python execute 直接返回 tensor；
- 关闭 `PyCausalLM::forward()` 旁路。

Phase 1 完成后仍使用现有 attention 和 runner 行为，eager 与 decode graph 必须可运行。

### Phase 2：FlashInferBackend、Context 与 Decode Graph

- 增加 `FlashInferBackend` 和 scoped `ForwardContext`；
- 增加最小 `AttentionBackend` 和 `BaseRunner`；
- 将 eager、decode CUDA graph 和 Inductor 拆为平级 runner；
- 将 wrapper、workspace、plan、KV write 和 execute 从 model/layer 移入 backend；
- runner 通过 backend 完成 eager plan；
- attention layer 通过 context 调用 backend；
- 同时将 decode graph 的 metadata 更新和 plan 切换到 backend；
- 删除 model-owned attention state 和 `plan_attention()`。

backend、context 和 decode graph 在同一 phase 迁移，避免出现 eager 已切换而 graph 不可用
的中间状态。

## 10. 验收

### 10.1 Bridge 与对象身份

- `PyExecutorImpl` 注入的 model 与 `PyCausalLM.py_model_` 是同一个 Python object；
- model registry 每个 worker 只调用一次；
- load weights、forward 和 logits 使用同一个 model；
- KV cache Python tuple 和 per-layer wrapper 跨 step identity 不变；
- metadata view tensor 与 C++ tensor 使用相同 storage；
- 每个 step 只有一次 C++ 到 Python `execute()` 调用；
- Python execute 返回 tensor，C++ 正确构造 `ModelOutput`。

### 10.2 FlashInfer 与 Graph

- 每个 step 的 FlashInfer plan count 为 1；
- execute count 等于 Qwen3 attention layer 数；
- attention execute 不触发 plan 或 workspace 初始化；
- plan 不进入 CUDA graph captured region；
- graph stream 顺序保持不变；
- 奇数 batch 和 KV cache 接近满载时 padding metadata 不越界；
- eager 和 decode full CUDA graph 与重构前结果一致。

### 10.3 TP 回归

TP=1 和 TP=2 必须运行相同的功能和 graph case，包括相同的 batch size 集合。两种配置都要
验证 native/Python greedy parity，不能将 TP=2 缩减为 smoke test。
