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

# Python Model Execution 重构实施设计

## 1. 文档定位

本文给出 Python model execution 的可实施重构方案，包括跨语言接口、对象生命周期、
迁移顺序和验收条件。上层架构选择以
[`cpp_framework_python_model_architecture.md`](cpp_framework_python_model_architecture.md)
为准；本文不重复论证是否采用 Python model execution，也不改变该文档确定的 C++ 与
Python 所有权边界。

本文以 PR [#1891](https://github.com/xLLM-AI/xllm/pull/1891) 的当前实现为基线。
文中的目标类和接口尚未全部实现。为避免把待设计能力误写成已支持能力，本文使用以下
状态：

- **本次实现**：纳入本轮重构，必须完成对应测试后才能合入；
- **后续实现**：接口需要预留，但本轮不启用；
- **TODO/启动失败**：语义尚未完成。在解除 TODO 前，相关配置必须在启动阶段报错，
  不能进入 eager fallback 或部分执行路径。

## 2. 实施基线

### 2.1 保持不变的 C++ framework

Python model execution 接入 C++ worker 的模型计算位置。以下能力继续由 C++ 负责，本次
重构不复制到 Python：

- Input Processing 和 Output Processing；
- Continuous Batching Scheduler，包括 chunked prefill、mixed batch、优先级和 batch
  形成；
- Worker Runtime、rank/process 管理、多节点通信和 PD 分离；
- KV Cache Manager、block allocation、prefix cache、KV transfer 和 cache tensor
  生命周期；
- `Batch::prepare_forward_input()` 和 common attention metadata 构造；
- Sampling；
- Speculative Decoding 的整体流程，包括 MTP、Eagle、Suffix 的 draft、verification、
  token acceptance 和回滚；
- checkpoint 读取、`StateDict` 生命周期、metrics、profiling 和 device monitoring；
- native C++ model 与 native `ExecutorImpl` 路径。

Python 只负责 model execution：model、`ModelExecutor`/runner、attention backend、模型
内部 TP collective 和设备计算调用。

### 2.2 当前 PR 的可运行路径

PR #1891 已经提供以下基线能力：

- `model_impl=python` 创建 `PyCausalLM`；
- Python Qwen3 model、权重加载和 logits；
- `torch.ops.xllm_ops` 无状态算子；
- FlashInfer Python attention；
- PyTorch distributed TP=1/TP=2；
- eager prefill/chunked prefill 和 decode full CUDA graph；
- flag/JSON 配置、wheel 和 CUDA image 运行时。

当前生产链路为：

```text
LLMWorkerImpl
  -> BaseExecutorImpl
  -> PyCausalLM::forward()
  -> Python PyModelBase.forward()
  -> GraphRunner
  -> Qwen3Model.forward()
```

这条路径能够回归功能，但存在以下结构问题：

1. Python model 自己创建并调用 runner，model 与 executor 无法独立演进；
2. model/layer 持有 FlashInfer wrapper、workspace 和 plan state；
3. runner 直接修改 FlashInfer metadata；
4. C++ 每步构造 metadata `dict`；
5. C++ 每步遍历全部 layer，重新构造 KV cache Python list 和 tuple；
6. 当前 `forward(input_ids, positions)` 只覆盖基本文本输入，未定义后续 multimodal、
   mRoPE 和 speculative/MTP 输入如何扩展；
7. dummy forward、schedule overlap 和 speculative/MTP 在 Python execution path 中没有
   完整语义。

### 2.3 本轮支持矩阵

| 场景 | 本轮状态 | 行为 |
|---|---|---|
| Native C++ model + native executor | 保持支持 | 不经过 Python |
| Python text model + eager | 本次实现 | prefill、chunked prefill、mixed、decode |
| Python text model + decode full CUDA graph | 本次实现 | 只允许纯 decode batch |
| Python prefill/mixed graph | 后续实现 | 当前走 eager，不宣称 graph 支持 |
| Python mode + schedule overlap/dummy forward | TODO | 启动失败 |
| Python mode + speculative decoding/MTP | TODO | 启动失败 |
| Python multimodal model | 后续实现 | 完成 typed input contract 后启用 |
| Python model + native C++ graph executor | 非目标 | 启动失败 |
| Native C++ model + Python `ModelExecutor` | 非目标 | 不注册该组合 |

这里的启动失败是阶段性能力门禁，不代表目标架构排除 schedule overlap、speculative
decoding/MTP 或 multimodal model。

## 3. 目标结构与固定约束

### 3.1 对象关系

目标结构保留 C++ 现有 model/executor 两层，并分别增加 Python adapter：

```text
LLMWorkerImpl
  |-- model_: CausalLM
  |     `-- PyCausalLM
  |           `-- py_model_ ------------------------.
  |                                                  |
  `-- model_executor_: Executor                      |
        `-- PyExecutorImpl                           |
              `-- Python ModelExecutor(model=py_model_)
                    |-- Runner
                    `-- AttentionBackend
```

`PyCausalLM` 与 `PyExecutorImpl` 是 C++/Python 边界上的 adapter，因此使用 `Py` 前缀。
Python package 已经表达语言归属，Python 执行对象命名为 `ModelExecutor`。

### 3.2 每步执行链路

```text
C++ scheduler/Batch
  -> ForwardInput + AttentionMetadata
  -> PyExecutorImpl::run()                 # 每 step 一次 C++ -> Python
  -> ModelExecutor.execute(step_inputs)
  -> Runner.run(step_inputs)
  -> AttentionBackend.prepare_step()       # 每 step 一次，内部每 group 准备一次
  -> scoped ForwardContext
  -> Model.forward(model_inputs)
  -> AttentionBackend.execute()            # 每 attention layer 一次
  -> hidden states
```

该链路必须满足：

1. 每个 worker 的 Python model 只创建一次；
2. load weights、forward 和 logits 使用同一个 Python model 对象；
3. 每 step 只有一次 C++ 到 Python execution call；
4. tensor storage 零拷贝；
5. KV cache 等稳定对象只绑定一次；
6. C++ 已构造的 sequence length、page table 和 slot mapping 不在 Python 重算；
7. model 不持有 runner、attention backend、padding 或 graph state；
8. runner 不读取 backend 私有 plan/state。

## 4. Phase 0：冻结跨语言契约

Phase 0 必须先于实际搬迁代码完成。它的产物是下面各节的 C++/Python schema、生命周期
测试和配置门禁。后续 phase 只能在兼容该契约的前提下修改内部实现。

### 4.1 Model 与 executor 身份

初始化时只允许一次 model registry lookup：

1. `create_llm_model()` 创建 `PyCausalLM`；
2. `PyCausalLM` 初始化解释器并创建 `py_model_`；
3. `ExecutorImplFactory` 把同一个 `CausalLM*` 传给 `PyExecutorImpl`；
4. `PyExecutorImpl` checked cast 到 `PyCausalLM*`；
5. `PyExecutorImpl` 取得 `py_model_`，并调用 `ModelExecutor(py_model_, config)`；
6. `ModelExecutor` 持有该 model 的 Python 强引用，不得再次访问 model registry。

```cpp
PyExecutorImpl::PyExecutorImpl(CausalLM* model, /* ... */) {
  py_causal_lm_ = dynamic_cast<PyCausalLM*>(model);
  CHECK(py_causal_lm_ != nullptr)
      << "PyExecutorImpl requires PyCausalLM";

  pybind11::gil_scoped_acquire gil;
  py_executor_ = create_model_executor(
      py_causal_lm_->python_model(), /* config */);
}
```

`PyExecutorImpl` 对 `PyCausalLM` 持有 non-owning pointer；`LLMWorkerImpl` 保证 model 的
生命周期长于 executor。Python `ModelExecutor` 对 `py_model_` 持有强引用。

### 4.2 ExecutorImpl 接口扩展

KV cache 在 model/executor 构造后才分配，因此给 `ExecutorImpl` 增加一次性绑定 hook：

```cpp
class ExecutorImpl {
 public:
  virtual void bind_kv_caches(std::vector<KVCache>&) {}

  virtual ModelOutput run(const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          std::vector<KVCache>& kv_caches,
                          const ModelInputParams& params) = 0;
};
```

`Executor` 增加同名转发方法。`WorkerImpl::allocate_kv_cache_storage()` 在
`allocate_kv_caches()` 成功返回后调用一次 `model_executor_->bind_kv_caches(kv_caches_)`。
所有普通分配、带 transfer 分配和 async 分配最终都经过该 owning path，不在各调用方
分别增加 Python 特判。

native executor 使用默认 no-op。`PyExecutorImpl` 必须实现该 hook，并拒绝重复绑定。
现有 `run()` 参数中的 `kv_caches` 为兼容 native executor 暂时保留；Python 路径只检查
已经完成绑定，不再按 layer 遍历和转换该参数。

`prepare_graph_input()` 对 `PyExecutorImpl` 保持有意的 no-op。Python graph input 由
Python runner 管理，不通过 native graph hook 准备。

### 4.3 StepInputsView

C++ 每步向 Python 传递一个只读 `StepInputsView`，替代多个位置参数和动态字典：

```cpp
class PyStepInputsView final {
 public:
  const torch::Tensor& input_ids() const;
  const torch::Tensor& positions() const;
  const torch::Tensor& input_embeddings() const;  // optional
  const PyAttentionMetadataView& attention() const;
  BatchKind batch_kind() const;  // delegates to attention().batch_kind()
  int64_t num_tokens() const;
  int64_t num_sequences() const;
  int64_t actual_num_sequences() const;

  // Typed optional facets. Phase 0 only exposes a facet after its schema is
  // implemented and tested.
  std::optional<PyMultimodalInputsView> multimodal() const;
  std::optional<PySpeculativeInputsView> speculative() const;
};
```

`StepInputsView` 持有本次 forward 所需对象的强引用或 shared ownership，生命周期至少
覆盖 `ModelExecutor.execute()`。属性返回现有 tensor handle，不复制 storage。

输入扩展采用 typed optional facet，不使用 `dict[str, Any]`：

- 公共 text 字段变化时更新 `StepInputsView` schema；
- multimodal 字段进入 `MultimodalInputsView`；
- speculative/MTP 字段进入 `SpeculativeInputsView`；
- 新 facet 必须有版本、必需字段检查和独立测试；
- runner 对不理解的 facet 不做解释。具体 model 消费 model input，相关 runner 通过
  capability 声明是否支持 graph；
- 本轮只实现 text core。检测到未支持的 multimodal/speculative facet 时必须报错，不能
  丢弃字段后继续 forward。

`positions` 的 rank/shape 由具体 model contract 解释，因此公共 bridge 不把它限制为
一维。这样 mRoPE 可以在后续 model contract 中使用结构化 position tensor，而不需要
改 executor 的控制流。

### 4.4 AttentionMetadataView 完整 schema

`AttentionMetadataView` 是 C++ common metadata 到 Python attention backend 的只读
契约。它持有 `std::shared_ptr<const AttentionMetadata>`，不得暴露 C++ FlashInfer
workspace、native `plan_info` 或 native graph state。

`PyExecutorImpl::run()` 在构造 view 前确保 common metadata 已经存在：如果
`ModelInputParams::attn_metadata` 已定义则直接复用；否则调用一次
`AttentionMetadataBuilder::build()` 并放入本 step 的 `shared_ptr`。model 和 Python
backend 都不能再次 build 或修改 common metadata。`StepInputsView.batch_kind()` 只委托给
`AttentionMetadataView.batch_kind()`，不保存第二份 batch 状态。

批次类型使用单一枚举，避免多个 bool 产生不一致组合：

```text
BatchKind = PREFILL | CHUNKED_PREFILL | MIXED | DECODE | DUMMY
```

`is_prefill` 等便捷属性只能从 `BatchKind` 派生，不能作为第二份状态存储。baseline schema
如下：

| 字段 | 位置/形式 | 必需场景 | 主要消费方 |
|---|---|---|---|
| `batch_kind` | host enum | 全部 | executor、runner、backend |
| `num_tokens` | host scalar | 全部 | runner、graph slicing |
| `num_sequences` | host scalar | 全部 | runner、backend |
| `actual_num_sequences` | host scalar | 全部 | padding、output slicing |
| `max_query_len` | host scalar | 全部 | backend plan/dispatch |
| `max_seq_len` | host scalar | 全部 | backend plan/dispatch |
| `total_kv_len` | host scalar | prefill/mixed | workspace/plan |
| `is_causal` | host bool | 全部 | backend kernel selection |
| `q_seq_lens` | device tensor | backend capability 决定 | attention execute/plan |
| `kv_seq_lens` | device tensor | backend capability 决定 | attention execute/plan |
| `q_cu_seq_lens` | device tensor | prefill/mixed | ragged/paged plan |
| `kv_cu_seq_lens` | device tensor | prefill | ragged plan |
| `block_table` | device tensor | paged attention | backend execute |
| `slot_mapping` | device tensor | 非 dummy | KV cache write |
| `paged_kv_indptr` | device tensor | paged attention | backend plan |
| `paged_kv_indices` | device tensor | paged attention | backend plan |
| `paged_kv_last_page_len` | device tensor | paged attention | backend plan |
| `qo_indptr` | optional device tensor | backend capability 决定 | paged prefill/decode |
| `q_seq_lens_host` | read-only host view | backend capability 决定 | plan/shape logic |
| `kv_seq_lens_host` | read-only host view | backend capability 决定 | plan/shape logic |
| `q_cu_seq_lens_host` | read-only host view | prefill backend 需要时 | plan |
| `kv_cu_seq_lens_host` | read-only host view | prefill backend 需要时 | plan |

host sequence length 来自 C++ `AttentionHostInput`，不能通过 device tensor `.cpu()` 构造。
pybind view 在构造时缓存只读 host buffer/memoryview，backend 访问它不得触发 D2H 或再次
构造逐元素 Python list。

view 对 length 字段规定统一语义和 dtype：

- `*_seq_lens` 是长度为 `num_sequences` 的逐序列长度；
- `*_cu_seq_lens` 是长度为 `num_sequences + 1`、首元素为 0 的累计长度；
- page index、slot mapping 和 length tensor 使用连续 `int32`，除非 backend capability
  明确声明其他 dtype；
- 当前 C++ 不同设备路径中的历史字段如果语义不同，由 metadata builder 在 host 侧归一化，
  不把歧义带入 Python schema。

当前 `AttentionMetadata` 缺少 `num_tokens`、`num_sequences`、
`actual_num_sequences` 和完整 host cumulative length view。Phase 0 需要从
`ModelInputParams::meta/attention.host` 补齐 owning schema；不能在 Python 通过 tensor
shape 或同步 device tensor 猜测这些值。

backend 声明 required fields：

```python
@dataclass(frozen=True, slots=True)
class AttentionCapabilities:
    supported_batch_kinds: frozenset[BatchKind]
    required_device_fields: frozenset[str]
    required_host_fields: frozenset[str]
    supports_graph: frozenset[BatchKind]
    requires_plan: bool
```

`ModelExecutor` 在初始化时校验静态 capability，并在每个 step 校验当前 `BatchKind` 的
必需字段是否 defined、dtype/rank 是否符合 schema。实现可以缓存字段 accessor 和不变量，
但不能只校验第一次出现的 batch。缺字段直接报错；不允许 backend 内部通过 D2H 或重新
展开 block table 补齐。

### 4.5 KV cache 一次性绑定

`PyExecutorImpl::bind_kv_caches()` 只执行一次：

```text
std::vector<KVCache>
  -> one Python tuple[(k_cache, v_cache), ...]
  -> ModelExecutor.bind_kv_caches(...)
  -> AttentionBackend.bind_kv_caches(...)
```

约束如下：

- tuple 和每层 `(k_cache, v_cache)` wrapper 在 worker 生命周期内复用；
- `ModelExecutor.execute()` 和 `ForwardContext` 不再接收 per-step `kv_caches` 参数；
- backend 按 `layer_id` 读取已绑定 cache；
- layer 数、dtype、device、layout 和 storage pointer 在绑定时一次性校验；
- execute 前未绑定、重复绑定或 cache 数量不匹配都要报错；
- sleep/wake 只允许保持 tensor identity 的 storage remap。如果未来需要替换 tensor，必须
  增加显式 `unbind/rebind` 生命周期，不能让旧 Python wrapper 静默存活。

### 4.6 返回值契约

Python model execution 返回固定结构，不返回任意 dict：

```python
@dataclass(frozen=True, slots=True)
class ModelExecutionOutput:
    hidden_states: torch.Tensor
    dsa_topk_indices: torch.Tensor | None = None
```

本轮 text model 只要求 `hidden_states`。新增输出字段必须先进入 C++ `ModelOutput` 和上述
schema，再由 worker 的 owning path 消费。

## 5. C++ adapter 实现

### 5.1 PyCausalLM

`PyCausalLM` 负责：

- 初始化 embedded CPython 和 package import path；
- 通过 registry 创建一次 Python model；
- `load_model()` 转发到 `model.load_weights()`；
- `logits()` 转发到 `model.compute_logits()`；
- 仅向 `PyExecutorImpl` 提供受控的 `python_model()` accessor。

`PyCausalLM` 不创建 runner、attention backend、workspace 或 graph state。

Python mode 下 `PyCausalLM::forward()` 必须明确失败：

```cpp
LOG(FATAL) << "Python model forward must run through PyExecutorImpl";
```

这是为了满足 `CausalLM` 现有虚接口，同时关闭未经过 `ModelExecutor` 的旁路。不得保留一条
“仍然可运行但生产代码不调用”的旧 forward。

### 5.2 PyExecutorImpl

`PyExecutorImpl` 负责：

- checked cast 并注入已有 Python model；
- 创建唯一的 Python `ModelExecutor`；
- 一次性绑定 KV cache；
- 每 step 创建只读 `StepInputsView`/`AttentionMetadataView`；
- 调用一次 `ModelExecutor.execute()`；
- 将 `ModelExecutionOutput` 转回 `ModelOutput`；
- 记录 bridge latency、错误和对象分配指标。

`PyExecutorImpl` 不负责 padding、attention plan、workspace 或 graph capture。

### 5.3 Executor 选择与配置校验

executor 选择先判断 model implementation：

```text
model_impl=python -> backend="python" -> PyExecutorImpl
model_impl=native -> existing backend selection
```

选择逻辑必须集中在 config validation 和 `Executor` factory input 构造处，而不是让 Python
model 先进入 native executor 后再做运行时判断。

Python mode 下：

- `enable_graph` 不得使 factory 选择 native graph executor；
- 本轮要求 `enable_graph=false`；设置为 `true` 时启动失败；
- Python graph 只由 `python_graph_backend`/runner registry 选择；
- 未注册的 graph backend 在启动时失败，不能把任意字符串直接传给 `torch.compile`；
- `PyExecutorImpl` 收到非 `PyCausalLM` 时立即失败。

### 5.4 解释器、线程与 GIL

运行时采用 one rank per process、one interpreter per rank：

- interpreter 在第一个 Python object 前初始化，在所有 Python object 析构后结束；
- `PyExecutorImpl` 必须先于 `PyCausalLM` 析构；
- 构造、绑定、execute、logits、load weights 和析构 Python object 时都显式持有 GIL；
- 同一个 `ModelExecutor` 不允许并发 execute；
- C++ scheduler、request processing 和 KV cache 管理线程不进入 Python；
- Python exception 必须带 traceback 转为明确的 C++ error，不允许清除异常后继续执行。

schedule overlap 会改变 in-flight step 与输出生命周期，因此在对应 TODO 完成前由启动门禁
阻止，不能只依赖单线程 queue 推断安全。

由于当前 `enable_schedule_overlap` 默认值为 `true`，Python mode 在 TODO 解除前必须在
启动配置中显式设置为 `false`；validation error 需要直接给出该修正方式。

## 6. Python Model 与 ModelExecutor

### 6.1 Model contract

```python
class Model(Protocol):
    def forward(self, inputs: ModelInputs) -> torch.Tensor: ...

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        selected_idxes: torch.Tensor | None,
    ) -> torch.Tensor: ...

    def load_weights(
        self,
        state_dicts: Sequence[StateDict],
        tp_rank: int,
        tp_size: int,
    ) -> None: ...
```

`ModelInputs` 是 `StepInputsView` 中 model-visible 字段的 typed Python facade。本轮至少
包含 `input_ids`、`positions` 和 optional `input_embeddings`。attention metadata 和 KV
cache 不通过 `ModelInputs` 传给每层，而是由已准备好的 `ForwardContext` 发布。

具体 model 不包含：

- `_init_runner()`；
- `plan_attention()`；
- graph capture/replay hook；
- backend workspace；
- graph padding；
- ProcessGroup 初始化和销毁。

使用现有 input facet 的新增 model 可以增加自己的内部 module 和 weight loader，不应修改
C++ adapter、公共 runner 或现有 model 的 forward 调用方式。只有新增一类跨语言输入时才
扩展对应 typed facet；该扩展属于 bridge schema 变更，需要先完成 Phase 0 的字段和测试
要求。

### 6.2 ModelExecutor contract

```python
class ModelExecutor:
    def __init__(self, model: Model, config: ExecutorConfig) -> None:
        self.model = model
        groups = collect_attention_groups(model, config)
        self.attention_backend = AttentionBackendRegistry.create(config, groups)
        self.runner = RunnerRegistry.create(
            config, model, self.attention_backend
        )
        self._kv_caches = None

    def bind_kv_caches(
        self,
        kv_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        if self._kv_caches is not None:
            raise RuntimeError("KV caches are already bound")
        self.attention_backend.bind_kv_caches(kv_caches)
        self._kv_caches = kv_caches

    def execute(self, step: StepInputsView) -> ModelExecutionOutput:
        if self._kv_caches is None:
            raise RuntimeError("KV caches are not bound")
        return self.runner.run(step)
```

`ModelExecutor` 不创建 model、不加载权重、不计算 logits。它是 Python execution 的
composition root，负责组装 runner、attention backend 和 distributed state。

### 6.3 ForwardContext

`ForwardContext` 只发布已经准备完成的 runtime state：

```python
@dataclass(frozen=True, slots=True)
class ForwardContext:
    attention_backend: AttentionBackend
    attention_state: AttentionStepState
```

它不发布 raw `AttentionMetadataView` 或 KV cache list。layer 不能绕过 backend 读取未
padding metadata；backend 从一次绑定的 cache 中按 `layer_id` 取 cache。

context 使用作用域恢复：

```python
@contextmanager
def forward_context(context: ForwardContext) -> Iterator[None]:
    token = _current_context.set(context)
    try:
        yield
    finally:
        _current_context.reset(token)
```

实现可以使用 `contextvars.ContextVar` 或等价的线程/上下文隔离机制，但必须通过嵌套、异常
和连续 step 测试。未设置 context 时 attention layer 直接报错。

## 7. AttentionBackend

### 7.1 薄 Attention layer

model 中的 attention layer 只持有静态配置、`layer_id` 和 `group_id`：

```python
class Attention(nn.Module):
    def forward(self, q, k, v, output=None):
        context = get_forward_context()
        return context.attention_backend.execute(
            layer=self,
            q=q,
            k=k,
            v=v,
            state=context.attention_state,
            output=output,
        )
```

model/layer 不 import FlashInfer，不创建 wrapper/workspace，不调用 plan。

### 7.2 AttentionSpec 与 group

`AttentionSpec` 包含影响 kernel、workspace 或 plan 的静态字段：head 数、KV head 数、
head dim、value head dim、scale、sliding window、dtype、KV cache dtype、block size、
cache layout、attention type 和 quantization。

executor 初始化时根据 spec 建立 `AttentionGroup` 和 `layer_id -> group_id`。group key 不
包含 `layer_id`；逐层 scale 或 layer index 等不影响共享 plan 的数据仍由 layer 持有。

### 7.3 Backend contract

```python
class AttentionBackend(ABC):
    def bind_kv_caches(self, kv_caches: KVCaches) -> None: ...

    def prepare_step(
        self,
        metadata: AttentionMetadataView,
    ) -> AttentionStepState: ...

    def create_graph_state(
        self,
        bucket_size: int,
        capacity: GraphCapacity,
    ) -> AttentionGraphState: ...

    def prepare_graph_step(
        self,
        graph_state: AttentionGraphState,
        metadata: AttentionMetadataView,
    ) -> AttentionStepState: ...

    def execute(self, *, layer, q, k, v, state, output=None): ...
```

`AttentionStepState` 按 `group_id` 保存 opaque state。runner 只持有和发布它，不读取内部
字段。

`prepare_step()` 每个 live step 调用一次，但不是所有 backend 都必须执行 plan：

- `capabilities.requires_plan=True` 的 backend 在该方法中每 group plan 一次；
- 不需要 plan 的 backend 可以只准备 tiling/shape state 或返回 no-op state；
- `execute()` 不得首次分配 workspace、触发 plan 或读取 raw metadata。

### 7.4 FlashInfer 生命周期

`FlashInferBackend` 每个 attention group 长期持有 wrapper 和 workspace。对于当前 Qwen3
单 group：

```text
one scheduler step
  -> one FlashInfer plan
  -> num_layers FlashInfer execute
```

prefill、chunked prefill/mixed 和 decode 分别选择对应 wrapper。路由规则固定为：

| BatchKind | backend path | graph |
|---|---|---|
| `PREFILL` | ragged prefill | 本轮 eager |
| `CHUNKED_PREFILL` | paged prefill | 本轮 eager |
| `MIXED` | paged/chunked prefill | 本轮 eager |
| `DECODE` | paged decode | 可进入 decode full graph |
| `DUMMY` | 未实现 | 报错 |

mixed batch 只要包含 prefill token，就按 chunked prefill 处理，不能进入 decode full graph。
plan 必须在 model forward/captured region 外完成，并在同一 graph stream 上先于 capture/replay
提交。

## 8. Runner 与 graph

### 8.1 公共 runner contract

```python
class ModelRunner(ABC):
    @abstractmethod
    def run(self, step: StepInputsView) -> ModelExecutionOutput: ...
```

- `EagerRunner` 调用 `backend.prepare_step()`，发布 context，再执行 model；
- `CompiledRunner` 使用相同准备流程，只替换 model call；
- `CudaGraphRunner` 管理 CUDA bucket、persistent input、capture stream 和 graph object。

公共 `ModelRunner` 不定义设备 graph API。不同硬件 runner 通过 registry/capability 选择，
不要求继承 `CudaGraphRunner` 或共享 `DeviceGraphBackend`。

### 8.2 Decode full CUDA graph

纯 decode step 的执行顺序为：

```text
CudaGraphRunner
  -> validate BatchKind == DECODE
  -> select bucket
  -> copy model inputs to runner-owned persistent buffers
  -> FlashInferBackend.prepare_graph_step()
       - copy common metadata to backend-owned persistent buffers
       - append padding metadata
       - plan once on graph stream
  -> publish ForwardContext
  -> capture or replay model
  -> slice output to actual_num_sequences
```

runner 只管理 model-level input/output；paged KV metadata、dummy page entries、workspace 和
plan state 由 backend graph state 管理。

`batch_size > max_seqs_per_batch` 表示 scheduler/executor contract 被破坏，必须报错，不能
eager fallback。否则同一配置会因 batch shape 静默切换执行语义。

### 8.3 Padding 与容量

每个 bucket 的 backend graph state 必须显式记录：

- `bucket_size`；
- `max_blocks_per_sequence`；
- real KV index capacity；
- 至少 `bucket_size` 个额外 padding index capacity；
- runner/model static input/output bytes；
- backend static metadata、workspace 和 graph pool bytes。

`paged_kv_indices` 容量至少为：

```text
num_kv_blocks + bucket_size
```

或使用 backend 定义的、等价且经过验证的 dummy page metadata。无论采用哪种形式，
`indptr[-1]` 不能超过 indices capacity，dummy block 必须是 KV manager 保留且不会影响真实
sequence 的页。

### 8.4 最大 bucket 与显存预算

Python decode graph 的最大 bucket 直接使用 `max_seqs_per_batch`，不恢复
`python_graph_max_batch`。这要求 graph runner 在初始化时：

1. 对最大 bucket 做 capacity 计算；
2. 输出可观测的 persistent buffer/workspace/graph pool 预算；
3. 检查 bucket list 的最后一项等于 scheduler 上限向上取整后的合法 bucket；
4. 在 capture 前检查实际可用显存；
5. 对最大 bucket 进行集成测试，而不只测试 batch 100。

如果默认 `max_seqs_per_batch` 导致不可接受的 graph 显存，用户需要调整 scheduler 的同一
上限。不能增加一个与 scheduler 脱节的 Python graph 上限，也不能超过上限后静默 eager。

### 8.5 Python prefill graph

Python prefill、chunked prefill 和 mixed graph 是后续目标。当前实现尚未达到性能要求，
因此本轮 `cudagraphs` 对这些 batch 明确使用 eager。

后续启用前必须作为 runner/backend capability 接入，并满足：

- prefill、chunked prefill、mixed 的 eager parity；
- TP=1/TP=2；
- plan 在 captured region 外；
- dynamic shape/padding 和 KV index capacity 正确；
- TTFT、吞吐和显存达到预先记录的门槛。

不能保留一个默认关闭、没有测试的 piecewise prefill graph alternative path。

## 9. Distributed 与设备算子

### 9.1 TP distributed

Python TP 分成两个模块：

```text
distributed/parallel_state.py
  -> group initialization/query/destruction

distributed/communication_op.py
  -> all_reduce/all_gather/reduce_scatter
```

group 使用 PyTorch distributed。初始化根据当前 tensor device/platform 选择 PyTorch 支持的
backend，不直接构造 `ProcessGroupNCCL`。model layer 只调用 communication API。

启动时校验 rank、world size、local device 和 rendezvous 参数；TP>1 且 group 未初始化时
collective 必须报错，不能把 collective 当 identity 返回。

### 9.2 Stateless ops

RMSNorm、SwiGLU、RoPE、KV cache write 等无状态 op 继续使用：

```text
torch.ops.xllm_ops.*
```

设备实现通过 PyTorch DispatchKey 选择。Python wrapper 只提供类型友好的薄封装和 fake
implementation，不再增加第二套 device registry。

attention 有 workspace、plan、wrapper 和 graph state，因此由 stateful backend 管理，
不强制改造成无状态 custom op。

## 10. 尚未完成的运行时状态

### 10.1 Dummy forward 与 schedule overlap

现有 C++ `AttentionMetadata::is_dummy` 表明 schedule overlap 可能提交 dummy forward。
dummy 不是普通的 padded decode：它还涉及上一步输出、stream/event、输出 buffer 生命周期
和是否允许 replay 已捕获 graph。本轮不推测这些语义。

状态：**TODO/启动失败**。

在支持前，`model_impl=python && enable_schedule_overlap=true` 必须在 config validation 阶段
报错。不能等到第一次 dummy batch 才失败。

解除 TODO 需要先补充独立设计并完成：

1. dummy step 是否调用 backend `prepare_step()`；
2. dummy 是否执行 KV write/plan；
3. decode graph 是否 capture/replay dummy；
4. 上一步输出如何进入下一步 persistent input；
5. graph stream、scheduler stream 和 output clone 的 event 顺序；
6. eager/graph、TP=1/TP=2 的连续 overlap 回归。

### 10.2 Speculative decoding/MTP

C++ 继续拥有 speculative decoding/MTP 的 draft、verification、acceptance 和 cache correction
流程。Python 侧最终只执行 C++ 已经编排好的 target/draft model step，但当前
`StepInputsView` 还没有冻结以下 typed schema：

- spec draft 与 verify batch kind；
- `is_spec_verify`；
- accepted token count 的 host/device 表示；
- shifted token ids、bootstrap embedding 和 draft hidden state；
- target/draft KV cache 绑定关系；
- cache correction 与 graph persistent state；
- logits/hidden state 的返回字段。

状态：**TODO/启动失败**。

在支持前，只要 `enable_speculative_decode=true`、`num_speculative_tokens>0`、配置了 draft
model，或运行时选择 MTP/Eagle/Suffix Python model path，config validation 就必须报错。
不能忽略 speculative 字段后按普通 decode 执行。

解除 TODO 时，先定义 `SpeculativeInputsView` 和输出 schema，再分别验证 eager 与 graph。
MTP 是 speculative decoding 的一种实现，不能只为某个 MTP model 在公共 bridge 中增加
未命名 tensor 参数。

## 11. 配置语义

本轮 Python mode 接受以下 execution 配置：

| 配置 | 语义 |
|---|---|
| `model_impl=python` 或规范化后的 `py` | `PyCausalLM + PyExecutorImpl` |
| `enable_graph=false` | Python mode 必需；native graph 开关不控制 Python runner |
| `python_graph_backend=off` | `EagerRunner` |
| `python_graph_backend=cudagraphs` | decode full graph；其他 batch eager |
| `python_graph_backend=inductor` | 可选 `CompiledRunner`，需要显式注册和独立测试 |
| 其他 backend 名 | 启动失败 |
| `max_seqs_per_batch` | scheduler 上限和 decode graph 最大 bucket |
| `python_model_path` | external package import root；built-in package 正常安装 |

`torch.compile` backend 不使用“任意非 off/cudagraphs 字符串”规则。新增 compile backend
必须在 runner registry 显式注册，并定义支持的 model/device/batch kind。

Python mode 的初始门禁：

```text
enable_schedule_overlap == false
enable_speculative_decode == false
num_speculative_tokens == 0
no draft model configured
no unsupported multimodal input facet
```

flag 和 JSON 使用同一 validation function；dump JSON 必须能完整复现上述生效配置。

## 12. 目标目录结构

```text
xllm/
  core/runtime/
    py_executor_impl.h
    py_executor_impl.cpp
    py_step_inputs_view.h
    py_step_inputs_view.cpp

  models/llm/
    py_causal_lm.h
    py_causal_lm.cpp

  python/xllm/model_executor/
    executor.py
    inputs.py
    forward_context.py

    models/
      base.py
      registry.py
      qwen3.py

    layers/
      attention.py
      linear.py
      embedding.py
      layernorm.py

    attention/
      base.py
      registry.py
      flashinfer.py

    runners/
      base.py
      eager.py
      compiled.py
      cuda_graph.py

    distributed/
      parallel_state.py
      communication_op.py

    ops/
      __init__.py
```

依赖方向固定为：executor 是 composition root；model 不 import runner 或具体 attention
backend；runner 只依赖 backend public contract；设备专用 import 必须 lazy load。

## 13. 初始化、执行与析构时序

### 13.1 初始化

```text
LLMWorkerImpl::init_model()
  -> validate Python-mode configuration
  -> create PyCausalLM
       -> initialize interpreter
       -> create py_model exactly once
  -> create Executor
       -> select PyExecutorImpl
       -> checked cast PyCausalLM
       -> create ModelExecutor(py_model)
  -> load weights into py_model in place

WorkerImpl::allocate_kv_cache_storage()
  -> allocate_kv_caches()
  -> Executor::bind_kv_caches()
       -> PyExecutorImpl::bind_kv_caches()
       -> ModelExecutor.bind_kv_caches()
```

FlashInfer 只在选择 `FlashInferBackend` 时导入和创建 workspace。创建 model 时不能导入
FlashInfer 或分配 backend workspace。

### 13.2 Eager step

```text
Batch::prepare_forward_input()
  -> PyExecutorImpl::run()
       -> create StepInputsView (zero-copy)
       -> ModelExecutor.execute()
            -> EagerRunner.run()
                 -> validate metadata/capability
                 -> backend.prepare_step()
                 -> scoped ForwardContext
                 -> model.forward(ModelInputs)
       -> ModelExecutionOutput -> ModelOutput
  -> PyCausalLM::logits()
  -> C++ sampling/output processing
```

### 13.3 析构

```text
destroy PyExecutorImpl / ModelExecutor / runner / backend
  -> destroy Python references under GIL
destroy PyCausalLM / py_model
finalize interpreter only after all Python objects are gone
```

析构顺序需要 C++ unit test 和 sanitizer/重复启动测试，不依赖成员声明顺序的偶然行为。

## 14. 文件级改动清单

| 所有者 | 主要改动 |
|---|---|
| config | 规范化 `model_impl`；集中校验 executor、graph backend、overlap/spec 冲突 |
| `Executor`/`ExecutorImpl` | 增加 `bind_kv_caches()`；Python backend 的确定性选择 |
| `WorkerImpl` | KV allocation owning path 完成一次绑定 |
| `PyCausalLM` | 只保留 model/load/logits；关闭 direct forward |
| `PyExecutorImpl` | model 注入、view 构造、execute 和 output bridge |
| attention metadata | 补齐 batch counts、host view 和 `BatchKind` 一致性 |
| Python model base | 移除 runner 初始化和 plan hook |
| Python `ModelExecutor` | 组装 model/backend/runner，管理绑定和 execute |
| Python attention | layer 变薄；FlashInfer state 移入 backend |
| Python runner | eager/compiled/CUDA graph 分离，严格 batch routing |
| Python distributed | group 生命周期与 collective op 分离 |
| packaging | 安装正式 package，删除旧 import path，不留 compatibility shim |

## 15. 分阶段实施

每个 phase 都必须保持单一生产路径。允许临时保留尚未重构的内部实现，但不允许同时注册
两套可选 execution path。后一 phase 的测试通过后，立即删除被替代代码。

### Phase 0：契约与门禁

工作项：

- 冻结 `StepInputsView`、`AttentionMetadataView`、KV binding 和 output schema；
- 增加 `BatchKind` 及字段 location/capability validation；
- 增加 Python mode config validation；
- 对 overlap/dummy、speculative/MTP、unsupported multimodal 启动失败；
- 冻结 one-rank/one-interpreter、GIL 和析构顺序。

退出条件：schema unit tests、JSON/flag config tests 和所有 fail-fast tests 通过。

### Phase 1：双 adapter 与单 model 注入

工作项：

- 新增并注册 `PyExecutorImpl`；
- Python mode 强制选择该 executor；
- checked cast 并注入 `PyCausalLM.py_model_`；
- 将 runner 调用从 `PyCausalLM::forward()` 移到 `ModelExecutor.execute()`；
- `PyCausalLM::forward()` 改为明确失败；
- 保持 weight loading 和 logits 仍由同一 `PyCausalLM` model 处理。

退出条件：registry 每 worker 调用一次；`executor.model is py_model_`；sentinel weight 在
forward/logits 可见；Python mode 不创建 native graph executor。

### Phase 2：typed view 与 KV cache 一次绑定

工作项：

- 增加 `bind_kv_caches()` owning hook；
- 增加 `StepInputsView` 和完整 `AttentionMetadataView`；
- 删除 per-forward metadata dict；
- 删除 per-forward KV cache list/tuple 构造；
- 加入 host metadata，确保 plan 不发生 D2H。

退出条件：tensor storage pointer alias 测试通过；KV wrapper identity 跨 step 不变；每步
Python 对象分配量达到验收门槛；profiler 中无新增 D2H/synchronize。

### Phase 3：AttentionBackend

工作项：

- 引入 `AttentionSpec`、`AttentionGroup`、capability 和 backend registry；
- 将 FlashInfer wrapper/workspace/plan/KV write 移入 backend；
- Qwen3 使用薄 attention layer；
- 删除 `Qwen3Model.plan_attention()` 和共享 `PagedAttention` 注入；
- 固化 prefill/chunked/mixed/decode 路由。

退出条件：Qwen3 每 step `plan_count == 1`、`execute_count == num_layers`；mixed batch 走
chunked prefill；execute 内不触发 plan 或首次 workspace allocation。

### Phase 4：Runner 与 decode graph

工作项：

- 拆分 `EagerRunner`、`CompiledRunner`、`CudaGraphRunner`；
- 引入 scoped `ForwardContext`；
- graph padding/static metadata 移入 backend graph state；
- 加入最大 bucket capacity 和显存预算；
- 删除超限 eager fallback 和未验证的 prefill graph alternative path。

退出条件：decode graph parity、奇数 batch、KV 满载 padding、最大 bucket、stream ordering
和 graph memory 测试通过。

### Phase 5：Distributed

工作项：

- 拆分 `parallel_state.py` 和 `communication_op.py`；
- 根据 device/platform 选择 PyTorch distributed backend；
- linear/embedding 只调用 communication API；
- 删除 `ops/collectives.py` 中的 group 生命周期；
- group 未初始化时显式失败。

退出条件：TP=1/TP=2 使用完全相同的功能 case；重复初始化/销毁和错误 rendezvous 测试
通过。

### Phase 6：Package、第二模型与清理

工作项：

- 迁移到正式 `xllm.model_executor` package；
- 更新 wheel staging、C++ import 和启动工具；
- 删除旧 package 和 compatibility shim；
- 接入第二个结构不同的 text model 或 out-of-tree test model；
- 删除旧 runner、旧 metadata wrapper 和未接入生产链的实现。

退出条件：第二模型不修改 C++ bridge、公共 runner 或 Qwen3 实现即可接入；wheel/image
回归通过；代码搜索确认旧 path 不再被 import。

### 后续 phase：prefill graph、dummy/overlap、speculative/MTP、multimodal

这些能力分别编写补充设计和验收项，不能为了复用已有 test case 合并为一个未定义的
“advanced inputs”接口。每项 capability 只有在 eager 语义、graph 语义和 C++ owning path
都明确后才能从启动门禁中移除。

## 16. 测试与验收

### 16.1 C++ bridge

- Python/native executor 选择；
- `PyExecutorImpl` checked cast failure；
- model registry 每 worker 一次；
- `ModelExecutor.model is PyCausalLM.py_model_`；
- load/forward/logits 使用同一 object identity；
- KV cache bind exactly once；
- execute before bind、duplicate bind 和 layer mismatch 失败；
- Python object 在 interpreter finalization 前析构；
- Python exception traceback 保留。

### 16.2 Schema 与 Python unit tests

- `StepInputsView`/`AttentionMetadataView` 字段完整性和只读行为；
- device tensor storage pointer 零拷贝；
- host length view 不触发 D2H 且生命周期覆盖 plan；
- required-field/capability validation；
- `ForwardContext` 嵌套、异常恢复和未设置错误；
- fake model/backend 验证 prepare/context/model/execute 顺序；
- KV cache Python tuple 和 per-layer wrapper 跨 step identity 不变；
- Qwen3 single-group plan/execute count；
- mixed-attention multi-group plan count；
- graph state fixed address、padding 和 capacity invariant。

### 16.3 配置与未支持状态

- flag 与 JSON 得到相同 executor/config；
- `model_impl=python` 和 `py` 规范化一致；
- 未注册 `python_graph_backend` 启动失败；
- Python mode + native graph 冲突启动失败；
- `enable_schedule_overlap=true` 启动失败；
- speculative decode/MTP 配置启动失败；
- unsupported multimodal facet 不得被忽略；
- dump JSON reload 后门禁和 runner 选择不变化。

### 16.4 功能回归

- JSON-only Python executor 启动；
- TP=1/TP=2 native/Python greedy parity；
- eager prefill、chunked prefill、mixed 和 decode；
- decode full CUDA graph；
- batch `1/2/3/5/7/8/13/16/31/47/100`；
- KV cache 接近满载时的 padded decode graph；
- `max_seqs_per_batch` 对应的最大 bucket；
- 第二个结构不同的 model/out-of-tree model；
- wheel 和 CUDA image 中 built-in package import；
- 非 CUDA build import 公共 package 时不加载 FlashInfer。

TP=1 和 TP=2 必须运行同一组 case，不能用缩减的 TP2 smoke test 代替。

### 16.5 性能验收

- 每 step 一次 C++ -> Python execute call；
- metadata tensor 零拷贝；
- 无 per-step metadata dict；
- 无 per-step KV cache list/tuple 重建；
- FlashInfer 每 group 每 step 一次 plan；
- execute 内无 plan、workspace allocation 或 D2H；
- 对比重构前后每 step Python object allocation count；
- eager 和 graph 分别记录 TTFT/TPOT；
- Python decode graph TPOT 不劣于重构前基线；
- TP2 不重复创建 workspace 或 ProcessGroup；
- 最大 bucket graph 的 persistent bytes、graph pool 和峰值显存有记录且不 OOM。

## 17. 长期约束

1. Python model 不得 import runner、具体 attention backend 或 distributed group state。
2. `ModelExecutor` 必须接收已有 model，不得调用 model registry。
3. 每个 worker 的 Python model 只创建一次，weight update 不替换 model object。
4. Python mode 必须选择 `PyExecutorImpl`，不得进入 native graph executor。
5. `PyCausalLM::forward()` 不得成为可工作的替代执行路径。
6. KV cache 只通过显式 bind/rebind 生命周期进入 Python，不得每步重建 wrapper。
7. runner 不读取 backend-specific state；attention layer 不读取 raw metadata。
8. C++ 不持有 Python workspace、wrapper 或 plan state。
9. Python 不重新计算 C++ 已提供的 common metadata。
10. host plan input 不得从 device tensor 同步获得。
11. 无状态设备算子使用 PyTorch dispatcher；有状态 attention 生命周期归 backend。
12. `ForwardContext` 必须作用域恢复。
13. TP>1 group 未初始化必须失败，collective 不得静默变为 identity。
14. 新 backend/runner 必须显式注册并通过 capability validation。
15. `prepare_step()`/`prepare_graph_step()` 每 live step 调用一次，attention `execute()`
    不得 plan。
16. mixed batch 走 chunked prefill path，不得进入 decode full graph。
17. 超过 graph/scheduler 上限必须失败，不得 eager fallback。
18. dummy、speculative/MTP 等未支持状态必须由启动门禁阻止。
19. 不保留未接入生产调用链的 alternative implementation 或 package compatibility shim。
20. 本设计不承诺 Python model 与现有 C++ `CudaGraphExecutorImpl` 兼容。
