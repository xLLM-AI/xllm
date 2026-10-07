---
title: "代码结构"
sidebar:
  order: 1
---

## 第一次阅读，从这些文件开始

建议按以下顺序阅读，先看列出的函数及其直接调用的代码，再深入具体平台的实现。

| 顺序 | 代码入口 | 帮你回答的问题 |
| --- | --- | --- |
| 1 | [`xllm/xllm.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/xllm.cpp)：`main()`、`run()` | 配置、master 和 HTTP 服务如何连接起来？ |
| 2 | [`chat_service_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/api_service/chat_service_impl.cpp)：`ChatServiceImpl::process_async_impl()` | 聊天消息如何转换为内部请求，并绑定输出回调？ |
| 3 | [`llm_master.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/llm_master.cpp)：`LLMMaster::handle_request()`、`run()` | 请求如何进入调度器？谁在驱动调度循环？ |
| 4 | [`continuous_scheduler.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/scheduler/continuous_scheduler.cpp)：`ContinuousSchedulerBase::step()` | 一轮批次如何被调度、执行并处理结果？ |
| 5 | [`llm_engine.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/llm_engine.cpp)：`LLMEngine::step()` | 批次如何发送到各个 worker，执行结果如何汇总？ |
| 6 | [`llm_worker_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/llm_worker_impl.cpp)：`LLMWorkerImpl::step_internal()` | 模型执行、logits 计算和 token 采样发生在哪里？ |

第一遍先沿着未启用投机解码、PD 分离和调度重叠的路径阅读。这些功能是在基本流程
上增加分支，理解主流程后再展开会更容易。

## 先认识代码中的几个概念

| 概念 | 在本页中的含义 | 源码入口 |
| --- | --- | --- |
| Token、prefill、decode | Tokenizer 将文本转换成 token ID。Prefill 处理输入提示词；decode 复用已有计算状态，继续生成输出。Chunked prefill 把提示词拆到多个调度轮次处理。 | [`tokenizer/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/tokenizer)、[`scheduler_policy.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/scheduler/scheduler_policy.cpp) |
| `Request` | 一次推理请求，包含生成参数、输出回调以及一个或多个 sequence。 | [`request.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/request.h)、[`request_state.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/request_state.h) |
| `Sequence` | 一条持续增长的 token 序列，记录生成进度、停止状态和缓存状态。同一个请求可以有多个候选序列。 | [`sequence.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/sequence.h) |
| `Batch`、`BatchGroup` | `Batch` 是一次执行步骤选中的工作集合；`BatchGroup` 汇总该步骤中各个数据并行（DP）rank 的 batch。 | [`batch.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/batch/batch.h)、[`batch_group.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/batch/batch_group.h) |
| Rank 与并行方式 | Rank 标识并行执行中的参与者。DP 将请求工作分配到不同组；张量并行（TP）把模型的张量计算拆分到多个 rank；专家并行（EP）分布部署混合专家模型中的专家。 | [`parallel_state/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/parallel_state) |
| Continuous batching（连续批处理） | 调度器每轮重新选择工作，让新请求加入、已完成请求退出。一个 batch 的成员不会在整个请求生命周期内保持固定。 | [`continuous_scheduler.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/scheduler/continuous_scheduler.h) |
| KV cache、block、prefix cache | KV cache 保存可复用的注意力状态；block manager 分配和回收缓存容量；prefix cache 让满足复用条件的请求共享已缓存的提示词前缀。 | [`kv_cache/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/kv_cache)、[`block/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/block)、[`prefix_cache/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/prefix_cache) |
| Logits 与采样 | Logits 是模型对词表中各 token 给出的分数；采样根据生成参数，从这些分数中选择下一个 token。 | [`sampling/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/sampling) |
| Master、engine、worker、executor | Master 连接请求处理与调度；engine 协调各 worker 执行批次；worker 管理设备上的执行；其中的 executor 选择模型的执行方式。 | [`distributed_runtime/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/distributed_runtime)、[`runtime/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/runtime) |

## 仓库目录导航

下面只列出用于定位代码的主要目录，并非完整文件清单。C++ 可执行程序的入口是
源码目录内的 **`xllm/xllm.cpp`**。

```text
.
├── xllm/
│   ├── xllm.cpp                           # C++ 可执行程序入口
│   ├── launch_server.py                   # xllm serve 的 Python 启动器
│   ├── server/                            # HTTP 服务生命周期与路由注册
│   ├── api_service/                       # 请求解析、校验与响应构造
│   ├── core/
│   │   ├── common/                        # 公共类型、选项、指标与辅助定义
│   │   ├── distributed_runtime/           # Master、engine、worker RPC 与协同管理
│   │   ├── scheduler/                     # 队列、调度策略与输出处理
│   │   ├── framework/                     # 推理数据结构与公共机制
│   │   │   ├── config/                    # 分类配置与命令行选项
│   │   │   ├── request/                   # 请求、序列与停止状态
│   │   │   ├── batch/                     # 批次与模型输入构造
│   │   │   ├── block/                     # 缓存分配与块管理
│   │   │   ├── kv_cache/                  # 缓存张量、布局与容量估算
│   │   │   ├── prefix_cache/              # 提示词前缀复用
│   │   │   ├── kv_cache_transfer/         # 缓存传输与外部缓存存储
│   │   │   ├── model/                     # 模型接口与输入输出类型
│   │   │   ├── model_loader/              # 模型配置与 checkpoint 加载
│   │   │   ├── tokenizer/                 # 文本与 token 转换
│   │   │   ├── chat_template/             # 聊天消息转换为模型提示词
│   │   │   ├── sampling/                  # Token 采样与约束解码
│   │   │   ├── parallel_state/            # 并行分组与通信状态
│   │   │   ├── eplb/                      # 专家并行负载均衡
│   │   │   └── xtensor/                   # 虚拟内存与模型内存管理
│   │   ├── runtime/                       # Worker 与模型 executor
│   │   ├── layers/                        # C++ 神经网络层
│   │   ├── kernels/                       # 设备算子实现与封装
│   │   ├── platform/                      # 设备、流、内存与通信接口
│   │   └── util/                          # 辅助工具
│   ├── models/                            # C++ 模型注册与实现
│   │   ├── llm/                           # 语言模型及 Python 模型桥接
│   │   ├── vlm/                           # 视觉语言模型
│   │   ├── dit/                           # 扩散模型
│   │   └── rec/                           # 生成式推荐模型
│   ├── python/                            # Python 模型执行
│   │   ├── models/                        # Python 模型定义
│   │   ├── model_executor/                # 执行协调与 eager/graph runner
│   │   ├── attention/                     # 注意力后端与元数据
│   │   ├── model_loader/                  # 权重加载与切分
│   │   ├── layers/                        # Python 神经网络层
│   │   ├── kernels_cuda/                  # CUDA 算子
│   │   ├── kernels_npu/                   # NPU 算子，含 TileLang/Triton 源码
│   │   └── distributed/                   # 模型内部的集合通信
│   ├── processors/                        # 多模态输入预处理
│   ├── parser/                            # 推理过程输出解析器
│   ├── function_call/                     # 工具调用解析器
│   ├── proto/                             # 服务与内部 RPC 协议定义
│   ├── pybind/                            # 面向用户的 Python 推理 API 与 C++ 绑定
│   ├── c_api/                             # C 集成接口
│   ├── cc_api/                            # C++ 集成接口
│   ├── compiler/                          # 算子编译与 AOT 构建工具
│   └── auto_config/                       # 按模型和硬件生成启动配置
├── examples/                              # 离线推理示例
├── tests/                                 # C++、Python 测试及设备测试
├── scripts/                               # 构建、测试、启动与开发辅助脚本
├── tools/                                 # 转换、张量比较与 profiling 工具
├── third_party/                           # 外部依赖与算子库
├── docs/                                  # 中英文文档站点
├── CMakeLists.txt                         # C++ 构建配置
└── setup.py                               # Python 打包与构建入口
```

阅读时先区分**状态、调度、执行**：`framework/` 定义请求、批次、缓存和模型接口；
`scheduler/` 决定下一轮运行哪些工作；`distributed_runtime/` 和 `runtime/` 负责
派发并执行这些工作；`models/` 或 `python/models/` 中的模型描述网络计算。

## 跟踪一次聊天请求

### 启动阶段

[`launch_server.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm/launch_server.py)
处理 `xllm serve` 启动命令。进入 C++ 可执行程序后，`main()` 初始化配置，`run()`
通过 [`create_master()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/master_factory.cpp)
创建 master。对于 `llm` 后端，`LLMMaster` 建立 engine、scheduler、tokenizer
和 request factory。`master->run()` 启动循环，提供服务的 rank 通过
[`server/xllm_server.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/server/xllm_server.cpp)
启动 `APIService`。

### 请求与响应流程

下面是文本生成的逻辑流程。它跨越队列、worker 通信和响应回调，并非一个同步调用栈。
调度器会重复执行其中的计算步骤，直到生成结束。

```text
POST /v1/chat/completions
  -> HTTP route / APIService / ChatServiceImpl
  -> LLMMaster / LLMRequestFactory
       messages -> chat template -> prompt -> token IDs -> Request
  -> ContinuousSchedulerBase / SchedulerPolicy
       choose sequences and token budgets -> BatchGroup
  -> LLMEngine
       build inputs -> dispatch to worker ranks
  -> LLMWorkerImpl / Executor
       model forward -> logits -> sampling -> token results
  -> update sequences / AsyncResponseProcessor
       token IDs -> text -> RequestOutput
  -> ChatServiceImpl callback -> JSON response or SSE stream
```

1. **接收并解析请求。** 路由在 `server/xllm_server.cpp` 中注册。
   [`api_service.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/api_service/api_service.cpp)、
   [`openai_request.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/api_service/openai_request.cpp)
   和 [`chat_request_decoder.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/api_service/chat_request_decoder.cpp)
   处理 HTTP 请求体、规范化、参数校验及内部协议转换。`ChatServiceImpl` 选择模型
   master，准备聊天消息、生成参数和输出回调。

2. **构造推理请求。** `LLMMaster::handle_request()` 把工作交给请求线程池。
   [`LLMRequestFactory`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/llm_request_factory.cpp)
   应用聊天模板，对提示词分词并校验，创建请求和序列状态。随后 master 调用
   `scheduler_->add_request()` 将请求加入调度器。

3. **选择下一轮执行的工作。** `ContinuousSchedulerBase` 管理请求队列和执行循环。
   [`scheduler_policy.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/scheduler/scheduler_policy.cpp)
   中的 `PrefillFirstPolicy`、`DecodeFirstPolicy` 和 `UnifiedPolicy` 负责组批，
   考虑 token/序列预算与缓存容量。想了解哪些请求会一起执行，应继续阅读策略文件；
   策略选择与调度循环本身是分开的。

4. **构造输入并派发。** `LLMEngine::step()` 为各 DP batch 准备输入，通过
   [`WorkerClient`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/worker_client.h)
   派发给 worker。定位 token 位置、序列长度和缓存映射等问题时，先阅读
   [`batch/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/batch)，
   尤其是 `ForwardInputBuilder`，再进入模型层。

5. **执行模型并采样。** `LLMWorkerImpl::step_internal()` 调用
   [`Executor`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/executor.cpp)，
   获取模型输出和 logits，并通过
   [`sampling/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/framework/sampling)
   选择输出 token。Executor 选择 C++ 或 Python 执行实现，具体边界见下一节。

6. **返回结果，继续生成或结束。** Engine 的执行结果用于更新序列状态；调度器
   处理批次输出和完成状态；
   [`AsyncResponseProcessor`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/scheduler/async_response_processor.cpp)
   生成解码后的 `RequestOutput` 并调用回调。聊天服务将结果组织成完整 JSON 响应，
   或 Server-Sent Events（SSE）流。配置了推理过程或工具调用解析器时，它们也会
   参与输出处理。停止条件见
   [`stopping_checker.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/stopping_checker.h)，
   资源释放则继续跟踪调度器与缓存管理器。

## C++ 与 Python 模型执行路径

两条路径共用上面的 HTTP 服务和调度流程。`--model_impl=python` 选择 Python
模型执行，而请求、调度、缓存管理、worker 协调与采样仍由 C++ 负责。

| 关注点 | C++ 模型路径 | Python 模型路径 |
| --- | --- | --- |
| 选择方式 | `model_impl` 为空，或 `--model_impl=native` | `--model_impl=python`，同时需要模型和平台支持 |
| 模型入口 | [`models/model_registry.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/model_registry.h)，可从 [`models/llm/qwen3.h`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/llm/qwen3.h) 开始阅读 | [`models/llm/py_causal_lm.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/llm/py_causal_lm.cpp)、[`python/registry.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm/python/registry.py) 与 [`python/models/qwen3.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm/python/models/qwen3.py) |
| 执行方式 | [`core/runtime/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/core/runtime) 中的基础 executor 和设备专用 executor | [`py_executor_impl.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/py_executor_impl.cpp) 桥接到 [`ModelExecutor`](https://github.com/xLLM-AI/xllm/blob/main/xllm/python/model_executor/executor.py) 及其 runner |
| 层与算子 | `core/layers/`、`core/kernels/`、`core/platform/` | `python/layers/`、`python/attention/`、`python/kernels_cuda/`、`python/kernels_npu/` |
| 权重加载 | `core/framework/model_loader/`、`core/framework/state_dict/` 及模型/层的 loader | C++ 模型桥接、`python/model_loader/` 及模型自身的加载逻辑 |

跟踪配置时，请区分三个选择：

- **服务后端**（`llm`、`vlm`、`dit`、`rec`）决定 master/engine 类型，从
  `master_factory.cpp` 开始查看。
- **模型实现**（`native` 或 `python`）决定模型执行路径，从
  `core/runtime/executor.cpp` 和
  [`model_config.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/config/model_config.cpp)
  开始查看。
- **硬件与执行模式**决定可用的设备算子和 eager/graph executor。Python 路径中，
  `python_graph_backend` 的处理见 `ModelExecutor` 与
  [`runners/`](https://github.com/xLLM-AI/xllm/tree/main/xllm/python/model_executor/runners)。
  可用组合依赖模型和平台；启用图模式不代表每一种 batch 都会走图执行。

`xllm/pybind/` 提供面向用户的 Python 推理 API，例如
[`examples/generate.py`](https://github.com/xLLM-AI/xllm/blob/main/examples/generate.py)
使用的接口；`xllm/python/` 实现模型执行。使用 Python 推理 API 本身并不意味着
选择了 Python 模型实现。

更详细的说明见
[C++ 框架与 Python 模型设计](/zh/design/cpp_framework_python_model_architecture/)、
[图模式设计](/zh/design/graph_mode_design/)以及
[`xllm/python/` 包说明](https://github.com/xLLM-AI/xllm/blob/main/xllm/python/README_zh.md)。
