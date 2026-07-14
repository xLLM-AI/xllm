---
title: "C++ Serving Framework + Python Model Execution 架构决策"
sidebar:
  order: 3
---
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

本文讨论 xLLM 为什么在现有 C++ serving framework 中增加 Python model execution，
以及为什么 Python 侧同时包含 model 和 `ModelExecutor`。本文只讨论架构选择，不描述
实现类、接口细节或迁移步骤。

## 1. Situation

### 1.1 xLLM 已有完整的 C++ serving framework

xLLM 当前的 C++ framework 已经覆盖从请求进入到结果返回的完整流程：

```text
Input Processing
    -> Continuous Batching Scheduler
    -> Worker Runtime
    -> Sampling / Speculative Decoding
    -> Output Processing
```

- **Input Processing** 负责 tokenization、chat template 和 multimodal input processing。
- **Continuous Batching Scheduler** 负责 chunked prefill、prefill/decode scheduling、
  request priority 和 schedule overlap。
- **Worker Runtime** 接收 scheduler 生成的 Batch，构造模型输入，并管理设备资源和执行
  流程。
- **Sampling / Speculative Decoding** 负责 next-token selection，以及 MTP、Eagle、
  Suffix 等 speculative decoding method 的 draft、verification 和 token acceptance。
- **Output Processing** 负责异步响应、detokenization 和 streaming output。

`KV Cache Manager`、`Distributed Runtime` 和 `Observability` 支撑整条流程：前两者负责
KV block、prefix cache、多级 KV cache、master/worker、多 rank、多节点、PD 分离和 KV
transfer，后者负责 profiling、metrics 和 device monitoring。

这套 C++ framework 是本次架构决策的基础。Python model execution 接入其中的模型计算
位置，不改变上述 serving 和 runtime 职责。

### 1.2 当前 C++ libtorch model 的主要问题

现有 C++ libtorch model 已经能够提供稳定、可优化的模型执行能力。本次引入 Python 不是
因为 C++ model 无法达到性能要求，而是因为新模型接入和长期维护越来越受以下问题限制。

#### 社区模型需要重新实现

新模型发布后，可直接参考的实现通常来自模型厂商、Hugging Face、vLLM 或 SGLang，主要
形式是 Python `nn.Module`、Python weight loader 和 Python layer API。接入 C++ libtorch
model 时，需要重新表达模型结构、TP 切分、权重映射和 attention 调用，再单独完成数值
对齐。

这不是简单的语言翻译。社区实现后续增加新 attention 变体、量化格式或 checkpoint 字段
时，C++ model 也需要人工跟进。社区 Python 实现和 xLLM C++ 实现因此成为两份需要持续
同步的模型代码。

#### Python-first 工具需要额外接入成本

xLLM 已经支持以 AOT 方式接入 Triton 和 TileLang kernel。问题在于 kernel 往往需要同时
适配不同 model size、TP size、batch/sequence shape、dtype、量化格式和硬件型号。AOT
需要提前准备这些组合，也难以在实际部署硬件上根据运行时 shape 做 autotuning。

Triton、TileLang、量化 loader 和许多新 kernel 通常先提供 Python API。C++ model 可以
继续调用已经注册好的 native op，但增加或调优 Python-first kernel 时，需要额外处理 AOT
构建、wrapper、cache、类型转换和生命周期。

Python 也更适合使用 hook、逐层 tensor dump 和 profiler 与社区参考实现做数值对齐。
C++ model 的修改通常还需要重新编译和链接，使模型迁移阶段的调试周期更长。

`torch.compile`、Dynamo、FX 和 Inductor 也是考虑因素，但优先级低于社区模型迁移、JIT
kernel、量化和调试能力。它们可以直接处理 Python model，却不能直接处理 C++
`torch::nn::Module` 定义的模型图。

## 2. Task

需要解决的问题是：**在保留现有 C++ serving framework 的前提下，减少社区 Python 模型
迁移到 xLLM 时的结构重写，同时保持生产推理所需的性能和多硬件扩展能力。**

这个任务有以下约束：

1. C++ 继续负责 Input/Output Processing、Continuous Batching Scheduler、KV Cache
   Manager、Distributed Runtime、Worker Runtime、Sampling 和 Speculative Decoding。
   MTP 等 speculative decoding method 的整体控制流程仍在 C++，Python 只执行相关模型
   的 tensor 计算。
2. Python model 需要保持接近社区 `nn.Module` 的职责，只管理模型结构、权重和模型计算，
   不能把 scheduler、KV cache manager 或请求状态复制到 Python。
3. 持久化输入、padding、attention plan 和 graph state 不能散落在 model layer 中，需要
   有独立的 model execution owner。
4. C++ 已经构造的 Batch、page table、sequence length、slot mapping 等数据不能在 Python
   重算。跨语言边界必须窄，并保持 tensor 零拷贝。
5. 目标架构需要支持不同 attention backend、collective backend 和硬件 graph runtime，
   公共 model 不能依赖 FlashInfer、CUDA graph 或其他设备专用实现。
6. Python eager 的每步开销必须受到控制。达到目标生产性能需要覆盖 decode、prefill 和
   mixed batch 的 graph execution。
7. 每个 rank 保持独立进程和 Python 解释器。C++ serving 线程不进入 Python，只有 worker
   的 model execution 路径跨越解释器边界。

## 3. Solution

xLLM 采用 **C++ Serving Framework + Python Model Execution**：C++ framework 保持现有
职责，Python 负责 model 以及与模型执行直接相关的状态和执行模式。

```text
┌─────────────────────────────────────────────────────────────┐
│ C++ Serving Framework                                       │
│ Input/Output Processing · Continuous Batching Scheduler      │
│ KV Cache Manager · Distributed Runtime · Worker Runtime      │
│ Sampling · Speculative Decoding · Observability              │
└───────────────────────────┬─────────────────────────────────┘
                            │ PyTorch tensor + metadata view
                            │ once per step
┌───────────────────────────┴─────────────────────────────────┐
│ Python Model Execution                                      │
│ Model：model structure · weights · forward · logits          │
│ ModelExecutor/Runner：persistent inputs · padding · graph    │
│ AttentionBackend：plan · execute · backend state            │
│ Distributed：TP group · collective                          │
└───────────────────────────┬─────────────────────────────────┘
                            │ torch.ops / backend API
┌───────────────────────────┴─────────────────────────────────┐
│ Device Kernels and Runtime                                  │
│ xllm_ops · Triton/TileLang · attention kernels · graph API   │
└─────────────────────────────────────────────────────────────┘
```

职责划分如下：

- **C++ framework** 负责请求、调度、Batch、KV cache、分布式 worker、speculative
  decoding、sampling 和输出处理，并在每个 step 调用一次 Python model execution。
- **Python Model** 负责模型结构、权重加载、forward 和 logits，不持有 runner、padding、
  attention plan 或 graph state。
- **Python ModelExecutor/Runner** 负责持久化输入、padding、execution mode 和 graph
  lifecycle，并调用已经创建的 model。
- **AttentionBackend** 负责 attention wrapper、workspace、每步 plan 和 backend-specific
  state。对于需要 plan 的 backend，同一 attention group 每个 step 只 plan 一次，所有
  layer 复用该 plan。
- **Python Distributed** 只负责 model 内部的 TP group 和 collective；worker/rank 管理、
  多节点协同、PD 分离和 KV transfer 仍由 C++ `Distributed Runtime` 负责。
- **Device ops** 通过 PyTorch dispatch 或具体 backend 选择硬件实现，model layer 不选择
  CUDA、NPU 或其他设备实现。

common attention metadata 仍属于 C++ InputBuilder 的输出。C++ 负责构造，Python
attention backend 只消费 metadata view，并转换为当前 backend 的 runtime state。单独
说明 attention metadata，是为了明确这条跨语言 ownership，不是把它视为独立的 C++
framework 子系统。

graph 由具体 Python runner 和 attention backend 管理。公共 model 和公共 runner 接口
不规定 CUDA graph、ACL graph 或其他设备 graph 的类型；不同硬件可以提供各自的具体
runner，而不需要共享同一套 graph 继承结构。

## 4. 依据

### 4.1 不同解法的比较

本文只比较实际与任务相关的四种解法：

| 解法 | 能解决什么 | 主要限制 |
|---|---|---|
| 继续只使用 C++ libtorch model | 保持现有执行路径和运行时依赖 | 社区 Python 模型仍需重新实现，不能解决模型迁移和长期同步成本 |
| Python model 在 `forward()` 中直接使用 runner | 只增加一个 Python model 入口，接入简单 | model 同时持有权重、persistent input、padding、attention plan 和 graph state，难以保持社区 model 的职责和结构 |
| Python model + C++ model executor | 可以复用社区 model，并复用部分 C++ executor | persistent input、padding、attention plan 和 graph state 横跨 C++/Python；新增 Python attention backend 或硬件 graph 时仍需修改 C++ executor |
| Python model + Python `ModelExecutor` | model 和 execution state 都能复用 Python 生态；runner、attention backend 和 graph 可以在 Python 内协同演进 | 需要嵌入式 Python、跨语言 metadata contract 和 Python graph 能力 |

**继续只使用 C++ libtorch model** 可以保持已经验证的 native 路径，仍适合现有模型和
不接受 Python runtime 的部署。但它没有解决本次任务的首要问题：社区模型仍要重新实现，
后续模型变化仍需维护两份代码。因此它继续存在，但不能单独承担新模型扩展。

**让 Python model 直接使用 runner** 是最短的接入路径，但把两种变化频率不同的职责放进
了同一个对象。model structure、weight 和 raw forward 应尽量贴近社区实现；persistent
buffer、padding、attention plan 和 graph 则随 backend、batch shape 和硬件变化。两者
耦合后，移植社区 model 时仍需理解 xLLM 的完整 execution state，model 也无法脱离 runner
独立测试和复用。

**Python model + C++ model executor** 能保持 model/executor 分层，也可以让 C++ 准备
persistent input 和 graph。问题在于 attention plan、workspace、padding metadata 和
graph capture 必须与 Python attention backend 保持一致：要么把 backend 私有状态暴露
给 C++，要么在 C++ 和 Python 各维护一套状态转换。后续增加 Python backend 或硬件 graph
时，修改范围仍会跨越两种语言。

**Python model + Python `ModelExecutor`** 让 model 保持接近社区 `nn.Module`，并让
execution state 与 Python runner、attention backend 和 graph 实现在同一侧演进。C++
只提供已经准备好的 Batch/tensor/metadata，并继续负责外围 serving runtime。代价是需要
维护嵌入式 Python 和跨语言 contract，但这两项成本边界明确，也可以独立验收。

因此选择第四种解法。关键原因不是“Python 比 C++ 快”，而是它同时满足社区 model 迁移、
model/executor 职责分离，以及 backend/graph 在 Python 内演进这三个任务要求。

将 model 与 `ModelExecutor` 都放在 Python，不表示 model 自己管理 graph。两者仍然分层：
model 只负责权重和计算，`ModelExecutor`/runner 负责执行状态。这与 vLLM 和 SGLang 中
Model 与 ModelRunner/ModelExecutor 分离的基本职责一致。

### 4.2 社区模型迁移是首要决策依据

社区模型实现不是只有 model class，还包括 layer 组合、权重名称、量化处理、TP 切分和
模型特有的输入规则。Python model execution 可以复用其中大部分结构和开发方式，使迁移
工作集中在 xLLM 的 layer、attention 和 weight loader contract 上。

vLLM 和 SGLang 的新模型支持文档都以 Python model class 和 ModelRunner 为扩展入口。
这不能单独证明 xLLM 必须选择 Python，但说明社区模型实现、调试方法和贡献者经验已经
围绕 Python 形成。xLLM 采用相近的 model/layer vocabulary，可以减少移植时不必要的
结构改写。

### 4.3 JIT kernel、量化和调试是第二层依据

Python model execution 可以直接调用 Triton/TileLang JIT、Python-first 量化 loader 和
社区 kernel package。关键 kernel 仍然可以由 C++、CUDA、AscendC 或其他 native 实现
提供；选择 Python model 不等于用 Python 重写 kernel。

Python 还可以直接使用 hook、tensor dump 和 profiler 与参考实现逐层对比。对新模型
移植而言，这些工具主要缩短正确性验证和性能定位周期。

`torch.compile` 是额外收益，重要性低于模型迁移、JIT、量化和调试。它适合已经验证稳定
的模型或子模块，例如部分 VLM encoder；是否使用需要按模型验证编译时间、正确性和性能，
不作为所有 Python model 的强制路径。

compile 路线的长期价值还包括编译器扩展。Dynamo backend 以及公开的 FX/ATen pass 可以
接入其他图优化或 fused-op replacement，并继续复用同一份 Python model。设计只依赖这些
公开扩展点，不把长期接口建立在变化较快的 Inductor 内部 IR 上。

AOTInductor 可以把 Python model 编译为由 C++ 加载的产物，但不作为本方案的主路径。它
需要更严格的 export 约束，kernel config 在构建时确定，也不能在部署环境中按新 shape
重新 JIT 和 autotune。它更适合 shape 范围较窄、交付环境不允许 Python runtime 的场景。

### 4.4 性能成立依赖 graph 和严格的跨语言边界

Python eager 会增加 dispatch 和对象构造开销，尤其影响单步设备计算时间较短的 decode。
decode 的执行路径和 shape 相对稳定，适合使用 full graph，通过 capture/replay 减少这部分
host 开销。

prefill、chunked prefill 和 mixed batch 的动态性更强，attention planning、sequence 组合和
执行 shape 难以统一纳入 full graph。目标架构使用 piecewise graph：将执行路径中稳定的部分
捕获为 graph pieces，并把依赖动态 metadata 的部分保留在 graph 之外。通过 full graph 与
piecewise graph 分别覆盖不同执行场景，是 Python model execution 达到生产性能的必要条件。

无论使用何种 graph，跨语言边界都需要满足：

- 每个 step 只进行一次 C++ 到 Python execute 调用；
- tensor 和 metadata tensor 零拷贝；
- KV cache tensor 等稳定对象一次绑定，不在每步重建 Python wrapper；
- C++ 已构造的数据不在 Python 重算；
- attention plan 位于 captured model 之外，并由同组 layer 复用。

进程模型也是性能条件。每个 rank 使用独立进程和解释器时，C++ request processing、
scheduler 和 KV cache 线程不会争用 Python GIL。若改成单进程内用多个线程执行多个 rank，
需要重新评估 GIL 和解释器生命周期，不能直接沿用本设计的性能判断。

### 4.5 多硬件兼容取决于 backend 边界

Python model 本身不保证多硬件兼容。兼容性来自 model、attention backend、graph runner、
collective 和 device ops 之间的边界：

1. model 不 import FlashInfer、CUDA graph 或其他设备专用 runtime；
2. attention layer 只调用当前 backend，不解释 plan/state；
3. 具体 graph runner 自己管理设备 graph，公共接口不暴露 CUDA 类型；
4. 无状态算子通过 PyTorch dispatch 选择设备实现；
5. TP communication 根据 tensor device 使用对应 collective backend。

vllm-ascend 复用 Python model，并通过 NPU runner、attention backend、设备算子和
`torch.npu.NPUGraph` 适配 Ascend，说明这种边界可以支持非 CUDA 硬件。这里引用它只为
验证架构边界，不表示 xLLM 需要复制其 runner 继承关系或 patch 方式。

### 4.6 接受的代价

选择 Python model execution 后，需要长期承担以下成本：

- 部署包含嵌入式 Python、wheel、PyTorch 和设备 runtime 的版本组合；
- metadata view 成为需要明确 ownership、生命周期和字段范围的跨语言 contract；
- C++ native model 与 Python model 在一段时间内并存；
- graph 未覆盖的场景仍会承担 Python eager 开销；
- model、runner 和 backend 的边界需要持续审查，避免执行状态重新进入 model layer。

如果主要 workload 长期无法获得 graph 覆盖、跨语言开销不能达到性能目标，或者社区模型
迁移效率没有实际改善，就需要重新评估这项架构选择。
