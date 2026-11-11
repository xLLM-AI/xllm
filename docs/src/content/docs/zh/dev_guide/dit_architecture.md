---
title: "DiT 架构与能力矩阵"
---

<!-- Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

本开发者指南说明当前 C++ 扩散推理的实际调用链和能力边界。部署与请求示例见
[模型使用文档](/zh/cookbook/diffusion_models/)，参数定义与默认值见
[CLI 参考](/zh/cli_reference/)。下列矩阵描述源码接入情况，不代表所有权重、
设备、batch size 和并行组合均已完成验证。

## 配置与模型初始化

| 边界 | 源码与职责 |
| --- | --- |
| 启动与配置 | [`xllm.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/xllm.cpp) 初始化配置、组装运行选项并创建 master。[`DiTConfig`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/config/dit_config.h) 管理 DiT 参数；[`ParallelConfig`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/config/parallel_config.h) 管理并行规模。 |
| Master 与 engine | [`DiTMaster`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/dit_master.cpp) 连接 engine、调度器和请求处理。[`DiTEngine::init_model()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/dit_engine.cpp) 通过已有 distributed worker manager 初始化 worker。 |
| 组件与权重 | [`DiTModelLoader`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/model_loader/dit_model_loader.cpp) 读取 `model_index.json` 和各组件配置，组件 loader 随后转移给 pipeline 加载权重。即使 pipeline 不执行某个组件，索引声明的目录也可能仍须存在。 |
| 模型构造 | [`DiTWorkerImpl::init_model()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_worker_impl.cpp) 构建 `DiTModelContext`，调用 [`create_dit_model()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/model_registry.cpp)，加载权重、创建 `DiTExecutor` 并初始化 `DiTCache`。Pipeline 注册决定实现；worker 可回退到配置中的 model ID。 |

模型和组件参数、并行组沿用现有
[`DiTModelContext`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/model_context/dit_model_context.h)。
配置中存在某个字段，不等于所选模型会使用它；需要继续追踪到 pipeline、transformer
或算子中的读取位置。

## 请求与执行调用链

```text
HTTP service → DiTRequestParams → DiTMaster → DiTDynamicBatchScheduler
  → DiTBatch / DiTForwardInputBuilder → DiTEngine → worker RPC
  → DiTWorkerImpl → DiTExecutor → DiTModel → 模型 pipeline
      条件准备 → latent/timestep 准备 → 去噪循环
        → transformer blocks → attention / 通信 / 算子
      → VAE 解码 → DiTForwardOutput
  → DiTBatch → DiTRequest::generate_output() → HTTP 回调
```

| 步骤 | 接口与边界 |
| --- | --- |
| 接收请求 | [`xllm_server.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/server/xllm_server.cpp) 注册路由。以视频为例，[`VideoGenerationServiceImpl::process_async_impl()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/api_service/video_generation_service_impl.cpp) 构造 [`DiTRequestParams`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_request_params.cpp) 并绑定回调。请求层负责输入解码和请求校验；pipeline 内部还存在模型专用检查。 |
| 入队与组批 | `DiTMaster::handle_request()` 构造 `DiTRequestState` 并入队。[`DiTDynamicBatchScheduler::prepare_batch()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/scheduler/dit_scheduler.cpp) 按生成参数及媒体/张量签名选择兼容请求。调度器允许组批，不代表具体 pipeline 支持该 batch size。 |
| 构造输入与分发 | [`DiTForwardInputBuilder`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/batch/dit_forward_input_builder.cpp) 组装 `DiTModelParams`，保留命名张量及其参数。`DiTEngine::step()` 向 worker 分发原生 [`DiTForwardInput`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_forward_params.h)。输入 RPC 转换由 [`dit_forward_params.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_forward_params.cpp) 中的 `DiTForwardInput::to_proto()` / `from_proto()` 完成；[`worker_model_codec_dit.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/worker_model_codec_dit.cpp) 负责 worker 输出转换。 |
| 执行 | `DiTWorkerImpl::step()` 将输入移到设备，调用 [`DiTExecutor::forward()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_executor.cpp)。[`DiTModelImpl`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/model/dit_model.h) 委托给已注册 pipeline。Worker 同步设备执行，仅由 driver 返回结果；当前结果通路假设 DP = 1。 |
| 完成请求 | [`DiTBatch::process_forward_output()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/batch/dit_batch.cpp) 将结果分配给请求。调度器的异步响应处理器调用 [`DiTRequest::generate_output()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_request.cpp) 编码媒体，再触发 service 回调。 |

对图像/视频 pipeline，一次 worker forward 通常执行完整去噪循环和解码。
调度器不会在每个 diffusion timestep 重新选取请求。调度和响应阶段的取消检查，
也不代表支持在去噪循环内部中断；不能直接套用 LLM 逐 token 调度的理解。

## 状态归属与复用边界

| 状态 | 所有者与生命周期 |
| --- | --- |
| 请求输入、参数、回调与输出 | `DiTRequest` / [`DiTRequestState`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_request_state.h) 持有，生命周期随请求。媒体、张量及其参数沿用 [`DiTMediaSources` / `DiTTensorSources`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_input_sources.h) 表示。 |
| Batch 与 forward input | 调度器将请求选入 `DiTBatch`，builder 构造本次执行的 `DiTForwardInput`。Pipeline 在 forward 内管理局部 latent、timestep 和中间张量。 |
| 模型、executor 与权重 | `DiTWorkerImpl` 拥有模型和 executor；executor 引用模型。已注册的 pipeline 模块持有加载后的组件，可跨请求复用。 |
| 并行组 | 已有 parallel-state 初始化流程创建进程组，经 `DiTModelContext` 中的 `ParallelArgs` 传入。模型保存组指针，不需要另建 rank 映射。 |
| DiT 近似缓存 | [`DiTCache`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/dit_cache/dit_cache.h) 是进程级单例，按 `use_cfg` 选择两个独立 policy 实例。[`DiTCacheMixin`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/utils/dit_cache_mixin.h) 本身无状态。必须一起检查 context/reset 与 transformer 回调；缓存状态不属于每个请求。 |
| SP padding 元数据 | [`SequenceParallelMixin`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/utils/dit_parallel_mixin.h) 用 inline static map 记录 padding，在 `sequence_parallel_forward()` 入口清空。Qwen-Image 使用另一个进程级单例 [`SequenceParallelPadManager`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/utils/sequence_parallel_pad_manager.h)。两种表示的 padding 状态均不归属于单个请求。 |

新增请求数据优先复用 builder/RPC codec；新增 pipeline 复用 model registry 和组件
loader；张量分布复用并行线性层和 collective。复用 mixin 前，应核对调用处的张量
布局和状态约定。

## 模型能力矩阵

“已接入”表示链接源码中存在执行路径；“未接入”表示该实现没有连接相应能力，
不能通过全局开关或相关模型的支持自动获得。并行列指 diffusion transformer，
不包含文本编码器的并行能力。

| Pipeline 家族 / 源码 | 条件输入 | Transformer TP / SP | 通用 DiT Cache 回调 | Laser / 稀疏 Attention | VAE 路径 |
| --- | --- | --- | --- | --- | --- |
| [Flux](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux.h)、[Fill](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux_fill.h)、[Control](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux_control.h) | Pipeline 内文本编码器 | [Flux transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_flux.h) 未显式接入 DiT TP/SP | 共享 step/双流/单流回调；Flux 设置每次生成的 context，Fill/Control 未设置 | 未接入 | Pipeline 内图像 VAE |
| [Flux2](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux2.h) | Pipeline 内文本编码器 | [Transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_flux2.h) 已接入 TP，未接入 SP | 已接入 step 和 block helper | 未接入 | Pipeline 内图像 VAE |
| [Qwen-Image Edit Plus](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_qwenimage_edit_plus.h) | Pipeline 内 Qwen2.5-VL / 外部 embedding | [Transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_qwen_image.h) 已接入 SP，未显式接入 DiT TP | 已接入 step 和双流 helper | 未接入 | 按宽度并行的 VAE；pipeline 读取 `dit_enable_vae_tiling` |
| [JoyImage Edit Plus](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_joyimage_edit_plus.h) | Pipeline 内 Qwen3-VL / 外部 embedding | [Transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_joyimage_edit_plus.h) 已接入 SP，未显式接入 DiT TP | 已接入 step 和双流 helper | 未接入 | 复用 Wan VAE，支持宽度并行 |
| [Wan I2V](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_wan_i2v.h) | Pipeline 内 UMT5 / 外部 embedding | [Transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_wan.h) 已接入 TP、SP | 存在 step/单流回调；pipeline 未设置每次生成的 context | NPU Laser 用于 Q/K 等长路径；稀疏分支优先于 dense/Laser | 按宽度并行的视频 VAE |
| [MiniMax-H3](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_minimax_h3.h) | 外部 H3 embedding 和 token tags；不执行原始文本编码 | [Transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_minimax_h3.h) 已接入 TP、SP | 已接入 step 和 packed 单流回调；支持 None、TaylorSeer 和 FBCache | 未接入；使用 packed dense attention | 视频 VAE 支持空间 tile 分发；音频 VAE 在 TP driver 解码 |

其他已注册路径包括 [LongCat-Image](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_longcat_image.h)、
[LongCat-Image Edit](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_longcat_image_edit.h)、
[LongCat AudioDiT](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_longcat_audiodit.h)
和 [COLA DLM](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_cola_dlm.h)。
其 transformer 未接入通用 DiT Cache mixin 或 Laser 开关。COLA 是文本扩散路径，
其 KV cache 与 DiT 去噪步骤间的近似复用属于不同机制。模型已注册不等于具备表中
图像/视频模型的能力。

### 配置作用范围与约束

| 配置 | 真实作用范围与约束 |
| --- | --- |
| `tp_size`、`sp_size`、`cfg_size` | [`DiTMapping`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/parallel_state/dit_mapping.cpp) 检查 `TP × SP × CFG × DP = world_size`；CFG 为 1 或 2。当前 DiT 结果收集假设 DP = 1，且模型必须实际使用相应组。JoyImage/Wan 使用 `CFGParallelMixin`；Qwen-Image 有自己的 CFG 分支。 |
| `vae_size`、`text_encoder_tp_size` | 在同一个 world 上建立子组，不额外乘入 world size；组大小须整除 world size。所选 VAE/编码器还必须实际使用该组。H3 不执行文本编码器。 |
| `dit_sp_communication_overlap` | 作用于模型/attention 调用处，不是新的调度模式。H3 用它控制 SP 通信；启用 overlap 不会合并独立的 Q/K/V collective。 |
| `dit_cache_policy` 及缓存参数 | Worker 初始化会创建所选 policy，即使模型没有缓存回调。获得加速还需要回调和对应 context。Step 复用可跳过 transformer 主体，block 复用包围单个 block 调用。单流 helper 不提供 `encoder_hidden_states`；只有请求 context 显式选择 `single_stream` 时，FBCache 才会复用 packed 主干。既有双流/单流混合调用方仍保持直通行为。Fill/Control/Wan 缺少每次生成的 context 设置，不能视为已完整接入依赖该信息的 policy。选择 policy 不代表已发生缓存命中。 |
| `dit_laser_attention_enabled` | 由 Wan attention 的 NPU dense 分支读取，Q/K 序列等长时选择 Laser，不等长时选择 `npu_fusion_attention`；稀疏分支优先。选择条件是序列长度，不是 self/cross-attention 的身份。[Laser wrapper](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/kernels/npu/xllm_ops/laser_attention.cpp) 接受 BNSD 张量，转为 FP16，并 padding 序列和 head 维度；接口不接受 H3 packed 序列边界。 |
| `dit_sparse_attention_*` | 接入 Wan 稀疏 attention 路径，不是 H3 或表中其他 transformer 的通用开关。 |
| `dit_enable_vae_tiling` | 由 Qwen-Image Edit Plus pipeline 读取；不控制 H3 专用 tiling，且与 `vae_size` 含义不同。 |

### MiniMax-H3 的具体边界

当前 H3 pipeline 要求 batch size 为 1，每个 prompt 生成一个视频。条件输入契约见
[H3 使用文档](/zh/cookbook/diffusion_models/minimax_h3/minimax_h3/)。
`prepare_conditioning()` 在去噪前完成已有特征的投影/refiner 和 rotary 数据准备，
不会对原始 prompt 执行分词或文本编码。

H3 默认执行 dense 推理。TaylorSeer 和 FBCache 需要显式选择，并针对实际任务验证视频和音频质量。每次生成的 context 会重置缓存历史；命中缓存时，依赖当前 timestep 的输出归一化和媒体投影仍执行。H3 FBCache 明确拒绝滚动加载；TP1 关闭滚动加载后仍须有足够显存容纳完整常驻权重及工作区，关闭该开关不保证模型能够装下。H3 不支持 FBCacheTaylorSeer 组合策略。

H3 通过一个输出宽度为 `3 × inner_dim` 的 `ColumnParallelLinear` 融合 QKV
投影，loader 接受分开的 Q/K/V 权重前缀。SP 在 attention 前仍分别
交换 Q、K、V，计算后再反向交换输出。NPU attention 使用带有效段/padding 边界的
packed TND 数据。启用 SP 时，切分前的总序列长度补齐到 `lcm(SP, 64)` 的倍数；
head 数必须能被 TP × SP 整除，pipeline
还检查视频/音频投影的整除约束。替换 attention 内核必须保留 packed 边界、padding
排除、dtype 和布局语义。

[视频 VAE](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/autoencoders/autoencoder_kl_minimax_h3_video.h)
已有按空间 tile 向 `dit_vae_group_` 分发、gather 和重叠混合的实现。组大小为 1，
或 tile 数少于 rank 数时，执行本地 tile 计算。这与 `VaeParallelMixin` 的宽度切分
不同。`vae_size=1` 时，H3 在 TP driver 本地解码；仅增加 SP 不会分布 VAE 解码。
最终 RPC 结果只由 worker driver 返回。
