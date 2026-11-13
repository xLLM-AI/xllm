---
title: "DiT Architecture and Capability Matrix"
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

This developer guide maps the current C++ diffusion inference path and capability boundaries.
For deployment and request examples, use the [model recipes](/en/cookbook/diffusion_models/).
For flag definitions and defaults, use the [CLI reference](/en/cli_reference/).
The matrices below describe source integration, not a guarantee that every
checkpoint, device, batch size, or parallel configuration has been validated.

## Configuration and model initialization

| Boundary | Source and responsibility |
| --- | --- |
| Launch and configuration | [`xllm.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/xllm.cpp) initializes configuration, assembles runtime options, and creates the master. [`DiTConfig`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/config/dit_config.h) owns DiT flags; [`ParallelConfig`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/config/parallel_config.h) owns parallel sizes. |
| Master and engine | [`DiTMaster`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/dit_master.cpp) connects the engine, scheduler, and request handling. [`DiTEngine::init_model()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/dit_engine.cpp) initializes workers through the existing distributed worker manager. |
| Components and weights | [`DiTModelLoader`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/model_loader/dit_model_loader.cpp) reads `model_index.json` and component configurations. Component loaders are transferred to the pipeline for weight loading. A component directory declared in the index may be required even if the pipeline does not execute that component. |
| Model construction | [`DiTWorkerImpl::init_model()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_worker_impl.cpp) builds `DiTModelContext`, calls [`create_dit_model()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/model_registry.cpp), loads weights, creates `DiTExecutor`, and initializes `DiTCache`. Pipeline registrations select the implementation; the worker can fall back to the configured model ID. |

Keep model/component arguments and parallel groups in the existing
[`DiTModelContext`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/model_context/dit_model_context.h).
A configuration field being present does not establish that the selected model
consumes it. Follow its reader to the pipeline, transformer, or kernel.

## Request and execution flow

```text
HTTP service → DiTRequestParams → DiTMaster → DiTDynamicBatchScheduler
  → DiTBatch / DiTForwardInputBuilder → DiTEngine → worker RPC
  → DiTWorkerImpl → DiTExecutor → DiTModel → model pipeline
      conditioning → latents/timesteps → denoising loop
        → transformer blocks → attention / collectives / kernels
      → VAE decode → DiTForwardOutput
  → DiTBatch → DiTRequest::generate_output() → HTTP callback
```

| Step | Interface and boundary |
| --- | --- |
| Accept | Routes are registered in [`xllm_server.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/server/xllm_server.cpp). For video, [`VideoGenerationServiceImpl::process_async_impl()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/api_service/video_generation_service_impl.cpp) constructs [`DiTRequestParams`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_request_params.cpp) and binds the response callback. The request layer decodes inputs and performs request validation; model-specific checks also exist inside pipelines. |
| Enqueue and batch | `DiTMaster::handle_request()` constructs `DiTRequestState` and enqueues a request. [`DiTDynamicBatchScheduler::prepare_batch()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/scheduler/dit_scheduler.cpp) groups compatible generation parameters and media/tensor signatures. Scheduler compatibility does not override a pipeline's batch-size limits. |
| Build and dispatch | [`DiTForwardInputBuilder`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/batch/dit_forward_input_builder.cpp) assembles `DiTModelParams`, including named tensors and their parameters. `DiTEngine::step()` sends native [`DiTForwardInput`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_forward_params.h) to workers. Input RPC conversion uses `DiTForwardInput::to_proto()` / `from_proto()` in [`dit_forward_params.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_forward_params.cpp); [`worker_model_codec_dit.cpp`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/distributed_runtime/worker_model_codec_dit.cpp) handles the worker's output conversion. |
| Execute | `DiTWorkerImpl::step()` moves inputs to the device and calls [`DiTExecutor::forward()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/runtime/dit_executor.cpp). [`DiTModelImpl`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/model/dit_model.h) delegates to the registered pipeline. The worker synchronizes device execution and returns output only from the driver. The current result path assumes DP = 1. |
| Complete | [`DiTBatch::process_forward_output()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/batch/dit_batch.cpp) distributes results to requests. The scheduler's asynchronous response processor calls [`DiTRequest::generate_output()`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_request.cpp), which encodes media, before invoking the service callback. |

For the image/video pipelines, one worker forward normally runs the entire
denoising loop and decoding. The scheduler does not select new work at each
diffusion timestep. Request cancellation checks around scheduling and response
delivery do not imply cancellation inside that loop. Do not apply the LLM
token-by-token scheduling model to this path.

## State ownership and reuse

| State | Owner and lifetime |
| --- | --- |
| Request inputs, settings, callback, output | `DiTRequest` / [`DiTRequestState`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_request_state.h) for the request lifetime. Preserve media and tensor parameters through the existing [`DiTMediaSources` / `DiTTensorSources`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/request/dit_input_sources.h) representations. |
| Batch and forward input | The scheduler selects requests into `DiTBatch`; the builder creates one execution's `DiTForwardInput`. The pipeline owns its local latents, timesteps, and intermediate tensors during forward. |
| Model, executor, weights | `DiTWorkerImpl` owns the model and executor; the executor references the model. Registered pipeline modules own loaded components across requests. |
| Parallel groups | Existing parallel-state initialization supplies process groups through `ParallelArgs` in `DiTModelContext`. Models retain group pointers rather than creating another rank mapping. |
| Approximate DiT cache | [`DiTCache`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/dit_cache/dit_cache.h) is a process-wide singleton, with separate policy instances selected by `use_cfg`. [`DiTCacheMixin`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/utils/dit_cache_mixin.h) is stateless. Context/reset and transformer callbacks must be checked together; cache state is not owned by each request. |
| SP padding metadata | [`SequenceParallelMixin`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/utils/dit_parallel_mixin.h) records padding in an inline static map, cleared at `sequence_parallel_forward()` entry. Qwen-Image uses a separate process-wide singleton, [`SequenceParallelPadManager`](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/utils/sequence_parallel_pad_manager.h). Neither representation provides request ownership of padding state. |

Reuse the input builder/RPC codecs for new request data, the model registry and
component loader for new pipelines, and existing parallel linear layers and
collectives for tensor distribution. Before reusing a mixin, check its tensor
layout and state contract at the intended call site.

## Model capability matrix

“Connected” means an execution path exists in the linked implementation.
“Not connected” means that implementation does not wire the capability; a
global flag or a related model's support does not enable it automatically.
The parallel column concerns the diffusion transformer, not its text encoder.

| Pipeline family / source | Conditioning | Transformer TP / SP | Generic DiT Cache callbacks | Laser / sparse attention | VAE path |
| --- | --- | --- | --- | --- | --- |
| [Flux](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux.h), [Fill](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux_fill.h), [Control](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux_control.h) | Pipeline text encoders | No explicit DiT TP/SP wiring in [Flux transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_flux.h) | Shared step/dual-stream/single-stream callbacks. Flux sets per-generation context; Fill/Control do not | Not connected | Pipeline-local image VAE |
| [Flux2](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_flux2.h) | Pipeline text encoder | TP connected; SP not connected in [transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_flux2.h) | Step and block helpers connected | Not connected | Pipeline-local image VAE |
| [Qwen-Image Edit Plus](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_qwenimage_edit_plus.h) | Pipeline Qwen2.5-VL / supplied embeddings | SP connected in [transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_qwen_image.h); no explicit DiT TP wiring | Step and dual-stream helpers connected | Not connected | Width-parallel VAE; pipeline reads `dit_enable_vae_tiling` |
| [JoyImage Edit Plus](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_joyimage_edit_plus.h) | Pipeline Qwen3-VL / supplied embeddings | SP connected in [transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_joyimage_edit_plus.h); no explicit DiT TP wiring | Step and dual-stream helpers connected | Not connected | Reuses Wan VAE with width parallelism |
| [Wan I2V](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_wan_i2v.h) | Pipeline UMT5 / supplied embeddings | TP and SP connected in [transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_wan.h) | Step/single-stream callbacks exist; pipeline does not set per-generation context | NPU Laser for equal-length Q/K; sparse path selected before dense/Laser | Width-parallel video VAE |
| [MiniMax-H3](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_minimax_h3.h) | Supplied H3 embeddings and token tags; no raw-text encoder execution | TP and SP connected in [transformer](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/transformers/transformer_minimax_h3.h) | Step and packed single-stream callbacks connected; supports None, TaylorSeer, and FBCache | Opt-in NPU Laser; packed dense attention by default; no sparse path | Video VAE with spatial tile distribution; audio VAE decoded on TP drivers |

Other registered paths include [LongCat-Image](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_longcat_image.h),
[LongCat-Image Edit](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_longcat_image_edit.h),
[LongCat AudioDiT](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_longcat_audiodit.h),
and [COLA DLM](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/pipelines/pipeline_cola_dlm.h).
Their transformers do not wire the generic DiT Cache mixin or the Laser flag.
COLA is a text diffusion path; its KV cache is a different mechanism from
approximate reuse across DiT denoising steps. Registration alone does not imply
the image/video capabilities in the table.

### Configuration scope and constraints

| Configuration | Actual scope and constraints |
| --- | --- |
| `tp_size`, `sp_size`, `cfg_size` | [`DiTMapping`](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/framework/parallel_state/dit_mapping.cpp) checks `TP × SP × CFG × DP = world_size`; CFG is 1 or 2. Current DiT result collection assumes DP = 1. The model must actually use the groups. JoyImage/Wan use `CFGParallelMixin`; Qwen-Image has its own CFG branch. |
| `vae_size`, `text_encoder_tp_size` | Overlay subgroups on the same world, not extra ranks multiplied into the world size. Group sizes must divide the world size. The selected VAE/encoder must use the group. H3 does not execute a text encoder. |
| `dit_sp_communication_overlap` | A model/attention call-site choice, not a new scheduling mode. H3 uses it for SP exchange; overlap does not fuse the separate Q/K/V collectives. |
| `dit_cache_policy` and cache parameters | Worker initialization constructs the selected policy, including for models without cache callbacks. Acceleration requires callbacks and the appropriate context. Step reuse can skip a transformer body; block reuse surrounds individual block calls. The single-stream helper omits `encoder_hidden_states`; FBCache reuses a packed body only when the request context explicitly selects `single_stream`. Legacy mixed dual-/single-stream callers retain their pass-through behavior. Fill/Control/Wan omit per-generation context setup and must not be treated as complete integrations for policies that require it. Policy selection alone is not proof of a cache hit. |
| `dit_laser_attention_enabled` | Read by Wan and H3 NPU attention. Wan selects Laser for equal-length Q/K after the sparse branch. H3 selects Laser for at least 2,048 live tokens and head dimensions up to 128, excluding SP padding first. The [Laser wrapper](https://github.com/xLLM-AI/xllm/blob/main/xllm/core/kernels/npu/xllm_ops/laser_attention.cpp) accepts BNSD tensors, preserves BF16/FP16 input dtype, and pads sequence/head dimensions. |
| `dit_sparse_attention_*` | Wired through Wan's sparse attention path. Not a generic switch for H3 or the other transformers in the table. |
| `dit_enable_vae_tiling` | Read by the Qwen-Image Edit Plus pipeline. It does not control H3's model-specific tiling and is separate from `vae_size`. |

### MiniMax-H3 boundaries

The current H3 pipeline requires batch size 1 and one video per prompt. Its
conditioning contract is documented in the [H3 recipe](/en/cookbook/diffusion_models/minimax_h3/minimax_h3/).
`prepare_conditioning()` projects/refines supplied features and prepares rotary
data once before denoising; it does not tokenize or encode a raw prompt.

H3 defaults to dense inference. TaylorSeer and FBCache require explicit policy selection and workload-specific video/audio quality validation. Per-generation context resets cache history; timestep-dependent output normalization and media projection still run on cache hits. H3 FBCache rejects rolling weight loading. TP1 without rolling requires enough memory for all resident weights and workspace; disabling rolling does not guarantee that the model fits. H3 does not support the combined FBCacheTaylorSeer policy.

H3 uses one `ColumnParallelLinear` with output width `3 × inner_dim` for fused
QKV projection; its loader accepts the separate Q/K/V checkpoint prefixes. SP still
exchanges Q, K, and V separately before attention and exchanges the output back.
On NPU, the default attention path uses packed TND data with live/padding
boundaries. Opt-in Laser excludes SP padding, uses BNSD tensors, and restores
padding before the output exchange. When SP is enabled, the total sequence
length is padded to a multiple of `lcm(SP, 64)`
before partitioning; head counts must be divisible by TP × SP, and the
pipeline also checks video/audio projection divisibility. A replacement attention
kernel must preserve packed boundaries, padding exclusion, dtype, and layout.

The [video VAE](https://github.com/xLLM-AI/xllm/blob/main/xllm/models/dit/autoencoders/autoencoder_kl_minimax_h3_video.h)
already distributes spatial tiles across `dit_vae_group_`, gathers them, and
blends overlaps. It runs tiles locally when the group has one rank or there are
fewer tiles than ranks. This differs from `VaeParallelMixin`'s width partition.
With `vae_size=1`, H3 decodes locally on TP drivers; increasing SP alone does not
distribute VAE decoding. Only the worker driver returns the final RPC result.
