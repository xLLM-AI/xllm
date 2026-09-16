# Release xllm 0.11.0

## **Major Features and Improvements**

### Highlights

- **DeepSeek-V4 end-to-end hardening on NPU**: PD disaggregation with prefill context parallelism, FlashComm1 sequence parallelism with schedule overlap, host/device prefix cache spanning SWA + C4 + C128 units, DSpark decoding, and EPLB integrated with the fused MoE / MC2 path. MLU gained its own DeepSeek-V4 attention, MoE, PD transfer, and decode-path optimizations.
- **DeepSeek-V3.2 and GLM-5.2** move onto the Python model stack: expert-parallel (EP) support for DeepSeek-V3.2, DeepSeek-V3.2 MTP under ACL graph, and GLM-5.2 EPLv2 with shared-expert overlap, on top of existing W8A8 PyTorch adaptation, DSA top-k sharing, cache elision, and Mooncake PD.
- **Qwen3.5 / Qwen3.6 / Qwen3.8** round out with a CUDA Python model implementation, optional FIA (fused-infer-attention) decode, Qwen3.8-27B chat templates, context-parallel sequence sharding, NPU Python ACL-graph data-parallel decode, and a new Qwen3-VL-dense Python backend with ViT + deepstack + mRoPE — alongside the existing MegaMoE kernel, MegaChunkGDN fused operator, FlashComm1+MMRS fusion, and heterogeneous PD disaggregation.
- **Speculative decoding grows an adaptive layer**: adaptive speculative decoding for DFlash and DSpark on NPU, extended to graph mode and per-sequence validate pruning, plus JSON-object constrained decoding integrated with MTP verification.
- **KV cache gets a new sharding and offload story**: layerwise split KV cache sharding (shard persistent KV by static layer owner so PD cache ids stay aligned across prefill/decode/spec-verify), asynchronous host KV cache transfers, Mooncake Store as a host-cache backend, and host KV offload wired through PD and MTP.
- **Context parallelism lands on MLU**: foundational decode context-parallel (DCP) support with dedicated attention-merge and decode-context kernels for MLU DeepSeek-V2/GLM-5.2.
- **Multimodal / diffusion**: Flux2 text encoder, DiT, and VAE on NPU; Wan2.2 pipeline with distilled I2V, laser attention, RainFusion sparse attention, fused RoPE/norm operators, and single-NPU rolling weight load; QwenImageEditPlus, joy-image-edit-plus, and DFlash block-diffusion speculative decoding.
- **Multi-hardware breadth**: MUSA (Moore Threads) gains a graph executor, FP8/MoE/sampling kernels, Qwen3.5 dense-model layers, standalone builds, and a dedicated runtime-integration refactor; DCU adds DeepSeek-V2/V3 FP8 W8A8, MiniMax-M2.7 Channel FP8, MiMo-MTP, Qwen3.5, Flux image generation, Mooncake PD, and PD-OOC.
- **Developer experience**: the Python executor tree is now linted and formatted with ruff + codespell, and platform-specific typing/dispatch code was consolidated into a single torch-based module.
- A large wave of **MTP / PD / SWA correctness fixes** stabilizes speculative decoding and disaggregated serving under sustained, high-concurrency load — see [Bugfix](#bugfix) below.

### Model Support

#### NPU
- Support DeepSeek-V4 PD disaggregation, prefill context parallelism, FlashComm1 sequence parallelism with schedule overlap, and host/device prefix cache (SWA + C4 + C128).
- Support DeepSeek-V4 DSpark decoding and integrate DeepSeek-V4 EPLB with NPU fused MoE and MC2.
- Support DeepSeek-V3.2 and GLM-5.2 W8A8 PyTorch adaptation, GLM-5.2 DSA top-k sharing, cache elision, and Mooncake PD; add expert-parallel (EP) support for the DeepSeek-V3.2 Python model and DeepSeek-V3.2 Python MTP under ACL graph; enable GLM-5.2 EPLv2 and shared-expert overlap.
- Support KIMI-K25 W4A8 with ACL graph.
- Support Qwen3.5 / Qwen3.6 with MegaMoE kernel, MegaChunkGDN fused operator, FlashComm1 + MMRS fusion, gated-delta layers, and heterogeneous PD disaggregation; add a Qwen3.5 Python CUDA-parity model implementation, optional FIA decode, context-parallel sequence sharding, and NPU Python ACL-graph data-parallel decode for dense Qwen3.
- Support Qwen3.8-27B chat templates and add Qwen3-VL-dense to the Python model backend with ViT, deepstack, and mRoPE.
- Support Ascend950 attention, paged KV cache, TileLang, and causal convolution.
- Support Flux2 text encoder, DiT, and VAE, and W8A8 dynamic quantization for QwenImageEdit / Wan2.2 DiT models.
- Support Wan2.2 pipeline with distill I2V, laser attention, RainFusion sparse attention, fused RoPE / norm operators, and single-NPU rolling weight load.
- Support QwenImageEditPlus, joy-image-edit-plus, and DFlash block-diffusion speculative decoding.

#### CUDA
- Support RWKV-7-World model.
- Support Cola-DLM model.
- Support MiMo-MTP model and DeepSeek / Qwen3.5 Triton kernels.
- Support embedded-python model executor for Qwen3.
- Add a Qwen3.5 Python model CUDA implementation; the DeepSeek-V3.2 EP work also lands CUDA-side MoE kernel plumbing.

#### MLU
- Support DeepSeek-V4 attention layers, MoE, selected-MoE DP path, and MTP; add DeepSeek-V4 PD transfer and optimize DeepSeek-V4 MLU decode paths.
- Support Qwen3.5 gated-delta layers, kernels, Triton JIT, MTP, and prefill context parallelism for GLM-5.2.
- Support linear prefix cache and host KV cache transfer primitives; add foundational decode context-parallel (DCP) support with attention-merge and decode-context kernels for DeepSeek-V2 and GLM-5.2.

#### DCU
- Support DeepSeek-V2, DeepSeek-V3 FP8 W8A8, MiniMax-M2.7 Channel FP8, MiMo-MTP, and Qwen3.5.
- Support Flux image generation, Mooncake disaggregated PD, and PD-OOC.

#### MUSA
- Support Moore Threads MUSA platform, including graph executor, FP8 / MoE / sampling kernels, and Qwen3.5 dense-model layers.
- Support standalone MUSA builds, and refactor MUSA runtime integration and layer code style for maintainability.

### Feature

#### Speculative Decoding
- Add adaptive speculative decoding for DFlash and DSpark on NPU, extend it to graph mode, and add per-sequence validate pruning.
- Add JSON-object constrained decoding integrated with MTP draft/verify state.
- Add speculative-decoding per-token latency metrics and MTP support across NPU / MLU / DCU / CUDA.
- Move non-worker speculative/MTP files from `runtime/` into `framework/speculative/`, and extract `AuxHiddenCapture` for spec-draft hidden states.
- Improve speculative decoding by overlapping MTP graph updates, eliminating validate-to-draft bubbles, skipping greedy token broadcasts, and optimizing GLM MTP scheduler overlap.

#### KV Cache & Offload
- Add layerwise split KV cache sharding: shard persistent KV by static layer owner with one shared scratch layer so PD cache ids stay aligned across prefill, decode, and spec-verify; reject incompatible `layerwise_split_size` configs at startup.
- Add asynchronous host KV cache transfers and Mooncake Store support for the host cache.
- Support host KV offload for PD and MTP.
- Add a linear-state prefix cache subsystem (hash primitives, block manager, scheduler plumbing, restore, and capacity estimation) with VLM linear-state prefix cache support.
- Add in-batch prefix cache, PD-aware / DP-aware graph warmup, and device prefix cache in PD-disaggregated mode for DSV4 and Qwen3.5.

#### Parallelism
- Add foundational decode context-parallel (DCP) support for MLU and context-parallel sequence sharding for the Qwen3 Python model.
- Support DeepSeek-V4 prefill context parallelism in PD disaggregation, and DeepSeek-V4 FlashComm1 + PCP + schedule-overlap.
- Add DiT SP+TP parallelism, CFG parallelism, VAE parallelism, and configurable QwenImageEdit VAE size.

#### Platform, Config & Serving
- Add embedded-python model executor with NPU ACL-graph backend, TP, and ProcessGroupHCCL, plus separated platform backends; simplify Python model platform support and consolidate platform typing into a single torch-based module.
- Add auto-tuning xLLM server configuration, an enhanced command-line interface, and an experimental unified launch method for online and offline services.
- Add CLI-over-JSON gflags precedence, JSON config import/export, and config-struct-based flag initialization.
- Add EPLB expert rebalancing with reliable runtime lifecycle, load aggregation, and configurable placement policies; integrate DeepSeek-V4 EPLB with NPU fused MoE and MC2, and enable GLM-5.2 EPLv2 with shared-expert overlap.
- Support `trace_id` as `x-request-id`, request-ID propagation through inference paths, and asynchronous verbose request-trace logging.
- Support `include_stop_str_in_output` and OpenAI-style integer-array prompts for completions.
- Manage request rate-limit slots via RAII lifetime.
- Add multimodal processor cache, custom headers for the multimodal downloader, and VLM embedding support in offline inference.
- Add RL deep-sleep for co-located training and pause/resume for fully async RL.
- Add Python linting and formatting with ruff + codespell across the Python executor tree.

#### Performance
- Optimize NPU execution with GammaAddRmsNorm / fused LayerNorm integration, HCCL AIV small-tensor communication, cached device-side scalars, and H2D sync-bubble elimination.
- Optimize Qwen3.5 causal-conv1d, MegaChunkGDN, prefill projection, and MoE all-reduce overhead; adapt Qwen W8A8 quantization for FlashComm.
- Optimize GLM MTP scheduler overlap and DeepSeek-V4 MLU decode paths.
- Support CANN aclnn operators for ATB layers, ACL-graph decode double buffering, and event-driven recommendation scheduling.
- Add MaCa and additional platform-compatibility layers, and promote `xllm_atb_layers` to main.

### Bugfix

#### Speculative Decoding
- Fix MTP correctness under asynchronous execution, including cross-TP-rank state divergence, TPOT latency accounting, DP synchronization, overlap input preparation, and acceptance-rate regressions.
- Fix DeepSeek-V4 MTP hidden-state flow, schedule-overlap, and multi-device MTP input handling.
- Restructure MTP step prelaunch scheduling and fix `glm_moe_dsa` draft prelaunch.
- Cover ACL MTP decode graph buckets in graph warmup — a bucket ladder gated to MLU left NPU ACL executors capturing graphs mid-serving for shapes like DSV4 + DSpark at `dp_size=4`, contributing to an OOM cascade under high load; NPU now shares the same bucket-coverage helper as the executor.
- Preserve raw hidden states across NPU MTP drafts and keep the true `num_accepted_tokens` for the Qwen3.5 GDN spec-verify checkpoint.
- Skip metadata rebuild for prelaunched MTP drafts, retain no-sync inputs until validation completes, and pass `block_size` into `build_expanded_spec_verify_graph_input` on the chunked-prefill path.
- Enable speculative decode under DP4 when some DP ranks are idle, and enable draft-engine `n_layers` reduction on NPU for KV-cache estimation — the draft engine previously sized its KV cache off the full model's layer count instead of `num_nextn_predict_layers`, costing roughly 80% of the intended KV blocks on NPU.
- Support DFlash2 for Qwen3.8-series.

#### PD Disaggregation
- Fix a Prefill-Decode disaggregation schedule bug, align `pd_link_cluster` with kv-split owners, and fix kv-split chunked-prefill alignment.
- Fix prefix-cache propagation across PD, prefix-cached block skipping during KV transfer, and KV-cache completion guards.
- Fix Mooncake PD SSM/conv state transfer for hybrid MTP models by remapping state block ids using the checkpoint stride, and fix PD+MTP correctness for Qwen3.5-35B-A3B (linear-state slot mapping, speculative-wrapper restore skip, and schedule-overlap stopping check).
- Honor Mooncake Store transport selection, and fix PD decode latency stats after a prefetch timeout.

#### Memory, KV Cache & Capacity
- Align DSV4 SWA capacity and prefix cache with C128 units, correct SWA sizing and balance DP cache allocation, bound DSV4 SWA allocation during host-cache restore, and retain all SWA blocks covering the decode window.
- Fix two memory-capacity defects for hybrid models with MTP + graph mode and reject prompts exceeding decode KV capacity.
- Build the attention mask for the chunked-prefill stage, avoid linear-state H2D copies during graph capture, and skip the canonical validity mask for placeholder-only linear state.

#### Graph Capture & Execution
- Fix graph-capture issues including padded decode CUDA-graph metadata, MLU graph linear-state padding, and graph-prepare stream waits.
- Fix A5 TileLang and mRoPE support, after reverting an earlier attempt that regressed correctness.
- Fix MLU attention multi-device setup and the Qwen3.5 VLM SmoothQuant MLU vision path.
- Disable dense FIA warmup capture that was producing garbled decode output, and fall back uneven Qwen3.5 DP graph steps to eager execution.
- Fix Qwen3.5 DP empty-shard crashes, causal-conv decode, W8A8 weight loading, and quant weight loading on MLU.
- Move `torch_npu` calls from `models/` into `kernels_npu/` to keep the architectural boundary clean.

#### Multimodal
- Fix multimodal data races in parallel batch building, DiT image precision during resize, and Qwen2.5-VL M-RoPE handling.

#### Service, API & Reliability
- Fix service and API reliability including empty tool-call omission, OpenAI named tool choice, streaming, `ignore_eos` behavior, and health-report responses.
- Add missing Anthropic reasoning request fields, and always close the GLM-4.7 streaming tool-call root JSON object.
- Fix a bug from mixed usage of the common / ATB model namespace registries, and fix control-reaches-end in profile start/stop.
- Guard a `size_t` underflow in request log statistics when a cancelled request produced zero tokens (previously logged as `18446744073709551615`).
- Fix the executor test match pattern after ACL-graph support landed, and remove a redundant/useless test (from the layerwise-split follow-up work).

#### Build & CI
- Fix NPU Python runtime device pinning and initialization for standalone C++ tests, and multi-node worker address selection.
- Support standalone MUSA builds, and bump `xllm_atb_layers` to include the merged layerwise-split KV cache sharding and its follow-up bugfix.
- Fix build and CI issues around CUDA 13, recursive submodule checks, Mooncake dependencies, and `bdist_wheel` packaging.


# Release xllm 0.10.0

## **Major Features and Improvements**

### Model Support

#### NPU
- Support DeepSeek-V4 model.
- Support KIMI-K25 model.
- Support MiniMax-M2.7 model.
- Support JoyAI-LLM-Flash model.
- Support QwenImageEditPlus model.
- Support Qwen3-VL video inference.

#### CUDA
- Support Qwen3-MoE model.
- Support MiMo-7B-Base model.
- Support LongCat-AudioDiT model.

#### MLU
- Support DeepSeek-V4 model.
- Support Qwen3.5 model.
- Support OxygenVLM model.
- Support Flux model.

### Feature
- Support CANN 9.0.0 toolkit and torch_npu 2.9.0 for NPU devices.
- Support DCU backend.
- Support Torch 2.10.0 + CUDA 13.0 builds.
- Update the MLU container to version 26.04.
- Support xLLM service Anthropic PD protocol and improve Anthropic tool-call compatibility.
- Support online profiling.
- Support Python-based startup for online and offline services.
- Support importing and exporting xLLM server startup flags through JSON config files.
- Support cached token usage in responses and `best_of_n` in normal and disaggregated PD modes.
- Support pause and resume for fully async RL.
- Support offline speculative decoding and graph mode in offline inference.
- Support multi-priority scheduling for PD disaggregation and prefix-cache-aware PD chunk budgeting.
- Support namespace prefixes and optional authentication for etcd keys.
- Add typed brpc handlers for API service endpoints and improve xLLM server startup routing.
- Add encoder cache and request-transfer parallelization for multimodal models.
- Support prefix cache for multimodal models and embedding interfaces for generate VLM models.
- Add REC enhancements, including OneRec XAttention on NPU, REC XAttention for Qwen3-MoE on CUDA, constrained top-k sampling, beam-search `num_return_sequences`, extended item information, logprobs, and multi-item outputs.
- Add Wan2.2 text encoder, scheduler, TP parallelism, VAE, and `/v1/video/generation`.
- Improve KV cache internals with hybrid attention block manager, separated transfer logic, llmdatadist tensor registration, and centralized NPU allocation paths.
- Add persistent Triton NPU runtime binary cache and improve xllm_ops incremental build and CI cache behavior.
- Improve CUDA shared-memory tensor handling and add CUDA block-copy kernel support.
- Add more manual model loaders and support parsing dtype fields from model config.

### Bugfix
- Fix DeepSeek-V3.2 graph failures, prefix-cache starvation, and MLA option propagation in speculative paths.
- Fix Qwen3.5 / Qwen3.6 decode reshape, TileLang gating, causal-conv1d tiling with padded batches, and repeated weight adjustment.
- Fix NPU int4 MoE / torch communication initialization, prompt/context length limits, xattention accuracy, and offline TP initialization when selecting NPU kernel backend.
- Fix multimodal build breaks, header paths, encoder-cache linking, QwenImageEdit precision, and preprocessing accuracy.
- Fix MLU layer tests, MLU build issues, Mooncake PD paths, and local bind address handling.
- Fix CUDA and ILU build errors, chunked-prefill CUDA failures, and pipeline LongCat Image Edit compile failures.
- Fix service and API reliability issues including HTTP content types, console script metadata, brpc callback blocking, request cancellation/lifecycle shutdown, and model name extraction.
- Fix REC XAttention fallback, REC tokenizer forwarding, OneRec input copies, ND format preservation, and beam helper exposure for non-NPU builds.
- Fix build, CI, submodule cleanup, and release packaging issues around Triton NPU runtime assets and custom xllm math operator install paths.


# Release xllm 0.9.0

## **Major Features and Improvements**

### Model Support

#### NPU
- Support GLM-5 model.
- Support GLM4.7-Flash model.
- Support Qwen3-next model.
- Support OneRec model.
- Support Qwen3.5/Qwen3.5-MoE model.
#### CUDA
- Support LongCat-Image model.
- Support LongCat-Image-Edit model.
#### MLU
- Support DeepSeek-V3.2 W4A8 MoE model.
- Support GLM-5 W8A8 model.
#### ILU
- Support Qwen3-8B model.
- Support Qwen3-30B-MoE model.

### Feature
- Adapt NPU builds to CANN 8.5 and PyTorch 2.7.1.
- Support graph mode for the LLM part of VLM models on NPU devices.
- Support context parallelism for NPU DeepSeek-V3.2 / GLM-5.
- Support DeepSeek-V3.2 prefill sequence parallel on MLU devices.
- Support rolling weight loading and loading model weights with varied prefixes.
- Support dynamic and scalable multi-model serving.
- Support bidirectional remote-host to local-device KV cache transfer and batch offload.
- Support Qwen3 xattention on NPU devices.
- Support prefix cache for DeepSeek-V3.2.
- Support chunked prefill on CUDA devices.
- Support embedding interface for all generate LLM models.
- Support Anthropic Messages API.
- Support the new `v1/sample` interface.
- Support a single xLLM instance connecting to multiple xLLM services.
- Support startup progress bar, worker health check, and unified request statistics logging.
- Optimize Qwen3 MoE performance on NPU devices.
- Add CUDA Graph Executor and piecewise prefill graph.
- Support KV cache quantization on MLU devices.
- Add VMM-based allocators to reuse graph buffers and physical memory.
- Improve FP8 GEMM, fused RMSNorm, fused MoE, xattention, and activation kernel performance.

### Bugfix
- Support the new `compressed-tensors` FP8 config and fix Qwen2 prompt length.
- Fix Qwen3 MoE VL parameter settings on MLU devices.
- Fix Qwen VL issues on MLU devices and Qwen2.5 chunked-prefill accuracy on NPU.
- Fix DeepSeek tool-call, prefix-cache, DP/MTP, and PD-disagg related issues.
- Fix GLM-4.7 streaming function call issues and GLM detector stability issues.
- Fix graph mode, schedule overlap, KV cache, and REC multi-round stability issues.
- Fix multiple compile, link, env setup, and worker lifecycle issues.


# Release xllm 0.8.0

## **Major Features and Improvements**

### Model Support

#### NPU
- Support DeepSeek-v3.2 model.
- Support GLM4.7 model.
- Support GLM4.6Vmodel.
- Support GME-Qwen2-VL model.
- Support FluxControl model.
#### CUDA
- Support Qwen2/3 Dense model.
#### MLU
- Support DeepSeek-v3.2 model.
- Support Qwen2_5_vl/Qwen3_vl/Qwen3_vl_moe model.
#### ILU
- Support Qwen3-0.6B model.

### Feature
- Implement chunked prefill and prefix cache for Qwen3 MoE.
- Support GLM-4.6V model.
- Add wrappers for ATB and ACLNN fused operators.
- Optimize prefetch from kv cache store.
- Support Qwen2-VL & GME-Qwen2-VL model on npu device.
- Fix hang issue when enable schedule overlap.
- Add GLM-4.7 detector implementation and update tool call parser.
- Adapt hierarchy block manager for disagg PD.
- Support deepseek-v3.2-Exp for npu.
- Support acl_graph for qwen3/qwen3_moe.
- Support prefix cache for deepseek-v3/r1 models.
- Support disagg PD for MTP.
- Add mooncake kv cache transfer.
- Add GLM-4.7 support to reasoning detector registry.
- Support nd-to-nz continuous memory copy.
- Support RPC-based link/unlink for PD disaggregation.
- Support IntraLayerAddNorm, aclgraph, etc for DeepSeek V3.2.
- Add activation, norm and rope ops for cuda device.
- Support fused norm for Qwen3 and DeepSeek for cuda device.
- Build deepseek v2 decoder layer and related model files for mlu device.
- Support qwen2_5_vl/qwen3_vl/qwen3_vl_moe on mlu device.
- Add moe all2all kernels and deep ep layer on mlu device.
- Support deepseek mtp on mlu device.
- Support graph executor on mlu device.
- Support dp+ep moe and all2all computation on mlu device.
- Support parallelized shared experts in fused moe on mlu device.
- Support qwen3 0.6B model on iluvatar device.
- Add rec proto,service and utils for rec framework
- Support C api for llm inference.
- Add constrained decoding for generative recommendation.
- Add rec scheduler master and engine for rec framework.
- Add rec_type and onerec batch input builder for rec framework.
- Add onerec worker impl for rec framework.
- Add qwen3/LlmRec support in rec framework.

### Bugfix
- Resolve core dump of stream chat completion request when backend is VLM.
- Resolve duplicate content in multi-turn tool call conversations.
- Fix core dump issue triggered by client disconnection.
- Fix the memory leak issue in the completions interface.
- Fix wrong positions of validate input when enable MTP.
- Resolve kv_cache_num mismatch in ChunkedPrefill due to H2D block copy.
- Fix the missing index shape in the allocate kv cache transfer.
- Fix MiMo-VL weights loading crash on NPU device.
- Fix inaccurate metrics issue when enabling schedule overlap.
- Fix potential out-of-range and block leaks during deallocate in D2H copy.
- Fix allocation failure in HierarchyBlockManagerPool::allocate.
- Fix deepseek accuracy issues with prefix cache enabled.
- Resolve Deepseek execution failure caused by invalid input.
- Fix DeepSeek failing to run when enabling DP.
- Fix the rate_limit bug for stream and non-stream request in PD disagg and refactor some callback logics.
- Correct attn mask when prefix cache and MTP are both enabled in deepseek.
- Correct precision loss when enabling prefixcache with disagg pd.
- Fix incorrect async implementation in rerank interface.
- Fix acl_graph_executor not handling q_cu_seq_lens parameter for deepseekv3.2.
- Fix precision issue when enabling MTP in PD disaggregation mode.
- Fix mrope calculation in the multimodal situation.
- Fix core dump of large beam width.


# Release xllm 0.7.0

## **Major Features and Improvements**

### Model Support

- Support GLM-4.5.
- Support Qwen3-Embedding.
- Support Qwen3-VL.
- Support FluxFill.

### Feature
- Support MLU backend, currently supports Qwen3 series models.
- Support dynamic disaggregated PD, with dynamic switching between P and D phases based on strategy.
- Support multi-stream parallel overlap optimization.
- Support beam-search capability in generative models.
- Support virtual memory continuous kv-cache capability.
- Support ACL graph executor.
- Support unified online-offline co-location scheduling in disaggregated PD scenarios.
- Support PrefillOnly Scheduler.
- Support v1/rerank model service interface.
- Support communication between devices via shared memory instead of RPC on a single machine.
- Support function call.
- Support reasoning output in chat interface.
- Support top-k+add fusion in the router component of MoE models.
- Support offline inference for LLM, VLM, and Embedding models.
- Optimized certain runtime performance.

### Bugfix
- Skip cancelled requests when processing stream output.
- Resolve segmentation fault during qwen3 quantized inference.
- Fix the alignment of monitoring metrics format for Prometheus.
- Clear outdated tensors to save memory when loading model weights.
- Fix attention mask to support long sequence requests.
- Fix bugs caused by enabling scheduler overlap.

# Release xllm 0.6.0

## **Major Features and Improvements**

### Model Support

- Support DeepSeek-V3/R1.
- Support DeepSeek-R1-Distill-Qwen.
- Support Kimi-k2.
- Support Llama2/3.
- Support Qwen2/2.5/QwQ.
- Support Qwen3/Qwen3-MoE.
- Support MiniCPM-V.
- Support MiMo-VL.
- Support Qwen2.5-VL .

### Feature

- Support KV cache store.
- Support Expert Parallelism Load Balance.
- Support multi-priority on/offline scheduler.
- Support latency-aware scheduler.
- Support serving early stop.
- Optimize ppmatmul kernel.
- Support image url input for VLM.
- Support disaggregated prefill and decoding.
- Support large-scale EP parallelism.
- Support Hash-based PrefixCache matching.
- Support Multi-Token Prediction for DeepSeek.
- Support asynchronous scheduling, allowing the scheduling and computational pipeline to execute in parallel.
- Support EP, DP, TP model parallel.
- Support multiple process and multiple nodes.

### Docs

- Add getting started docs.
- Add features docs.