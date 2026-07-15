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
- Add rec proto,serivce and utils for rec framework
- Support C api for llm inference.
- Add constrained decoding for generative recommendation.
- Add rec scheduler master and engine for rec framework.
- Add rec_type and onerec batch input builder for rec framework.
- Add onerec worker impl for rec framework.
- Add qwen3/LlmRec support in rec framework.

### Bugfix
- Reslove core dump of stream chat completion request when backend is VLM.
- Resolve duplicate content in multi-turn tool call conversations.
- Fix core dump issue triggered by client disconnection.
- Fix the memory leak issue in the completions interface.
- Fix wrong positons of validate input when enable MTP.
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