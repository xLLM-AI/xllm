---
title: "Service Startup Parameters"
sidebar:
  order: 100
---

xLLM uses gflags to manage service startup parameters. `--model <PATH>` is the only required flag. When `--config_json_file` is used, values in the JSON file override command-line flag values. The tables below are grouped by the Config classes in `/xllm/core/framework/config`, with one Config per section. The `ConfigJsonUtils` section contains the common JSON config-file flags.

> **Device selection**: xLLM no longer provides `--devices` / `--device_id` / `--draft_devices`. The available devices are determined by the visible-device mask environment variables (`ASCEND_RT_VISIBLE_DEVICES` for NPU, `CUDA_VISIBLE_DEVICES` for NVIDIA, `MLU_VISIBLE_DEVICES` for Cambricon, `HIP_VISIBLE_DEVICES` for DCU, `MUSA_VISIBLE_DEVICES` for Moore Threads). Each service process selects one runtime logical device from its visible devices according to its global `node_rank`; visible-device subsetting and reordering are resolved by the hardware runtime. The draft model always shares the selected device with the target model.

## ConfigJsonUtils

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `config_json_file` | `string` | `""` | Path to a JSON config file. Values in the file override command-line flag values. |
| `enable_dump_config_json` | `bool` | `false` | Whether to dump the resolved startup config as JSON. |
| `dump_config_json_file` | `string` | `"xllm_config.json"` | Path to write the resolved startup config as JSON. Used only when `enable_dump_config_json=true`. |

## ServiceConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `host` | `string` | `""` | Host name or IP for the bRPC server. |
| `port` | `int32` | `8010` | Port for the bRPC server. |
| `rpc_idle_timeout_s` | `int32` | `-1` | Close the connection when there are no read/write operations during the last `rpc_idle_timeout_s` seconds. `-1` waits indefinitely. |
| `rpc_channel_timeout_ms` | `int32` | `-1` | Maximum bRPC Channel duration in milliseconds. `-1` waits indefinitely. |
| `max_reconnect_count` | `int32` | `40` | Maximum number of reconnect attempts from a worker to a server. |
| `num_threads` | `int32` | `8` | Number of threads used to process requests. |
| `max_concurrent_requests` | `int32` | `200` | Maximum number of concurrent requests the xLLM instance can handle. Set to `0` for no limit. |
| `num_request_handling_threads` | `int32` | `4` | Number of threads for handling input requests. |
| `num_response_handling_threads` | `int32` | `4` | Number of threads for handling responses. |
| `health_check_interval_ms` | `int32` | `3000` | Worker health-check interval in milliseconds. |

## ModelConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_id` | `string` | `""` | Hugging Face model name, not a path. |
| `model` | `string` | `""` | Hugging Face model name or model path. |
| `backend` | `string` | `""` | Backend model type: `llm` for text-only models, `vlm` for multimodal models, or `dit` for diffusion models. |
| `task` | `string` | `"generate"` | Model task, for example `generate`, `embed`, or `mm_embed`. |
| `limit_image_per_prompt` | `int32` | `8` | Maximum number of images per prompt. Only applies to multimodal models. |
| `max_encoder_cache_size` | `int64` | `0` | Maximum GPU/NPU memory size in MB for the encoder cache per worker. `0` disables the encoder cache. |
| `reasoning_parser` | `string` | `""` | Reasoning parser, for example `auto`, `glm45`, `glm47`, `glm5`, `qwen3`, `qwen35`, or `deepseek-r1`. |
| `tool_call_parser` | `string` | `""` | Tool-call parser, for example `auto`, `qwen25`, `qwen3`, `qwen35`, `qwen3_coder`, `kimi_k2`, `deepseekv3`, `deepseekv32`, `deepseekv4`, `glm45`, `glm47`, or `glm5`. |
| `enable_qwen3_reranker` | `bool` | `false` | Whether to enable the Qwen3 reranker. |
| `flashinfer_workspace_buffer_size` | `int32` | `134217728` | Reserved FlashInfer workspace buffer for intermediate attention results in split-k attention. Default is 128 MiB. |
| `enable_return_mm_full_embeddings` | `bool` | `false` | Whether VLM models return ViT embeddings and sequence embeddings. |
| `mm_download_headers` | `string` | `""` | Service-level default HTTP headers for multimodal downloads, as a JSON object. Per-request headers take precedence. Example: `{"Authorization":"Bearer xxx"}`. |
| `use_audio_in_video` | `bool` | `false` | Whether to decode both audio and video when the input is a video. |
| `use_cpp_chat_template` | `bool` | `true` | Use native C++ chat templates for supported models, for example `deepseek_v32` and `deepseek_v4`. Set to `false` to fall back to Jinja for debugging. |

## LoadConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_manual_loader` | `bool` | `false` | Pin decoder layer weights to host memory and use async H2D transfer. Required by `enable_rolling_load`; also implied by `enable_xtensor`. |
| `enable_rolling_load` | `bool` | `false` | Enable rolling weight load: keep only N decoder layer weight slots in HBM and stream-load each layer just in time. Requires `enable_manual_loader=true`. NPU only. |
| `rolling_load_num_cached_layers` | `int32` | `2` | Number of decoder layer weight slots to keep in HBM when `enable_rolling_load=true`. |
| `rolling_load_num_rolling_slots` | `int32` | `-1` | Number of rolling slots used by decoder rolling load. Fixed slots are `rolling_load_num_cached_layers - rolling_load_num_rolling_slots`. `-1` means auto, `min(2, preload_count)`. Must be in `[-1, rolling_load_num_cached_layers]`. |
| `enable_prefetch_weight` | `bool` | `false` | Whether to enable weight prefetching. Only applies to Qwen3-dense models. The default gateup weight prefetch ratio is 40%; adjust with `PREFETCH_COEFFOCIENT`. |

## KVCacheConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `block_size` | `int32` | `128` | Number of slots per KV Cache block. |
| `max_cache_size` | `int64` | `0` | Maximum GPU memory size for KV Cache. `0` means calculated from available memory. |
| `max_memory_utilization` | `double` | `0.8` | Fraction of GPU memory used for model inference, including model weights and KV Cache. |
| `kv_cache_dtype` | `string` | `"auto"` | KV Cache dtype for quantization. `auto` aligns with model dtype and disables quantization. `int8` enables INT8 quantization and is only supported on the MLU backend. |
| `indexer_cache_dtype` | `string` | `"auto"` | Indexer cache dtype for models with an indexer cache. Supported values are `auto` and `int8`. `auto` aligns with model dtype and disables indexer cache quantization. `int8` enables INT8 indexer cache quantization. |
| `enable_prefix_cache` | `bool` | `true` | Whether to enable prefix cache in the block manager. See [Prefix Cache](/en/features/prefix_cache/). |
| `enable_in_batch_prefix_cache` | `bool` | `false` | Whether to cache admitted prefill full blocks into the prefix cache so that later requests in the same batch can share them. |
| `max_linear_state_cache_slots` | `int64` | `0` | Maximum number of active linear-attention state cache slots. `0` derives an automatic capacity from the available KV Cache budget. |
| `xxh3_128bits_seed` | `uint32` | `1024` | Default XXH3 128-bit hash seed. |
| `enable_xtensor` | `bool` | `false` | Whether to enable XTensor for model weights with the physical page pool. |
| `phy_page_granularity_size` | `int64` | `2097152` | Granularity size of one physical page in bytes, default 2 MiB, for continuous KV Cache. |

## KVCacheStoreConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `prefetch_timeout` | `uint32` | `0` | Timeout for prefetching from KV Cache Store. |
| `prefetch_batch_size` | `uint32` | `2` | Copy batch size for prefetching from KV Cache Store. |
| `layers_wise_copy_batchs` | `uint32` | `4` | Number of batches for layer-wise H2D copy. |
| `host_blocks_factor` | `double` | `0.0` | Host block factor, for example `host block num = host_blocks_factor * hbm block num`. |
| `enable_kvcache_store` | `bool` | `false` | Whether to enable KV Cache Store. |
| `store_protocol` | `string` | `"tcp"` | KV Cache Store protocol, for example `tcp` or `rdma`. |
| `store_master_server_address` | `string` | `""` | Address information of the Store master service. |
| `store_metadata_server` | `string` | `""` | Address of the KV Cache Store metadata service. |
| `store_local_hostname` | `string` | `""` | Local host name of the KV Cache Store client. |
| `enable_control_h2d_block_num` | `bool` | `false` | Whether to control the number of H2D copy blocks. |

## BeamSearchConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_beam_search_kernel` | `bool` | `false` | Whether to enable the beam search kernel. |
| `beam_width` | `int32` | `1` | Beam width for beam search. |
| `enable_block_copy_kernel` | `bool` | `true` (NPU/CUDA); `false` (other backends) | Whether to use the block copy kernel on supported backends. |
| `enable_topk_sorted` | `bool` | `true` | Whether to enable sorted top-k output. |

## SchedulerConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `max_tokens_per_batch` | `int32` | `10240` | Maximum number of tokens per batch. |
| `max_seqs_per_batch` | `int32` | `1024` | Maximum number of sequences per batch. |
| `enable_schedule_overlap` | `bool` | `false` | Whether to enable schedule overlap, also known as asynchronous scheduling. See [Async Scheduling](/en/features/async_schedule/). |
| `prefill_scheduling_memory_usage_threshold` | `double` | `0.95` | Memory usage threshold during prefill scheduling. |
| `enable_chunked_prefill` | `bool` | `true` | Whether to enable chunked prefill. |
| `max_tokens_per_chunk_for_prefill` | `int32` | `-1` | Maximum number of tokens per chunk in the prefill stage. `-1` uses the default policy. |
| `chunked_match_frequency` | `int32` | `2` | Sequence prefix-cache match frequency. |
| `use_zero_evict` | `bool` | `false` | Whether to use ZeroEvictionScheduler. See [Zero Evict Scheduler](/en/features/zero_evict_scheduler/). |
| `max_decode_token_per_sequence` | `int32` | `256` | Maximum decode tokens per sequence for ZeroEvictionScheduler. |
| `priority_strategy` | `string` | `"fcfs"` | Request priority strategy, for example `fcfs`, `priority`, or `deadline`. |
| `use_mix_scheduler` | `bool` | `false` | Whether to use MixScheduler to handle prefill and decode uniformly. |
| `enable_online_preempt_offline` | `bool` | `true` | Whether online requests can preempt offline requests. |
| `aggressive_coeff` | `double` | `1.0` | Aggressive coefficient for MixScheduler urgency judgment. |
| `starve_threshold` | `double` | `1.0` | Starvation threshold coefficient for MixScheduler. |
| `enable_starve_prevent` | `bool` | `true` | Whether to enable anti-starvation behavior in MixScheduler. |

## ParallelConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `dp_size` | `int32` | `1` | Data parallel size for MLA attention. |
| `ep_size` | `int32` | `1` | Expert parallel size for MoE models. |
| `cp_size` | `int32` | `1` | Context parallel size. Backend and model support is required; `1` disables context parallelism. See [Cambricon MLU](/en/hardware/cambricon_mlu/) for MLU-specific constraints. |
| `kv_split_size` | `int32` | `1` | KV Cache split width. `0` falls back to `cp_size` (legacy). `1` disables KV split: each CP rank stores the full KV and skips the prefix AllGather. Other K values (K divides `cp_size`) shard KV across K ranks while token-CP still uses `cp_size`. |
| `tp_size` | `int64` | `1` | Tensor parallelism size. Only used for DiT models. |
| `sp_size` | `int64` | `1` | Sequence parallelism size. Only used for DiT models. |
| `cfg_size` | `int64` | `1` | Classifier-free guidance parallelism size. Only used for DiT models. |
| `vae_size` | `int64` | `1` | VAE patch parallelism size. Only used for DiT models. |
| `communication_backend` | `string` | `"hccl"` | NPU communication backend, for example `lccl` or `hccl`. Uses `hccl` when DP is enabled. |
| `enable_mm_encoder_dp` | `bool` | `false` | Whether to enable encoder data parallelism for multimodal models. |
| `enable_multi_stream_parallel` | `bool` | `false` | Whether to enable computation/communication parallelism with two streams and two micro batches in the prefill stage. See [Multi-Stream Parallelism](/en/features/multi_streams/). |
| `micro_batch_num` | `int32` | `1` | Number of micro batches used for multi-stream parallelism. |
| `enable_dp_balance` | `bool` | `false` | Whether to enable DP load balancing. If true, sequences within a single DP batch are shuffled. |

## EPLBConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_eplb` | `bool` | `false` | Whether to enable expert parallel load balance. See [EPLB](/en/features/eplb/). |
| `redundant_experts_num` | `int32` | `1` | Number of redundant experts per device. |
| `eplb_update_interval` | `int64` | `1000` | EPLB update interval. |
| `eplb_update_threshold` | `double` | `0.8` | EPLB update threshold. |
| `expert_parallel_degree` | `int32` | `0` | Expert parallel degree. |
| `rank_tablefile` | `string` | `""` | ATB HCCL rank table file. |

## DistributedConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `master_node_addr` | `string` | `"127.0.0.1:19888"` | Master address for multi-node distributed serving, for example `10.18.1.1:9999`. |
| `xtensor_master_node_addr` | `string` | `"127.0.0.1:19889"` | Master address for the XTensor distributed service, for example `10.18.1.1:9999`. |
| `nnodes` | `int32` | `1` | Number of multi-node nodes. |
| `node_rank` | `int32` | `0` | Rank of the current node. |
| `etcd_addr` | `string` | `""` | etcd address used to save instance metadata. |
| `etcd_namespace` | `string` | `""` | Optional etcd namespace prefix for all xLLM keys, for example `prod-a`. |
| `enable_service_routing` | `bool` | `false` | Whether to enable xLLM service routing. |
| `heart_beat_interval` | `double` | `0.5` | Heartbeat interval. |
| `etcd_ttl` | `int32` | `3` | Time to live for etcd keys. |

## DisaggPDConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_disagg_pd` | `bool` | `false` | Whether to enable disaggregated prefill and decode execution. See [P-D Separation](/en/features/disagg_pd/). |
| `enable_pd_ooc` | `bool` | `false` | Whether to enable online-offline co-location in disaggregated PD mode. |
| `disagg_pd_port` | `int32` | `7777` | Listening port for the disaggregated PD bRPC server. |
| `instance_role` | `string` | `"DEFAULT"` | Instance role, for example `DEFAULT`, `PREFILL`, `DECODE`, or `MIX`. |
| `kv_cache_transfer_type` | `string` | `"LlmDataDist"` | KV Cache transfer type, for example `LlmDataDist`, `Mooncake`, or `HCCL`. |
| `kv_cache_transfer_mode` | `string` | `"PUSH"` | KV Cache transfer mode, for example `PUSH` or `PULL`. |
| `transfer_listen_port` | `int32` | `26000` | Listening port for KV Cache Transfer. |
| `kv_push_dst_rotate` | `bool` | `false` | Rotate the destination-worker traversal order in `push_kv_blocks` per KV-split rank to spread incast across decode workers. |

## SpeculativeConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `draft_model` | `string` | `""` | Draft model path. See [MTP](/en/features/mtp/) for MTP usage. |
| `num_speculative_tokens` | `int32` | `0` | Number of speculative tokens generated per speculative decoding step. |
| `speculative_algorithm` | `string` | `"MTP"` | Speculative decoding algorithm. Supported values: `MTP`, `Eagle3`, `Suffix`, `DFlash`. |
| `speculative_suffix_cache_max_depth` | `int32` | `64` | Maximum suffix-tree depth for suffix speculative decoding. |
| `speculative_suffix_max_spec_factor` | `double` | `1.0` | Maximum suffix speculation token factor relative to match length. |
| `speculative_suffix_max_spec_offset` | `double` | `0.0` | Maximum additive token offset for suffix speculation. |
| `speculative_suffix_min_token_prob` | `double` | `0.1` | Minimum token probability used in suffix speculation. |
| `speculative_suffix_max_cached_requests` | `int32` | `-1` | Maximum number of globally cached requests for suffix speculation. `-1` means unlimited; `0` disables it. |
| `speculative_suffix_use_tree_spec` | `bool` | `false` | Whether to use tree-based suffix speculation instead of path speculation. |
| `enable_opt_validate_probs` | `bool` | `false` | Whether validation uses selected-only `draft_probs [B,S]` directly. If false, selected-only cache values are restored to dense `[B,S,V]`. |
| `enable_atb_spec_kernel` | `bool` | `false` | Whether to use the ATB speculative kernel. |

## ProfileConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_profile_step_time` | `bool` | `false` | Whether to enable step-time profiling. |
| `enable_profile_token_budget` | `bool` | `false` | Whether to enable token-budget profiling. |
| `enable_latency_aware_schedule` | `bool` | `false` | Whether to use predicted latency for latency-aware scheduling. |
| `profile_max_prompt_length` | `int32` | `2048` | Maximum prompt length used for profiling. |
| `max_global_ttft_ms` | `int32` | `std::numeric_limits<int32_t>::max()` | Global TTFT threshold in milliseconds. |
| `max_global_tpot_ms` | `int32` | `std::numeric_limits<int32_t>::max()` | Global TPOT threshold in milliseconds. |
| `enable_profile_kv_blocks` | `bool` | `true` | Whether to generate KV Cache blocks for profiling. |
| `disable_ttft_profiling` | `bool` | `false` | Whether to disable TTFT profiling. |
| `enable_forward_interruption` | `bool` | `false` | Whether to enable forward interruption. |
| `enable_online_profile` | `bool` | `false` | Whether to enable the online timeline profiling endpoints (`/start_profile` and `/stop_profile`). CUDA only for now; pair with launching the server under `nsys --capture-range=cudaProfilerApi`. |
| `profile_backend` | `string` | `"torch"` | Online profiling backend. `torch` records CPU+CUDA activities in-process and writes a Chrome trace on `/stop_profile`, no external profiler needed. `cuda` only toggles the CUDA profiler capture range and requires launching under `nsys --capture-range=cudaProfilerApi`. |
| `profile_dir` | `string` | `""` | Directory the `torch` online profiling backend writes timeline traces to. Empty means the current working directory. |

## ExecutionConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_graph` | `bool` | `false` | Whether to enable graph execution for the decode phase to reduce kernel-launch overhead and device idle time. Supports CUDA Graph, ACL Graph (NPU), MLU Graph, and DCU Graph. See [Graph Mode](/en/features/graph_mode/). |
| `enable_graph_double_buffer` | `bool` | `true` | Whether to enable double-buffered ACL graph persistent params and graph instances for NPU schedule-overlap decode. |
| `enable_graph_mode_decode_no_padding` | `bool` | `false` | Whether decode graph capture uses the actual `num_tokens` instead of a padded shape. |
| `enable_prefill_piecewise_graph` | `bool` | `false` | Whether to enable piecewise CUDA graph for the prefill phase. Attention runs in eager mode while other operations are captured in CUDA graphs. |
| `enable_graph_vmm_pool` | `bool` | `true` | Whether to enable a VMM-backed CUDA graph memory pool for multi-shape graph memory reuse. |
| `max_tokens_for_graph_mode` | `int32` | `2048` | Maximum number of tokens for graph execution. `0` means no limit. |
| `acl_graph_decode_batch_size_limit` | `int32` | `16` | Decode batch size threshold for ACL graph on NPU. When the actual decode batch size exceeds this value, ACL graph decode falls back to eager mode to avoid OOM. |
| `enable_shm` | `bool` | `false` | Whether to enable shared memory for model execution. |
| `use_contiguous_input_buffer` | `bool` | `true` | Whether to use a contiguous device input buffer for model execution. |
| `input_shm_size` | `uint64` | `1024` | Input shared-memory size. Default is 1GB. |
| `output_shm_size` | `uint64` | `128` | Output shared-memory size. Default is 128MB. |
| `random_seed` | `int32` | `-1` | Random seed for the random number generator. `-1` means no fixed seed. |

## KernelConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_customize_mla_kernel` | `bool` | `false` | Whether to enable the customized MLA kernel. |
| `npu_kernel_backend` | `string` | `"AUTO"` | NPU kernel backend. Supported values: `AUTO`, `ATB`, `TORCH`. |
| `enable_intralayer_addnorm` | `bool` | `false` | Whether to enable fused intralayer addnorm ops. |
| `enable_fused_mc2` | `int32` | `-1` | Fused MC2 mode for NPU. `-1` uses the auto default, `0` disables fused MC2, positive values enable dense matmul-allreduce, `1` uses DispatchFFNCombine for MoE, `2` uses DispatchGmmCombineDecode for MoE. |
| `enable_interlayer_addnorm` | `bool` | `false` | Whether to enable fused interlayer addnorm ops. |
| `enable_split_rmsnorm_rope` | `bool` | `false` | Whether to enable fused split rmsnorm rope ops. |
| `enable_aclnn_matmul` | `bool` | `false` | Whether to enable the ACLNN matmul backend for supported NPU ATB layers. |
| `enable_aclnn_swiglu` | `bool` | `false` | Whether to enable the ACLNN SwiGLU backend for supported NPU ATB layers. |

## DiTConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `max_requests_per_batch` | `int32` | `1` | Maximum number of requests per batch. |
| `dit_cache_policy` | `string` | `"TaylorSeer"` | DiT cache policy, for example `None`, `FBCache`, `TaylorSeer`, `FBCacheTaylorSeer`, or `ResidualCache`. |
| `dit_cache_warmup_steps` | `int64` | `0` | Number of warmup steps. |
| `dit_cache_n_derivatives` | `int64` | `3` | Number of derivatives to use in TaylorSeer. |
| `dit_cache_skip_interval_steps` | `int64` | `3` | Interval steps to skip for derivative calculation. |
| `dit_cache_residual_diff_threshold` | `double` | `0.09` | Residual difference threshold for cache reuse. |
| `dit_cache_start_steps` | `int64` | `5` | Number of steps to skip at the start. |
| `dit_cache_end_steps` | `int64` | `5` | Number of steps to skip at the end. |
| `dit_cache_start_blocks` | `int64` | `5` | Number of blocks to skip at the start. |
| `dit_cache_end_blocks` | `int64` | `5` | Number of blocks to skip at the end. |
| `dit_sp_communication_overlap` | `bool` | `true` | Whether to overlap communication and computation for sequence parallelism. |
| `dit_debug_print` | `bool` | `false` | Whether to print debug information for DiT models. |
| `dit_generation_image_area_max` | `int64` | `0` | Maximum allowed image area (`width * height`) for image generation requests. `0` means no limit. |
| `dit_enable_vae_tiling` | `bool` | `false` | Whether to enable VAE tiling. Currently only supported for `qwen-image-edit-plus`. |
| `dit_vae_image_size` | `int64` | `1048576` | Qwen-Image-Edit-Plus VAE image size used to calculate dimensions. |
| `dit_sparse_attention_enabled` | `bool` | `false` | Whether to enable block-wise sparse attention / RainFusion for WAN. |
| `dit_sparse_attention_sparsity` | `double` | `0.5` | Sparse attention sparsity ratio in `[0.0, 1.0)`. `0.0` means dense attention, `0.5` drops 50 percent of blocks. |
| `dit_sparse_attention_pool_size` | `int64` | `128` | Sparse attention pooling window size for block-wise mask generation. |
| `dit_sparse_attention_sparse_start_step` | `int64` | `0` | Sparse attention step index to start sparse attention. Steps before this use dense attention. |
| `dit_sparse_attention_version` | `string` | `"rain_fusion"` | Sparse attention version: `rain_fusion` (frame-pairing + `aclnnRainFusionAttention`) or `sparse_attention` (block-decompose + `aclnnBlockSparseAttention`). |
| `dit_sparse_attention_mask_refresh_steps` | `int64` | `1` | Recompute the block sparse mask every N diffusion steps. `1` means every step, higher values reuse the mask longer. |

## RecConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_rec_fast_sampler` | `bool` | `true` | Whether to enable the RecSampler fast sampling path for Rec pipelines. |
| `enable_rec_prefill_only` | `bool` | `false` | Whether to enable Rec prefill-only mode without decoder self-attention block allocation. |
| `enable_xattention_one_stage` | `bool` | `false` | Whether to force xattention one-stage decode for Rec multi-round mode. |
| `max_decode_rounds` | `int32` | `0` | Maximum number of decode rounds for multi-step decoding. `0` means disabled. |
| `enable_constrained_decoding` | `bool` | `false` | Whether to enable constrained decoding with predefined rules for output format or structure. |
| `output_rec_logprobs` | `bool` | `false` | Whether to output Rec multi-round token-aligned logprobs. Missing per-token logprobs are filled with the final beam logprob. |
| `enable_convert_tokens_to_item` | `bool` | `false` | Whether to convert token IDs to item IDs in REC/OneRec responses. |
| `enable_output_sku_logprobs` | `bool` | `false` | Whether to output REC/OneRec token-aligned logprobs tensors. |
| `enable_extended_item_info` | `bool` | `false` | Whether to parse and output REC extended item info tensors. |
| `each_conversion_threshold` | `int32` | `50` | Maximum number of items emitted for each REC token triplet. |
| `total_conversion_threshold` | `int32` | `1000` | Maximum total number of items emitted in one REC response. |
| `request_queue_size` | `int32` | `100000` | Scheduler request queue size. |
| `rec_worker_max_concurrency` | `uint32` | `1` | Concurrency for Rec worker parallel execution. Values less than or equal to `1` disable concurrent Rec workers. |
