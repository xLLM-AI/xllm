/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "global_flags.h"

#include <limits>

#include "brpc/reloadable_flags.h"

// NOTE:
// 1. related flags should be placed together.
// 2. when adding new flags, plz add the flag name to the appropriate
//    category in help_formatter.h so it appears in the help output.

// --- xllm service config ---

DEFINE_string(host, "", "Host name for brpc server.");

DEFINE_int32(port, 8010, "Port for brpc server.");

DEFINE_int32(
    rpc_idle_timeout_s,
    -1,
    "Connection will be closed if there is no read/write operations "
    "during the last `rpc_idle_timeout_s`. -1 means wait indefinitely.");

DEFINE_int32(rpc_channel_timeout_ms,
             -1,
             "Max duration of bRPC Channel. -1 means wait indefinitely.");

DEFINE_int32(max_reconnect_count,
             40,
             "The max count for worker try to connect to server.");

DEFINE_int32(num_threads, 8, "Number of threads to process requests.");

DEFINE_int32(max_concurrent_requests,
             200,
             "Maximum number of concurrent requests the xllm service can "
             "handle. If set to 0, there is no limit.");

// --- model config ---

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model, "", "Name or path of the huggingface model to use.");

DEFINE_string(
    backend,
    "",
    "Choose the backend model type. 'llm' for text-only, "
    "'vlm' for multimodal (text and images), 'dit' for diffusion models.");

DEFINE_string(task,
              "generate",
              "The task to use the model for(e.g. generate, embed, mm_embed).");

DEFINE_string(devices,
              "npu:0",
              "Devices to run the model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_bool(enable_customize_mla_kernel, false, "enable customize mla kernel");

// --- graph mode execution config ---

DEFINE_bool(
    enable_graph,
    false,
    "Whether to enable graph execution for decode phase. When enabled, "
    "the engine uses graph mode (CUDA Graph for GPU, ACL Graph for NPU, "
    "or MLU Graph) to optimize decode performance by reducing kernel "
    "launch overhead and device idle time.");

DEFINE_bool(enable_graph_mode_decode_no_padding,
            false,
            "Whether to enable graph execution for decode phase without "
            "padding. If true, graph will be caputured with every actual num "
            "tokens, as stride is 1.");

DEFINE_bool(enable_prefill_piecewise_graph,
            false,
            "Whether to enable piecewise CUDA graph for prefill phase. "
            "When enabled, attention operations use eager mode while other "
            "operations are captured in CUDA graphs.");

DEFINE_bool(enable_graph_vmm_pool,
            true,
            "Whether to enable VMM-backed CUDA graph memory pool for "
            "multi-shape graph memory reuse.");

DEFINE_int32(max_tokens_for_graph_mode,
             2048,
             "Maximum number of tokens for graph execution. "
             "If 0, no limit is applied.");
// --- vlm config ---

DEFINE_int32(limit_image_per_prompt,
             4,
             "Maximum number of image per prompt. Only applicable for "
             "multimodal models.");

// --- threading config ---

DEFINE_int32(num_request_handling_threads,
             4,
             "Number of threads for handling input requests.");

DEFINE_int32(num_response_handling_threads,
             4,
             "Number of threads for handling responses.");

// --- kvcache config ---

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_int64(max_cache_size,
             0,
             "Max gpu memory size for kv cache. Default is 0, which means "
             "cache size is caculated by available memory.");

DEFINE_double(max_memory_utilization,
              0.8,
              "The fraction of GPU memory to be used for model inference, "
              "including model weights and kv cache.");

DEFINE_string(
    kv_cache_dtype,
    "auto",
    "KV cache data type for quantization. \"auto\" (default): KV "
    "cache dtype aligns with model dtype (no quantization). "
    "\"int8\": Enables INT8 quantization. Only supported on MLU backend.");

// --- scheduler config ---

DEFINE_int32(max_tokens_per_batch, 10240, "Max number of tokens per batch.");

DEFINE_int32(max_seqs_per_batch, 1024, "Max number of sequences per batch.");

DEFINE_bool(enable_schedule_overlap,
            false,
            "Whether to enable schedule overlap.");

DEFINE_double(prefill_scheduling_memory_usage_threshold,
              0.95,
              "The memory usage threshold during prefill scheduling.");

DEFINE_bool(enable_chunked_prefill, true, "Whether to enable chunked prefill.");

DEFINE_bool(enable_prefill_sp,
            false,
            "Whether to enable prefill-only sequence parallel.");

DEFINE_int32(max_tokens_per_chunk_for_prefill,
             -1,
             "Max number of token per chunk in prefill stage.");

DEFINE_int32(chunked_match_frequency,
             2,
             "Number of sequence prefix cache match frequency.");

DEFINE_bool(use_zero_evict,
            false,
            "Use ZeroEvictionScheduler but ContinuousScheduler.");

DEFINE_int32(
    max_decode_token_per_sequence,
    256,
    "Max decode token per sequence which used for ZeroEvictionScheduler.");

DEFINE_string(priority_strategy,
              "fcfs",
              "Priority strategy for requests(e.g. fcfs, priority, deadline).");

DEFINE_bool(use_mix_scheduler,
            false,
            "Use MixScheduler for handling prefill and decode uniformly.");

DEFINE_bool(enable_online_preempt_offline,
            true,
            "Whether to enable online preempt offline.");

// --- mix scheduler scheduling config ---

DEFINE_double(aggressive_coeff,
              1.0,
              "Aggressive coefficient for MixScheduler urgency judgment.");

DEFINE_double(starve_threshold,
              1.0,
              "Starvation threshold coefficient for MixScheduler.");

DEFINE_bool(enable_starve_prevent,
            true,
            "Whether to enable anti-starvation in MixScheduler.");

// for rec, it's better to set to 100;
DEFINE_int32(request_queue_size,
             100000,
             "The request queue size of the scheduler");

// --- parallel config ---

DEFINE_int32(dp_size, 1, "Data parallel size for MLA attention.");

DEFINE_int32(ep_size, 1, "Expert parallel size for MoE model.");

DEFINE_int32(cp_size, 1, "Context parallel size for DSA attention.");

DEFINE_string(
    communication_backend,
    "hccl",
    "NPU communication backend.(e.g. lccl, hccl). When enable dp, use hccl.");

// --- ep load balance config ---

DEFINE_bool(enable_eplb, false, "Whether to use expert parallel load balance.");

DEFINE_int32(redundant_experts_num,
             1,
             "Number of redundant experts on per device.");

DEFINE_int64(eplb_update_interval, 1000, "EPLB update rate.");

DEFINE_double(eplb_update_threshold, 0.8, "EPLB update threshold.");

DEFINE_int32(expert_parallel_degree, 0, "Expert parallel degree.");

DEFINE_string(rank_tablefile, "", "ATB HCCL rank table file.");

// --- profile config ---

DEFINE_bool(enable_profile_step_time,
            false,
            "Whether to enable profile step time.");

DEFINE_bool(enable_profile_token_budget,
            false,
            "Whether to enable profile token budget.");

DEFINE_bool(enable_latency_aware_schedule,
            false,
            "use predicted latency for latency aware schedule.");

DEFINE_int32(profile_max_prompt_length,
             2048,
             "The max prompt length for profile.");

DEFINE_bool(enable_profile_kv_blocks,
            true,
            "true if generate kv cache for profile");

DEFINE_bool(disable_ttft_profiling,
            false,
            "Whether to disable TTFT profiling.");

DEFINE_bool(enable_forward_interruption,
            false,
            "Whether to enable forward interruption.");

DEFINE_int32(max_global_ttft_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

DEFINE_int32(max_global_tpot_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

// --- prefix cache config ---

DEFINE_bool(enable_prefix_cache,
            true,
            "Whether to enable the prefix cache for the block manager.");

DEFINE_bool(enable_cache_upload,
            false,
            "Whether to upload cache info to service. This feature is only "
            "available when service routing is enabled.");

DEFINE_uint32(xxh3_128bits_seed, 1024, "Default XXH3 128-bits hash seed.");

// --- serving on multi-nodes config ---

DEFINE_string(master_node_addr,
              "127.0.0.1:19888",
              "The master address for multi-node distributed serving(e.g. "
              "10.18.1.1:9999).");

DEFINE_string(
    xtensor_master_node_addr,
    "127.0.0.1:19889",
    "The master address for XTensor distributed service(e.g. 10.18.1.1:9999).");

DEFINE_int32(nnodes, 1, "The number of multi-nodes.");

DEFINE_int32(node_rank, 0, "The node rank.");

DEFINE_bool(enable_shm,
            false,
            "Whether to enable shared memory for executing model.");

DEFINE_bool(use_contiguous_input_buffer,
            true,
            "Whether to use contiguous device input buffer for executing "
            "model. Currently only effective when enable_shm is true.");

// --- disaggregated prefill and decode config ---

DEFINE_bool(enable_disagg_pd,
            false,
            "Whether to enable disaggregated prefill and decode execution.");

DEFINE_bool(
    enable_pd_ooc,
    false,
    "Whether to enable online-offline co-location in disaggregated PD mode.");

DEFINE_int32(disagg_pd_port, 7777, "Port for brpc disagg pd server.");

DEFINE_string(instance_role,
              "DEFAULT",
              "The role of instance(e.g. DEFAULT, PREFILL, DECODE, MIX).");

DEFINE_string(
    kv_cache_transfer_type,
    "LlmDataDist",
    "The type of kv cache transfer(e.g. LlmDataDist, Mooncake, HCCL).");

DEFINE_string(kv_cache_transfer_mode,
              "PUSH",
              "The mode of kv cache transfer(e.g. PUSH, PULL).");

DEFINE_int32(transfer_listen_port, 26000, "The KVCacheTranfer listen port.");

DEFINE_uint64(input_shm_size,
              1024,
              "Input shared memory size, default is 1GB.");

DEFINE_uint64(output_shm_size,
              128,
              "Output shared memory size, default is 128MB.");

// --- speculative config ---

DEFINE_string(draft_model, "", "draft hf model path to the model file.");

DEFINE_string(draft_devices,
              "npu:0",
              "Devices to run the draft model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_int32(num_speculative_tokens, 0, "Number of speculative tokens.");

DEFINE_string(speculative_algorithm,
              "MTP",
              "Speculative decoding algorithm. Supported options: MTP, Eagle3, "
              "Suffix. Default is MTP.");

DEFINE_int32(speculative_suffix_cache_max_depth,
             64,
             "Maximum suffix tree depth for suffix speculative decoding.");

DEFINE_double(speculative_suffix_max_spec_factor,
              1.0,
              "Suffix speculation max tokens factor relative to match length.");

DEFINE_double(speculative_suffix_max_spec_offset,
              0.0,
              "Suffix speculation max tokens additive offset.");

DEFINE_double(speculative_suffix_min_token_prob,
              0.1,
              "Minimum token probability used in suffix speculation.");

DEFINE_int32(speculative_suffix_max_cached_requests,
             -1,
             "Maximum globally cached requests for suffix speculation (-1 "
             "unlimited, 0 disabled).");

DEFINE_bool(speculative_suffix_use_tree_spec,
            false,
            "Whether to use tree-based suffix speculation instead of path "
            "speculation.");

DEFINE_bool(enable_opt_validate_probs,
            false,
            "Whether validate uses selected-only draft_probs [B,S] directly. "
            "If false, selected-only cache values are restored to dense "
            "[B,S,V].");

DEFINE_bool(enable_atb_spec_kernel,
            false,
            "Whether to use ATB speculative kernel.");

// --- block copy config ---

#if defined(USE_NPU) || defined(USE_CUDA)
DEFINE_bool(enable_block_copy_kernel,
            true,
            "Whether to use block copy kernel on supported backends.");
#else
DEFINE_bool(enable_block_copy_kernel,
            false,
            "Whether to use block copy kernel on supported backends.");
#endif

// --- service routing config ---

DEFINE_string(etcd_addr, "", "Etcd adderss for save instance meta info.");

DEFINE_string(etcd_namespace,
              "",
              "Optional etcd namespace prefix for all xllm keys, e.g. prod-a.");

DEFINE_bool(enable_service_routing,
            false,
            "Whether to use xllm service routing.");

DEFINE_double(heart_beat_interval, 0.5, "Heart beat interval.");

DEFINE_int32(etcd_ttl, 3, "Time to live for etcd.");

// --- kvcache store config ---

DEFINE_uint32(prefetch_timeout,
              0,
              "Prefetch timeout for prefetch from kv cache store.");

DEFINE_uint32(prefetch_bacth_size,
              2,
              "Prefetch from kvcache store copy batch size.");

DEFINE_uint32(layers_wise_copy_batchs, 4, "Layer wise H2D copy batchs.");

DEFINE_double(host_blocks_factor,
              0.0,
              "Host block factor, e.g. host block num = host_blocks_factor * "
              "hbm block num.");

DEFINE_bool(enable_kvcache_store, false, "Whether to use kvcache store.");

DEFINE_string(store_protocol,
              "tcp",
              "KV cache store protocol(e.g. tcp, rdma).");

DEFINE_string(store_master_server_address,
              "",
              "The address information of the store master service.");

DEFINE_string(store_metadata_server,
              "",
              "The address of the kv cache store metadata service.");

DEFINE_string(store_local_hostname,
              "",
              "The local host name of the kv cache store client.");

DEFINE_bool(enable_control_h2d_block_num,
            false,
            "Whether to control h2d copy block num.");

// --- computation communication parallel config ---

DEFINE_bool(
    enable_multi_stream_parallel,
    false,
    "Whether to enable computation communication parallel by two streams "
    "and two micro batches in prefill stage.");

DEFINE_int32(micro_batch_num,
             1,
             "Default use two micro batches for multi-stream parallel.");

// --- dit config ---

DEFINE_int32(max_requests_per_batch, 1, "Max number of request per batch.");

DEFINE_bool(enable_manual_loader,
            false,
            "Pin decoder layer weights to host memory and use async H2D "
            "transfer. Required by enable_rolling_load; also implied by "
            "enable_xtensor.");

DEFINE_bool(
    enable_xtensor,
    false,
    "Whether to enable xtensor for model weights with physical page pool.");

// --- rolling load config ---

DEFINE_bool(enable_rolling_load,
            false,
            "Enable rolling weight load: keep only N decoder layer weight "
            "slots in HBM and stream-load each layer just-in-time. "
            "Requires enable_manual_loader=true. NPU only.");

DEFINE_int32(rolling_load_num_cached_layers,
             2,
             "Number of decoder layer weight slots to keep in HBM when "
             "enable_rolling_load=true.");

DEFINE_int32(rolling_load_num_rolling_slots,
             -1,
             "Number of rolling slots used by decoder rolling load. "
             "Fixed slots are computed as "
             "rolling_load_num_cached_layers - rolling_load_num_rolling_slots."
             " -1 means auto (min(2, preload_count)). "
             "Must be in [-1, rolling_load_num_cached_layers].");

DEFINE_int64(
    phy_page_granularity_size,
    2 * 1024 * 1024,
    "Granularity size for one physical page in bytes, default 2MB, when enable "
    "continuous kv cache.");

// --- beam search config ---

DEFINE_bool(enable_beam_search_kernel,
            false,
            "Whether to enable beam search kernel.");

DEFINE_bool(
    enable_rec_fast_sampler,
    true,
    "Whether to enable RecSampler fast sampling path for Rec pipelines.");

DEFINE_bool(enable_topk_sorted,
            true,
            "Whether to enable sorted output for topk.");

DEFINE_bool(
    output_rec_logprobs,
    false,
    "Whether to output rec multi-round token-aligned logprobs. "
    "When enabled, missing per-token logprobs are filled with the final "
    "beam logprob.");

// --- reasoning parser config ---

DEFINE_string(reasoning_parser,
              "",
              "Specify the reasoning parser for handling reasoning "
              "interactions(e.g. auto, glm45, glm47, glm5, qwen3, qwen35, "
              "deepseek-r1).");

// --- function call config ---

DEFINE_string(tool_call_parser,
              "",
              "Specify the parser for handling tool-call interactions(e.g. "
              "auto, qwen25, qwen3, qwen35, qwen3_coder, kimi_k2, "
              "deepseekv3, glm45, glm47, glm5).");

// --- qwen3 reranker config ---

DEFINE_bool(enable_qwen3_reranker, false, "Whether to enable qwen3 reranker.");

// --- flashinfer config ---

DEFINE_int32(flashinfer_workspace_buffer_size,
             128 * 1024 * 1024,
             "The user reserved workspace buffer used to store intermediate "
             "attention results in split-k algorithm for flashinfer.");

// --- prefetch weight config ---

DEFINE_bool(
    enable_prefetch_weight,
    false,
    "Whether to enable prefetch weight,only applicable to Qwen3-dense model."
    "The default prefetching ratio for gateup weight is 40%."
    "If adjustments are needed, e.g. export PREFETCH_COEFFOCIENT=0.5");

// --- rec prefill-only mode ---
DEFINE_bool(enable_rec_prefill_only,
            false,
            "Enable rec prefill-only mode (no decoder self-attention blocks "
            "allocation).");

// --- dp load balance ---

DEFINE_bool(
    enable_dp_balance,
    false,
    "Whether to enable dp load balance, if true, sequences within a single "
    "dp batch will be shuffled.");

// --- the seed for random number generator ---
DEFINE_int32(random_seed, -1, "Random seed for random number generator.");

// --- dit cache config ---

DEFINE_string(dit_cache_policy,
              "TaylorSeer",
              "The policy of dit cache(e.g. None, FBCache, TaylorSeer, "
              "FBCacheTaylorSeer, ResidualCache).");

DEFINE_int64(dit_cache_warmup_steps, 5, "The number of warmup steps.");
DEFINE_int64(
    dit_cache_probe_depth,
    2,
    "Number of DiT blocks to run before RACFGCache makes a joint decision.");

DEFINE_double(dit_cache_tau,
              0.28,
              "Accumulated guided-risk threshold for RACFGCache.");

DEFINE_bool(dit_cache_use_prop_weight,
            true,
            "Whether RACFGCache uses propagation-aware reweighting.");

DEFINE_double(dit_cache_prop_a,
              1.7997948016,
              "Propagation-aware fitted parameter a for RACFGCache.");

DEFINE_double(dit_cache_prop_alpha,
              1.1729645804,
              "Propagation-aware fitted parameter alpha for RACFGCache.");

DEFINE_double(dit_cache_prop_b,
              -0.1016712615,
              "Propagation-aware fitted parameter b for RACFGCache.");

DEFINE_int64(dit_cache_proxy_error_type,
             0,
             "Proxy error type for RACFGCache: 0=delta_y, 1=delta_minus.");

DEFINE_string(dit_cache_rho_table_path,
              "qwen_image_edit_cfg4_steps40",
              "Path to the offline rho lookup table used by RACFGCache.");

DEFINE_string(
    dit_cache_model_name,
    "qwen_image_edit_plus",
    "Model name for selecting hardcoded rho table when using RACFGCache.");

DEFINE_double(true_cfg_scale,
              4.0,
              "Optional CFG scale recorded with the rho table for RACFGCache.");
DEFINE_int64(dit_cache_n_derivatives,
             3,
             "The number of derivatives to use in TaylorSeer.");

DEFINE_int64(dit_cache_skip_interval_steps,
             3,
             "The interval steps to skip for derivative calculation.");

DEFINE_double(dit_cache_residual_diff_threshold,
              0.09f,
              "The residual difference threshold for cache reuse.");

DEFINE_bool(enable_constrained_decoding,
            false,
            "Whether to enable constrained decoding, which is used to ensure "
            "that the output meets specific format or structural requirements "
            "through pre-defined rules.");

DEFINE_bool(enable_convert_tokens_to_item,
            false,
            "Enable token ids conversion to item id in REC/OneRec response.");

DEFINE_int64(dit_cache_start_steps,
             5,
             "The number of steps to skip at the start");

DEFINE_int64(dit_cache_end_steps, 5, "The number of steps to skip at the end.");

DEFINE_int64(dit_cache_start_blocks,
             5,
             "The number of blocks to skip at the start.");

DEFINE_int64(dit_cache_end_blocks,
             5,
             "The number of blocks to skip at the end.");

// --- dit parallel config ---

DEFINE_int64(tp_size, 1, "Tensor parallelism size");

DEFINE_int64(sp_size, 1, "Sequence parallelism size");

DEFINE_int64(cfg_size, 1, "Classifier-free guidiance parallelism size");

DEFINE_int64(dit_sp_communication_overlap,
             1,
             "Communication & Computation overlap for sequence parallel");

// --- dit debug ---

DEFINE_bool(dit_debug_print,
            false,
            "whether print the debug info for dit models");

// --- embedding type ---

DEFINE_bool(enable_return_mm_full_embeddings,
            false,
            "return vit and sequence embeddings for vlm models");

DEFINE_bool(enable_output_sku_logprobs,
            false,
            "Enable REC / OneRec token-aligned logprobs tensor output.");

DEFINE_int32(each_conversion_threshold,
             50,
             "Maximum number of items emitted for each REC token triplet.");

DEFINE_int32(total_conversion_threshold,
             1000,
             "Maximum total number of items emitted in one REC response.");

DEFINE_bool(
    use_audio_in_video,
    false,
    "Whether to decode both audio and video when the input is a video.");

// --- concurrent rec worker config ---
DEFINE_uint32(rec_worker_max_concurrency,
              1,
              "Concurrency for rec worker parallel execution. Less than or "
              "equal to 1 means disable concurrent rec worker.");

#if defined(USE_NPU)
DEFINE_string(npu_kernel_backend,
              "AUTO",
              "NPU kernel backend. Supported options: AUTO, ATB, TORCH.");

DEFINE_bool(enable_intralayer_addnorm,
            false,
            "enable fused intralayer addnorm ops.");
#endif

// --- multi-step decode config ---

DEFINE_int32(max_decode_rounds,
             0,
             "Maximum number of decode rounds for multi-step decoding. "
             "0 means disabled.");

DEFINE_int32(beam_width, 1, "Beam width for beam search.");

// --- health check config ---
DEFINE_int32(health_check_interval_ms,
             3000,
             "Worker health check interval in milliseconds.");
DEFINE_bool(enable_xattention_one_stage,
            false,
            "Whether to force xattention one-stage decode for rec "
            "multi-round mode.");
