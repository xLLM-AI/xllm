---
title: "服务启动参数"
sidebar:
  order: 100
---

xLLM 使用 gflags 管理服务启动参数。`--model <PATH>` 是唯一必填参数。使用 `--config_json_file` 时，JSON 文件中的值会覆盖命令行 flag 值。下表按 `/xllm/core/framework/config` 下的 Config 类分组，一个 Config 对应一节；`ConfigJsonUtils` 一节包含配置文件相关的通用参数。

> **设备选择**：xLLM 不再提供 `--devices` / `--device_id` / `--draft_devices` 参数。可用设备由可见设备掩码环境变量决定（NPU 用 `ASCEND_RT_VISIBLE_DEVICES`，NVIDIA 用 `CUDA_VISIBLE_DEVICES`，寒武纪用 `MLU_VISIBLE_DEVICES`，DCU 用 `HIP_VISIBLE_DEVICES`，摩尔线程用 `MUSA_VISIBLE_DEVICES`）。每个服务进程根据全局 `node_rank` 从其可见设备中选择一个运行时逻辑设备；可见设备的子集化与重排由硬件运行时解析。draft 模型始终与 target 模型共享所选设备。

## ConfigJsonUtils

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `config_json_file` | `string` | `""` | JSON 配置文件路径；文件中的值会覆盖命令行 flag 值。 |
| `enable_dump_config_json` | `bool` | `false` | 是否将最终生效的启动配置导出为 JSON。 |
| `dump_config_json_file` | `string` | `"xllm_config.json"` | 导出启动配置 JSON 的路径，仅在 `enable_dump_config_json=true` 时使用。 |

## ServiceConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `host` | `string` | `""` | bRPC 服务监听的主机名或 IP。 |
| `port` | `int32` | `8010` | bRPC 服务监听端口。 |
| `rpc_idle_timeout_s` | `int32` | `-1` | 连接在最近 `rpc_idle_timeout_s` 秒内无读写操作时关闭；`-1` 表示无限等待。 |
| `rpc_channel_timeout_ms` | `int32` | `-1` | bRPC Channel 的最大等待时间，单位毫秒；`-1` 表示无限等待。 |
| `max_reconnect_count` | `int32` | `40` | worker 尝试连接 server 的最大重连次数。 |
| `num_threads` | `int32` | `8` | 处理请求的线程数。 |
| `max_concurrent_requests` | `int32` | `200` | 实例可同时处理的最大请求数；设为 `0` 表示不限流。 |
| `num_request_handling_threads` | `int32` | `4` | 处理输入请求的线程数。 |
| `num_response_handling_threads` | `int32` | `4` | 处理响应输出的线程数。 |
| `health_check_interval_ms` | `int32` | `3000` | worker 健康检查间隔，单位毫秒。 |

## ModelConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `model_id` | `string` | `""` | Hugging Face 模型名称，非路径。 |
| `model` | `string` | `""` | Hugging Face 模型名称或模型路径。 |
| `backend` | `string` | `""` | 后端模型类型；`llm` 表示纯文本模型，`vlm` 表示多模态模型，`dit` 表示扩散模型。 |
| `task` | `string` | `"generate"` | 模型任务类型，例如 `generate`、`embed`、`mm_embed`。 |
| `limit_image_per_prompt` | `int32` | `8` | 每个 prompt 允许的最大图片数量，仅用于多模态模型。 |
| `max_encoder_cache_size` | `int64` | `0` | 每个 worker 的 encoder cache 最大显存大小，单位 MB；`0` 表示禁用 encoder cache。 |
| `reasoning_parser` | `string` | `""` | reasoning 交互解析器，例如 `auto`、`glm45`、`glm47`、`glm5`、`qwen3`、`qwen35`、`deepseek-r1`。 |
| `tool_call_parser` | `string` | `""` | tool-call 交互解析器，例如 `auto`、`qwen25`、`qwen3`、`qwen35`、`qwen3_coder`、`kimi_k2`、`deepseekv3`、`deepseekv32`、`deepseekv4`、`glm45`、`glm47`、`glm5`。 |
| `enable_qwen3_reranker` | `bool` | `false` | 是否启用 Qwen3 reranker。 |
| `flashinfer_workspace_buffer_size` | `int32` | `134217728` | FlashInfer split-k attention 中用于保存中间 attention 结果的预留 workspace 大小，默认 128 MiB。 |
| `enable_return_mm_full_embeddings` | `bool` | `false` | VLM 模型是否返回 ViT embedding 与序列 embedding。 |
| `mm_download_headers` | `string` | `""` | 多模态下载的 service 级默认 HTTP header，为 JSON 对象；per-request header 优先级更高。示例：`{"Authorization":"Bearer xxx"}`。 |
| `use_audio_in_video` | `bool` | `false` | 输入为视频时，是否同时解码音频和视频。 |
| `use_cpp_chat_template` | `bool` | `true` | 对支持的模型使用原生 C++ chat template，例如 `deepseek_v32`、`deepseek_v4`；设为 `false` 可回退到 Jinja 以便调试。 |

## LoadConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_manual_loader` | `bool` | `false` | 将 decoder layer 权重固定在 host 内存并使用异步 H2D 传输；`enable_rolling_load` 依赖该参数，`enable_xtensor` 也会隐式启用该能力。 |
| `enable_rolling_load` | `bool` | `false` | 启用 rolling weight load：HBM 中仅保留 N 个 decoder layer 权重槽位，并按层即时加载；需要 `enable_manual_loader=true`，仅 NPU 支持。 |
| `rolling_load_num_cached_layers` | `int32` | `2` | `enable_rolling_load=true` 时 HBM 中保留的 decoder layer 权重槽位数量。 |
| `rolling_load_num_rolling_slots` | `int32` | `-1` | decoder rolling load 使用的 rolling 槽位数量；固定槽位数为 `rolling_load_num_cached_layers - rolling_load_num_rolling_slots`。`-1` 表示自动设置为 `min(2, preload_count)`，取值需在 `[-1, rolling_load_num_cached_layers]` 范围内。 |
| `enable_prefetch_weight` | `bool` | `false` | 是否启用权重预取，仅适用于 Qwen3-dense 模型；gateup 权重默认预取比例为 40%，可通过环境变量 `PREFETCH_COEFFOCIENT` 调整。 |

## KVCacheConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `block_size` | `int32` | `128` | 每个 KV Cache block 的 slot 数。 |
| `max_cache_size` | `int64` | `0` | KV Cache 可使用的 GPU 显存大小；`0` 表示根据可用显存自动计算。 |
| `max_memory_utilization` | `double` | `0.8` | 模型推理可使用的 GPU 显存比例，包括模型权重和 KV Cache。 |
| `kv_cache_dtype` | `string` | `"auto"` | KV Cache 量化数据类型；`auto` 表示与模型 dtype 对齐且不量化，`int8` 表示启用 INT8 量化，仅 MLU 后端支持。 |
| `indexer_cache_dtype` | `string` | `"auto"` | 带 indexer cache 的模型所使用的 indexer cache 数据类型。支持 `auto` 和 `int8`；`auto` 表示与模型 dtype 对齐且不量化，`int8` 表示启用 INT8 indexer cache 量化。 |
| `enable_prefix_cache` | `bool` | `true` | 是否在 block manager 中启用 prefix cache；详见 [Prefix Cache](/zh/features/prefix_cache/)。 |
| `enable_in_batch_prefix_cache` | `bool` | `false` | 是否将已准入的 prefill 完整 block 缓存进 prefix cache，使同一 batch 内的后续请求可以共享。 |
| `max_linear_state_cache_slots` | `int64` | `0` | linear-attention state cache 的最大活跃槽位数；`0` 表示根据可用 KV Cache 预算自动推导容量。 |
| `xxh3_128bits_seed` | `uint32` | `1024` | XXH3 128-bit 哈希的默认 seed。 |
| `enable_xtensor` | `bool` | `false` | 是否为模型权重启用基于物理页池的 XTensor。 |
| `phy_page_granularity_size` | `int64` | `2097152` | 单个物理页的粒度大小，单位 byte，默认 2 MiB；用于连续 KV Cache。 |

## KVCacheStoreConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `prefetch_timeout` | `uint32` | `0` | 从 KV Cache Store 预取数据的超时时间。 |
| `prefetch_batch_size` | `uint32` | `2` | 从 KV Cache Store 预取并拷贝的 batch size。 |
| `layers_wise_copy_batchs` | `uint32` | `4` | 按层执行 H2D 拷贝的 batch 数。 |
| `host_blocks_factor` | `double` | `0.0` | host block 系数，例如 `host block num = host_blocks_factor * hbm block num`。 |
| `enable_kvcache_store` | `bool` | `false` | 是否启用 KV Cache Store。 |
| `store_protocol` | `string` | `"tcp"` | KV Cache Store 协议，例如 `tcp`、`rdma`。 |
| `store_master_server_address` | `string` | `""` | Store master service 的地址信息。 |
| `store_metadata_server` | `string` | `""` | KV Cache Store metadata service 的地址。 |
| `store_local_hostname` | `string` | `""` | KV Cache Store client 的本地主机名。 |
| `enable_control_h2d_block_num` | `bool` | `false` | 是否控制 H2D 拷贝的 block 数。 |

## BeamSearchConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_beam_search_kernel` | `bool` | `false` | 是否启用 beam search kernel。 |
| `beam_width` | `int32` | `1` | Beam search 的 beam width。 |
| `enable_block_copy_kernel` | `bool` | `true`（NPU/CUDA）；`false`（其他后端） | 是否在支持的后端使用 block copy kernel。 |
| `enable_topk_sorted` | `bool` | `true` | 是否启用 top-k 结果排序输出。 |

## SchedulerConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `max_tokens_per_batch` | `int32` | `10240` | 每个 batch 可处理的最大 token 数。 |
| `max_seqs_per_batch` | `int32` | `1024` | 每个 batch 可处理的最大 sequence 数。 |
| `enable_schedule_overlap` | `bool` | `false` | 是否启用 schedule overlap（异步调度）；详见 [异步调度](/zh/features/async_schedule/)。 |
| `prefill_scheduling_memory_usage_threshold` | `double` | `0.95` | prefill 调度时的内存使用阈值。 |
| `enable_chunked_prefill` | `bool` | `true` | 是否启用 chunked prefill。 |
| `max_tokens_per_chunk_for_prefill` | `int32` | `-1` | prefill 阶段每个 chunk 的最大 token 数；`-1` 表示使用默认策略。 |
| `chunked_match_frequency` | `int32` | `2` | sequence prefix cache 匹配频率。 |
| `use_zero_evict` | `bool` | `false` | 是否使用 ZeroEvictionScheduler；详见 [Zero Evict Scheduler](/zh/features/zero_evict_scheduler/)。 |
| `max_decode_token_per_sequence` | `int32` | `256` | ZeroEvictionScheduler 中每个 sequence 的最大 decode token 数。 |
| `priority_strategy` | `string` | `"fcfs"` | 请求优先级策略，例如 `fcfs`、`priority`、`deadline`。 |
| `use_mix_scheduler` | `bool` | `false` | 是否使用 MixScheduler 统一处理 prefill 和 decode。 |
| `enable_online_preempt_offline` | `bool` | `true` | 是否允许在线请求抢占离线请求。 |
| `aggressive_coeff` | `double` | `1.0` | MixScheduler 紧急度判断的激进系数。 |
| `starve_threshold` | `double` | `1.0` | MixScheduler 的饥饿阈值系数。 |
| `enable_starve_prevent` | `bool` | `true` | 是否启用 MixScheduler 的防饥饿机制。 |

## ParallelConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `dp_size` | `int32` | `1` | MLA attention 的 data parallel 规模。 |
| `ep_size` | `int32` | `1` | MoE 模型的 expert parallel 规模。 |
| `cp_size` | `int32` | `1` | Context parallel 规模，需要后端和模型支持；设为 `1` 表示禁用 context parallel。MLU 的具体限制见[寒武纪 MLU](/zh/hardware/cambricon_mlu/)。 |
| `kv_split_size` | `int32` | `1` | KV Cache split 宽度；`0` 表示沿用 `cp_size`（旧行为），`1` 表示不切分 KV（每个 CP rank 存储完整 KV 并跳过 prefix AllGather），其他可整除 `cp_size` 的 K 表示 KV 在 K 个 rank 间分片，而 token-CP 仍使用 `cp_size`。 |
| `tp_size` | `int64` | `1` | Tensor parallel 规模，仅 DiT 模型使用。 |
| `sp_size` | `int64` | `1` | Sequence parallel 规模，仅 DiT 模型使用。 |
| `cfg_size` | `int64` | `1` | Classifier-free guidance parallel 规模，仅 DiT 模型使用。 |
| `vae_size` | `int64` | `1` | VAE patch parallel 规模，仅 DiT 模型使用。 |
| `communication_backend` | `string` | `"hccl"` | NPU 通信后端，例如 `lccl`、`hccl`；启用 DP 时使用 `hccl`。 |
| `enable_mm_encoder_dp` | `bool` | `false` | 是否为多模态模型启用 encoder data parallel。 |
| `enable_multi_stream_parallel` | `bool` | `false` | 是否在 prefill 阶段通过双 stream 和双 micro batch 启用计算通信并行；详见 [多流并行](/zh/features/multi_streams/)。 |
| `micro_batch_num` | `int32` | `1` | 多流并行使用的 micro batch 数。 |
| `enable_dp_balance` | `bool` | `false` | 是否启用 DP 负载均衡；启用后会 shuffle 单个 DP batch 内的 sequences。 |

## EPLBConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_eplb` | `bool` | `false` | 是否启用 expert parallel load balance；详见 [EPLB](/zh/features/eplb/)。 |
| `redundant_experts_num` | `int32` | `1` | 每个 device 上的冗余 expert 数量。 |
| `eplb_update_interval` | `int64` | `1000` | EPLB 更新间隔。 |
| `eplb_update_threshold` | `double` | `0.8` | EPLB 更新阈值。 |
| `expert_parallel_degree` | `int32` | `0` | Expert parallel degree。 |
| `rank_tablefile` | `string` | `""` | ATB HCCL rank table 文件。 |

## DistributedConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `master_node_addr` | `string` | `"127.0.0.1:19888"` | 多机分布式服务的 master 地址，例如 `10.18.1.1:9999`。 |
| `xtensor_master_node_addr` | `string` | `"127.0.0.1:19889"` | XTensor 分布式服务的 master 地址，例如 `10.18.1.1:9999`。 |
| `nnodes` | `int32` | `1` | 多机节点数量。 |
| `node_rank` | `int32` | `0` | 当前节点 rank。 |
| `etcd_addr` | `string` | `""` | 保存实例元信息的 etcd 地址。 |
| `etcd_namespace` | `string` | `""` | xLLM etcd key 使用的可选 namespace 前缀，例如 `prod-a`。 |
| `enable_service_routing` | `bool` | `false` | 是否启用 xLLM service routing。 |
| `heart_beat_interval` | `double` | `0.5` | 心跳间隔。 |
| `etcd_ttl` | `int32` | `3` | etcd key 的 TTL。 |

## DisaggPDConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_disagg_pd` | `bool` | `false` | 是否启用 Prefill-Decode 分离执行；详见 [PD 分离](/zh/features/disagg_pd/)。 |
| `enable_pd_ooc` | `bool` | `false` | 是否在 PD 分离模式下启用在线/离线混部。 |
| `disagg_pd_port` | `int32` | `7777` | PD 分离 bRPC server 的监听端口。 |
| `instance_role` | `string` | `"DEFAULT"` | 当前实例角色，例如 `DEFAULT`、`PREFILL`、`DECODE`、`MIX`。 |
| `kv_cache_transfer_type` | `string` | `"LlmDataDist"` | KV Cache 传输类型，例如 `LlmDataDist`、`Mooncake`、`HCCL`。 |
| `kv_cache_transfer_mode` | `string` | `"PUSH"` | KV Cache 传输模式，例如 `PUSH`、`PULL`。 |
| `transfer_listen_port` | `int32` | `26000` | KV Cache Transfer 的监听端口。 |
| `kv_push_dst_rotate` | `bool` | `false` | 在 `push_kv_blocks` 中按 KV-split rank 轮转遍历目标 worker，用于分散对 decode worker 的流量。 |

## SpeculativeConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `draft_model` | `string` | `""` | draft 模型路径；MTP 使用方式详见 [MTP](/zh/features/mtp/)。 |
| `num_speculative_tokens` | `int32` | `0` | 每轮 speculative decoding 生成的 speculative token 数。 |
| `speculative_algorithm` | `string` | `"MTP"` | Speculative decoding 算法，支持 `MTP`、`Eagle3`、`Suffix`、`DFlash`。 |
| `speculative_suffix_cache_max_depth` | `int32` | `64` | Suffix speculative decoding 的后缀树最大深度。 |
| `speculative_suffix_max_spec_factor` | `double` | `1.0` | Suffix speculation 相对于匹配长度的最大 token 系数。 |
| `speculative_suffix_max_spec_offset` | `double` | `0.0` | Suffix speculation 的最大 token 加性偏移。 |
| `speculative_suffix_min_token_prob` | `double` | `0.1` | Suffix speculation 使用的最小 token 概率。 |
| `speculative_suffix_max_cached_requests` | `int32` | `-1` | Suffix speculation 全局最大缓存请求数；`-1` 表示不限，`0` 表示禁用。 |
| `speculative_suffix_use_tree_spec` | `bool` | `false` | 是否使用 tree-based suffix speculation，而不是 path speculation。 |
| `enable_opt_validate_probs` | `bool` | `false` | validate 阶段是否直接使用 selected-only `draft_probs [B,S]`；设为 `false` 时会将 selected-only cache 值恢复为 dense `[B,S,V]`。 |
| `enable_atb_spec_kernel` | `bool` | `false` | 是否使用 ATB speculative kernel。 |

## ProfileConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_profile_step_time` | `bool` | `false` | 是否启用 step time profiling。 |
| `enable_profile_token_budget` | `bool` | `false` | 是否启用 token budget profiling。 |
| `enable_latency_aware_schedule` | `bool` | `false` | 是否使用预测 latency 进行 latency-aware schedule。 |
| `profile_max_prompt_length` | `int32` | `2048` | profiling 使用的最大 prompt 长度。 |
| `max_global_ttft_ms` | `int32` | `std::numeric_limits<int32_t>::max()` | 全局 TTFT 阈值，单位毫秒。 |
| `max_global_tpot_ms` | `int32` | `std::numeric_limits<int32_t>::max()` | 全局 TPOT 阈值，单位毫秒。 |
| `enable_profile_kv_blocks` | `bool` | `true` | profiling 时是否生成 KV Cache blocks。 |
| `disable_ttft_profiling` | `bool` | `false` | 是否禁用 TTFT profiling。 |
| `enable_forward_interruption` | `bool` | `false` | 是否启用 forward interruption。 |
| `enable_online_profile` | `bool` | `false` | 是否启用在线 timeline profiling 端点（`/start_profile` 和 `/stop_profile`）；目前仅支持 CUDA，需配合以 `nsys --capture-range=cudaProfilerApi` 启动 server。 |
| `profile_backend` | `string` | `"torch"` | 在线 profiling 后端。`torch` 在进程内记录 CPU+CUDA 活动，并在 `/stop_profile` 时写出 Chrome trace，无需外部 profiler；`cuda` 仅切换 CUDA profiler 的 capture range，需配合以 `nsys --capture-range=cudaProfilerApi` 启动。 |
| `profile_dir` | `string` | `""` | `torch` 在线 profiling 后端写出 timeline trace 的目录；为空表示当前工作目录。 |

## ExecutionConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_graph` | `bool` | `false` | 是否在 decode 阶段启用图执行以降低 kernel launch 开销和设备空闲时间；支持 CUDA Graph、ACL Graph（NPU）、MLU Graph 和 DCU Graph，详见 [图执行](/zh/features/graph_mode/)。 |
| `enable_graph_double_buffer` | `bool` | `true` | 是否为 NPU schedule-overlap decode 启用双缓冲 ACL graph 持久化参数与 graph 实例。 |
| `enable_graph_mode_decode_no_padding` | `bool` | `false` | decode 阶段是否按实际 `num_tokens` 捕获 graph，而不是按 padding 后 shape 捕获。 |
| `enable_prefill_piecewise_graph` | `bool` | `false` | 是否在 prefill 阶段启用 piecewise CUDA graph；attention 使用 eager 模式，其他操作捕获进 CUDA graph。 |
| `enable_graph_vmm_pool` | `bool` | `true` | 是否启用 VMM-backed CUDA graph memory pool，用于多 shape graph 的显存复用。 |
| `max_tokens_for_graph_mode` | `int32` | `2048` | 图执行最大 token 数；`0` 表示不限制。 |
| `acl_graph_decode_batch_size_limit` | `int32` | `16` | NPU 上 ACL graph 的 decode batch size 阈值；实际 decode batch size 超过该值时，ACL graph decode 会回退到 eager 模式以避免 OOM。 |
| `enable_shm` | `bool` | `false` | 是否为模型执行启用共享内存。 |
| `use_contiguous_input_buffer` | `bool` | `true` | 是否使用连续的 device input buffer 执行模型。 |
| `input_shm_size` | `uint64` | `1024` | 输入共享内存大小，默认 1GB。 |
| `output_shm_size` | `uint64` | `128` | 输出共享内存大小，默认 128MB。 |
| `random_seed` | `int32` | `-1` | 随机数生成器 seed；`-1` 表示不固定 seed。 |

## KernelConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_customize_mla_kernel` | `bool` | `false` | 是否启用自定义 MLA kernel。 |
| `npu_kernel_backend` | `string` | `"AUTO"` | NPU kernel 后端，支持 `AUTO`、`ATB`、`TORCH`。 |
| `enable_intralayer_addnorm` | `bool` | `false` | 是否启用 fused intralayer addnorm ops。 |
| `enable_fused_mc2` | `int32` | `-1` | NPU 的 Fused MC2 模式；`-1` 使用自动默认值，`0` 禁用 fused MC2，正值启用 dense matmul-allreduce，`1` 对 MoE 使用 DispatchFFNCombine，`2` 对 MoE 使用 DispatchGmmCombineDecode。 |
| `enable_interlayer_addnorm` | `bool` | `false` | 是否启用 fused interlayer addnorm ops。 |
| `enable_split_rmsnorm_rope` | `bool` | `false` | 是否启用 fused split rmsnorm rope ops。 |
| `enable_aclnn_matmul` | `bool` | `false` | 是否为支持的 NPU ATB layer 启用 ACLNN matmul 后端。 |
| `enable_aclnn_swiglu` | `bool` | `false` | 是否为支持的 NPU ATB layer 启用 ACLNN SwiGLU 后端。 |

## DiTConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `max_requests_per_batch` | `int32` | `1` | 每个 batch 的最大 request 数。 |
| `dit_cache_policy` | `string` | `"TaylorSeer"` | DiT cache 策略，例如 `None`、`FBCache`、`TaylorSeer`、`FBCacheTaylorSeer`、`ResidualCache`。 |
| `dit_cache_warmup_steps` | `int64` | `0` | warmup step 数量。 |
| `dit_cache_n_derivatives` | `int64` | `3` | TaylorSeer 使用的 derivative 数量。 |
| `dit_cache_skip_interval_steps` | `int64` | `3` | derivative 计算的跳步间隔。 |
| `dit_cache_residual_diff_threshold` | `double` | `0.09` | cache reuse 的 residual difference 阈值。 |
| `dit_cache_start_steps` | `int64` | `5` | 起始阶段跳过的 step 数。 |
| `dit_cache_end_steps` | `int64` | `5` | 末尾阶段跳过的 step 数。 |
| `dit_cache_start_blocks` | `int64` | `5` | 起始阶段跳过的 block 数。 |
| `dit_cache_end_blocks` | `int64` | `5` | 末尾阶段跳过的 block 数。 |
| `dit_sp_communication_overlap` | `bool` | `true` | 是否为 sequence parallel 启用通信与计算的 overlap。 |
| `dit_debug_print` | `bool` | `false` | 是否打印 DiT 模型 debug 信息。 |
| `dit_generation_image_area_max` | `int64` | `0` | 图像生成请求允许的最大图像面积（宽 * 高）；`0` 表示不限制。 |
| `dit_enable_vae_tiling` | `bool` | `false` | 是否启用 VAE tiling，目前仅支持 `qwen-image-edit-plus`。 |
| `dit_vae_image_size` | `int64` | `1048576` | Qwen-Image-Edit-Plus VAE 用于计算维度的 image size。 |
| `dit_sparse_attention_enabled` | `bool` | `false` | 是否为 WAN 启用 block-wise sparse attention / RainFusion。 |
| `dit_sparse_attention_sparsity` | `double` | `0.5` | sparse attention 稀疏度，取值 `[0.0, 1.0)`；`0.0` 表示 dense attention，`0.5` 表示丢弃 50% 的 block。 |
| `dit_sparse_attention_pool_size` | `int64` | `128` | block-wise mask 生成时 sparse attention 的 pooling 窗口大小。 |
| `dit_sparse_attention_sparse_start_step` | `int64` | `0` | 开始使用 sparse attention 的 step 索引；此前的 step 使用 dense attention。 |
| `dit_sparse_attention_version` | `string` | `"rain_fusion"` | sparse attention 版本：`rain_fusion`（frame-pairing + `aclnnRainFusionAttention`）或 `sparse_attention`（block-decompose + `aclnnBlockSparseAttention`）。 |
| `dit_sparse_attention_mask_refresh_steps` | `int64` | `1` | 每 N 个 diffusion step 重新计算一次 block sparse mask；`1` 表示每步都算，值越大 mask 复用越久。 |

## RecConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_rec_fast_sampler` | `bool` | `true` | 是否为 Rec pipeline 启用 RecSampler fast sampling path。 |
| `enable_rec_prefill_only` | `bool` | `false` | 是否启用 Rec prefill-only 模式，不分配 decoder self-attention blocks。 |
| `enable_xattention_one_stage` | `bool` | `false` | 是否在 Rec multi-round 模式下强制使用 xattention one-stage decode。 |
| `max_decode_rounds` | `int32` | `0` | multi-step decoding 的最大 decode round 数；`0` 表示禁用。 |
| `enable_constrained_decoding` | `bool` | `false` | 是否启用 constrained decoding，用预定义规则约束输出格式或结构。 |
| `output_rec_logprobs` | `bool` | `false` | 是否输出 Rec multi-round token-aligned logprobs；启用后缺失的 per-token logprobs 会用最终 beam logprob 填充。 |
| `enable_convert_tokens_to_item` | `bool` | `false` | 是否在 REC/OneRec response 中将 token ids 转换为 item id。 |
| `enable_output_sku_logprobs` | `bool` | `false` | 是否输出 REC/OneRec token-aligned logprobs tensor。 |
| `enable_extended_item_info` | `bool` | `false` | 是否解析并输出 REC extended item info tensors。 |
| `each_conversion_threshold` | `int32` | `50` | 每个 REC token triplet 最多输出的 item 数。 |
| `total_conversion_threshold` | `int32` | `1000` | 单个 REC response 最多输出的 item 总数。 |
| `request_queue_size` | `int32` | `100000` | scheduler request queue 大小。 |
| `rec_worker_max_concurrency` | `uint32` | `1` | Rec worker 并行执行并发度；小于等于 `1` 表示禁用并发 Rec worker。 |
