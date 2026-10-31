---
title: "服务启动参数"
sidebar:
  order: 100
---

xLLM 使用 gflags 管理服务启动参数。`--model <PATH>` 是唯一必填参数。对于原生配置，显式指定的命令行 flag 优先于 JSON 配置值，后者优先于编译时默认值。下表按 `/xllm/core/framework/config` 下的 Config 类分组，一个 Config 对应一节；`ConfigJsonUtils` 一节包含配置文件相关的通用参数。

## ConfigJsonUtils

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `config_json_file` | `string` | `""` | JSON 配置文件路径；对于原生配置，显式指定的命令行 flag 优先于文件中的值。 |
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
| `enable_return_embedding_modality_tags` | `bool` | `false` | 是否在完整序列 embedding 中返回每个 token 的 modality tag。 |
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
| `enable_manual_loader` | `bool` | `false` | 将 decoder layer 权重固定在 host 内存并使用异步 H2D 传输；`enable_rolling_load` 依赖该参数，`enable_vmm_weights` 也会隐式启用该能力。 |
| `enable_vmm_weights` | `bool` | `false` | 为模型权重启用 VMM，并隐式启用 manual loader。与 `enable_vmm_kvcache` 独立，仅 NPU 支持。 |
| `enable_rolling_load` | `bool` | `false` | 启用 rolling weight load：HBM 中仅保留 N 个 decoder layer 权重槽位，并按层即时加载；需要 `enable_manual_loader=true`，仅 NPU 支持。 |
| `rolling_load_num_cached_layers` | `int32` | `2` | `enable_rolling_load=true` 时 HBM 中保留的 decoder layer 权重槽位数量。 |
| `rolling_load_num_rolling_slots` | `int32` | `-1` | decoder rolling load 使用的 rolling 槽位数量；固定槽位数为 `rolling_load_num_cached_layers - rolling_load_num_rolling_slots`。`-1` 表示自动设置为 `min(2, preload_count)`，取值需在 `[-1, rolling_load_num_cached_layers]` 范围内。 |
| `enable_prefetch_weight` | `bool` | `false` | 是否启用权重预取，仅适用于 Qwen3-dense 模型；gateup 权重默认预取比例为 40%，可通过环境变量 `PREFETCH_COEFFOCIENT` 调整。 |

## KVCacheConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `block_size` | `int32` | `128` | 每个 KV Cache block 的 slot 数。 |
| `max_cache_size` | `int64` | `0` | KV Cache 的容量上限，单位 byte；虚拟内存模式下限制 KV 与权重共享池的容量。`0` 表示不设显式字节上限，根据可用显存推导容量。 |
| `max_memory_utilization` | `double` | `0.8` | 模型推理可使用的设备显存比例，包括模型权重和 KV Cache。原有物理 KV Cache 和固定 VMM 要求取值在 `(0, 1]` 内。弹性 VMM 仅在显式配置时应用该上限；不配置或设为 `0` 时按实时可用显存扩张。 |
| `kv_cache_dtype` | `string` | `"auto"` | KV Cache 量化数据类型；`auto` 表示与模型 dtype 对齐且不量化，`int8` 表示启用 INT8 量化，仅 MLU 后端支持。 |
| `indexer_cache_dtype` | `string` | `"auto"` | 带 indexer cache 的模型所使用的 indexer cache 数据类型。支持 `auto` 和 `int8`；`auto` 表示与模型 dtype 对齐且不量化，`int8` 表示启用 INT8 indexer cache 量化。 |
| `enable_prefix_cache` | `bool` | `true` | 是否在 block manager 中启用 prefix cache；详见 [Prefix Cache](/zh/features/prefix_cache/)。 |
| `kv_events_config` | `string` | `""` | KV 驻留事件发布配置，为 JSON 对象字符串；空字符串禁用上报。JSON 字段和示例见下文。 |
| `enable_in_batch_prefix_cache` | `bool` | `false` | 是否将已准入的 prefill 完整 block 缓存进 prefix cache，使同一 batch 内的后续请求可以共享。 |
| `max_linear_state_cache_slots` | `int64` | `0` | linear-attention state cache 的最大活跃槽位数；`0` 表示根据可用 KV Cache 预算自动推导容量。 |
| `xxh3_128bits_seed` | `uint32` | `1024` | XXH3 128-bit 哈希的默认 seed。 |
| `enable_vmm_kvcache` | `bool` | `false` | 为 KV Cache 启用 VMM，与 `enable_vmm_weights` 独立；`false` 使用原有物理 KV Cache 实现，仅 NPU 支持。 |
| `vmm_elastic` | `bool` | `true` | 任一 VMM 开关启用后，按需扩张活动逻辑槽位及物理页，并释放空闲页的物理内存；`false` 使用固定驻留池。 |
| `vmm_memory_reserve_bytes` | `int64` | `0` | 弹性虚拟内存增长时额外保留的设备空闲显存，单位 byte，须非负；在运行时执行预留空间之外叠加生效。 |
| `vmm_startup_timeout_ms` | `int32` | `120000` | VMM 启动时，master 等待节点注册、各非 master 等待物理池就绪分别使用的超时，单位毫秒，须为正数。独立于单次 RPC 超时，每个等待阶段单独计时。 |
| `phy_page_granularity_size` | `int64` | `2097152` | 单个物理页的粒度大小，单位 byte，默认 2 MiB；用于连续 KV Cache。 |

VMM 的使用方式和容量配置说明见[KV Cache 与模型权重的虚拟内存](/zh/features/virtual_memory/)。

### kv_events_config

以下字段位于 `kv_events_config` 的 JSON 对象中：

| 字段 | 类型 | 默认值 | 含义 |
|:-----|:-----|:-------|:-----|
| `publisher` | `string` | `"zmq"` | 发布器类型，当前仅支持 `zmq`。 |
| `endpoint` | `string` | 必填 | vLLM 风格 KV 事件的 ZeroMQ 发布基准地址，例如 `tcp://*:5557`。必须为非空 TCP 地址。 |
| `replay_endpoint` | `string` | `""` | 快照恢复请求的 ZeroMQ 基准地址，例如 `tcp://*:6557`；空字符串禁用该入口。恢复回复使用当前全量快照和对应的实时流序号。 |
| `motor_pool_endpoint` | `string` | `""` | Motor 本地 Host 副本确认事件的 ZeroMQ 发布基准地址，例如 `tcp://*:7557`；空字符串禁用该流。 |
| `topic` | `string` | `""` | 发布的 ZeroMQ topic；消费端的订阅过滤器需要匹配该 topic。 |
| `hwm` | `int32` | `10000` | ZeroMQ 发送队列高水位，单位为消息数；必须为正数。 |
| `snapshot_interval_seconds` | `int32` | `60` | 周期性发布全量驻留快照的间隔，单位为秒；必须为正数。 |

每个 DP rank 使用基准 TCP 端口加 `dp_rank`，所有事件、恢复和 Host 确认端口范围须互不重叠，且端口须位于 `1` 到 `65535`。例如 DP=2、`endpoint=tcp://*:5557` 时，两个 rank 分别发布到 `5557` 和 `5558`。

可在原有启动命令中加入以下参数，启用事件发布和快照恢复：

```bash
--kv_events_config='{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:6557"}'
```

当前支持开启 prefix cache 的普通非 PD 分离文本推理、flat KV 布局和本地 Host offload。上报的是可复用前缀 block 的 Device/Host 驻留元数据；Host 副本在拷贝成功确认后发布。发布器在后台线程运行。多资源复合/linear-state 布局、多模态、虚拟内存 KV、PD 分离和外部 KV 存储尚未支持。

## KVCacheStoreConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `prefetch_timeout` | `uint32` | `0` | 到期后停止下发新的 KV Cache Store 预取批次，并等待在途批次完成；`0` 表示无限等待。 |
| `prefetch_batch_size` | `uint32` | `2` | 从 KV Cache Store 预取并拷贝的 batch size。 |
| `layers_wise_copy_batchs` | `uint32` | `4` | 按层执行 H2D 拷贝的 batch 数。 |
| `host_blocks_factor` | `double` | `0.0` | host block 系数，例如 `host block num = host_blocks_factor * hbm block num`。 |
| `enable_kvcache_store` | `bool` | `false` | 是否启用 KV Cache Store。 |
| `store_protocol` | `string` | `"tcp"` | KV Cache Store 协议，例如 `tcp`、`rdma`。 |
| `store_rdma_devices` | `string` | `""` | xLLM Worker 内嵌 Store client 使用的 RDMA HCA 列表，多个设备以逗号分隔；空值表示由 Mooncake 自动发现。 |
| `store_master_server_address` | `string` | `""` | Store master 地址。单机模式使用 `IP:Port`；etcd 高可用模式使用 `etcd://IP:Port;IP:Port;...`。 |
| `store_metadata_server` | `string` | `""` | KV Cache Store metadata service 的地址。 |
| `store_local_hostname` | `string` | `""` | KV Cache Store client 的本地主机名。 |
| `enable_control_h2d_block_num` | `bool` | `false` | 是否控制 H2D 拷贝的 block 数。 |

## BeamSearchConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_beam_search_kernel` | `bool` | `false` | 是否启用 beam search kernel。 |
| `beam_width` | `int32` | `1` | Beam search 的 beam width。 |
| `enable_block_copy_kernel` | `bool` | `true`（NPU/CUDA/MUSA/DCU）；`false`（其他后端） | 是否在支持的后端使用 block copy kernel。 |
| `enable_topk_sorted` | `bool` | `true` | 是否启用 top-k 结果排序输出。 |

## SchedulerConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `max_tokens_per_batch` | `int32` | `10240` | 每个 batch 可处理的最大 token 数。 |
| `max_seqs_per_batch` | `int32` | `200` | 每个 batch 可处理的最大 sequence 数。 |
| `enable_schedule_overlap` | `bool` | `false` | 是否启用 schedule overlap（异步调度）；详见 [异步调度](/zh/features/async_schedule/)。 |
| `prefill_scheduling_memory_usage_threshold` | `double` | `0.95` | prefill 调度时的内存使用阈值。 |
| `enable_chunked_prefill` | `bool` | `true` | 是否启用 chunked prefill。 |
| `max_tokens_per_chunk_for_prefill` | `int32` | `-1` | prefill 阶段每个 chunk 的最大 token 数；`-1` 表示使用默认策略。 |
| `chunked_match_frequency` | `int32` | `2` | sequence prefix cache 匹配频率。 |
| `use_zero_evict` | `bool` | `false` | 是否使用 ZeroEvictionScheduler；详见 [Zero Evict Scheduler](/zh/features/zero_evict_scheduler/)。 |
| `max_decode_token_per_sequence` | `int32` | `256` | ZeroEvictionScheduler 中每个 sequence 的最大 decode token 数。 |
| `priority_strategy` | `string` | `"fcfs"` | 请求优先级策略，例如 `fcfs`、`priority`、`deadline`。 |
| `enable_mix_batch` | `bool` | `true` | 是否在同一 batch 中运行 prefill 和 decode。启用 CP 或 MTP 时会被强制设为 `false`。 |
| `aggressive_coeff` | `double` | `1.0` | MixScheduler 紧急度判断的激进系数。 |
| `starve_threshold` | `double` | `1.0` | MixScheduler 的饥饿阈值系数。 |
| `enable_starve_prevent` | `bool` | `true` | 是否启用 MixScheduler 的防饥饿机制。 |
| `enable_dp_fair_token_budget` | `bool` | `true` | 是否在 PD 分离的 PREFILL 实例上把每个 DP group 每轮调度的 token 数限制为 `max_tokens_per_batch / dp_size`（下限为一个 prefill chunk）。 |
| `request_queue_size` | `int32` | `100000` | 调度器请求队列容量；队列满时新到的请求会在准入阶段被拒绝。 |

## ParallelConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `dp_size` | `int32` | `1` | MLA attention 的 data parallel 规模。 |
| `ep_size` | `int32` | `1` | MoE 模型的 expert parallel 规模。 |
| `cp_size` | `int32` | `1` | Context parallel 规模，需要后端和模型支持；设为 `1` 表示禁用 context parallel。|
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
| `expert_parallel_degree` | `int32` | `0` | Expert parallel degree。 |
| `rank_tablefile` | `string` | `""` | ATB HCCL rank table 文件。 |

## DistributedConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `master_node_addr` | `string` | `"127.0.0.1:19888"` | 多机分布式服务的 master 地址，例如 `10.18.1.1:9999`。 |
| `worker_memory_master_node_addr` | `string` | `"127.0.0.1:19889"` | rank 0 上收集 worker memory RPC 服务地址的注册地址，例如 `10.18.1.1:9999`；仅启用虚拟内存且 `nnodes > 1` 时使用。 |
| `nnodes` | `int32` | `1` | 多机节点数量。 |
| `node_rank` | `int32` | `0` | 当前节点 rank。 |
| `etcd_addr` | `string` | `""` | 保存实例元信息的 etcd 地址。 |
| `etcd_namespace` | `string` | `""` | xLLM etcd key 使用的可选 namespace 前缀，例如 `prod-a`。 |
| `enable_service_routing` | `bool` | `false` | 是否启用 xLLM service routing。 |
| `xservice_rpc_protocol` | `string` | `"baidu_std"` | `XServiceClient` 与服务层通信使用的协议，支持 `baidu_std` 和 `http`。不改变 P/D 引擎间的 RPC 和 KV 传输协议。 |
| `xservice_address` | `string` | `""` | HTTP 服务层的直连控制地址（`host:port`），用于实例注册、心跳和元信息查询。要求 `xservice_rpc_protocol=http`，与 `etcd_addr` 互斥。具体请求的结果回传使用请求中携带的服务层地址。 |
| `heart_beat_interval` | `double` | `0.5` | 心跳间隔。 |
| `etcd_ttl` | `int32` | `3` | etcd key 的 TTL。 |

## DisaggPDConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_disagg_pd` | `bool` | `false` | 是否启用 Prefill-Decode 分离执行；详见 [PD 分离](/zh/features/disagg_pd/)。 |
| `disagg_pd_port` | `int32` | `7777` | PD 分离 bRPC server 的监听端口。 |
| `instance_role` | `string` | `"DEFAULT"` | 当前实例角色，例如 `DEFAULT`、`PREFILL`、`DECODE`、`MIX`。 |
| `kv_cache_transfer_mode` | `string` | `"PUSH"` | KV Cache 传输模式，例如 `PUSH`、`PULL`。 |
| `transfer_listen_port` | `int32` | `26000` | KV Cache Transfer 的监听端口。 |
| `kv_push_dst_rotate` | `bool` | `false` | 在 `push_kv_blocks` 中按 KV-split rank 轮转遍历目标 worker，用于分散对 decode worker 的流量。 |

## SpeculativeConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `draft_model` | `string` | `""` | draft 模型路径；MTP 使用方式详见 [MTP](/zh/features/mtp/)。 |
| `num_speculative_tokens` | `int32` | `0` | 每轮 speculative decoding 生成的 speculative token 数。 |
| `speculative_algorithm` | `string` | `"MTP"` | Speculative decoding 算法，支持 `MTP`、`Eagle3`、`Suffix`、`DFlash`、`DFlash2`（仅 NPU）、`DSpark`。 |
| `speculative_suffix_cache_max_depth` | `int32` | `64` | Suffix speculative decoding 的后缀树最大深度。 |
| `speculative_suffix_max_spec_factor` | `double` | `1.0` | Suffix speculation 相对于匹配长度的最大 token 系数。 |
| `speculative_suffix_max_spec_offset` | `double` | `0.0` | Suffix speculation 的最大 token 加性偏移。 |
| `speculative_suffix_min_token_prob` | `double` | `0.1` | Suffix speculation 使用的最小 token 概率。 |
| `speculative_suffix_max_cached_requests` | `int32` | `-1` | Suffix speculation 全局最大缓存请求数；`-1` 表示不限，`0` 表示禁用。 |
| `speculative_suffix_use_tree_spec` | `bool` | `false` | 是否使用 tree-based suffix speculation，而不是 path speculation。 |
| `enable_probabilistic_draft` | `bool` | false | draft 从自身分布采样 proposal（携带 q 供拒绝采样）。默认为 greedy argmax。仅 `MTP`、`DSpark`、`DFlash2`、`Eagle3` 支持。 |
| `enable_atb_spec_kernel` | `bool` | `false` | 是否使用 ATB speculative kernel。 |
| `enable_adaptive_target_verify` | `bool` | `false` | 根据历史 confidence 和完整轮成本，调整已经生成的 draft 候选中交给 target 验证的数量，不改变 draft 生成宽度。支持 Python ACL graph、task pipeline 下的 DSpark 和 MTP greedy Draft5/7。MTP 使用选中 token 的 logits 概率，要求普通 graph 和 DP=CP=1；两者普通 graph 均支持 greedy、随机或混合 target，draft 必须为 greedy；DSpark unified graph 要求 greedy target。见[自适应投机推理](/zh/features/adaptive_speculation/)。 |
| `enable_adaptive_draft_step` | `bool` | `false` | 调整一次 target 验证前实际执行的顺序 draft forward 步数，以最大化预测的完整轮吞吐。适合自回归生成候选的路径，当前接入 MTP；`num_speculative_tokens` 是最大步数（1–64），而非当前实际步数。要求 Python GLM DSA、普通预热 ACL graph、异步 task pipeline、DP=CP=1；不支持 unified MTP 和 PD 分离。见[自适应投机推理](/zh/features/adaptive_speculation/)。 |
| `adaptive_draft_step_config` | `string` | `""` | 可选 JSON 配置文件的路径：`policy` 配置接受率窗口、更新间隔与切换门槛，`calibration` 配置启动采样与拟合；不是成本表输入。不传时使用默认值并打印日志，仍在就绪前实测当前 MTP 服务各候选步数的完整轮成本，复用 SPS/二维加性拟合代码，不复用 DSpark 的实测值。见[自适应投机推理](/zh/features/adaptive_speculation/)及下文示例。 |
| `adaptive_target_verify_min_gain` | `double` | `0.0` | DSpark 改变上一个有效验证预算时，要求的最小预测相对吞吐收益。 |
| `adaptive_target_verify_calibration` | `string` | `""` | 可选的启动标定 JSON 文件，`model` 支持 `sps`、`additive` 或 `graph`；为空时使用内置默认配置，两种情况都会打印实际生效值。不能与已保存成本表输入同时指定。 |
| `adaptive_target_verify_cost_table` | `string` | `""` | 加载兼容的已保存完整步成本表，跳过启动采样和拟合；支持 SPS、加性模型和实测 `graph` 成本；graph 成本需要带兼容性信息的版本化 bundle，仍需 graph warmup。 |
| `adaptive_target_verify_cost_table_output` | `string` | `""` | 可选的启动标定结果输出文件，保存为带版本和配置身份的 bundle；为空则只保留在内存，绝不覆盖已有文件。加载已保存表时不能指定输出。 |

### 吞吐感知的 draft 步数选择

`enable_adaptive_draft_step` 调整 draft 模型实际执行的步数，
`enable_adaptive_target_verify` 则在 target 验证前裁剪候选。
两个开关按执行能力命名，不绑定算法名称。共享的 Host 成本模型、预算规划、
启动标定和稳态窗口采集统一放在 `core/framework/speculative/adaptive/`，
使用 `Spec*` 类型；步数选择器复用这套基础设施，但独立测量自身的成本。
两者分别控制不同阶段：当前前者支持 MTP，后者支持 DSpark Draft5/7，
暂不支持同时开启。

原来的 `enable_adaptive_mtp`、`adaptive_mtp_config`、
`enable_adaptive_spec_prune` 分别改为 `enable_adaptive_draft_step`、
`adaptive_draft_step_config`、`enable_adaptive_target_verify`。
命令行和 JSON 配置均需更新；旧 JSON 字段会明确报错并提示迁移，
不会静默忽略而关闭功能。

target 验证相关配置也统一按能力命名：

| 原名称 | 新名称 |
| --- | --- |
| `adaptive_dspark_calibration` | `adaptive_target_verify_calibration` |
| `adaptive_dspark_cost_table` | `adaptive_target_verify_cost_table` |
| `adaptive_dspark_cost_table_output` | `adaptive_target_verify_cost_table_output` |
| `adaptive_speculative_min_gain` | `adaptive_target_verify_min_gain` |

命令行参数和 JSON 字段需一起迁移。旧 flag 不再注册，旧 JSON 字段会明确提示新名称。
已保存成本表的 JSON 格式及其算法、后端兼容身份保持不变。
`SpecCostModel` 提供一维 SPS／二维加性成本，`SpecCostProfileStore` 按物理请求桶
与 KV 覆盖范围保存成本，`VerifyBudgetPlanner` 负责验证预算规划。
具体 draft 的 token 布局和 kernel 宽度限制保留在 runtime 接入层，不放进共享成本存储。

启动参数示例：

```bash
--model_impl=python --speculative_algorithm=MTP \
--num_speculative_tokens=3 --enable_task_pipeline=true \
--enable_graph=true --enable_unified_mtp_graph=false \
--enable_schedule_overlap=true --enable_adaptive_draft_step=true \
--adaptive_draft_step_config=mtp_profile.json
```

另外指定正常的 target/draft 模型路径和并行参数。可选 JSON 只指定采样方式，
不提供成本数值。不传 `adaptive_draft_step_config` 时使用默认策略与默认采样配置，
两种情况都会打印实际生效值。以下示例要求图覆盖请求桶 1、2、4、8：

```json
{
  "version": 1,
  "policy": {
    "window_size": 20, "update_interval": 5,
    "switch_min_gain": 0.05
  },
  "calibration": {
    "model": "additive", "request_buckets": [1, 2, 4, 8],
    "contexts": [{"prefill_tokens": 2048, "min_kv_length": 1, "max_kv_length": 4096}],
    "warmup_steps": 8, "window_steps": 32,
    "windows_per_point": 3, "token_seed": 1
  }
}
```

启动时复用 DSpark 的内部请求驱动和稳态结果栅栏窗口，实际执行 MTP
prefill/decode，无需重启服务或构造 HTTP 客户端。每个候选步数独立测量、拟合
SPS（`model: "sps"`）或二维加性模型。二维轴分别是第一步 draft 的物理请求
容量和 target 的 padding 后行数；同时测量合法的更大 target 桶以区分两个轴。
成本包含逐步 draft、target 验证、采样、prefix refresh 和正常结果回收。
拟合结果保留在 engine 内存，日志记录实际生效配置与标定进度；不要求现成成本
JSON，更不套用 DSpark 的实测数值。默认使用二维加性模型、实际捕获的请求桶（最多 128）、
最多 2048 的 prefill、最多 4096 的 KV，并预留最坏接受长度的空间。
显式配置必须落在真实捕获的图和 KV 容量内；覆盖不足时启动失败，不猜测成本。
遍历候选步数和图桶会增加启动时间，不引入 NPU Event 测时。

engine 从正常回收的结果统计逐位置前缀接受概率
`p[j] = P(接受的 draft 前缀长度 > j)`，选择使
`B * (1 + sum(p[:步数])) / cost_ms` 最大的步数。每个回收的验证 batch
向各位置窗口记录一次“该 batch 中接受完整前缀的请求比例”，不同 batch
等权，不因请求数较多而增加权重。默认窗口为最近 20 个已完成 batch，
在各 batch size 间共享。当前深度的所有位置都填满窗口后，每完成 5 个 batch
重新评分，切换要求超过 5% 的预测收益。未验证的位置不记作拒绝。
降步时清空更深位置的窗口；缺少历史的位置按已知前缀概率做几何衰减外推，
只有一个已知位置时延续该概率。不再周期性强制运行最大深度。
旧策略字段 `min_observations`、`exploration_interval` 不再接受，
改用 `window_size` 与 `update_interval` 配置。

不新增 confidence D2H 或 TP 决策广播；异步模式使用已有结果回收中的历史，
不修改已提交 task。CPU 侧记录各在途 batch 的统计版本，防止降步前的旧结果
重新填回已清空的深层窗口。冷启动、成本覆盖不足或超出已预热 graph
请求容量时，按明确策略使用配置的最大步数，启动时打印实际生效配置。
启动预热覆盖各候选步数，Slot 保留最大容量的存储，不以在线重新捕获实现切换。
开关默认关闭，不影响默认的固定步数 MTP。

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
| `enable_online_profile` | `bool` | `false` | 是否启用在线 timeline 接口（`/start_profile` 和 `/stop_profile`），支持 Ascend NPU 及文档中列出的 profiling 后端。 |
| `profile_backend` | `string` | `"torch"` | 在线 profiling 后端。`torch`（默认）在 Ascend 上通过 torch_npu 采集 CPU+NPU，在其他支持的平台通过 Kineto 采集 CPU+CUDA；`ascend` 采集 CANN 原始数据，使用 msprof 导出；`cuda` 控制外部 nsys 的 capture range。部署限制见[在线服务性能分析](/zh/dev_guide/online_profiling/)。 |
| `profile_dir` | `string` | `""` | `torch` 和 `ascend` 后端的 trace 输出目录；为空表示当前工作目录。 |

## ExecutionConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `extension_impl` | `string` | `""` | 为模型、执行器等注册组件选择原生扩展实现族；空值保留原有选择方式。所选实现族必须已注册；该实现族未注册的组件继续使用原有选择方式。 |
| `enable_graph` | `bool` | `false` | 是否在 decode 阶段启用图执行以降低 kernel launch 开销和设备空闲时间；支持 CUDA Graph、ACL Graph（NPU）、MLU Graph 和 DCU Graph，详见 [图执行](/zh/features/graph_mode/)。 |
| `enable_unified_mtp_graph` | `bool` | `false` | 将支持的 NPU speculative decode 请求交由统一 Python graph 执行。支持 GLM DSA MTP，以及以 GLM DSA 为目标模型的 `DFlash`/`DFlash2`/`DSpark`。要求 `enable_graph=true`；`DFlash`/`DFlash2`/`DSpark` 还要求 `enable_task_pipeline=true`。关闭此开关不影响普通模型的图执行。 |
| `enable_graph_double_buffer` | `bool` | `true` | 是否为 NPU schedule-overlap decode 启用双缓冲 ACL graph 持久化参数与 graph 实例。 |
| `enable_graph_mode_decode_no_padding` | `bool` | `false` | decode 阶段是否按实际 `num_tokens` 捕获 graph，而不是按 padding 后 shape 捕获。 |
| `enable_prefill_piecewise_graph` | `bool` | `false` | 是否在 prefill 阶段启用 piecewise CUDA graph；attention 使用 eager 模式，其他操作捕获进 CUDA graph。 |
| `enable_graph_vmm_pool` | `bool` | `true` | 是否启用 VMM-backed CUDA graph memory pool，用于多 shape graph 的显存复用。 |
| `max_tokens_for_graph_mode` | `int32` | `2048` | 图执行最大 token 数；`0` 表示不限制。 |
| `acl_graph_decode_batch_size_limit` | `int32` | `16` | NPU 上 ACL graph 的 decode batch size 阈值；实际 decode batch size 超过该值时，ACL graph decode 会回退到 eager 模式以避免 OOM。 |
| `block_draft_graph_warmup_max_kv_len` | `int32` | `0` | 仅为 Python block draft 的独立 ACL graph（DSpark/DFlash2，包括 GLM MLA draft）预捕获按 2 的幂增长的 KV 容量桶。`0` 覆盖完整模型上下文；正数预捕获到覆盖该长度的容量桶；`-1` 保留按需捕获。已有 warmup 必需的形状始终保留，即使超过正数上限。超过预捕获上界的请求仍可在线扩桶。更多容量桶会增加启动时间和 draft 图显存占用；不改变 target、MTP、context-write 或可选整轮 unified graph 的预热策略。 |
| `enable_shm` | `bool` | `false` | 是否为模型执行启用共享内存。 |
| `use_contiguous_input_buffer` | `bool` | `true` | 是否使用连续的 device input buffer 执行模型。 |
| `input_shm_size` | `uint64` | `1024` | 输入共享内存大小，默认 1GB。 |
| `output_shm_size` | `uint64` | `128` | 输出共享内存大小，默认 128MB。 |
| `random_seed` | `int32` | `-1` | 随机数生成器 seed；`-1` 表示不固定 seed。 |
| `enable_dsa_multi_stream` | `bool` | `false` | 为 Python GLM indexer 启用局部并行：K/weights 投影在辅助流执行，query/cache 行相同时保留融合投影；主流同步后再更新 index cache。非 interleaved RoPE 还可并行 Q 投影/RoPE 与 cache 准备。主 attention 准备完成后才执行 indexer。要求模型后端和设备流能力均受支持。 |
| `enable_attn_dp_weight_sharding` | `bool` | `false` | 在 Python GLM MLA 中，将依赖 attention head 的权重按 TP × DP 切分。要求 DP > 1、CP = 1、layerwise split = 1，attention head 数能被 TP × DP 整除。每个 token owner 仍执行普通 TP 的本地 heads，每层使用三次通信；MoE 使用独立的 MoE TP/EP 分组。 |

## KernelConfig

| 参数名称 | 类型 | 默认值 | 参数含义 |
|:---------|:-----|:-------|:---------|
| `enable_customize_mla_kernel` | `bool` | `false` | 是否启用自定义 MLA kernel。 |
| `npu_kernel_backend` | `string` | `"AUTO"` | NPU kernel 后端，支持 `AUTO`、`ATB`、`TORCH`、`XLITE`。 |
| `enable_intralayer_addnorm` | `bool` | `false` | 是否启用 fused intralayer addnorm ops。 |
| `enable_fused_mc2` | `int32` | `-1` | NPU 的 Fused MC2 模式；`-1` 使用自动默认值，`0` 禁用 fused MC2，正值启用 dense matmul-allreduce，`1` 对 MoE 使用 DispatchFFNCombine，`2` 对 MoE 使用 DispatchGmmCombineDecode。 |
| `enable_interlayer_addnorm` | `bool` | `false` | 是否启用 fused interlayer addnorm ops。 |
| `enable_split_rmsnorm_rope` | `bool` | `false` | 是否启用 fused split rmsnorm rope ops。 |
| `enable_aclnn_matmul` | `bool` | `false` | 是否为支持的 NPU ATB layer 启用 ACLNN matmul 后端。 |
| `enable_aclnn_swiglu` | `bool` | `false` | 是否为支持的 NPU ATB layer 启用 ACLNN SwiGLU 后端。 |
| `enable_dspark_native_sas` | `bool` | `false` | 启用 NPU DSpark 原生 SparseAttnSharedkv 语义。旧版算子若不接受非空 `ori_sparse_indices`，可能在 tiling 阶段终止进程；请保持关闭以使用 q_len=1 兼容模式。 |

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
| `dit_h3_task_type` | `string` | `"fl2va"` | MiniMax-H3 任务类型：`fl2va` 或 `ref2va`。 |
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
| `rec_worker_max_concurrency` | `uint32` | `1` | Rec worker 并行执行并发度；小于等于 `1` 表示禁用并发 Rec worker。 |
