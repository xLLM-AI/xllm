---
title: "Service Startup Parameters"
sidebar:
  order: 100
---

xLLM uses gflags to manage service startup parameters. `--model <PATH>` is the only required flag. For native configuration, explicitly supplied command-line flags override JSON configuration values, which override compiled defaults. The tables below are grouped by the Config classes in `/xllm/core/framework/config`, with one Config per section. The `ConfigJsonUtils` section contains the common JSON config-file flags.

## ConfigJsonUtils

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `config_json_file` | `string` | `""` | Path to a JSON config file. For native configuration, explicit command-line flags take precedence over file values. |
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
| `enable_return_embedding_modality_tags` | `bool` | `false` | Whether full sequence embeddings include per-token modality tags. |
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
| `enable_manual_loader` | `bool` | `false` | Pin decoder layer weights to host memory and use async H2D transfer. Required by `enable_rolling_load`; also implied by `enable_vmm_weights`. |
| `enable_vmm_weights` | `bool` | `false` | Enable VMM for model weights and implicitly enable the manual loader. Independent of `enable_vmm_kvcache`. NPU only. |
| `enable_rolling_load` | `bool` | `false` | Enable rolling weight load: keep only N decoder layer weight slots in HBM and stream-load each layer just in time. Requires `enable_manual_loader=true`. NPU only. |
| `rolling_load_num_cached_layers` | `int32` | `2` | Number of decoder layer weight slots to keep in HBM when `enable_rolling_load=true`. |
| `rolling_load_num_rolling_slots` | `int32` | `-1` | Number of rolling slots used by decoder rolling load. Fixed slots are `rolling_load_num_cached_layers - rolling_load_num_rolling_slots`. `-1` means auto, `min(2, preload_count)`. Must be in `[-1, rolling_load_num_cached_layers]`. |
| `enable_prefetch_weight` | `bool` | `false` | Whether to enable weight prefetching. Only applies to Qwen3-dense models. The default gateup weight prefetch ratio is 40%; adjust with `PREFETCH_COEFFOCIENT`. |

## KVCacheConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `block_size` | `int32` | `128` | Number of slots per KV Cache block. |
| `max_cache_size` | `int64` | `0` | Maximum KV Cache size in bytes. With virtual memory, caps the shared KV/weight pool. `0` leaves the explicit byte ceiling unset and derives capacity from available memory. |
| `max_memory_utilization` | `double` | `0.8` | Fraction of device memory used for model inference, including model weights and KV Cache. Must be in `(0, 1]` for physical KV Cache and fixed VMM. Elastic VMM uses this ceiling only when explicitly configured; omitting it or setting `0` allows growth according to live available memory. |
| `kv_cache_dtype` | `string` | `"auto"` | KV Cache dtype for quantization. `auto` aligns with model dtype and disables quantization. `int8` enables INT8 quantization and is only supported on the MLU backend. |
| `indexer_cache_dtype` | `string` | `"auto"` | Indexer cache dtype for models with an indexer cache. Supported values are `auto` and `int8`. `auto` aligns with model dtype and disables indexer cache quantization. `int8` enables INT8 indexer cache quantization. |
| `enable_prefix_cache` | `bool` | `true` | Whether to enable prefix cache in the block manager. See [Prefix Cache](/en/features/prefix_cache/). |
| `kv_events_config` | `string` | `""` | KV residency event publisher configuration as a JSON object string. An empty string disables reporting. See the fields and example below. |
| `enable_in_batch_prefix_cache` | `bool` | `false` | Whether to cache admitted prefill full blocks into the prefix cache so that later requests in the same batch can share them. |
| `max_linear_state_cache_slots` | `int64` | `0` | Maximum number of active linear-attention state cache slots. `0` derives an automatic capacity from the available KV Cache budget. |
| `xxh3_128bits_seed` | `uint32` | `1024` | Default XXH3 128-bit hash seed. |
| `enable_vmm_kvcache` | `bool` | `false` | Enable VMM for KV Cache. Independent of `enable_vmm_weights`; `false` uses the original physical KV Cache implementation. NPU only. |
| `vmm_elastic` | `bool` | `true` | When either VMM flag is enabled, grow active logical slots and physical backing on demand, and release idle backing. `false` uses a fixed resident pool. |
| `vmm_memory_reserve_bytes` | `int64` | `0` | Additional device memory to keep free when elastic virtual memory grows, in bytes. Supplements runtime execution headroom; must be nonnegative. |
| `vmm_startup_timeout_ms` | `int32` | `120000` | Positive timeout in milliseconds applied separately to the master's node-registration wait and each non-master's physical-pool readiness wait during VMM startup. Independent of per-request RPC timeouts; each wait has its own timer. |
| `phy_page_granularity_size` | `int64` | `2097152` | Granularity size of one physical page in bytes, default 2 MiB, for continuous KV Cache. |

For VMM usage and capacity guidance, see [Virtual Memory for KV Cache and Model Weights](/en/features/virtual_memory/).

### kv_events_config

The following fields belong to the JSON object in `kv_events_config`:

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `publisher` | `string` | `"zmq"` | Publisher type. Currently only `zmq` is supported. |
| `endpoint` | `string` | Required | Base ZeroMQ publish address for vLLM-style KV events, for example `tcp://*:5557`. Must be a nonempty TCP address. |
| `replay_endpoint` | `string` | `""` | Base ZeroMQ address for snapshot recovery requests, for example `tcp://*:6557`. An empty string disables this endpoint. Recovery replies contain the current full snapshot and its matching live stream sequence. |
| `motor_pool_endpoint` | `string` | `""` | Base ZeroMQ publish address for Motor local Host copy confirmation events, for example `tcp://*:7557`. An empty string disables this stream. |
| `topic` | `string` | `""` | Published ZeroMQ topic. Consumer subscription filters must match this topic. |
| `hwm` | `int32` | `10000` | ZeroMQ send queue high-water mark in messages. Must be positive. |
| `snapshot_interval_seconds` | `int32` | `60` | Interval between periodic full residency snapshots, in seconds. Must be positive. |

Each DP rank uses the base TCP port plus `dp_rank`. The event, replay and Host confirmation port ranges must not overlap, and every port must be between `1` and `65535`. For example, with DP=2 and `endpoint=tcp://*:5557`, the two ranks publish on `5557` and `5558`.

Add the following flag to an existing launch command to enable event publication and snapshot recovery:

```bash
--kv_events_config='{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:6557"}'
```

Current support covers ordinary non-PD text inference with prefix caching enabled, flat KV layouts and local Host offload. Events describe the Device/Host residency of reusable prefix blocks; Host copies are advertised after successful copy acknowledgements. The publisher runs on a background thread. Multi-resource composite/linear-state layouts, multimodal inference, virtual-memory KV, PD separation and external KV stores are not yet supported.

## KVCacheStoreConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `prefetch_timeout` | `uint32` | `0` | Stops issuing new KV Cache Store prefetch batches after the timeout and waits for in-flight batches; `0` waits indefinitely. |
| `prefetch_batch_size` | `uint32` | `2` | Copy batch size for prefetching from KV Cache Store. |
| `layers_wise_copy_batchs` | `uint32` | `4` | Number of batches for layer-wise H2D copy. |
| `host_blocks_factor` | `double` | `0.0` | Host block factor, for example `host block num = host_blocks_factor * hbm block num`. |
| `enable_kvcache_store` | `bool` | `false` | Whether to enable KV Cache Store. |
| `store_protocol` | `string` | `"tcp"` | KV Cache Store protocol, for example `tcp` or `rdma`. |
| `store_rdma_devices` | `string` | `""` | Comma-separated RDMA HCAs for the embedded xLLM Store client. Empty means Mooncake auto-discovery. |
| `store_master_server_address` | `string` | `""` | Store master address. Use `IP:Port` in standalone mode or `etcd://IP:Port;IP:Port;...` in etcd-backed HA mode. |
| `store_metadata_server` | `string` | `""` | Address of the KV Cache Store metadata service. |
| `store_local_hostname` | `string` | `""` | Local host name of the KV Cache Store client. |
| `enable_control_h2d_block_num` | `bool` | `false` | Whether to control the number of H2D copy blocks. |

## BeamSearchConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_beam_search_kernel` | `bool` | `false` | Whether to enable the beam search kernel. |
| `beam_width` | `int32` | `1` | Beam width for beam search. |
| `enable_block_copy_kernel` | `bool` | `true` (NPU/CUDA/MUSA/DCU); `false` (other backends) | Whether to use the block copy kernel on supported backends. |
| `enable_topk_sorted` | `bool` | `true` | Whether to enable sorted top-k output. |

## SchedulerConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `max_tokens_per_batch` | `int32` | `10240` | Maximum number of tokens per batch. |
| `max_seqs_per_batch` | `int32` | `200` | Maximum number of sequences per batch. |
| `enable_schedule_overlap` | `bool` | `false` | Whether to enable schedule overlap, also known as asynchronous scheduling. See [Async Scheduling](/en/features/async_schedule/). |
| `prefill_scheduling_memory_usage_threshold` | `double` | `0.95` | Memory usage threshold during prefill scheduling. |
| `enable_chunked_prefill` | `bool` | `true` | Whether to enable chunked prefill. |
| `max_tokens_per_chunk_for_prefill` | `int32` | `-1` | Maximum number of tokens per chunk in the prefill stage. `-1` uses the default policy. |
| `chunked_match_frequency` | `int32` | `2` | Sequence prefix-cache match frequency. |
| `use_zero_evict` | `bool` | `false` | Whether to use ZeroEvictionScheduler. See [Zero Evict Scheduler](/en/features/zero_evict_scheduler/). |
| `max_decode_token_per_sequence` | `int32` | `256` | Maximum decode tokens per sequence for ZeroEvictionScheduler. |
| `priority_strategy` | `string` | `"fcfs"` | Request priority strategy, for example `fcfs`, `priority`, or `deadline`. |
| `enable_mix_batch` | `bool` | `true` | Whether to run prefill and decode in the same batch. Forced to `false` when CP or MTP is active. |
| `aggressive_coeff` | `double` | `1.0` | Aggressive coefficient for MixScheduler urgency judgment. |
| `starve_threshold` | `double` | `1.0` | Starvation threshold coefficient for MixScheduler. |
| `enable_starve_prevent` | `bool` | `true` | Whether to enable anti-starvation behavior in MixScheduler. |
| `enable_dp_fair_token_budget` | `bool` | `true` | Whether to cap each DP group at `max_tokens_per_batch / dp_size` tokens per scheduling round on disaggregated PD PREFILL instances, floored at one prefill chunk. |
| `request_queue_size` | `int32` | `100000` | Capacity of the scheduler request queue. Requests arriving while the queue is full are rejected at admission. |

## ParallelConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `dp_size` | `int32` | `1` | Data parallel size for MLA attention. |
| `ep_size` | `int32` | `1` | Expert parallel size for MoE models. |
| `cp_size` | `int32` | `1` | Context parallel size. Backend and model support is required; `1` disables context parallelism. |
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
| `expert_parallel_degree` | `int32` | `0` | Expert parallel degree. |
| `rank_tablefile` | `string` | `""` | ATB HCCL rank table file. |

## DistributedConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `master_node_addr` | `string` | `"127.0.0.1:19888"` | Master address for multi-node distributed serving, for example `10.18.1.1:9999`. |
| `worker_memory_master_node_addr` | `string` | `"127.0.0.1:19889"` | Rank 0 address where worker memory RPC servers register their endpoints, for example `10.18.1.1:9999`. Used only when virtual memory is enabled and `nnodes > 1`. |
| `nnodes` | `int32` | `1` | Number of multi-node nodes. |
| `node_rank` | `int32` | `0` | Rank of the current node. |
| `etcd_addr` | `string` | `""` | etcd address used to save instance metadata. |
| `etcd_namespace` | `string` | `""` | Optional etcd namespace prefix for all xLLM keys, for example `prod-a`. |
| `enable_service_routing` | `bool` | `false` | Whether to enable xLLM service routing. |
| `xservice_rpc_protocol` | `string` | `"baidu_std"` | Protocol used by `XServiceClient` to communicate with the service layer: `baidu_std` or `http`. Does not change P/D engine RPC or KV transfer protocols. |
| `xservice_address` | `string` | `""` | Direct HTTP service control address (`host:port`) for instance registration, heartbeats and metadata queries. Requires `xservice_rpc_protocol=http` and is mutually exclusive with `etcd_addr`. Per-request result callbacks use the service address supplied in the request. |
| `heart_beat_interval` | `double` | `0.5` | Heartbeat interval. |
| `etcd_ttl` | `int32` | `3` | Time to live for etcd keys. |

## DisaggPDConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_disagg_pd` | `bool` | `false` | Whether to enable disaggregated prefill and decode execution. See [P-D Separation](/en/features/disagg_pd/). |
| `disagg_pd_port` | `int32` | `7777` | Listening port for the disaggregated PD bRPC server. |
| `instance_role` | `string` | `"DEFAULT"` | Instance role, for example `DEFAULT`, `PREFILL`, `DECODE`, or `MIX`. |
| `kv_cache_transfer_mode` | `string` | `"PUSH"` | KV Cache transfer mode, for example `PUSH` or `PULL`. |
| `transfer_listen_port` | `int32` | `26000` | Listening port for KV Cache Transfer. |
| `kv_push_dst_rotate` | `bool` | `false` | Rotate the destination-worker traversal order in `push_kv_blocks` per KV-split rank to spread incast across decode workers. |

## SpeculativeConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `draft_model` | `string` | `""` | Draft model path. See [MTP](/en/features/mtp/) for MTP usage. |
| `num_speculative_tokens` | `int32` | `0` | Number of speculative tokens generated per speculative decoding step. |
| `speculative_algorithm` | `string` | `"MTP"` | Speculative decoding algorithm. Supported values: `MTP`, `Eagle3`, `Suffix`, `DFlash`, `DFlash2` (NPU only), `DSpark`. |
| `speculative_suffix_cache_max_depth` | `int32` | `64` | Maximum suffix-tree depth for suffix speculative decoding. |
| `speculative_suffix_max_spec_factor` | `double` | `1.0` | Maximum suffix speculation token factor relative to match length. |
| `speculative_suffix_max_spec_offset` | `double` | `0.0` | Maximum additive token offset for suffix speculation. |
| `speculative_suffix_min_token_prob` | `double` | `0.1` | Minimum token probability used in suffix speculation. |
| `speculative_suffix_max_cached_requests` | `int32` | `-1` | Maximum number of globally cached requests for suffix speculation. `-1` means unlimited; `0` disables it. |
| `speculative_suffix_use_tree_spec` | `bool` | `false` | Whether to use tree-based suffix speculation instead of path speculation. |
| `enable_probabilistic_draft` | `bool` | false | Drafts sample proposals from their distribution (emitting q for rejection sampling). The default is greedy argmax. Only `MTP`, `DSpark`, `DFlash2`, `Eagle3` support it. |
| `enable_atb_spec_kernel` | `bool` | `false` | Whether to use the ATB speculative kernel. |
| `enable_adaptive_target_verify` | `bool` | `false` | Adapt how many already-generated draft proposals the target verifies, using historical confidence and calibrated full-cycle costs; draft generation width is unchanged. Supports DSpark and MTP greedy Draft5/7 with Python ACL graphs and task pipeline. MTP uses selected-token logits probabilities and requires ordinary graphs and DP=CP=1. Both ordinary-graph paths support greedy, random or mixed targets with a greedy draft; DSpark unified graphs require greedy targets. See [Adaptive Speculation](/en/features/adaptive_speculation/). |
| `enable_adaptive_draft_step` | `bool` | `false` | Adapt the actual number of sequential draft forward steps before verification to maximize predicted full-cycle throughput. This suits autoregressive drafts, currently MTP; `num_speculative_tokens` is the maximum (1–64), not the current depth. Requires Python GLM DSA, ordinary prewarmed ACL graphs, async task pipeline and DP=CP=1; unified MTP and PD disaggregation are not supported. See [Adaptive Speculation](/en/features/adaptive_speculation/). |
| `adaptive_draft_step_config` | `string` | `""` | Path to an optional JSON file: `policy` controls the acceptance window, update interval and switch margin; `calibration` controls startup sampling and fitting. It is not a cost-table input. Empty uses logged defaults and still measures the current MTP service's per-depth full-cycle costs before ready, sharing SPS/additive fitting code, not DSpark's measured values. See [Adaptive Speculation](/en/features/adaptive_speculation/) and the example below. |
| `adaptive_target_verify_min_gain` | `double` | `0.0` | Minimum predicted relative throughput gain for changing the DSpark verification budget from the previous eligible plan. |
| `adaptive_target_verify_calibration` | `string` | `""` | Optional startup-calibration JSON file; `model` accepts `sps`, `additive`, or `graph`. Empty selects built-in defaults; the effective settings are always logged. Cannot be combined with a saved-table input. |
| `adaptive_target_verify_cost_table` | `string` | `""` | Load compatible saved full-step costs instead of sampling and fitting. Supports SPS, additive, and exact measured `graph` costs. Graph costs require a versioned bundle with compatibility metadata. Graph warmup still runs. |
| `adaptive_target_verify_cost_table_output` | `string` | `""` | Optionally save startup-calibrated costs as a versioned bundle. Empty keeps costs in memory. An existing output file is never overwritten. Not valid when loading a saved table. |

### Throughput-aware draft-step configuration

`enable_adaptive_draft_step` changes the number of draft forward steps;
`enable_adaptive_target_verify` prunes proposals before target verification.
Both flags name execution capabilities, not algorithms. Shared Host-side
cost models, budget planning, calibration and steady-window collection live
under `core/framework/speculative/adaptive/` and use `Spec*` types; the
draft-step selector uses the same infrastructure with its own measured costs.
They are separate controls: currently the former supports MTP and the latter
supports DSpark Draft5/7. They cannot be enabled together.

The previous names `enable_adaptive_mtp`, `adaptive_mtp_config`, and
`enable_adaptive_spec_prune` are replaced by `enable_adaptive_draft_step`,
`adaptive_draft_step_config`, and `enable_adaptive_target_verify`, respectively.
Update both CLI arguments and JSON configurations; old JSON keys are rejected
with migration instructions instead of silently disabling the feature.

Target-verification settings follow the same capability-oriented names:

| Previous name | Current name |
| --- | --- |
| `adaptive_dspark_calibration` | `adaptive_target_verify_calibration` |
| `adaptive_dspark_cost_table` | `adaptive_target_verify_cost_table` |
| `adaptive_dspark_cost_table_output` | `adaptive_target_verify_cost_table_output` |
| `adaptive_speculative_min_gain` | `adaptive_target_verify_min_gain` |

Update CLI arguments and JSON keys together. Old flags are no longer registered;
old JSON keys report the corresponding replacement. Saved cost-table JSON and
its algorithm/backend compatibility identity are unchanged. `SpecCostModel`
provides SPS/additive costs, `SpecCostProfileStore` indexes physical request
buckets and KV coverage, and `VerifyBudgetPlanner` selects verification budgets.
Backend-specific draft token layouts and supported kernel widths stay at the
runtime boundary rather than in the shared profile store.

Example launch arguments:

```bash
--model_impl=python --speculative_algorithm=MTP \
--num_speculative_tokens=3 --enable_task_pipeline=true \
--enable_graph=true --enable_unified_mtp_graph=false \
--enable_schedule_overlap=true --enable_adaptive_draft_step=true \
--adaptive_draft_step_config=mtp_profile.json
```

Supply the usual target/draft model paths and parallel arguments separately.
The optional JSON controls sampling, not precomputed cost values. Omit
`adaptive_draft_step_config` to use defaults for both sections; the effective settings
are always logged. An example for graphs covering request buckets 1, 2, 4, 8:

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

Startup reuses DSpark's internal request driver and steady result-fence windows,
executing real MTP prefill/decode without restarting a service or sending HTTP
requests. Each depth gets its own measured SPS (`model: "sps"`) or additive
model. The additive axes are physical first-draft request capacity and padded
target rows; valid larger padded target buckets make the two axes identifiable.
Costs include sequential draft forwards, verification, sampling, prefix refresh
and ordinary result retirement. Fitted models are kept in engine memory;
effective settings and calibration progress are logged. No cost JSON is
required or taken from a DSpark run.
Defaults use additive fitting, actual captured request buckets (up to 128),
prefill up to 2048 and KV up to 4096 with worst-case decode headroom. Explicit
sampling must fit the captured shapes and KV capacity; invalid coverage fails
startup rather than silently using guessed costs. Sampling all depths and
captured shapes adds startup time. No NPU timing Events are introduced.

The engine learns `p[j] = P(accepted prefix length > j)` from completed result
metadata and chooses the depth maximizing `B * (1 + sum(p[:depth])) / cost_ms`.
Each retired verify batch contributes its fraction of requests accepting the
whole prefix through `j`; batches have equal window weight regardless of size.
The default window holds 20 completed batches, shared across batch sizes.
Rescoring occurs every 5 completed batches once every active position has a
full window, with a 5% minimum predicted gain for a switch. Positions beyond
the verified depth are unobserved, not rejections. A depth decrease clears
deeper windows; missing rates are geometrically extrapolated from the known
prefix rates (a single known position supplies a flat tail). There is no
periodic forced maximum-depth exploration. The former `min_observations` and
`exploration_interval` policy keys are rejected; configure `window_size` and
`update_interval` instead.

No confidence D2H or additional TP decision broadcast is introduced. The
normal async result harvest supplies historical feedback; an already-submitted
task is never modified. CPU-side dispatch epochs prevent results from older
in-flight batches from refilling windows cleared by a depth decrease. Cold
start, missing cost coverage and requests outside warmed graph capacity use
the configured maximum depth. The effective policy is logged at startup.
Graphs warm all candidate depths using maximum-capacity Slot storage, so step
selection does not deliberately trigger serving-time recapture. The enabling
flag defaults to false and leaves fixed-width MTP unchanged when disabled.

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
| `enable_online_profile` | `bool` | `false` | Enable the online timeline endpoints (`/start_profile`, `/stop_profile`). Supports Ascend NPU and the documented profiling backends. |
| `profile_backend` | `string` | `"torch"` | Online profiling backend. `torch` (default) records CPU+NPU using torch_npu on Ascend, or CPU+CUDA using Kineto. `ascend` records native CANN data for export with msprof. `cuda` toggles the external nsys capture range. See [Profiling a Running Service](/en/dev_guide/online_profiling/) for deployment requirements. |
| `profile_dir` | `string` | `""` | Output directory for `torch` and `ascend` traces. Empty means the current working directory. |

## ExecutionConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `extension_impl` | `string` | `""` | Native extension implementation family for registered components, including models and executors. An empty value preserves normal selection. The selected family must be registered; components without registrations in that family keep their normal selection. |
| `enable_graph` | `bool` | `false` | Whether to enable graph execution for the decode phase to reduce kernel-launch overhead and device idle time. Supports CUDA Graph, ACL Graph (NPU), MLU Graph, and DCU Graph. See [Graph Mode](/en/features/graph_mode/). |
| `enable_unified_mtp_graph` | `bool` | `false` | Route supported NPU speculative decode requests through a unified Python graph. Supports GLM DSA MTP and `DFlash`/`DFlash2`/`DSpark` with a GLM DSA target model. Requires `enable_graph=true`; `DFlash`/`DFlash2`/`DSpark` also require `enable_task_pipeline=true`. Disabling this flag leaves ordinary model graph execution unchanged. |
| `enable_graph_double_buffer` | `bool` | `true` | Whether to enable double-buffered ACL graph persistent params and graph instances for NPU schedule-overlap decode. |
| `enable_graph_mode_decode_no_padding` | `bool` | `false` | Whether decode graph capture uses the actual `num_tokens` instead of a padded shape. |
| `enable_prefill_piecewise_graph` | `bool` | `false` | Whether to enable piecewise CUDA graph for the prefill phase. Attention runs in eager mode while other operations are captured in CUDA graphs. |
| `enable_graph_vmm_pool` | `bool` | `true` | Whether to enable a VMM-backed CUDA graph memory pool for multi-shape graph memory reuse. |
| `max_tokens_for_graph_mode` | `int32` | `2048` | Maximum number of tokens for graph execution. `0` means no limit. |
| `acl_graph_decode_batch_size_limit` | `int32` | `16` | Decode batch size threshold for ACL graph on NPU. When the actual decode batch size exceeds this value, ACL graph decode falls back to eager mode to avoid OOM. |
| `block_draft_graph_warmup_max_kv_len` | `int32` | `0` | Pre-capture power-of-two KV capacity buckets for independent Python block-draft ACL graphs (DSpark/DFlash2, including GLM MLA drafts). `0` covers the full model context; a positive value selects the bucket covering that length; `-1` retains lazy capacity capture. Required existing warmup shapes are always retained, even above a positive limit. Requests beyond the warmup limit can still grow graphs lazily. More buckets increase startup time and draft graph memory. Does not change target, MTP, context-write or optional whole-round unified graph warmup. |
| `enable_shm` | `bool` | `false` | Whether to enable shared memory for model execution. |
| `use_contiguous_input_buffer` | `bool` | `true` | Whether to use a contiguous device input buffer for model execution. |
| `input_shm_size` | `uint64` | `1024` | Input shared-memory size. Default is 1GB. |
| `output_shm_size` | `uint64` | `128` | Output shared-memory size. Default is 128MB. |
| `random_seed` | `int32` | `-1` | Random seed for the random number generator. `-1` means no fixed seed. |
| `enable_dsa_multi_stream` | `bool` | `false` | Local overlap in the Python GLM indexer: K/weights projections run on a side stream and stay fused when query/cache rows match; the main stream joins before updating the index cache. Non-interleaved RoPE additionally overlaps Q projection/RoPE with cache preparation. Main attention preparation finishes before the indexer. Requires a supported model backend and stream-capable device. |
| `enable_attn_dp_weight_sharding` | `bool` | `false` | Shard Python GLM MLA head-dependent attention weights across TP and DP. Requires DP > 1, CP = 1, layerwise split = 1, and attention heads divisible by TP × DP. Attention still executes TP-local heads on each token owner, with three collectives per layer. MoE uses its independent MoE TP/EP groups. |

## KernelConfig

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_customize_mla_kernel` | `bool` | `false` | Whether to enable the customized MLA kernel. |
| `npu_kernel_backend` | `string` | `"AUTO"` | NPU kernel backend. Supported values: `AUTO`, `ATB`, `TORCH`, `XLITE`. |
| `enable_intralayer_addnorm` | `bool` | `false` | Whether to enable fused intralayer addnorm ops. |
| `enable_fused_mc2` | `int32` | `-1` | Fused MC2 mode for NPU. `-1` uses the auto default, `0` disables fused MC2, positive values enable dense matmul-allreduce, `1` uses DispatchFFNCombine for MoE, `2` uses DispatchGmmCombineDecode for MoE. |
| `enable_interlayer_addnorm` | `bool` | `false` | Whether to enable fused interlayer addnorm ops. |
| `enable_split_rmsnorm_rope` | `bool` | `false` | Whether to enable fused split rmsnorm rope ops. |
| `enable_aclnn_matmul` | `bool` | `false` | Whether to enable the ACLNN matmul backend for supported NPU ATB layers. |
| `enable_aclnn_swiglu` | `bool` | `false` | Whether to enable the ACLNN SwiGLU backend for supported NPU ATB layers. |
| `enable_dspark_native_sas` | `bool` | `false` | Enable native NPU DSpark SparseAttnSharedkv semantics. Older operators that reject non-empty `ori_sparse_indices` may terminate during tiling; keep this disabled to use q_len=1 compatibility mode. |

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
| `dit_h3_task_type` | `string` | `"fl2va"` | MiniMax-H3 task type: `fl2va` or `ref2va`. |
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
| `rec_worker_max_concurrency` | `uint32` | `1` | Concurrency for Rec worker parallel execution. Values less than or equal to `1` disable concurrent Rec workers. |
