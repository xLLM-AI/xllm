/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <gflags/gflags.h>

DECLARE_string(config_json_file);

DECLARE_bool(enable_dump_config_json);

DECLARE_string(dump_config_json_file);

DECLARE_string(host);

DECLARE_int32(port);

DECLARE_int32(rpc_idle_timeout_s);

DECLARE_int32(num_threads);

DECLARE_string(model_id);

DECLARE_string(model);

DECLARE_string(backend);

DECLARE_string(model_impl);

DECLARE_string(python_model_path);

DECLARE_string(task);

DECLARE_int32(limit_image_per_prompt);

DECLARE_string(mm_download_headers);

// --- kvcache config ---
DECLARE_int32(block_size);
DECLARE_int64(max_cache_size);
DECLARE_double(max_memory_utilization);
DECLARE_string(kv_cache_dtype);
DECLARE_string(indexer_cache_dtype);
DECLARE_int64(max_linear_state_cache_slots);

DECLARE_bool(enable_prefix_cache);

DECLARE_bool(enable_in_batch_prefix_cache);

DECLARE_int64(max_encoder_cache_size);

DECLARE_uint32(xxh3_128bits_seed);

DECLARE_int32(max_tokens_per_batch);

DECLARE_int32(max_seqs_per_batch);

DECLARE_int32(max_tokens_per_chunk_for_prefill);

// --- speculative inference config ---
DECLARE_string(draft_model);

DECLARE_int32(num_speculative_tokens);

DECLARE_string(speculative_algorithm);

DECLARE_bool(enable_opt_validate_probs);

DECLARE_bool(enable_mtp_draft_body_tp1);

DECLARE_int32(speculative_suffix_cache_max_depth);

DECLARE_double(speculative_suffix_max_spec_factor);

DECLARE_double(speculative_suffix_max_spec_offset);

DECLARE_double(speculative_suffix_min_token_prob);

DECLARE_int32(speculative_suffix_max_cached_requests);

DECLARE_bool(speculative_suffix_use_tree_spec);

DECLARE_int32(num_request_handling_threads);

DECLARE_int32(num_response_handling_threads);

// --- parallel config ---
DECLARE_string(communication_backend);

DECLARE_int32(dp_size);

DECLARE_int32(ep_size);

DECLARE_int32(cp_size);

DECLARE_int64(tp_size);

DECLARE_int64(sp_size);

DECLARE_int64(cfg_size);

DECLARE_int64(vae_size);

DECLARE_bool(enable_mm_encoder_dp);

DECLARE_bool(enable_multi_stream_parallel);

DECLARE_int32(micro_batch_num);

DECLARE_bool(enable_dp_balance);

// --- ep load balance config ---
DECLARE_bool(enable_eplb);

DECLARE_int32(redundant_experts_num);

DECLARE_int64(eplb_update_interval);

DECLARE_double(eplb_update_threshold);

DECLARE_int32(expert_parallel_degree);

DECLARE_string(rank_tablefile);

constexpr int32_t kGraphExecutorLogVerboseLevel = 50;

DECLARE_bool(enable_graph);

DECLARE_bool(enable_graph_double_buffer);

DECLARE_bool(enable_graph_mode_decode_no_padding);

DECLARE_bool(enable_prefill_piecewise_graph);

DECLARE_bool(enable_graph_vmm_pool);

DECLARE_int32(max_tokens_for_graph_mode);

DECLARE_int32(acl_graph_decode_batch_size_limit);

DECLARE_string(python_graph_backend);

DECLARE_bool(enable_chunked_prefill);

DECLARE_string(master_node_addr);

DECLARE_string(xtensor_master_node_addr);

// --- disaggregated prefill and decode config ---
DECLARE_bool(enable_disagg_pd);

DECLARE_bool(enable_pd_ooc);

DECLARE_int32(disagg_pd_port);

DECLARE_string(instance_role);

DECLARE_string(kv_cache_transfer_type);

DECLARE_string(kv_cache_transfer_mode);

DECLARE_int32(transfer_listen_port);

DECLARE_int32(nnodes);

DECLARE_int32(node_rank);

DECLARE_int32(max_concurrent_requests);

DECLARE_bool(enable_schedule_overlap);

DECLARE_double(prefill_scheduling_memory_usage_threshold);

DECLARE_int32(max_reconnect_count);

DECLARE_bool(enable_atb_comm_multiprocess);

DECLARE_bool(enable_atb_spec_kernel);

DECLARE_string(etcd_addr);

DECLARE_string(etcd_namespace);

DECLARE_bool(enable_service_routing);

DECLARE_double(heart_beat_interval);

DECLARE_int32(etcd_ttl);

DECLARE_int32(rpc_channel_timeout_ms);

DECLARE_int32(chunked_match_frequency);

DECLARE_bool(use_zero_evict);

DECLARE_int32(max_decode_token_per_sequence);

DECLARE_string(priority_strategy);

DECLARE_bool(enable_mix_batch);

DECLARE_bool(enable_online_preempt_offline);

// --- mix scheduler config ---
DECLARE_double(aggressive_coeff);

DECLARE_double(starve_threshold);

DECLARE_bool(enable_starve_prevent);

// --- kvcache store config ---
DECLARE_uint32(prefetch_timeout);

DECLARE_uint32(prefetch_batch_size);

DECLARE_uint32(layers_wise_copy_batchs);

DECLARE_double(host_blocks_factor);

DECLARE_bool(enable_kvcache_store);

DECLARE_string(store_protocol);

DECLARE_string(store_master_server_address);

DECLARE_string(store_metadata_server);

DECLARE_string(store_local_hostname);

DECLARE_bool(enable_control_h2d_block_num);

DECLARE_bool(enable_profile_step_time);

DECLARE_bool(enable_profile_token_budget);

DECLARE_bool(enable_latency_aware_schedule);

DECLARE_int32(profile_max_prompt_length);

DECLARE_bool(enable_profile_kv_blocks);

DECLARE_bool(disable_ttft_profiling);

DECLARE_bool(enable_forward_interruption);

DECLARE_bool(enable_online_profile);

DECLARE_string(profile_backend);

DECLARE_string(profile_dir);

DECLARE_int32(max_global_ttft_ms);

DECLARE_int32(max_global_tpot_ms);

DECLARE_int32(max_requests_per_batch);

DECLARE_bool(enable_xtensor);

DECLARE_int64(phy_page_granularity_size);

// --- load config ---
DECLARE_bool(enable_manual_loader);

DECLARE_bool(enable_rolling_load);

DECLARE_int32(rolling_load_num_cached_layers);
DECLARE_int32(rolling_load_num_rolling_slots);

DECLARE_bool(enable_prefetch_weight);

// --- beam search config ---
DECLARE_bool(enable_beam_search_kernel);

DECLARE_int32(beam_width);

DECLARE_bool(enable_block_copy_kernel);

DECLARE_bool(enable_topk_sorted);

// --- rec config ---
DECLARE_bool(enable_rec_fast_sampler);

DECLARE_bool(enable_rec_prefill_only);

DECLARE_bool(output_rec_logprobs);

DECLARE_bool(enable_xattention_one_stage);

DECLARE_int32(max_decode_rounds);

DECLARE_bool(enable_constrained_decoding);

DECLARE_int32(request_queue_size);

DECLARE_uint32(rec_worker_max_concurrency);

// --- qwen3 reranker config ---
DECLARE_bool(enable_qwen3_reranker);

DECLARE_string(reasoning_parser);

DECLARE_string(tool_call_parser);

// --- execution config ---
DECLARE_bool(enable_shm);

DECLARE_bool(use_contiguous_input_buffer);

DECLARE_uint64(input_shm_size);

DECLARE_uint64(output_shm_size);

DECLARE_int32(flashinfer_workspace_buffer_size);

DECLARE_int32(random_seed);

DECLARE_string(dit_cache_policy);

DECLARE_int64(dit_cache_warmup_steps);

DECLARE_int64(dit_cache_n_derivatives);

DECLARE_int64(dit_cache_skip_interval_steps);

DECLARE_double(dit_cache_residual_diff_threshold);

DECLARE_bool(enable_convert_tokens_to_item);
DECLARE_bool(enable_output_sku_logprobs);
DECLARE_bool(enable_extended_item_info);
DECLARE_int32(each_conversion_threshold);
DECLARE_int32(total_conversion_threshold);

DECLARE_bool(enable_return_mm_full_embeddings);

DECLARE_int64(dit_cache_start_steps);

DECLARE_int64(dit_cache_end_steps);

DECLARE_int64(dit_cache_start_blocks);

DECLARE_int64(dit_cache_end_blocks);

DECLARE_bool(dit_sp_communication_overlap);

DECLARE_int64(dit_generation_image_area_max);

DECLARE_int64(dit_vae_image_size);

DECLARE_bool(dit_debug_print);

DECLARE_bool(dit_enable_vae_tiling);

DECLARE_bool(dit_sparse_attention_enabled);

DECLARE_double(dit_sparse_attention_sparsity);

DECLARE_int64(dit_sparse_attention_pool_size);

DECLARE_int64(dit_sparse_attention_sparse_start_step);

DECLARE_string(dit_sparse_attention_version);

DECLARE_int64(dit_sparse_attention_mask_refresh_steps);

DECLARE_bool(dit_laser_attention_enabled);

DECLARE_bool(use_audio_in_video);

// --- kernel config ---
#if defined(USE_NPU)
DECLARE_bool(enable_customize_mla_kernel);

DECLARE_string(npu_kernel_backend);

DECLARE_bool(enable_intralayer_addnorm);

DECLARE_int32(enable_fused_mc2);

DECLARE_bool(enable_interlayer_addnorm);

DECLARE_bool(enable_split_rmsnorm_rope);

DECLARE_bool(enable_aclnn_matmul);

DECLARE_bool(enable_aclnn_swiglu);

DECLARE_bool(enable_flashcomm1);

DECLARE_int32(flashcomm1_min_prefill_tokens);

DECLARE_bool(enable_mmrs_fusion);

DECLARE_string(mmrs_comm_mode);
#endif

// --- chat template config ---
DECLARE_bool(use_cpp_chat_template);

DECLARE_int32(health_check_interval_ms);
