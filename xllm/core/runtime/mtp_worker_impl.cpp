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

#include "mtp_worker_impl.h"

#include <glog/logging.h>
#if defined(USE_NPU)
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>

#include "common/metrics.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#include "core/framework/block/block_utils.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/model/mtp_utils.h"
#include "core/framework/multimodal/mm_data.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#endif
#include "core/framework/speculative/adaptive_pruning_helpers.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/runtime/mtp_async_input_builder.h"
#include "core/runtime/mtp_async_state.h"
#include "spec_input_builder.h"
#include "util/pretty_print.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

namespace {

// Qwen3.5 GDN conv_state history capacity (kernel_conv_size - 1). Values of
// num_accepted_tokens beyond this describe history that has already rolled
// out of conv_state; passing them to aclnnCausalConv1d makes tiling fail when
// the current step's per_seq_val_tokens is small (e.g. adaptive prunes down
// to 2 while nat=5). Callers clamp accepted-prefix lengths to this cap before
// invoking the GDN spec-verify path.
constexpr int32_t kGdnConvHistoryCap = 3;

void clamp_gdn_conv_history(std::vector<int32_t>& accepted_prefix_lengths) {
  for (int32_t& v : accepted_prefix_lengths) {
    v = std::min(v, kGdnConvHistoryCap);
  }
}

bool has_active_dp_tokens(const ForwardInput& input) {
  const ParallelInput& parallel = input.input_params.parallel;
  const std::vector<int32_t>& token_nums =
      parallel.raw_dp_global_token_nums.empty()
          ? parallel.dp_global_token_nums
          : parallel.raw_dp_global_token_nums;
  return std::any_of(token_nums.begin(), token_nums.end(), [](int32_t count) {
    return count > 0;
  });
}

void broadcast_tokens_in_group(torch::Tensor& tokens,
                               ProcessGroup* process_group,
                               int32_t root_rank = 0) {
  if (process_group == nullptr || process_group->world_size() <= 1 ||
      !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  process_group->broadcast(tokens, root_rank);
}

bool should_broadcast_spec_tokens(const ParallelArgs& parallel_args,
                                  bool enable_spec_token_broadcast,
                                  bool all_greedy_sample) {
  const bool use_orthogonal_cp_consensus =
      parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
      parallel_args.cp_group_ != nullptr &&
      parallel_args.cp_group_ != parallel_args.tp_group_;
  return use_orthogonal_cp_consensus ||
         (enable_spec_token_broadcast && !all_greedy_sample);
}

constexpr int64_t kMaxSpecVerifyGraphUpdateBlockTableWidth = (1 << 15) - 1;

// Speculative state is replicated across every model rank within one DP
// replica. With orthogonal CP x TP, tp_group_ covers only one CP shard, while
// cp_group_ connects the same TP rank across CP shards. Broadcast along both
// axes so every replica caches and consumes the same sampled token, without
// crossing into another DP replica through the world group.
void broadcast_spec_tokens(torch::Tensor& tokens,
                           const ParallelArgs& parallel_args) {
  // DeepSeek-V4 TORCH publishes orthogonal TP and CP groups. Other backends
  // retain their existing single-group speculative broadcast behavior.
  ProcessGroup* tp_group = parallel_args.tp_group_ != nullptr
                               ? parallel_args.tp_group_
                               : parallel_args.process_group_;
  const bool use_orthogonal_cp_consensus =
      parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
      parallel_args.cp_group_ != nullptr &&
      parallel_args.cp_group_ != parallel_args.tp_group_;
  if (!use_orthogonal_cp_consensus) {
    broadcast_tokens_in_group(tokens, tp_group);
    return;
  }

  broadcast_tokens_in_group(tokens, parallel_args.tp_group_);
  if (parallel_args.cp_group_ != parallel_args.tp_group_) {
    broadcast_tokens_in_group(tokens, parallel_args.cp_group_);
  }
}

int64_t get_dp_local_tp_size(const ParallelArgs& parallel_args) {
  const int64_t dp_size = std::max<int64_t>(parallel_args.dp_size(), 1);
  const int64_t cp_size = std::max<int64_t>(parallel_args.cp_size(), 1);
  return std::max<int64_t>(parallel_args.world_size() / dp_size / cp_size, 1);
}

KVCacheEstimateOptions make_kv_cache_estimate_options(
    const ModelArgs& model_args,
    const runtime::Options& options,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    int64_t cache_size_in_bytes) {
  const int64_t dp_local_tp_size = get_dp_local_tp_size(parallel_args);
  const int64_t n_heads = model_args.n_heads();
  const int64_t n_kv_heads = model_args.n_kv_heads().value_or(n_heads);

  KVCacheEstimateOptions estimate_options;
  estimate_options.dtype = dtype;
  estimate_options.kv_cache_dtype = options.kv_cache_dtype();
  estimate_options.indexer_cache_dtype =
      ::xllm::KVCacheConfig::get_instance().indexer_cache_dtype();
  estimate_options.cache_size_in_bytes = cache_size_in_bytes;
  estimate_options.block_size = options.block_size();
  estimate_options.world_size = dp_local_tp_size;
  estimate_options.n_local_kv_heads =
      std::max<int64_t>(n_kv_heads / dp_local_tp_size, 1);
  if (has_linear_attention_layers(model_args)) {
    estimate_options.n_local_linear_k_heads = std::max<int64_t>(
        model_args.linear_num_key_heads() / dp_local_tp_size, 1);
    estimate_options.n_local_linear_v_heads = std::max<int64_t>(
        model_args.linear_num_value_heads() / dp_local_tp_size, 1);
  }
  estimate_options.max_seqs_per_batch =
      static_cast<int64_t>(options.max_seqs_per_batch());
  estimate_options.num_speculative_tokens =
      static_cast<int64_t>(options.num_speculative_tokens());
  estimate_options.max_tokens_per_batch =
      static_cast<int64_t>(options.max_tokens_per_batch());
  estimate_options.max_linear_state_cache_slots =
      options.max_linear_state_cache_slots();
  estimate_options.is_draft_engine = options.is_draft_engine();
  estimate_options.enable_prefix_cache =
      ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
  return estimate_options;
}

void record_metadata_ready_event(Stream& stream, ForwardInput& input) {
  input.metadata_ready_event = stream.record_event_or_sync();
}

void finish_metadata_prepare(Stream& stream, ForwardInput& input) {
  record_metadata_ready_event(stream, input);
}

void record_current_metadata_ready_event(ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
  record_metadata_ready_event(stream, input);
}

void wait_metadata_ready_event(const ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
}

void record_output_ready_event(ForwardOutput& output, Stream& stream) {
  StreamEventPtr event = stream.record_event();
  if (event == nullptr) {
    const int32_t ret = stream.synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  }
  output.ready_event = event;
}

void transfer_retained_inputs(ForwardOutput& destination,
                              ForwardOutput& source) {
  const size_t retained_input_count =
      source.retained_input_dependencies.size() +
      (source.retained_input != nullptr ? 1 : 0);
  destination.retained_input_dependencies.reserve(
      destination.retained_input_dependencies.size() + retained_input_count);
  if (source.retained_input != nullptr) {
    destination.retained_input_dependencies.emplace_back(
        std::move(source.retained_input));
  }
  for (std::shared_ptr<ForwardInput>& retained_input :
       source.retained_input_dependencies) {
    destination.retained_input_dependencies.emplace_back(
        std::move(retained_input));
  }
  source.retained_input_dependencies.clear();
}

void release_retained_inputs(ForwardOutput& output) {
  output.retained_input.reset();
  output.retained_input_dependencies.clear();
}

void finalize_output_on_stream(ForwardOutput& output,
                               Stream& stream,
                               bool allow_async) {
  if (allow_async) {
    record_output_ready_event(output, stream);
    return;
  }
  const int32_t ret = stream.synchronize();
  CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  release_retained_inputs(output);
}

#if defined(USE_NPU)
void clear_expanded_spec_verify_graph_input(ModelInputParams& input_params) {
  input_params.graph.use_expanded_decode_for_spec_verify_attention = false;
  input_params.graph.expanded_kv_seq_lens = torch::Tensor();
  input_params.graph.expanded_block_tables = torch::Tensor();
  input_params.graph.expanded_paged_kv_indptr = torch::Tensor();
  input_params.graph.expanded_paged_kv_indices = torch::Tensor();
  input_params.graph.expanded_paged_kv_last_page_len = torch::Tensor();
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
}

bool build_expanded_spec_verify_graph_host_input(
    ModelInputParams& input_params) {
  clear_expanded_spec_verify_graph_input(input_params);
  if (!input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return false;
  }

  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return false;
  }
  std::vector<int32_t> expanded_kv_seq_lens =
      layer::ExpandedDecodeMetadataBuilder::build_tokenwise_kv_seq_lens(
          q_seq_lens, kv_seq_lens);
  if (expanded_kv_seq_lens.empty()) {
    return false;
  }

  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
  input_params.graph.expanded_kv_seq_lens_vec = std::move(expanded_kv_seq_lens);
  return true;
}

void bind_expanded_spec_verify_graph_input(ModelInputParams& input_params,
                                           const torch::Device& device,
                                           bool kv_lens_already_bound,
                                           int32_t block_size) {
  if (!input_params.graph.use_expanded_decode_for_spec_verify_attention) {
    return;
  }
  CHECK(input_params.attention.device.block_tables.defined())
      << "spec verify block tables must be rebuilt before graph input";
  const auto& q_seq_lens = input_params.attention.host.q_seq_lens;
  CHECK_GE(input_params.attention.device.block_tables.size(0),
           static_cast<int64_t>(q_seq_lens.size()))
      << "spec verify block table rows are fewer than sequences";
  std::vector<torch::Tensor> expanded_block_rows;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_seq_lens.size());
       ++seq_idx) {
    for (int32_t token_idx = 0;
         token_idx < q_seq_lens[static_cast<size_t>(seq_idx)];
         ++token_idx) {
      expanded_block_rows.emplace_back(
          input_params.attention.device.block_tables.select(/*dim=*/0,
                                                            seq_idx));
    }
  }

  if (!kv_lens_already_bound) {
    torch::Tensor expanded_kv_seq_lens_host =
        torch::tensor(input_params.graph.expanded_kv_seq_lens_vec,
                      torch::TensorOptions()
                          .dtype(torch::kInt)
                          .device(torch::kCPU)
                          .pinned_memory(true));
    input_params.graph.expanded_kv_seq_lens =
        expanded_kv_seq_lens_host.to(device, /*non_blocking=*/true);
  }

  // ATB consumes this tensor as dense row-major storage. Keep the generic
  // fallback contiguous; a zero-stride expand view is rejected at runtime.
  torch::Tensor expanded_block_tables = torch::stack(expanded_block_rows, 0);
  layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
      input_params,
      input_params.graph.expanded_kv_seq_lens,
      expanded_block_tables,
      input_params.graph.expanded_kv_seq_lens_vec,
      block_size);
}

void build_expanded_spec_verify_graph_input(ModelInputParams& input_params,
                                            const torch::Device& device,
                                            int32_t block_size) {
  build_expanded_spec_verify_graph_host_input(input_params);
  bind_expanded_spec_verify_graph_input(
      input_params, device, false, block_size);
}
#endif

void clear_sample_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
}

void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

void clear_all_output_embeddings(ForwardOutput& output) {
  clear_sample_embeddings(output);
  clear_selected_embeddings(output);
}

void clear_ready_events(ForwardInput& input) {
  input.metadata_ready_event.reset();
}

std::optional<ForwardOutput> run_llm_no_sync_impl(
    LLMWorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput& processed_input) {
  worker.prepare_work_before_execute_on_stream(
      input,
      processed_input,
      prepare_stream,
      /*record_ready_event=*/&prepare_stream != &compute_stream);
  worker.set_hierarchy_layer_synchronizer(processed_input.input_params);
  return worker.execute_no_sync_on_stream(
      processed_input, compute_stream, /*record_ready_event=*/false);
}

torch::Tensor clone_host_tensor(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK(tensor.device().is_cpu()) << "expected a CPU host tensor";
  return tensor.contiguous().clone();
}

void stabilize_decode_host_tensors(ForwardInput& input) {
  input.token_ids_host = clone_host_tensor(input.token_ids_host);
  input.positions_host = clone_host_tensor(input.positions_host);
  input.input_params.attention.host.block_tables =
      clone_host_tensor(input.input_params.attention.host.block_tables);
  for (torch::Tensor& block_table : input.input_params.multi_block_tables) {
    block_table = clone_host_tensor(block_table);
  }
}

void set_token_ids_device_tensor(ForwardInput& input,
                                 const torch::Tensor& token_ids,
                                 const torch::TensorOptions& token_options,
                                 Stream& compute_stream) {
  CHECK(token_ids.defined()) << "draft token_ids must be defined";
  torch::Tensor flat_token_ids = token_ids.flatten();
  CHECK_EQ(flat_token_ids.numel(), input.input_params.meta.num_sequences)
      << "draft token count must match num_sequences";

  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  input.device_tensors_ready = false;
  input.token_ids_host = torch::Tensor();
  input.token_ids =
      safe_to(flat_token_ids, token_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

torch::Tensor to_cpu_int_tensor_for_read(const torch::Tensor& values) {
  return safe_to(values.flatten(),
                 torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU),
                 false)
      .contiguous();
}

void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token) {
  CHECK(!request_ids.empty())
      << "MTP decode requires request ids for bootstrap state validation";
  CHECK_EQ(states.size(), request_ids.size())
      << "MTP decode request/state count mismatch";
  CHECK_GE(token_ids_host.numel(), static_cast<int64_t>(states.size()))
      << "MTP decode token/state count mismatch";

  Slice<int32_t> token_ids = {token_ids_host.data_ptr<int32_t>(),
                              static_cast<size_t>(token_ids_host.numel())};
  for (int32_t i = 0; i < static_cast<int32_t>(states.size()); ++i) {
    const EmbeddingCache::DecodeState& state = states[i];
    const int32_t token_id = token_ids[i];
    CHECK(state.valid) << "MTP decode missing target state, request_id="
                       << request_ids[i];
    CHECK_EQ(state.request_id, request_ids[i])
        << "MTP decode target state request mismatch";
    CHECK(state.embedding.defined())
        << "MTP decode target state embedding is undefined, request_id="
        << request_ids[i];
    if (token_id < 0) {
      CHECK(allow_overlap_fake_token)
          << "MTP decode fake token is only allowed with schedule overlap, "
          << "request_id=" << request_ids[i];
      CHECK_GE(state.token_id, 0)
          << "MTP decode fake token requires a valid cached target token, "
          << "request_id=" << request_ids[i];
      continue;
    }
    CHECK_EQ(state.token_id, token_id)
        << "MTP decode target state token mismatch, request_id="
        << request_ids[i];
  }
}

void replace_host_token_placeholders(ForwardInput& input,
                                     int32_t placeholder,
                                     const torch::Tensor& replacements,
                                     const torch::TensorOptions& token_options,
                                     bool refresh_device = true) {
  CHECK(replacements.defined())
      << "speculative replacement tokens must be defined";
  CHECK(input.token_ids_host.defined())
      << "token_ids_host must be defined before speculative token update";
  CHECK(input.token_ids_host.device().is_cpu())
      << "token_ids_host must stay on CPU";
  CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
      << "token_ids_host must be int32";

  input.device_tensors_ready = false;
  torch::Tensor replacement_cpu = to_cpu_int_tensor_for_read(replacements);
  int32_t* token_ids = input.token_ids_host.data_ptr<int32_t>();
  const size_t num_token_ids =
      static_cast<size_t>(input.token_ids_host.numel());
  Slice<int32_t> replacement_ids = {
      replacement_cpu.data_ptr<int32_t>(),
      static_cast<size_t>(replacement_cpu.numel())};

  size_t replacement_idx = 0;
  for (size_t i = 0; i < num_token_ids; ++i) {
    if (token_ids[i] != placeholder) {
      continue;
    }
    CHECK_LT(replacement_idx, replacement_ids.size())
        << "not enough speculative replacement tokens";
    token_ids[i] = replacement_ids[replacement_idx++];
  }
  CHECK_EQ(replacement_idx, replacement_ids.size())
      << "unused speculative replacement tokens";

  if (refresh_device) {
    input.token_ids =
        safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
    input.device_tensors_ready = true;
  }
}

void set_positions_tensor(ForwardInput& input,
                          const std::vector<int32_t>& positions,
                          const torch::TensorOptions& device_options) {
  input.device_tensors_ready = false;
  input.positions_host = specBuilder::make_cpu_int_tensor(positions);
  input.positions =
      safe_to(input.positions_host, device_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

runtime::Options MTPTargetOptions(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options mtp_draft_options(const runtime::Options& options) {
  runtime::Options draft_options = options;
  draft_options.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(true);
  return draft_options;
}

ParallelArgs MTPDraftParallelArgs(const ParallelArgs& parallel_args,
                                  const runtime::Options& options) {
  if (!options.enable_mtp_draft_body_tp1()) {
    return parallel_args;
  }
  CHECK(parallel_args.single_rank_group_ != nullptr)
      << "MTP draft body TP1 requires a single-rank process group";
  ParallelArgs draft_args = parallel_args;
  draft_args.rank(0)
      .world_size(1)
      .dp_size(1)
      .ep_size(1)
      .cp_size(1)
      .tp_size(1)
      .sp_size(1);
  draft_args.mapping_data(nlohmann::json{});
  draft_args.process_group_ = parallel_args.single_rank_group_;
  draft_args.dp_local_process_group_ = parallel_args.single_rank_group_;
  draft_args.lm_head_group_ = parallel_args.tp_group_;
  draft_args.tp_group_ = parallel_args.single_rank_group_;
  draft_args.cp_group_ = parallel_args.single_rank_group_;
  draft_args.moe_ep_group_ = parallel_args.single_rank_group_;
  draft_args.moe_tp_group_ = parallel_args.single_rank_group_;
  return draft_args;
}

KVCacheShape MTPDraftKVCacheShape(const KVCacheShape& target_shape,
                                  const ModelArgs& draft_model_args,
                                  int64_t block_size) {
  KVCacheCapacity draft_capacity;
  draft_capacity.n_blocks(target_shape.key_cache_shape()[0])
      .block_size(block_size);
  return KVCacheShape(draft_capacity, draft_model_args, /*world_size=*/1);
}

bool is_qwen3_5_draft_model_type(const std::string& model_type) {
  return mtp_async::classify_combined_draft_execution_path(model_type) ==
         mtp_async::CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
}

uint32_t validate_paired_transfer_counts(uint32_t target_transferred,
                                         uint32_t draft_transferred) {
  if (target_transferred != draft_transferred) {
    LOG(ERROR) << "MTP target/draft KV block transfer count mismatch: target="
               << target_transferred << ", draft=" << draft_transferred;
    return 0;
  }
  return target_transferred;
}

}  // namespace

using adaptive_pruning::apply_pruned_prefix_lengths;
using adaptive_pruning::clamp_prefix_lengths;
using adaptive_pruning::has_selected_probs_by_step;
using adaptive_pruning::max_pruned_prefix_length;
using adaptive_pruning::selected_probs_by_step;
using adaptive_pruning::sync_pruned_boundary_outputs;
using adaptive_pruning::truncate_draft_outputs;

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : MTPWorkerImpl(
          parallel_args,
          device,
          options,
          MTPTargetOptions(options),
          mtp_draft_options(options),
          ::xllm::SpeculativeConfig::get_instance().enable_opt_validate_probs(),
          /*enable_adaptive_speculative_decode=*/true) {}

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             const runtime::Options& target_options,
                             const runtime::Options& draft_options,
                             bool enable_opt_validate_probs,
                             bool enable_adaptive_speculative_decode)
    : SpeculativeWorkerImpl(parallel_args, device, options, target_options),
      enable_opt_validate_probs_(enable_opt_validate_probs) {
  draft_impl_ = std::make_unique<LLMWorkerImpl>(
      MTPDraftParallelArgs(parallel_args, options),
      device,
      mtp_draft_options(draft_options));
  const bool enable_parallel_adaptive_sl =
      parallel_args.dp_size() <= 1 && parallel_args.ep_size() <= 1;
  if (enable_adaptive_speculative_decode && enable_parallel_adaptive_sl) {
    adaptive_spec_controller_ =
        std::make_unique<AdaptiveSpeculativeController>(options);
  } else if (enable_adaptive_speculative_decode &&
             options.enable_adaptive_speculative_decode()) {
    LOG(WARNING)
        << "Adaptive speculative decode is disabled for DP/EP parallelism "
        << "in v1. dp_size=" << parallel_args.dp_size()
        << ", ep_size=" << parallel_args.ep_size();
  }
}

bool MTPWorkerImpl::init_model(const std::string& model_weights_path,
                               int32_t random_seed,
                               MasterStatus master_status) {
  // Load target model via base class
  bool result = true;
  const bool loading_target =
      impl_->get_status() == WorkerImpl::Status::UNINITIALIZED;
  if (loading_target) {
    result = SpeculativeWorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  } else {
    CHECK_EQ(draft_impl_->get_status(), WorkerImpl::Status::UNINITIALIZED);
    result = draft_impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
  }

  if (impl_ != nullptr && impl_->get_status() == WorkerImpl::Status::LOADED) {
    context_ = impl_->context_;
    target_spec_verify_mode_ = mtp_async::classify_target_spec_verify_mode(
        context_.get_model_args().model_type());
  }

  if (draft_impl_ != nullptr &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    combined_draft_execution_path_ =
        mtp_async::classify_combined_draft_execution_path(
            draft_impl_->context_.get_model_args().model_type());
    const bool draft_owns_shared_weights =
        options_.enable_mtp_draft_body_tp1() &&
        combined_draft_execution_path_ ==
            mtp_async::CombinedDraftExecutionPath::QWEN3_5_PAGED_ATTENTION;
    // Qwen3.5 draft checkpoints contain complete embedding and LMHead weights.
    // Other MTP drafts retain their existing target-weight sharing contract;
    // only their transformer body is replicated with TP1 parallel arguments.
    if (!draft_owns_shared_weights) {
      const bool python_weights_shared =
          draft_impl_->share_weights_from(*impl_);
      if (!python_weights_shared) {
#if defined(USE_NPU)
        if (::xllm::KernelConfig::get_instance().npu_kernel_backend() !=
            "TORCH") {
          auto head = impl_->get_npu_lm_head();
          draft_impl_->set_npu_lm_head(head);
          auto word_embedding = impl_->get_npu_word_embedding();
          draft_impl_->set_npu_word_embedding(word_embedding);
        } else {
          auto head = impl_->get_lm_head();
          draft_impl_->set_lm_head(head);
          auto word_embedding = impl_->get_word_embedding();
          draft_impl_->set_word_embedding(word_embedding);
        }
#else
        auto head = impl_->get_lm_head();
        draft_impl_->set_lm_head(head);
        auto word_embedding = impl_->get_word_embedding();
        draft_impl_->set_word_embedding(word_embedding);
#endif
      }
    }
  }
#if defined(USE_NPU)
  if (result && supports_explicit_spec_verify_replay_update()) {
    CHECK_EQ(::xllm::KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
        << "Qwen3.5 MTP only supports NPU Torch backend";
  }
#endif
  return result;
}

std::tuple<int64_t, int64_t> MTPWorkerImpl::estimate_kv_cache_capacity() {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  const std::tuple<int64_t, int64_t> target_memory =
      impl_->estimate_kv_cache_capacity();
  const std::tuple<int64_t, int64_t> draft_memory =
      draft_impl_->estimate_kv_cache_capacity();
  const int64_t cache_size_in_bytes =
      std::min(std::get<0>(target_memory), std::get<0>(draft_memory));
  const int64_t total_memory =
      std::min(std::get<1>(target_memory), std::get<1>(draft_memory));

  const ModelArgs& target_model_args = impl_->context_.get_model_args();
  const ModelArgs& draft_model_args = draft_impl_->context_.get_model_args();
  if (!util::is_target_model_type(target_model_args.model_type(),
                                  /*target_model_type=*/"deepseek_v4",
                                  /*match_mtp=*/true)) {
    return {cache_size_in_bytes, total_memory};
  }

  // use for DSv4
  KVCacheEstimateOptions target_options =
      make_kv_cache_estimate_options(target_model_args,
                                     MTPTargetOptions(options_),
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  const KVCacheEstimateOptions draft_options =
      make_kv_cache_estimate_options(draft_model_args,
                                     mtp_draft_options(options_),
                                     parallel_args_,
                                     dtype_,
                                     cache_size_in_bytes);
  target_options.draft_model_args = &draft_model_args;
  target_options.draft_options = &draft_options;

  KVCacheCapacity kv_cache_cap =
      ::xllm::estimate_kv_cache_capacity(target_model_args, target_options);
  return {kv_cache_cap.cache_size_in_bytes(), total_memory};
}

int64_t MTPWorkerImpl::get_embedding_placeholder_size() {
  // DeepSeek-V4 MTP stashes the pre-hc_head 3D hidden flattened to
  // [num_tokens, hc_mult*hidden], so the cache placeholder must cover
  // hc_mult*hidden per row.
  if (impl_ != nullptr) {
    const ModelArgs& args = impl_->context_.get_model_args();
    return mtp_hidden_state_width(args);
  }
  return static_cast<int64_t>(embedding_size_);
}

bool MTPWorkerImpl::supports_explicit_spec_verify_replay_update() const {
  if (target_spec_verify_mode_ ==
      mtp_async::TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY) {
    return true;
  }
  // The Python NPU paged-attention runner consumes expanded metadata and can
  // replay its ACL graph for DeepSeek MLA. The native target executor has no
  // corresponding MLA spec-verify graph path yet, so it remains generic.
  return target_spec_verify_mode_ ==
             mtp_async::TargetSpecVerifyMode::DEEPSEEK_V32_EXPANDED_VERIFY &&
         ModelConfig::is_python_model_impl(context_.get_model_impl());
}

bool MTPWorkerImpl::requires_uniform_validate_width() const {
  // Currently only Qwen3.5's GDN spec-verify kernel requires uniform width;
  // this happens to coincide with the QWEN3_5_EXPANDED_VERIFY mode used by
  // supports_explicit_spec_verify_replay_update, but the two are semantically
  // distinct capabilities (graph-update capability vs. per-seq varlen kernel
  // support).
  return target_spec_verify_mode_ ==
         mtp_async::TargetSpecVerifyMode::QWEN3_5_EXPANDED_VERIFY;
}

bool MTPWorkerImpl::should_use_explicit_spec_verify_replay_update(
    const ForwardInput& input) const {
#if defined(USE_NPU)
  const torch::Tensor& block_tables =
      input.input_params.attention.host.block_tables;
  if (!::xllm::ExecutionConfig::get_instance().enable_graph() ||
      !::xllm::ExecutionConfig::get_instance()
           .enable_graph_mode_decode_no_padding() ||
      !supports_explicit_spec_verify_replay_update() ||
      options_.num_speculative_tokens() <= 0 ||
      input.input_params.meta.num_sequences <= 0 ||
      options_.block_size() <= 0 || !block_tables.defined() ||
      block_tables.dim() != 2 ||
      block_tables.size(0) != input.input_params.meta.num_sequences) {
    return false;
  }
  const int64_t block_table_width = spec_verify_block_table_width(block_tables);
  if (block_table_width <= 0 ||
      block_table_width > kMaxSpecVerifyGraphUpdateBlockTableWidth) {
    return false;
  }
  const int64_t spec_width =
      static_cast<int64_t>(options_.num_speculative_tokens()) + 1;
  return kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
      spec_width, options_.block_size());
#else
  (void)input;
  return false;
#endif
}

int64_t MTPWorkerImpl::spec_verify_block_table_width(
    const torch::Tensor& block_tables) const {
  CHECK(block_tables.defined() && block_tables.dim() == 2);
  CHECK_GT(options_.block_size(), 0);
  int64_t required_width = block_tables.size(1);
  if (impl_ != nullptr) {
    const int64_t declared_capacity =
        mtp_async::speculative_verify_block_table_capacity(
            impl_->context_.get_model_args().max_position_embeddings(),
            options_.block_size());
    CHECK_LE(required_width, declared_capacity)
        << "block table width exceeds the model position capacity";
    required_width = declared_capacity;
  }
  return required_width;
}

bool MTPWorkerImpl::use_chunked_prefill_spec_verify_path() const {
  return target_spec_verify_mode_ ==
             mtp_async::TargetSpecVerifyMode::CAUSAL_CHUNKED_PREFILL ||
         supports_explicit_spec_verify_replay_update();
}

bool MTPWorkerImpl::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  // init_model() must run first so dtype_/embedding_size_ are initialized.
  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  if (embedding_cache_) {
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(device_)));
    }
  }
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  bool target_allocated = true;
  const auto target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache(kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    if (options_.enable_mtp_draft_body_tp1()) {
      const KVCacheShape draft_shape =
          MTPDraftKVCacheShape(kv_cache_shape,
                               draft_impl_->context_.get_model_args(),
                               options_.block_size());
      draft_shape.print_shapes();
      draft_allocated = draft_impl_->allocate_kv_cache(draft_shape);
    } else {
      draft_allocated = draft_impl_->allocate_kv_cache(kv_cache_shape);
    }
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  return target_allocated && draft_allocated;
}

uint32_t MTPWorkerImpl::transfer_kv_blocks(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  const uint32_t target_transferred =
      impl_->transfer_kv_blocks(batch_id, block_transfer_info);
  const uint32_t draft_transferred =
      draft_impl_->transfer_kv_blocks(batch_id, block_transfer_info);
  return validate_paired_transfer_counts(target_transferred, draft_transferred);
}

uint32_t MTPWorkerImpl::transfer_kv_blocks(
    uint64_t batch_id,
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  const uint32_t target_transferred =
      impl_->transfer_kv_blocks(batch_id, block_transfer_info);
  const uint32_t draft_transferred =
      draft_impl_->transfer_kv_blocks(batch_id, block_transfer_info);
  return validate_paired_transfer_counts(target_transferred, draft_transferred);
}

#if defined(USE_NPU) || defined(USE_MLU)
bool MTPWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  if (kv_cache_transfer_ == nullptr) {
#if defined(USE_NPU)
    const std::string& transfer_type =
        ::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type();
    if (transfer_type == "LlmDataDist") {
      kv_cache_transfer_ = std::make_shared<SpecKVCacheTransfer>(
          options_.transfer_listen_port(),
          options_.instance_role(),
          context_.get_model_args().index_n_heads() > 0,
          context_.get_model_args().enable_mla(),
          options_.enable_mtp_draft_body_tp1());
    } else {
      CHECK_EQ(transfer_type, "Mooncake");
      kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
          device_.index(),
          options_.transfer_listen_port(),
          device_,
          context_.get_model_args().model_type());
    }
#elif defined(USE_MLU)
    CHECK_EQ(::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type(),
             "Mooncake")
        << "MLU MTP only supports Mooncake KV transfer.";
    kv_cache_transfer_ = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_.index(),
        options_.transfer_listen_port(),
        device_,
        context_.get_model_args().model_type());
#endif

    int32_t device_id = device_.index();
    kv_cache_transfer_->initialize(device_id);
  }

  bool target_allocated = true;
  const auto target_status = impl_->get_status();
  if (target_status == WorkerImpl::Status::LOADED) {
    target_allocated = impl_->allocate_kv_cache_with_transfer(
        kv_cache_transfer_, kv_cache_shape);
  } else {
    CHECK_EQ(target_status, WorkerImpl::Status::READY);
  }

  bool draft_allocated = true;
  const auto draft_status = draft_impl_->get_status();
  if (draft_status == WorkerImpl::Status::LOADED) {
    if (options_.enable_mtp_draft_body_tp1()) {
      const KVCacheShape draft_shape =
          MTPDraftKVCacheShape(kv_cache_shape,
                               draft_impl_->context_.get_model_args(),
                               options_.block_size());
      draft_shape.print_shapes();
      draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
          kv_cache_transfer_, draft_shape);
    } else {
      draft_allocated = draft_impl_->allocate_kv_cache_with_transfer(
          kv_cache_transfer_, kv_cache_shape);
    }
  } else {
    CHECK_EQ(draft_status, WorkerImpl::Status::READY);
  }

  embedding_cache_ = std::make_shared<EmbeddingCache>(num_blocks);
  if (embedding_cache_) {
    int64_t size = get_embedding_placeholder_size();
    if (size > 0) {
      embedding_cache_->set_placeholder(
          torch::zeros({size}, torch::dtype(dtype_).device(device_)));
    }
  }
  return target_allocated && draft_allocated;
}
#endif

ForwardInput MTPWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  return inputs;
}

void MTPWorkerImpl::prepare_work_before_execute(const ForwardInput& input,
                                                ForwardInput& processed_input) {
  // Composite skips CP prepare; leaves run it in run_llm_no_sync_impl.
  SpeculativeWorkerImpl::prepare_work_before_execute(input, processed_input);
}

bool MTPWorkerImpl::owns_npu_parallel_input_prepare() const { return false; }

std::optional<ForwardOutput> MTPWorkerImpl::step_empty(
    const ForwardInput& input) {
  const bool use_prelaunched_first_draft =
      input.input_params.meta.batch_forward_type.is_decode() &&
      can_use_combined_first_draft() && pending_draft_context_matches(input);
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // The preceding validation may have speculatively submitted draft-0 before
    // the scheduler learned that the batch had finished.  Keep its graph/input
    // buffers alive until the queued work completes, then discard the result.
    // This is a batch-exit slow path and is never taken in steady decode.
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain final MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  flush_pending_target_context();

  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    ForwardInput target_prepared;
    ForwardInput draft_prepared;
    auto output = run_llm_no_sync_impl(
        *impl_, input, *prepare_stream_, *compute_stream_, target_prepared);
    auto draft_output = run_llm_no_sync_impl(*draft_impl_,
                                             input,
                                             *prepare_stream_,
                                             *compute_stream_,
                                             draft_prepared);
    if (draft_output.has_value()) {
      transfer_retained_inputs(*output, draft_output.value());
    }
    clear_all_output_embeddings(*output);
    finalize_output_on_stream(
        *output, *compute_stream_, enable_schedule_overlap());
    return output;
  } else {
    ForwardInput draft_extend_prepared;
    std::vector<ForwardInput> draft_step_prepared(
        options_.num_speculative_tokens());
    ForwardInput target_prepared;
    std::vector<ForwardOutput> draft_outputs;
    draft_outputs.reserve(options_.num_speculative_tokens());

    ForwardInput new_input = input;
    for (int32_t& token_num :
         new_input.input_params.parallel.dp_global_token_nums) {
      token_num *= 2;
    }
    for (int32_t& token_num :
         new_input.input_params.parallel.raw_dp_global_token_nums) {
      token_num *= 2;
    }
    if (use_prelaunched_first_draft) {
      draft_outputs.emplace_back(
          std::move(pending_draft_context_.output.value()));
      draft_extend_prepared = std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      draft_outputs.emplace_back(run_llm_no_sync_impl(*draft_impl_,
                                                      new_input,
                                                      *prepare_stream_,
                                                      *compute_stream_,
                                                      draft_extend_prepared)
                                     .value());
    }

    for (int32_t i = 1; i < options_.num_speculative_tokens(); ++i) {
      draft_outputs.emplace_back(run_llm_no_sync_impl(*draft_impl_,
                                                      input,
                                                      *prepare_stream_,
                                                      *compute_stream_,
                                                      draft_step_prepared[i])
                                     .value());
    }

    new_input = input;
    for (int32_t& token_num :
         new_input.input_params.parallel.dp_global_token_nums) {
      token_num *= options_.num_speculative_tokens() + 1;
    }
    for (int32_t& token_num :
         new_input.input_params.parallel.raw_dp_global_token_nums) {
      token_num *= options_.num_speculative_tokens() + 1;
    }
    ForwardOutput output = run_llm_no_sync_impl(*impl_,
                                                new_input,
                                                *prepare_stream_,
                                                *compute_stream_,
                                                target_prepared)
                               .value();
    for (ForwardOutput& draft_output : draft_outputs) {
      transfer_retained_inputs(output, draft_output);
    }
    clear_all_output_embeddings(output);
    finalize_output_on_stream(
        output, *compute_stream_, enable_schedule_overlap());
    if (can_prelaunch_next_first_draft(input)) {
      ForwardInput next_first_draft_input = input;
      for (int32_t& token_num :
           next_first_draft_input.input_params.parallel.dp_global_token_nums) {
        token_num *= 2;
      }
      for (int32_t& token_num : next_first_draft_input.input_params.parallel
                                    .raw_dp_global_token_nums) {
        token_num *= 2;
      }
      submit_pending_first_draft(input, std::move(next_first_draft_input));
    }
    return output;
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::step_prefill(
    const ForwardInput& input) {
  flush_pending_target_context();

  Timer timer;
  ForwardInput target_prepared;
  ForwardInput draft_prepared;

  // run the target model to get first token and hidden states
  ForwardOutput output =
      run_llm_no_sync_impl(
          *impl_, input, *prepare_stream_, *compute_stream_, target_prepared)
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // MTP path that depends on hidden states.
  ForwardInput prefill_input;
  prepare_prefill_inputs(input, prefill_input);

  // prepare input for draft model
  auto& embeddings = output.sample_output.embeddings;

  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Target prefill seeds the MTP decode cache. Under orthogonal CP x TP each
    // CP shard samples independently unless this token is synchronized across
    // both axes; caching divergent tokens makes the first decode input disagree
    // with the non-driver CP shard's DecodeState.
    if (should_broadcast_spec_tokens(
            parallel_args_,
            get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.all_greedy_sample)) {
      broadcast_spec_tokens(output.sample_output.next_tokens, parallel_args_);
    }
    if (embeddings.defined()) {
      prefill_input.input_params.embedding.input_embedding = embeddings.clone();
    }
    if (output.sample_output.next_tokens.defined()) {
      replace_host_token_placeholders(prefill_input,
                                      -1,
                                      output.sample_output.next_tokens,
                                      prefill_input.token_ids.options());
    }
    if (embeddings.defined() || output.sample_output.next_tokens.defined()) {
      record_current_metadata_ready_event(prefill_input, *compute_stream_);
    }
  }
  // generate kv cache for draft model
  timer.reset();
  ForwardOutput draft_output = run_llm_no_sync_impl(*draft_impl_,
                                                    prefill_input,
                                                    *prepare_stream_,
                                                    *compute_stream_,
                                                    draft_prepared)
                                   .value();
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    process_draft_sample_output(draft_output.sample_output);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());

  if (input.sampling_params.selected_token_idxes.defined()) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    // Prefer embeddings (global-real after CP merge); selected_embeddings is
    // a fallback.
    const torch::Tensor& target_hidden =
        output.sample_output.selected_embeddings.defined()
            ? output.sample_output.selected_embeddings
            : embeddings;
    torch::Tensor bootstrap_embeddings = target_hidden;
    if (bootstrap_embeddings.size(0) !=
        static_cast<int64_t>(
            input.input_params.embedding.embedding_ids.size())) {
      torch::Tensor bootstrap_idxes =
          input.sampling_params.selected_token_idxes.to(
              torch::dtype(torch::kLong).device(bootstrap_embeddings.device()));
      bootstrap_embeddings =
          bootstrap_embeddings.index_select(/*dim=*/0, bootstrap_idxes);
    }
    output.sample_output.embeddings = bootstrap_embeddings.detach();
    embedding_cache_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        output.sample_output.next_tokens,
        target_hidden,
        input.sampling_params.selected_token_idxes);
    clear_selected_embeddings(output);
  } else {
    clear_all_output_embeddings(output);
  }

  transfer_retained_inputs(output, draft_output);
  finalize_output_on_stream(
      output, *compute_stream_, enable_schedule_overlap());

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

void MTPWorkerImpl::prepare_prefill_inputs(const ForwardInput& input,
                                           ForwardInput& prefill_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  prefill_input = input.to(device_, dtype_);
  prefill_input.sampling_params.return_probs = true;
  clear_ready_events(prefill_input);
  auto& input_params = prefill_input.input_params;
  auto& extra_token_ids = input_params.embedding.extra_token_ids;

  const torch::Tensor& token_ids = input.token_ids_host;
  Slice<int32_t> tokens_ids_slice = {token_ids.data_ptr<int32_t>(),
                                     static_cast<size_t>(token_ids.numel())};

  int32_t start_idx = 0;
  std::vector<int32_t> new_token_ids;
  new_token_ids.reserve(token_ids.numel());
  for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
    int32_t q_len = input_params.get_q_seq_len(i);
    Slice<int32_t> tokens_ids_slice_i =
        tokens_ids_slice.slice(start_idx + 1, start_idx + q_len);
    start_idx += q_len;
    new_token_ids.insert(new_token_ids.end(),
                         tokens_ids_slice_i.begin(),
                         tokens_ids_slice_i.end());
    new_token_ids.emplace_back(extra_token_ids[i]);
  }
  prefill_input.device_tensors_ready = false;
  prefill_input.token_ids_host =
      specBuilder::make_cpu_int_tensor(new_token_ids);
  prefill_input.token_ids = safe_to(prefill_input.token_ids_host,
                                    prefill_input.positions.options(),
                                    /*non_blocking=*/true);
  prefill_input.device_tensors_ready = true;
  finish_metadata_prepare(*prepare_stream_, prefill_input);
}

std::optional<ForwardOutput> MTPWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  if (use_chunked_prefill_spec_verify_path()) {
    stabilize_decode_host_tensors(input);
  }
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  // Reuse draft-0 prelaunched for this same batch.
  const bool use_prelaunched_first_draft =
      can_use_combined_first_draft() && pending_draft_context_matches(input);
  const bool matching_device_target_context =
      pending_target_context_matches(input);
  // Consume this batch's pending target context directly on device.
  const bool use_device_target_context =
      can_use_combined_first_draft() && matching_device_target_context &&
      device_target_context_ready_for_batch(input);
  // Keep the device-side accepted state alive across a first-transition Host
  // cache flush. The prelaunched draft can be valid before the batch is marked
  // device-context ready, while flush_pending_target_context() clears the
  // owning context below.
  const torch::Tensor accepted_tokens = pending_target_context_.accepted_tokens;
  const torch::Tensor accepted_embeddings =
      pending_target_context_.accepted_embeddings;
  const torch::Tensor target_base_positions =
      pending_target_context_.base_positions;
  const torch::Tensor target_base_kv_seq_lens =
      pending_target_context_.base_kv_seq_lens;
  const StreamEventPtr target_context_ready_event =
      pending_target_context_.ready_event;
  if (pending_draft_context_.output.has_value() &&
      !use_prelaunched_first_draft) {
    // A batch transition invalidates the speculative prelaunch.  Drain it
    // before releasing its graph/input buffers; this slow path is outside
    // steady decode and preserves cache/buffer lifetime correctness.
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to drain stale MTP draft prelaunch, ret="
                     << ret;
    pending_draft_context_ = PendingDraftContext();
  }
  if (!use_device_target_context) {
    // Batch transitions are uncommon in steady decode.  Materialize the most
    // recent target state only for that fallback; the normal path below never
    // synchronizes the worker thread with the NPU.
    flush_pending_target_context();
    if (matching_device_target_context) {
      // The first target-context publication for a new batch establishes the
      // scheduler's corrected position/KV base. Subsequent publications can
      // derive that base fully on device without waiting for the scheduler.
      device_context_ready_embedding_ids_ =
          input.input_params.embedding.embedding_ids;
      device_context_ready_request_ids_ =
          input.input_params.embedding.request_ids;
    } else if (!device_target_context_ready_for_batch(input)) {
      device_context_ready_embedding_ids_.clear();
      device_context_ready_request_ids_.clear();
    }
  }
  // Adaptive is enabled only after profile completes (registry has predictor),
  // ensuring profiling warmup never triggers the adaptive HCCL broadcast.
  const bool use_adaptive_speculative_decode =
      adaptive_enabled() &&
      SpeculativeProfileRegistry::get_instance().has_validate_time_predictor();

  std::vector<ForwardOutput> draft_outputs;
  ForwardInput current_draft_input, validate_input, next_step_input;
  std::vector<ForwardInput> draft_prepared(num_speculative_tokens);
  Timer timer;
  CHECK(embedding_cache_ != nullptr) << "MTP embedding cache is not allocated";

  const auto& embedding = input.input_params.embedding;
  if (embedding.mtp_bootstrap_embeddings.defined()) {
    CHECK(input.token_ids_host.defined())
        << "MTP bootstrap requires host token ids";
    CHECK(input.token_ids_host.device().is_cpu())
        << "MTP bootstrap host token ids must be on CPU";
    CHECK_EQ(input.token_ids_host.scalar_type(), torch::kInt)
        << "MTP bootstrap host token ids must be int32";

    torch::Tensor bootstrap_embeddings =
        safe_to(embedding.mtp_bootstrap_embeddings,
                torch::dtype(dtype_).device(device_));
    CHECK_EQ(bootstrap_embeddings.size(0),
             static_cast<int64_t>(embedding.mtp_bootstrap_row_idxes.size()))
        << "MTP bootstrap row count mismatch";

    Slice<int32_t> token_ids = {
        input.token_ids_host.data_ptr<int32_t>(),
        static_cast<size_t>(input.token_ids_host.numel())};
    for (int32_t i = 0;
         i < static_cast<int32_t>(embedding.mtp_bootstrap_row_idxes.size());
         ++i) {
      const int32_t row_idx = embedding.mtp_bootstrap_row_idxes[i];
      CHECK_GE(row_idx, 0) << "MTP bootstrap row index should be valid";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.embedding_ids.size()))
          << "MTP bootstrap row index exceeds embedding ids";
      CHECK_LT(row_idx, static_cast<int32_t>(embedding.request_ids.size()))
          << "MTP bootstrap row index exceeds request ids";
      CHECK_LT(static_cast<int64_t>(row_idx), input.token_ids_host.numel())
          << "MTP bootstrap row index exceeds token ids";
      embedding_cache_->write_mtp_bootstrap_context(
          embedding.embedding_ids[row_idx],
          embedding.request_ids[row_idx],
          token_ids[row_idx],
          bootstrap_embeddings[i]);
    }
  }

  ForwardInput metadata_template = input;
  if (use_prelaunched_first_draft) {
    // The first draft was fully prepared and submitted by the preceding
    // run_validate(). Host metadata is corrected below after the accepted-token
    // update; continuous DSA drafts use device correction, while target
    // verification still consumes the exact Host metadata.
  } else if (use_device_target_context) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    // Clone host tensors before mutating the shallow-copied template.
    metadata_template.token_ids_host =
        clone_host_tensor(metadata_template.token_ids_host);
    metadata_template.positions_host =
        clone_host_tensor(metadata_template.positions_host);

    // Build fixed-shape host metadata immediately while target verification is
    // still running. Use the maximum accepted draft offset for conservative
    // graph planning; actual values replace every device tensor below.
    int32_t* template_positions =
        metadata_template.positions_host.data_ptr<int32_t>();
    int32_t* template_tokens =
        metadata_template.token_ids_host.data_ptr<int32_t>();
    auto& template_kv_lens =
        metadata_template.input_params.attention.host.kv_seq_lens;
    for (int32_t seq_id = 0;
         seq_id < metadata_template.input_params.meta.num_sequences;
         ++seq_id) {
      template_positions[seq_id] += num_speculative_tokens;
      template_kv_lens[seq_id] += num_speculative_tokens;
      if (template_tokens[seq_id] < 0) {
        template_tokens[seq_id] = 0;
      }
    }

    std::vector<EmbeddingCache::DecodeState> template_states(
        metadata_template.input_params.meta.num_sequences);
    const torch::Tensor& placeholder =
        embedding_cache_->embedding_placeholder();
    for (int32_t seq_id = 0;
         seq_id < metadata_template.input_params.meta.num_sequences;
         ++seq_id) {
      template_states[seq_id].valid = true;
      template_states[seq_id].request_id =
          metadata_template.input_params.embedding.request_ids[seq_id];
      template_states[seq_id].token_id = template_tokens[seq_id];
      template_states[seq_id].embedding = placeholder;
    }
    prepare_draft_extend_inputs(metadata_template,
                                template_states,
                                current_draft_input,
                                /*force_two_rows=*/true);
    wait_metadata_ready_event(current_draft_input, *compute_stream_);
    clear_ready_events(current_draft_input);

    mtp_async::prepare_next_draft_from_accepted_state(
        current_draft_input,
        input,
        accepted_tokens,
        accepted_embeddings,
        embedding_cache_->embedding_placeholder(),
        target_base_positions,
        target_base_kv_seq_lens,
        /*use_chunked_prefill=*/false,
        /*rebuild_expanded_decode_metadata=*/true,
        options_.block_size());
  } else {
    // First decode after prefill and batch transitions use the host cache.
    std::vector<EmbeddingCache::DecodeState> last_states =
        embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    CHECK_EQ(last_states.size(),
             input.input_params.embedding.embedding_ids.size())
        << "decode target state count mismatch";
    check_mtp_decode_states(last_states,
                            input.input_params.embedding.request_ids,
                            input.token_ids_host,
                            enable_schedule_overlap());
    update_decode_step_input(input, last_states);
    metadata_template = input;
    prepare_draft_extend_inputs(input, last_states, current_draft_input);
  }

  const bool use_continuous_dsa_drafts =
      (use_device_target_context || use_prelaunched_first_draft) &&
      combined_draft_execution_path_ ==
          mtp_async::CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  std::vector<ForwardInput> later_draft_inputs;
  torch::Tensor accepted_base_positions;
  torch::Tensor accepted_base_kv_seq_lens;
  if (use_continuous_dsa_drafts) {
    later_draft_inputs.resize(num_speculative_tokens);
    const ForwardInput& combined_draft_input =
        use_prelaunched_first_draft ? pending_draft_context_.prepared_input
                                    : current_draft_input;
    const int64_t batch_size = input.input_params.meta.num_sequences;
    CHECK_EQ(combined_draft_input.positions.numel(), batch_size * 2)
        << "combined draft positions must contain [repair,current] rows";
    CHECK_EQ(
        combined_draft_input.input_params.attention.device.kv_seq_lens.numel(),
        batch_size * 2)
        << "combined draft KV lengths must contain [repair,current] rows";
    accepted_base_positions =
        combined_draft_input.positions.view({batch_size, 2}).select(1, 1);
    accepted_base_kv_seq_lens =
        combined_draft_input.input_params.attention.device.kv_seq_lens
            .view({batch_size, 2})
            .select(1, 1);

    for (int32_t draft_idx = 1; draft_idx < num_speculative_tokens;
         ++draft_idx) {
      // Only the fixed B layout is needed on prepare_stream. Reusing offset 0
      // avoids extending an already conservative Host template past the
      // scheduler-allocated block range; device metadata is replaced below.
      prepare_draft_inputs(metadata_template,
                           later_draft_inputs[draft_idx],
                           /*position_offset=*/0);
    }
  }

  const auto materialize_pending_target_host_state = [&]() {
    flush_pending_target_context();
    std::vector<EmbeddingCache::DecodeState> resolved_states =
        embedding_cache_->read_decode_states(
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    // The scheduler input contains the conservative overlap placeholder.
    // Force cache correction before comparing it with the accepted target.
    input.token_ids_host = torch::full_like(input.token_ids_host, -1);
    update_decode_step_input(input, resolved_states);
    check_mtp_decode_states(resolved_states,
                            input.input_params.embedding.request_ids,
                            input.token_ids_host,
                            /*allow_overlap_fake_token=*/false);
    metadata_template = input;
  };

  draft_outputs.reserve(num_speculative_tokens);
  const bool reuse_mtp_topk_state = layer::is_mtp_dsa_topk_reuse_enabled(
      draft_impl_->context_.get_model_args());
  MtpTopkStatePtr mtp_topk_state;
  timer.reset();
  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    const bool is_final_draft = draft_idx == num_speculative_tokens - 1;
    const bool static_graph_tasks_prepared =
        is_final_draft && !use_continuous_dsa_drafts &&
        prepare_static_mtp_graph_tasks_before_final_draft(input);
    if (reuse_mtp_topk_state) {
      current_draft_input.input_params.mtp_topk_state = mtp_topk_state;
    }
    std::optional<ForwardOutput> draft_output_opt;
    if (use_prelaunched_first_draft && draft_idx == 0) {
      draft_output_opt = std::move(pending_draft_context_.output);
      draft_prepared[draft_idx] =
          std::move(pending_draft_context_.prepared_input);
      pending_draft_context_ = PendingDraftContext();
    } else {
      draft_output_opt = run_llm_no_sync_impl(*draft_impl_,
                                              current_draft_input,
                                              *compute_stream_,
                                              *compute_stream_,
                                              draft_prepared[draft_idx]);
    }

    if ((use_device_target_context || use_prelaunched_first_draft) &&
        !use_continuous_dsa_drafts && draft_idx == 0) {
      // The next draft forward is already queued behind target validation.
      // It can start immediately when rejection sampling finishes while the
      // worker materializes the accepted state for later draft/target metadata
      // on CPU.  This synchronization is therefore outside the NPU critical
      // path rather than sitting between target and draft launches.
      materialize_pending_target_host_state();
    }

    if (use_continuous_dsa_drafts && is_final_draft) {
      // Queue target metadata before the Host target-context wait. The fixed
      // template can be copied immediately; its real device values are
      // corrected after a prepare-stream wait on the previous target event.
      prepare_validate_inputs(metadata_template,
                              validate_input,
                              /*static_graph_tasks_prepared=*/false,
                              /*record_ready_event=*/false);
      {
        c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
        CHECK(prepare_stream_->wait_event(target_context_ready_event))
            << "failed to wait pending target state on prepare stream";
        mtp_async::prepare_target_verify_from_accepted_state(
            validate_input,
            accepted_tokens,
            target_base_positions,
            target_base_kv_seq_lens,
            options_.block_size());
        validate_input.retained_device_tensors = {
            accepted_tokens, target_base_positions, target_base_kv_seq_lens};
        finish_metadata_prepare(*prepare_stream_, validate_input);
      }

      // Host cache materialization is still required before staging the next
      // target context, but it no longer blocks target metadata submission.
      materialize_pending_target_host_state();
    }

    // Overlap next-step input preparation with async draft forward.
    if (is_final_draft) {
      if (!use_continuous_dsa_drafts) {
        prepare_validate_inputs(
            metadata_template, validate_input, static_graph_tasks_prepared);
      }
    } else if (use_continuous_dsa_drafts) {
      next_step_input = std::move(later_draft_inputs[draft_idx + 1]);
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      wait_metadata_ready_event(next_step_input, *compute_stream_);
      clear_ready_events(next_step_input);
      mtp_async::prepare_later_draft_from_device_base(next_step_input,
                                                      input,
                                                      accepted_base_positions,
                                                      accepted_base_kv_seq_lens,
                                                      draft_idx + 1,
                                                      options_.block_size());
    } else {
      prepare_draft_inputs(metadata_template, next_step_input, draft_idx + 1);
    }

    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";

    draft_outputs.emplace_back(std::move(draft_output_opt.value()));
    const SamplingParameters& draft_sampling_params =
        draft_prepared[draft_idx].sampling_params;
    {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      if (reuse_mtp_topk_state) {
        mtp_topk_state = specBuilder::select_mtp_topk_state_for_next_step(
            draft_outputs.back().mtp_topk_state, draft_sampling_params);
      }
      // Unify this step's draft next_tokens across the consensus group before
      // process_draft_sample_output() compresses the still-full [batch, vocab]
      // probs into the cache: gathering the cached prob with a unified token
      // yields a unified prob, so we only broadcast the [batch] token tensor.
      if (should_broadcast_spec_tokens(
              parallel_args_,
              get_optimization_config().enable_spec_token_broadcast,
              draft_sampling_params.all_greedy_sample)) {
        SampleOutput& draft_sample = draft_outputs.back().sample_output;
        broadcast_spec_tokens(draft_sample.next_tokens, parallel_args_);
      }
      process_draft_sample_output(draft_outputs.back().sample_output);
    }
    if (draft_idx == num_speculative_tokens - 1) {
      continue;
    }

    const SampleOutput& last_output = draft_outputs.back().sample_output;
    current_draft_input = next_step_input;
    set_token_ids_device_tensor(current_draft_input,
                                last_output.next_tokens,
                                current_draft_input.token_ids.options(),
                                *compute_stream_);
    if (last_output.embeddings.defined()) {
      current_draft_input.input_params.embedding.input_embedding =
          last_output.embeddings;
      // input_embedding is produced and consumed on compute_stream_; FIFO
      // ordering replaces the same-stream EventRecord/EventWait pair.
    }
  }
  const double draft_latency_ms = timer.elapsed_milliseconds();
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_latency_ms / 1000.0);

  if (use_adaptive_speculative_decode) {
    return run_adaptive_validate(
        input, draft_outputs, validate_input, num_speculative_tokens);
  }

  return run_validate(
      input, draft_outputs, validate_input, num_speculative_tokens);
}

void MTPWorkerImpl::fill_validate_input_from_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens,
    Stream& compute_stream) {
  CHECK(!per_seq_val_tokens.empty()) << "per_seq_val_tokens must not be empty";
  const int32_t num_sequences = static_cast<int32_t>(per_seq_val_tokens.size());
  const int32_t max_val_tokens =
      *std::max_element(per_seq_val_tokens.begin(), per_seq_val_tokens.end());
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";

  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);

  validate_input.device_tensors_ready = false;
  auto& fused_draft_tokens =
      validate_input.input_params.graph.spec_verify_draft_token_sources;
  fused_draft_tokens.clear();
#if defined(USE_NPU)
  const bool use_fused_verify_token_update =
      validate_input.input_params.graph.spec_verify_source_addresses_stable &&
      validate_input.input_params.graph.input_tokens_override.defined() &&
      supports_explicit_spec_verify_replay_update() &&
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          max_val_tokens, options_.block_size());
  if (use_fused_verify_token_update) {
    fused_draft_tokens.reserve(draft_outputs.size());
    for (const ForwardOutput& draft_output : draft_outputs) {
      const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
      CHECK(next_tokens.defined() && next_tokens.numel() == num_sequences &&
            next_tokens.scalar_type() == torch::kInt64 &&
            next_tokens.device() == validate_input.token_ids.device() &&
            next_tokens.is_contiguous())
          << "fused speculative verify input update requires one contiguous "
             "int64 token per sequence and draft step";
      fused_draft_tokens.emplace_back(next_tokens.flatten());
    }
    validate_input.device_tensors_ready = true;
    return;
  }
#endif

  const int32_t total_val_tokens =
      static_cast<int32_t>(validate_input.token_ids.numel());
  const bool is_uniform = (total_val_tokens == num_sequences * max_val_tokens);

  if (is_uniform) {
    // Fast path: all seqs share the same val_tokens. Stack the draft-token
    // columns once into a contiguous [batch, num_draft_tokens] tensor and
    // issue a single strided slice-copy, instead of one device kernel per
    // draft step. This is the static (non-adaptive) MTP baseline and runs
    // every decode step; launch-bound NPU decode is measurably faster with
    // one op than num_speculative_tokens ops writing the same bytes.
    const int32_t num_draft_tokens = max_val_tokens - 1;
    torch::Tensor validate_token_rows = validate_input.token_ids.view(
        {static_cast<int64_t>(num_sequences), max_val_tokens});
    if (num_draft_tokens > 0) {
      std::vector<torch::Tensor> draft_token_columns;
      draft_token_columns.reserve(static_cast<size_t>(num_draft_tokens));
      for (int32_t i = 0; i < num_draft_tokens; ++i) {
        CHECK(static_cast<size_t>(i) < draft_outputs.size())
            << "draft_outputs index out of range for step " << i;
        const torch::Tensor& next_tokens =
            draft_outputs[static_cast<size_t>(i)].sample_output.next_tokens;
        CHECK(next_tokens.defined())
            << "draft next_tokens must be defined for validate token fill";
        draft_token_columns.push_back(safe_to(
            next_tokens.flatten(), token_options, /*non_blocking=*/true));
      }
      torch::Tensor packed_drafts =
          torch::stack(draft_token_columns, /*dim=*/1);
      validate_token_rows.slice(/*dim=*/1, /*start=*/1, /*end=*/max_val_tokens)
          .copy_(packed_drafts, /*non_blocking=*/true);
    }
  } else {
    // Slow path: per-seq variable-length, group by draft step.
    std::vector<int64_t> dst_idx_vec;
    std::vector<int64_t> src_idx_vec;
    std::vector<int64_t> step_vec;
    int32_t offset = 0;
    int32_t max_draft_step = -1;
    for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
      const int32_t seq_val_tokens =
          per_seq_val_tokens[static_cast<size_t>(seq_id)];
      const int32_t seq_draft_tokens = seq_val_tokens - 1;
      for (int32_t draft_idx = 0; draft_idx < seq_draft_tokens; ++draft_idx) {
        dst_idx_vec.push_back(offset + draft_idx + 1);
        src_idx_vec.push_back(seq_id);
        step_vec.push_back(draft_idx);
        max_draft_step = std::max(max_draft_step, draft_idx);
      }
      offset += seq_val_tokens;
    }

    if (!dst_idx_vec.empty()) {
      // Move all indices to device once (instead of a per-step H2D copy) and
      // select each step's entries on-device via a boolean mask.
      const torch::TensorOptions long_dev_opts =
          torch::TensorOptions()
              .dtype(torch::kLong)
              .device(validate_input.token_ids.device());
      torch::Tensor dst_idx_all =
          safe_to(torch::tensor(dst_idx_vec,
                                torch::TensorOptions().dtype(torch::kLong)),
                  long_dev_opts,
                  /*non_blocking=*/true);
      torch::Tensor src_idx_all =
          safe_to(torch::tensor(src_idx_vec,
                                torch::TensorOptions().dtype(torch::kLong)),
                  long_dev_opts,
                  /*non_blocking=*/true);
      torch::Tensor step_all = safe_to(
          torch::tensor(step_vec, torch::TensorOptions().dtype(torch::kLong)),
          long_dev_opts,
          /*non_blocking=*/true);
      for (int32_t step = 0; step <= max_draft_step; ++step) {
        CHECK(static_cast<size_t>(step) < draft_outputs.size())
            << "draft_outputs index out of range for step " << step;
        const torch::Tensor& next_tokens =
            draft_outputs[static_cast<size_t>(step)].sample_output.next_tokens;
        CHECK(next_tokens.defined())
            << "draft next_tokens must be defined for validate token fill";
        torch::Tensor flat_tokens = safe_to(
            next_tokens.flatten(), token_options, /*non_blocking=*/true);

        torch::Tensor step_mask = step_all.eq(step);
        torch::Tensor step_dst = dst_idx_all.masked_select(step_mask);
        if (step_dst.numel() == 0) {
          continue;
        }
        torch::Tensor step_src = src_idx_all.masked_select(step_mask);
        torch::Tensor gathered = flat_tokens.index_select(/*dim=*/0, step_src);
        validate_input.token_ids.index_copy_(/*dim=*/0, step_dst, gathered);
      }
    }
  }
  validate_input.device_tensors_ready = true;
  record_metadata_ready_event(compute_stream, validate_input);
}

std::optional<ForwardOutput> MTPWorkerImpl::run_adaptive_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    int32_t num_speculative_tokens) {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  std::vector<double> per_seq_kv_lens(static_cast<size_t>(batch_size), 0.0);
  const Slice<int32_t> kv_seq_lens =
      input.input_params.attention.host.kv_seq_lens;
  for (int32_t i = 0; i < batch_size; ++i) {
    if (static_cast<size_t>(i) < kv_seq_lens.size()) {
      per_seq_kv_lens[static_cast<size_t>(i)] = static_cast<double>(
          specBuilder::calc_kv_len(kv_seq_lens, i, /*offset=*/0));
    }
  }

  // All ranks compute pruning independently. Inputs must be deterministic
  // across ranks so every rank derives the same effective validate width.
  // The per-rank measured draft latency is NOT deterministic, so pruning is
  // driven purely by validate-time marginal cost (full_draft_time_ms = 0).
  std::vector<int32_t> prefix_lengths;
  const bool has_probs = has_selected_probs_by_step(draft_outputs);
  if (has_probs) {
    prefix_lengths = adaptive_spec_controller_->select_pruned_prefix_lengths(
        selected_probs_by_step(draft_outputs),
        /*full_draft_time_ms=*/0.0,
        per_seq_kv_lens);
  } else {
    prefix_lengths.assign(static_cast<size_t>(batch_size),
                          num_speculative_tokens);
  }
  clamp_prefix_lengths(prefix_lengths, batch_size, num_speculative_tokens);
  int32_t effective_speculative_tokens =
      max_pruned_prefix_length(prefix_lengths, num_speculative_tokens);
  if (effective_speculative_tokens <= 0) {
    effective_speculative_tokens = 1;
  }

  // The Qwen3.5 GDN CausalConv1d kernel produces aivec errors when the
  // per-seq validate segment length is smaller than num_accepted_tokens
  // (the previous step's accepted count). The tiling validation would
  // reject nat > lenI, but the underlying kernel itself is fragile at
  // those boundaries. Floor effective_speculative_tokens by the batch's
  // max num_accepted so uniform_val_tokens >= max(nat) + 1. Read from
  // embedding_cache directly since input.num_accepted_tokens_host is
  // populated by prepare_validate_inputs which hasn't run yet here.
  if (supports_explicit_spec_verify_replay_update() &&
      embedding_cache_ != nullptr &&
      !input.input_params.embedding.embedding_ids.empty()) {
    std::vector<int32_t> nat = embedding_cache_->read_accepted_prefix_lengths(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids);
    clamp_gdn_conv_history(nat);
    int32_t max_nat = 0;
    for (int32_t v : nat) {
      max_nat = std::max(max_nat, v);
    }
    effective_speculative_tokens =
        std::max(effective_speculative_tokens, max_nat);
    effective_speculative_tokens =
        std::min(effective_speculative_tokens, num_speculative_tokens);
  }

  std::vector<int32_t> per_seq_val_tokens(static_cast<size_t>(batch_size));
  // Qwen3.5 GatedDeltaNet spec-verify path requires dense same-length validate
  // tokens across sequences (see qwen3_gated_delta_net_base.cpp:405-408). On
  // Qwen3.5 we still take the batch-max pruning benefit (effective_sl < max_sl
  // when the controller decides to shrink), but every seq gets the same
  // validate width. On non-Qwen3.5 models we keep per-seq variable-length
  // tokens for maximum pruning benefit.
  const bool require_uniform_val_tokens = requires_uniform_validate_width();
  const int32_t uniform_val_tokens = effective_speculative_tokens + 1;
  for (int32_t i = 0; i < batch_size; ++i) {
    per_seq_val_tokens[static_cast<size_t>(i)] =
        require_uniform_val_tokens
            ? uniform_val_tokens
            : std::max(prefix_lengths[static_cast<size_t>(i)], 1) + 1;
  }
  std::vector<ForwardOutput> pruned_draft_outputs =
      truncate_draft_outputs(draft_outputs, effective_speculative_tokens);
  // If the controller did not actually prune any sequence, treat this as the
  // static path: pass nullptr so run_validate takes the async handoff tail
  // and skips the no-op pruned post-processing.
  const bool has_actual_prune =
      std::any_of(prefix_lengths.begin(),
                  prefix_lengths.end(),
                  [num_speculative_tokens](int32_t p) {
                    return p < num_speculative_tokens;
                  });
  prepare_validate_inputs(input, validate_input, per_seq_val_tokens);
  return run_validate(input,
                      pruned_draft_outputs,
                      validate_input,
                      effective_speculative_tokens,
                      per_seq_val_tokens,
                      has_actual_prune ? &prefix_lengths : nullptr);
}

std::optional<ForwardOutput> MTPWorkerImpl::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths) {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t val_tokens = num_speculative_tokens + 1;
  std::vector<int32_t> per_seq_val_tokens(static_cast<size_t>(batch_size),
                                          val_tokens);
  return run_validate(input,
                      draft_outputs,
                      validate_input,
                      num_speculative_tokens,
                      per_seq_val_tokens,
                      pruned_prefix_lengths);
}

// Run target model validate with per-seq variable-length support.
// When all seqs have the same val_tokens (uniform case), uses zero-copy view.
// When variable-length, pads logits to [batch * max_val_tokens, vocab] for
// RejectionSampler compatibility, with -inf at padding positions.
std::optional<ForwardOutput> MTPWorkerImpl::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>& per_seq_val_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths) {
  Timer timer;
  ForwardInput target_prepared;
  fill_validate_input_from_draft_outputs(
      draft_outputs, validate_input, per_seq_val_tokens, *compute_stream_);
  ForwardOutput target_output = run_llm_no_sync_impl(*impl_,
                                                     validate_input,
                                                     *compute_stream_,
                                                     *compute_stream_,
                                                     target_prepared)
                                    .value();
  const double target_latency_ms = timer.elapsed_milliseconds();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_latency_ms / 1000.0);

  const int32_t batch_size = static_cast<int32_t>(per_seq_val_tokens.size());
  const int32_t max_val_tokens = num_speculative_tokens + 1;
  const int32_t total_tokens =
      static_cast<int32_t>(target_output.logits.size(0));
  const int32_t vocab_size =
      static_cast<int32_t>(target_output.logits.size(-1));
  const int64_t padded_total =
      static_cast<int64_t>(batch_size) * max_val_tokens;

  // For the uniform fast path we only need to reinterpret target_output.logits
  // as `[padded_total, vocab]` — no ForwardOutput copy required. Only the
  // variable-length slow path materializes a separate padded output.
  std::optional<ForwardOutput> padded_target_output_slow;
  const bool needs_padding =
      (total_tokens != static_cast<int32_t>(padded_total));
  if (needs_padding) {
    // Slow path: per-seq variable-length, scatter into padded layout.
    padded_target_output_slow.emplace(target_output);
    ForwardOutput& padded_target_output = *padded_target_output_slow;
    std::vector<int64_t> dst_indices_vec;
    dst_indices_vec.reserve(static_cast<size_t>(total_tokens));
    for (int32_t i = 0; i < batch_size; ++i) {
      const int32_t seq_tokens = per_seq_val_tokens[static_cast<size_t>(i)];
      for (int32_t j = 0; j < seq_tokens; ++j) {
        dst_indices_vec.push_back(static_cast<int64_t>(i) * max_val_tokens + j);
      }
    }
    torch::Tensor dst_indices =
        torch::tensor(dst_indices_vec,
                      torch::TensorOptions()
                          .dtype(torch::kLong)
                          .device(target_output.logits.device()));

    // Only the padding rows need the -inf sentinel; using empty + a targeted
    // index_fill_ over the complement of dst_indices avoids paying vocab_size
    // × padded_total writes when most rows will be overwritten by index_copy_.
    torch::Tensor padded_logits = torch::empty({padded_total, vocab_size},
                                               target_output.logits.options());
    if (dst_indices_vec.size() < static_cast<size_t>(padded_total)) {
      std::vector<bool> valid(static_cast<size_t>(padded_total), false);
      for (int64_t idx : dst_indices_vec) {
        valid[static_cast<size_t>(idx)] = true;
      }
      std::vector<int64_t> pad_indices_vec;
      pad_indices_vec.reserve(static_cast<size_t>(padded_total) -
                              dst_indices_vec.size());
      for (int64_t i = 0; i < padded_total; ++i) {
        if (!valid[static_cast<size_t>(i)]) {
          pad_indices_vec.push_back(i);
        }
      }
      torch::Tensor pad_indices =
          torch::tensor(pad_indices_vec,
                        torch::TensorOptions()
                            .dtype(torch::kLong)
                            .device(target_output.logits.device()));
      padded_logits.index_fill_(/*dim=*/0, pad_indices, -1e9);
    }
    padded_logits.index_copy_(/*dim=*/0, dst_indices, target_output.logits);
    padded_target_output.logits = padded_logits;

    torch::Tensor padded_next_tokens = torch::zeros(
        {padded_total}, target_output.sample_output.next_tokens.options());
    padded_next_tokens.index_copy_(
        /*dim=*/0, dst_indices, target_output.sample_output.next_tokens);
    padded_target_output.sample_output.next_tokens = padded_next_tokens;

    if (target_output.sample_output.embeddings.defined()) {
      const int32_t hidden_size =
          static_cast<int32_t>(target_output.sample_output.embeddings.size(-1));
      torch::Tensor padded_embeddings =
          torch::zeros({padded_total, hidden_size},
                       target_output.sample_output.embeddings.options());
      padded_embeddings.index_copy_(
          /*dim=*/0, dst_indices, target_output.sample_output.embeddings);
      padded_target_output.sample_output.embeddings = padded_embeddings;
    }

    // Pad sampled logprobs / top_tokens / top_logprobs to [padded_total, ...]
    // so downstream sync_pruned_boundary_{logprobs,top_logprobs} can safely
    // view them as [batch, max_val_tokens]. Without this the shape CHECKs
    // in the helpers abort on any actually-pruned adaptive step when the
    // target sampler produced logprobs (non-Qwen3.5 targets + logprobs on).
    if (target_output.sample_output.logprobs.defined()) {
      torch::Tensor padded_logprobs = torch::zeros(
          {padded_total}, target_output.sample_output.logprobs.options());
      padded_logprobs.index_copy_(
          /*dim=*/0, dst_indices, target_output.sample_output.logprobs);
      padded_target_output.sample_output.logprobs = padded_logprobs;
    }
    if (target_output.sample_output.top_tokens.defined()) {
      const int64_t top_k = target_output.sample_output.top_tokens.size(-1);
      torch::Tensor padded_top_tokens =
          torch::zeros({padded_total, top_k},
                       target_output.sample_output.top_tokens.options());
      padded_top_tokens.index_copy_(
          /*dim=*/0, dst_indices, target_output.sample_output.top_tokens);
      padded_target_output.sample_output.top_tokens = padded_top_tokens;
    }
    if (target_output.sample_output.top_logprobs.defined()) {
      const int64_t top_k = target_output.sample_output.top_logprobs.size(-1);
      torch::Tensor padded_top_logprobs =
          torch::zeros({padded_total, top_k},
                       target_output.sample_output.top_logprobs.options());
      padded_top_logprobs.index_copy_(
          /*dim=*/0, dst_indices, target_output.sample_output.top_logprobs);
      padded_target_output.sample_output.top_logprobs = padded_top_logprobs;
    }
  }
  // Uniform fast path uses a scoped local ForwardOutput whose only diff is
  // logits viewed to [padded_total, vocab]; slow path uses the materialized
  // padded copy above. Both are const-ref'd into validate() below.
  ForwardOutput uniform_target_view;
  if (!needs_padding) {
    uniform_target_view = target_output;
    uniform_target_view.logits =
        target_output.logits.view({padded_total, vocab_size});
  }
  const ForwardOutput& target_output_for_validate =
      needs_padding ? *padded_target_output_slow : uniform_target_view;

  const bool prelaunch_next_first_draft =
      pruned_prefix_lengths == nullptr && can_prelaunch_next_first_draft(input);
  ForwardInput next_first_draft_input;
  if (prelaunch_next_first_draft) {
    // This input is independent of the accepted token.  Prepare it on the
    // auxiliary stream while target verification is still executing; the
    // compute stream consumes it through a device-side event after rejection
    // sampling, with no host synchronization.
    prepare_next_first_draft_template(input, next_first_draft_input);
  }

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    val_output = validate(input.sampling_params,
                          draft_outputs,
                          target_output_for_validate,
                          num_speculative_tokens,
                          pruned_prefix_lengths);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  if (pruned_prefix_lengths != nullptr) {
    // Adaptive pruning path: per-seq validate width is variable, which is
    // incompatible with the async handoff's fixed-width base-state derivation.
    // Use the synchronous tail: unify tokens, then write target context inline.
    if (should_broadcast_spec_tokens(
            parallel_args_,
            get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.all_greedy_sample)) {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      broadcast_spec_tokens(val_output.next_tokens, parallel_args_);
    }

    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
    release_retained_inputs(target_output);
    val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
    // Record adaptive-prune-aware draft/accept counts on the already-CPU
    // next_tokens. Static path lets worker_service count on the async-handoff
    // CPU tensor with no extra device sync.
    record_validate_metrics(
        val_output, num_speculative_tokens, pruned_prefix_lengths);
    write_target_context_to_cache(input, val_output, num_speculative_tokens);

    if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
      return std::nullopt;
    }
    clear_all_output_embeddings(target_output);
    val_output.embeddings = torch::Tensor();
    target_output.sample_output = val_output;
    return target_output;
  }

  const int64_t num_val_tokens = options_.num_speculative_tokens() + 1;
  CHECK_EQ(validate_input.positions.numel(), batch_size * num_val_tokens)
      << "validate positions must contain one row per speculative token";
  const torch::Tensor& validate_kv_seq_lens =
      validate_input.input_params.attention.device.kv_seq_lens;
  CHECK_GE(validate_kv_seq_lens.numel(), batch_size)
      << "validate KV lengths must be sequence-scoped";
  torch::Tensor accepted_tokens_host =
      acquire_accepted_tokens_host_buffer(val_output.next_tokens);
  torch::Tensor accepted_tokens_cpu_result = accepted_tokens_host;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
  StreamEventPtr target_context_ready_event;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

    // Catch-all for cross-rank RNG divergence: unify accepted tokens before
    // deriving any device-resident state used by the next draft iteration.
    if (should_broadcast_spec_tokens(
            parallel_args_,
            get_optimization_config().enable_spec_token_broadcast,
            input.sampling_params.all_greedy_sample)) {
      broadcast_spec_tokens(val_output.next_tokens, parallel_args_);
    }

    base_positions = validate_input.positions.view({batch_size, num_val_tokens})
                         .select(/*dim=*/1, /*index=*/0)
                         .contiguous();
    base_kv_seq_lens = mtp_async::extract_target_base_kv_seq_lens(
        validate_kv_seq_lens,
        batch_size,
        num_val_tokens,
        use_chunked_prefill_spec_verify_path());

    accepted_tokens_host.copy_(val_output.next_tokens,
                               /*non_blocking=*/true);
    // The event covers consensus, base-state derivation, and the D2H copy.
    target_context_ready_event = compute_stream_->record_event();
  }
  if (target_context_ready_event == nullptr) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize MTP target context, ret=" << ret;
  }
  stage_target_context_write(input,
                             val_output,
                             base_positions,
                             base_kv_seq_lens,
                             target_context_ready_event,
                             std::move(accepted_tokens_host));
  if (prelaunch_next_first_draft) {
    // Submit the next iteration's first draft before returning to the
    // scheduler.  This is the actual asynchronous boundary: scheduler/host
    // accepted-state work can no longer sit between target validation and the
    // next draft launch.
    enqueue_next_first_draft(input,
                             val_output,
                             base_positions,
                             base_kv_seq_lens,
                             std::move(next_first_draft_input));
  }
  target_output.ready_event = target_context_ready_event;

  // Target validation consumes all draft outputs on the same compute stream.
  // Keep their prepared inputs alive until the target-context event completes.
  for (const ForwardOutput& draft_output : draft_outputs) {
    if (draft_output.retained_input != nullptr) {
      target_output.retained_input_dependencies.emplace_back(
          draft_output.retained_input);
    }
    target_output.retained_input_dependencies.insert(
        target_output.retained_input_dependencies.end(),
        draft_output.retained_input_dependencies.begin(),
        draft_output.retained_input_dependencies.end());
  }

  if (!enable_schedule_overlap()) {
    flush_pending_target_context();
    release_retained_inputs(target_output);
    target_output.ready_event.reset();
    val_output.next_tokens = std::move(accepted_tokens_cpu_result);
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  clear_all_output_embeddings(target_output);
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void MTPWorkerImpl::stage_target_context_write(
    const ForwardInput& input,
    const SampleOutput& validate_output,
    torch::Tensor base_positions,
    torch::Tensor base_kv_seq_lens,
    StreamEventPtr ready_event,
    torch::Tensor accepted_tokens_host) {
  CHECK(!pending_target_context_.accepted_tokens.defined())
      << "previous MTP target context must be flushed before staging another";
  pending_target_context_.embedding_ids =
      input.input_params.embedding.embedding_ids;
  pending_target_context_.request_ids =
      input.input_params.embedding.request_ids;
  pending_target_context_.accepted_tokens = validate_output.next_tokens;
  pending_target_context_.accepted_tokens_host =
      std::move(accepted_tokens_host);
  pending_target_context_.accepted_embeddings = validate_output.embeddings;
  pending_target_context_.base_positions = std::move(base_positions);
  pending_target_context_.base_kv_seq_lens = std::move(base_kv_seq_lens);
  pending_target_context_.ready_event = std::move(ready_event);
}

torch::Tensor MTPWorkerImpl::acquire_accepted_tokens_host_buffer(
    const torch::Tensor& accepted_tokens) {
  CHECK(accepted_tokens.defined()) << "accepted tokens must be defined";
  CHECK_GT(accepted_tokens.numel(), 0) << "accepted tokens must not be empty";
  CHECK(!pending_target_context_.accepted_tokens.defined())
      << "accepted-token host buffer is still in use";

  const int64_t required_capacity = accepted_tokens.numel();
  const int64_t configured_capacity =
      static_cast<int64_t>(options_.max_seqs_per_batch()) *
      (static_cast<int64_t>(options_.num_speculative_tokens()) + 1);
  const bool needs_allocation =
      !accepted_tokens_host_buffer_.defined() ||
      accepted_tokens_host_buffer_.scalar_type() !=
          accepted_tokens.scalar_type() ||
      accepted_tokens_host_buffer_.numel() < required_capacity;
  if (needs_allocation) {
    const int64_t capacity = std::max(required_capacity, configured_capacity);
    accepted_tokens_host_buffer_ = torch::empty(
        {capacity},
        accepted_tokens.options().device(torch::kCPU).pinned_memory(true));
  }

  return accepted_tokens_host_buffer_
      .narrow(/*dim=*/0, /*start=*/0, required_capacity)
      .view(accepted_tokens.sizes());
}

bool MTPWorkerImpl::pending_target_context_matches(
    const ForwardInput& input) const {
  return pending_target_context_.accepted_tokens.defined() &&
         pending_target_context_.embedding_ids ==
             input.input_params.embedding.embedding_ids &&
         pending_target_context_.request_ids ==
             input.input_params.embedding.request_ids;
}

bool MTPWorkerImpl::device_target_context_ready_for_batch(
    const ForwardInput& input) const {
  return device_context_ready_embedding_ids_ ==
             input.input_params.embedding.embedding_ids &&
         device_context_ready_request_ids_ ==
             input.input_params.embedding.request_ids;
}

void MTPWorkerImpl::flush_pending_target_context() {
  if (!pending_target_context_.accepted_tokens.defined()) {
    return;
  }
  CHECK(pending_target_context_.ready_event == nullptr ||
        pending_target_context_.ready_event->synchronize())
      << "failed to wait for pending MTP target context";
  CHECK(embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before target cache write";
  embedding_cache_->write_target_context(
      pending_target_context_.embedding_ids,
      pending_target_context_.request_ids,
      pending_target_context_.accepted_tokens_host,
      pending_target_context_.accepted_embeddings,
      options_.num_speculative_tokens());
  pending_target_context_ = PendingTargetContext();
}

void MTPWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output,
    int32_t num_speculative_tokens) {
  CHECK(embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before target cache write";
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "target context cache write requires embedding ids";
  embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      num_speculative_tokens);
}

bool MTPWorkerImpl::supports_combined_first_draft_execution() const {
#if defined(USE_NPU)
  if (draft_impl_ == nullptr ||
      draft_impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    return false;
  }

  // The ATB speculative path expects CHUNKED_PREFILL metadata instead of the
  // eager two-row DECODE input used by the prelaunch.
  if (::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    return false;
  }

  const std::string& npu_backend =
      ::xllm::KernelConfig::get_instance().npu_kernel_backend();
  return device_.unwrap().is_privateuseone() &&
         mtp_async::supports_combined_draft_configuration(
             combined_draft_execution_path_,
             npu_backend,
             parallel_args_.dp_size());
#else
  return false;
#endif
}

bool MTPWorkerImpl::can_use_combined_first_draft() const {
  return enable_schedule_overlap() && supports_combined_first_draft_execution();
}

bool MTPWorkerImpl::can_prelaunch_next_first_draft(
    const ForwardInput& input) const {
  if (!can_use_combined_first_draft()) {
    return false;
  }
  const bool requires_dp_symmetric_prelaunch =
      parallel_args_.dp_size() > 1 &&
      combined_draft_execution_path_ ==
          mtp_async::CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION;
  if (requires_dp_symmetric_prelaunch) {
    return has_active_dp_tokens(input);
  }
  return device_target_context_ready_for_batch(input);
}

void MTPWorkerImpl::prepare_next_first_draft_template(
    const ForwardInput& input,
    ForwardInput& combined_input) {
  CHECK(embedding_cache_ != nullptr);

  ForwardInput metadata_template = input;
  // Clone host tensors before mutating the shallow-copied template.
  metadata_template.token_ids_host =
      clone_host_tensor(metadata_template.token_ids_host);
  metadata_template.positions_host =
      clone_host_tensor(metadata_template.positions_host);
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  int32_t* template_positions =
      metadata_template.positions_host.data_ptr<int32_t>();
  int32_t* template_tokens =
      metadata_template.token_ids_host.data_ptr<int32_t>();
  auto& template_kv_lens =
      metadata_template.input_params.attention.host.kv_seq_lens;
  for (int32_t seq_id = 0;
       seq_id < metadata_template.input_params.meta.num_sequences;
       ++seq_id) {
    template_positions[seq_id] += num_speculative_tokens;
    template_kv_lens[seq_id] += num_speculative_tokens;
    if (template_tokens[seq_id] < 0) {
      template_tokens[seq_id] = 0;
    }
  }

  std::vector<EmbeddingCache::DecodeState> template_states(
      metadata_template.input_params.meta.num_sequences);
  const torch::Tensor& placeholder = embedding_cache_->embedding_placeholder();
  for (int32_t seq_id = 0;
       seq_id < metadata_template.input_params.meta.num_sequences;
       ++seq_id) {
    template_states[seq_id].valid = true;
    template_states[seq_id].request_id =
        metadata_template.input_params.embedding.request_ids[seq_id];
    template_states[seq_id].token_id = template_tokens[seq_id];
    template_states[seq_id].embedding = placeholder;
  }

  prepare_draft_extend_inputs(metadata_template,
                              template_states,
                              combined_input,
                              /*force_two_rows=*/true,
                              /*wait_for_compute_stream=*/false);
  combined_input.skip_sampling_for_logits_only = false;
}

void MTPWorkerImpl::enqueue_next_first_draft(
    const ForwardInput& input,
    const SampleOutput& validate_output,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    ForwardInput combined_input) {
  CHECK(validate_output.next_tokens.defined());
  CHECK(validate_output.embeddings.defined());
  CHECK(embedding_cache_ != nullptr);

  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  wait_metadata_ready_event(combined_input, *compute_stream_);
  clear_ready_events(combined_input);

  // Interleave [repair, current] rows in one decode batch. Every transformer
  // layer projects both rows, writes both KV rows, and only then launches
  // PagedAttention. Same-stream ordering therefore makes repair KV visible to
  // the current row without a host wait or a separate repair forward.
  mtp_async::prepare_next_draft_from_accepted_state(
      combined_input,
      input,
      validate_output.next_tokens,
      validate_output.embeddings,
      embedding_cache_->embedding_placeholder(),
      base_positions,
      base_kv_seq_lens,
      /*use_chunked_prefill=*/false,
      /*rebuild_expanded_decode_metadata=*/false,
      options_.block_size());

  submit_pending_first_draft(input, std::move(combined_input));
}

void MTPWorkerImpl::submit_pending_first_draft(
    const ForwardInput& batch_identity_input,
    ForwardInput draft_input) {
  CHECK(!pending_draft_context_.output.has_value())
      << "MTP first-draft prelaunch was not consumed";
  pending_draft_context_.embedding_ids =
      batch_identity_input.input_params.embedding.embedding_ids;
  pending_draft_context_.request_ids =
      batch_identity_input.input_params.embedding.request_ids;
  pending_draft_context_.dp_global_token_nums =
      batch_identity_input.input_params.parallel.dp_global_token_nums;
  pending_draft_context_.raw_dp_global_token_nums =
      batch_identity_input.input_params.parallel.raw_dp_global_token_nums;
  pending_draft_context_.dp_global_batch_generations =
      batch_identity_input.input_params.parallel.dp_global_batch_generations;
  pending_draft_context_.output =
      run_llm_no_sync_impl(*draft_impl_,
                           draft_input,
                           *compute_stream_,
                           *compute_stream_,
                           pending_draft_context_.prepared_input);
  CHECK(pending_draft_context_.output.has_value())
      << "failed to prelaunch next MTP first draft";
}

bool MTPWorkerImpl::pending_draft_context_matches(
    const ForwardInput& input) const {
  return pending_draft_context_.output.has_value() &&
         pending_draft_context_.embedding_ids ==
             input.input_params.embedding.embedding_ids &&
         pending_draft_context_.request_ids ==
             input.input_params.embedding.request_ids &&
         pending_draft_context_.dp_global_token_nums ==
             input.input_params.parallel.dp_global_token_nums &&
         pending_draft_context_.raw_dp_global_token_nums ==
             input.input_params.parallel.raw_dp_global_token_nums &&
         pending_draft_context_.dp_global_batch_generations ==
             input.input_params.parallel.dp_global_batch_generations;
}

void MTPWorkerImpl::record_validate_metrics(
    const SampleOutput& validate_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths) const {
  CHECK(validate_output.next_tokens.defined())
      << "validate output tokens are undefined";
  CHECK_EQ(validate_output.next_tokens.dim(), 2)
      << "validate output tokens should be [batch, width]";
  const int32_t batch_size =
      static_cast<int32_t>(validate_output.next_tokens.size(0));
  CHECK_EQ(validate_output.next_tokens.size(1), num_speculative_tokens + 1)
      << "validate output width mismatch";

  CHECK(validate_output.next_tokens.device().is_cpu())
      << "record_validate_metrics expects next_tokens already on CPU to avoid "
         "a blocking device sync on the hot path";
  torch::Tensor next_tokens_cpu =
      validate_output.next_tokens.to(torch::kInt64).contiguous();
  const int64_t* token_data = next_tokens_cpu.const_data_ptr<int64_t>();
  int64_t num_draft_tokens = 0;
  int64_t accepted_count = 0;
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    int32_t prefix_len = num_speculative_tokens;
    if (pruned_prefix_lengths != nullptr) {
      CHECK_EQ(pruned_prefix_lengths->size(), static_cast<size_t>(batch_size))
          << "adaptive pruning prefix length batch mismatch";
      prefix_len =
          std::clamp((*pruned_prefix_lengths)[static_cast<size_t>(seq_id)],
                     0,
                     num_speculative_tokens);
    }
    num_draft_tokens += prefix_len;

    const int64_t row_offset = static_cast<int64_t>(seq_id) *
                               static_cast<int64_t>(num_speculative_tokens + 1);
    int32_t emitted_len = 0;
    for (int32_t token_idx = 0; token_idx <= num_speculative_tokens;
         ++token_idx) {
      if (token_data[row_offset + token_idx] < 0) {
        break;
      }
      ++emitted_len;
    }
    accepted_count += std::min(prefix_len, std::max(emitted_len - 1, 0));
  }
  COUNTER_ADD(speculative_num_draft_tokens_total, num_draft_tokens);
  COUNTER_ADD(speculative_num_accepted_tokens_total, accepted_count);
}

bool MTPWorkerImpl::adaptive_enabled() const {
  return adaptive_spec_controller_ != nullptr &&
         adaptive_spec_controller_->enabled();
}

void MTPWorkerImpl::process_draft_sample_output(SampleOutput& sample_output) {
  if (sample_output.probs.defined()) {
    CHECK(sample_output.next_tokens.defined())
        << "draft sample_output.next_tokens must be defined when probs exist";
    CHECK_EQ(sample_output.next_tokens.dim(), 1)
        << "MTP draft cache expects next_tokens [batch], got "
        << sample_output.next_tokens.sizes();
    CHECK(sample_output.probs.dim() == 1 || sample_output.probs.dim() == 2)
        << "MTP draft cache expects probs [batch] or [batch,vocab], got "
        << sample_output.probs.sizes();
    CHECK_EQ(sample_output.probs.size(0), sample_output.next_tokens.size(0))
        << "MTP draft cache probs/token batch mismatch";
    // Cache always stores selected-only draft probs [batch_size] to reduce HBM.
    sample_output.probs = specBuilder::draftProbs::compress_for_cache(
        sample_output.probs, sample_output.next_tokens);
  }
}

void MTPWorkerImpl::update_decode_step_input(
    ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states) const {
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "decode context state count mismatch";
  const bool enable_cache_correction = enable_schedule_overlap();

  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  token_ids_vec.reserve(num_sequences);
  positions_vec.reserve(num_sequences);
#if defined(USE_NPU)
  kv_seq_lens_vec.reserve(num_sequences);
#else
  kv_seq_lens_vec.reserve(num_sequences + 1);
#endif

  const torch::Tensor& token_ids_cpu = input.token_ids_host;
  const torch::Tensor& positions_cpu = input.positions_host;
  Slice<int32_t> input_token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(token_ids_cpu.numel())};
  Slice<int32_t> input_positions = {positions_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(positions_cpu.numel())};

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    CHECK_LT(static_cast<size_t>(seq_id), input_token_ids.size())
        << "decode context token seq_id out of range, seq_id=" << seq_id;
    CHECK_LT(static_cast<size_t>(seq_id), input_positions.size())
        << "decode context position seq_id out of range, seq_id=" << seq_id;
    const EmbeddingCache::DecodeState& state = last_states[seq_id];
    const int32_t input_token_id = input_token_ids[seq_id];
    const bool input_is_fake_token = input_token_id < 0;
    const bool use_cache_correction =
        enable_cache_correction && input_is_fake_token && state.valid;
    const bool use_fake_context =
        enable_cache_correction && input_is_fake_token && !state.valid;
    const int32_t position_offset =
        use_cache_correction ? state.position_offset : 0;
    int32_t current_position = input_positions[seq_id] + position_offset;
    int32_t current_kv_len = specBuilder::calc_kv_len(
        input.input_params.attention.host.kv_seq_lens, seq_id, position_offset);
    int32_t expected_kv_len = current_position + 1;
    if (use_chunked_prefill_spec_verify_path()) {
      const torch::Tensor& block_tables =
          input.input_params.attention.host.block_tables;
      if (block_tables.defined() && block_tables.dim() == 2 &&
          seq_id < block_tables.size(0)) {
        const int32_t allocated_kv_len =
            static_cast<int32_t>(block_tables.size(1)) * options_.block_size();
        const int32_t validate_width = options_.num_speculative_tokens() + 1;
        const int32_t max_valid_position = allocated_kv_len - validate_width;
        if (current_position > max_valid_position) {
          CHECK_GT(allocated_kv_len, 0)
              << "decode context has empty block table, seq_id=" << seq_id;
          CHECK_GE(max_valid_position, 0)
              << "decode context block table is too small for validation, "
              << "seq_id=" << seq_id
              << ", allocated_kv_len=" << allocated_kv_len
              << ", validate_width=" << validate_width;
          CHECK_LE(current_position - max_valid_position,
                   options_.num_speculative_tokens() + 1)
              << "decode context position exceeds allocated blocks, seq_id="
              << seq_id << ", current_position=" << current_position
              << ", current_kv_len=" << current_kv_len
              << ", allocated_kv_len=" << allocated_kv_len;
          current_position = max_valid_position;
          expected_kv_len = current_position + 1;
          current_kv_len = std::min(current_kv_len, expected_kv_len);
        }
      }
    }
    if (use_chunked_prefill_spec_verify_path() &&
        current_kv_len < expected_kv_len) {
      // Qwen3.5/MiMo MTP can receive a scheduler KV length that has not yet
      // caught up with the speculative placeholder resolved into
      // current_position. Normalize only the lag explainable by the current
      // speculative step.
      CHECK_LE(expected_kv_len - current_kv_len,
               options_.num_speculative_tokens() + 1)
          << "decode context kv_len lag is too large, seq_id=" << seq_id
          << ", current_position=" << current_position
          << ", current_kv_len=" << current_kv_len;
      current_kv_len = expected_kv_len;
    }
    if (use_chunked_prefill_spec_verify_path() &&
        current_kv_len > expected_kv_len) {
      // The first decode step can carry the prompt KV length while the decode
      // position is still initialized to zero. Align the position to the KV
      // context before building the MTP draft input.
      current_position = current_kv_len - 1;
      expected_kv_len = current_kv_len;
    }

    CHECK_EQ(expected_kv_len, current_kv_len)
        << "decode context position/kv_len mismatch, seq_id=" << seq_id
        << ", current_position=" << current_position
        << ", current_kv_len=" << current_kv_len;

    token_ids_vec.emplace_back((use_cache_correction || use_fake_context)
                                   ? state.token_id
                                   : input_token_id);
    positions_vec.emplace_back(current_position);
    specBuilder::append_seq_len_by_layout(kv_seq_lens_vec, current_kv_len);
  }

  input.token_ids_host = specBuilder::make_cpu_int_tensor(token_ids_vec);
  input.positions_host = specBuilder::make_cpu_int_tensor(positions_vec);
  input.input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input.device_tensors_ready = false;
}

void MTPWorkerImpl::prepare_validate_inputs(const ForwardInput& input,
                                            ForwardInput& validate_input,
                                            bool static_graph_tasks_prepared,
                                            bool record_ready_event) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input;
  clear_ready_events(validate_input);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t block_size = options_.block_size();
#if defined(USE_NPU)
  const bool use_explicit_spec_verify_replay_update =
      should_use_explicit_spec_verify_replay_update(input);
#else
  const bool use_explicit_spec_verify_replay_update = false;
#endif
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  Slice<int32_t> positions = {
      input.positions_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.positions_host.numel())};
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  const bool use_atb_spec_kernel =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      use_chunked_prefill_spec_verify_path();
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!use_atb_spec_kernel) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                                 row_ctx.block_table_stride);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec;
  std::vector<int32_t> atb_q_seq_lens_vec;
  std::vector<int32_t> atb_q_cu_seq_lens_vec;
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t start_position = positions[seq_id];
    const int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    CHECK_EQ(start_position + 1, kv_len)
        << "validate position/kv_len mismatch, seq_id=" << seq_id
        << ", start_position=" << start_position << ", kv_len=" << kv_len;

    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = val_idx == 0 ? token_ids[seq_id] : -val_idx;
      row.position_offset = val_idx;
      row.append_kv_len = !use_atb_spec_kernel;
      row.append_q_len_one = !use_atb_spec_kernel;
      row.append_block_table = !use_atb_spec_kernel;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
    }

    if (use_atb_spec_kernel) {
      const int32_t kv_len_after_validation =
          kv_len + options_.num_speculative_tokens();
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          atb_q_seq_lens_vec, atb_q_cu_seq_lens_vec, num_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  if (!use_explicit_spec_verify_replay_update) {
    specBuilder::set_token_position_tensors(validate_input,
                                            buf.out_token_ids,
                                            buf.out_positions,
                                            token_options,
                                            position_options);
  }
  if (!use_atb_spec_kernel) {
    input_params.meta.num_sequences = total_num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (use_atb_spec_kernel) {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     num_val_tokens,
                                     std::move(atb_q_seq_lens_vec),
                                     std::move(atb_q_cu_seq_lens_vec),
                                     atb_kv_max_seq_len,
                                     std::move(atb_kv_seq_lens_vec));
  } else {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }

  auto& validate_sampling_params = validate_input.sampling_params;
#if defined(USE_NPU)
  // update_sampling_params() uses repeat_interleave on device tensors.  For a
  // single greedy sequence that work only expands [0] and [false] into fixed
  // validation controls, yet it lands on the final-draft -> target dependency
  // chain.  Reuse stable controls after warmup and retain the generic builder
  // for sampling, penalties, multi-sequence batches and other backends.
  const bool use_stable_greedy_validate_sampling =
      use_explicit_spec_verify_replay_update &&
      validate_sampling_params.all_greedy_sample &&
      validate_sampling_params.selected_token_idxes.defined() &&
      validate_sampling_params.selected_token_idxes.numel() == 1 &&
      validate_sampling_params.sample_idxes.defined() &&
      validate_sampling_params.sample_idxes.numel() == 1 &&
      validate_sampling_params.do_sample.defined() &&
      validate_sampling_params.do_sample.numel() == 1 &&
      !validate_sampling_params.frequency_penalties.defined() &&
      !validate_sampling_params.presence_penalties.defined() &&
      !validate_sampling_params.repetition_penalties.defined() &&
      !validate_sampling_params.temperatures.defined() &&
      !validate_sampling_params.top_p.defined() &&
      !validate_sampling_params.top_k.defined() &&
      !validate_sampling_params.unique_token_ids.defined() &&
      !validate_sampling_params.unique_token_counts.defined() &&
      !validate_sampling_params.unique_token_ids_lens.defined();
  if (use_stable_greedy_validate_sampling) {
    if (!mtp_validate_greedy_indices_.defined() ||
        mtp_validate_greedy_indices_.numel() != total_num_val_tokens) {
      mtp_validate_greedy_indices_ = torch::arange(
          total_num_val_tokens,
          torch::TensorOptions().dtype(torch::kInt).device(device_));
      mtp_validate_greedy_do_sample_ = torch::zeros(
          {total_num_val_tokens},
          torch::TensorOptions().dtype(torch::kBool).device(device_));
    }
    validate_sampling_params.selected_token_idxes =
        mtp_validate_greedy_indices_;
    validate_sampling_params.sample_idxes = mtp_validate_greedy_indices_;
    validate_sampling_params.do_sample = mtp_validate_greedy_do_sample_;
    validate_sampling_params.all_random_sample = false;
    validate_sampling_params.all_greedy_sample = true;
  } else {
    update_sampling_params(
        validate_sampling_params, num_val_tokens, total_num_val_tokens);
  }
#else
  update_sampling_params(
      validate_sampling_params, num_val_tokens, total_num_val_tokens);
#endif

  for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
    token_num *= num_val_tokens;
  }
  for (int32_t& token_num : input_params.parallel.raw_dp_global_token_nums) {
    token_num *= num_val_tokens;
  }

  std::vector<int32_t> accepted_prefix_lengths;
  if (use_chunked_prefill_spec_verify_path()) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    if (!input_params.attention.host.q_seq_lens.empty()) {
      std::vector<int32_t> q_cu_seq_lens_vec;
      q_cu_seq_lens_vec.reserve(input_params.meta.num_sequences + 1);
      q_cu_seq_lens_vec.emplace_back(0);
      for (int32_t q_len : input_params.attention.host.q_seq_lens) {
        q_cu_seq_lens_vec.emplace_back(q_cu_seq_lens_vec.back() + q_len);
      }
      input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
    }
    accepted_prefix_lengths.assign(num_sequences, 1);
    if (embedding_cache_ != nullptr &&
        !input.input_params.embedding.embedding_ids.empty()) {
      accepted_prefix_lengths = embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
    }
    // Clamp num_accepted_tokens to Qwen3.5 GDN conv_state history capacity
    // (kernel_conv_size - 1 = 3). Larger values describe history that has
    // already rolled out of conv_state; passing them to aclnnCausalConv1d
    // makes tiling fail when the current step's per_seq_val_tokens is small
    // (e.g. adaptive prunes down to 2 while nat=5).
    if (supports_explicit_spec_verify_replay_update()) {
      clamp_gdn_conv_history(accepted_prefix_lengths);
    }
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
    if (!use_explicit_spec_verify_replay_update) {
      input_params.num_accepted_tokens =
          torch::tensor(accepted_prefix_lengths, token_options);
    }
  }

#if defined(USE_NPU)
  if (use_explicit_spec_verify_replay_update) {
    build_expanded_spec_verify_graph_host_input(input_params);

    auto& attention = input_params.attention;
    CHECK(attention.host.block_tables.defined());
    CHECK_EQ(attention.host.block_tables.dim(), 2);
    CHECK_EQ(attention.host.block_tables.size(0), num_sequences);
    CHECK_EQ(attention.host.block_tables.scalar_type(), torch::kInt32);
    const int64_t active_block_table_width =
        attention.host.block_tables.size(1);
    const int64_t verify_block_table_width =
        spec_verify_block_table_width(attention.host.block_tables);
    if (active_block_table_width != verify_block_table_width) {
      torch::Tensor padded_block_tables = torch::zeros(
          {num_sequences, verify_block_table_width},
          attention.host.block_tables.options().device(torch::kCPU));
      padded_block_tables.narrow(1, 0, active_block_table_width)
          .copy_(attention.host.block_tables);
      attention.host.block_tables = std::move(padded_block_tables);
    }

    const bool initialize_stable_buffer =
        !spec_verify_attention_host_buffer_.defined();
    attention.attention_host_buffer = spec_verify_attention_host_buffer_;
    attention.attention_device_buffer = spec_verify_attention_device_buffer_;
    attention.attention_buffer_bytes = 0;
    attention.attention_buffer_capacity =
        spec_verify_attention_buffer_capacity_;
    attention.attention_buffer_owner = spec_verify_attention_buffer_owner_;

    const int64_t expanded_block_rows = static_cast<int64_t>(
        input_params.graph.expanded_kv_seq_lens_vec.size());
    CHECK_EQ(expanded_block_rows, total_num_val_tokens);
    const torch::Tensor dense_block_source =
        attention.host.block_tables.contiguous();
    std::vector<int32_t> expanded_block_tables_dense(
        static_cast<size_t>(expanded_block_rows * verify_block_table_width));
    const int32_t* block_row = dense_block_source.data_ptr<int32_t>();
    for (int64_t row = 0; row < expanded_block_rows; ++row) {
      const int64_t sequence_index = row / num_val_tokens;
      std::memcpy(
          expanded_block_tables_dense.data() + row * verify_block_table_width,
          block_row + sequence_index * verify_block_table_width,
          static_cast<size_t>(verify_block_table_width) * sizeof(int32_t));
    }
    torch::Tensor expanded_block_tables_flat;
    std::vector<AttentionInput::PackedIntInput> extra_int_inputs;
    extra_int_inputs.push_back({&buf.out_token_ids,
                                &validate_input.token_ids_host,
                                &validate_input.token_ids});
    extra_int_inputs.push_back({&buf.out_positions,
                                &validate_input.positions_host,
                                &validate_input.positions});
    if (!input_params.embedding.linear_state_ids.empty()) {
      extra_int_inputs.push_back(
          {&input_params.embedding.linear_state_ids,
           nullptr,
           &input_params.embedding.linear_state_indices});
    }
    if (!input_params.num_accepted_tokens_host.empty()) {
      extra_int_inputs.push_back({&accepted_prefix_lengths,
                                  nullptr,
                                  &input_params.num_accepted_tokens});
    }
    if (!input_params.graph.expanded_kv_seq_lens_vec.empty()) {
      extra_int_inputs.push_back({&input_params.graph.expanded_kv_seq_lens_vec,
                                  nullptr,
                                  &input_params.graph.expanded_kv_seq_lens});
    }
    extra_int_inputs.push_back(
        {&expanded_block_tables_dense, nullptr, &expanded_block_tables_flat});
    attention.rebuild_device_buffer(
        device_,
        extra_int_inputs,
        initialize_stable_buffer
            ? AttentionInput::BufferReusePolicy::GROWABLE
            : AttentionInput::BufferReusePolicy::FIXED_CAPACITY);
    if (initialize_stable_buffer && num_sequences > 0) {
      // Source views are part of the explicit replay contract. Reserve enough
      // packed storage for any configured decode batch before the first graph
      // captures their addresses, so a later larger batch cannot relocate the
      // buffer and invalidate an already captured variant.
      const uint64_t max_sequences =
          static_cast<uint64_t>(options_.max_seqs_per_batch());
      const uint64_t batch_scale =
          (max_sequences + static_cast<uint64_t>(num_sequences) - 1) /
          static_cast<uint64_t>(num_sequences);
      const uint64_t reserve_capacity =
          attention.attention_buffer_bytes * batch_scale;
      if (reserve_capacity > attention.attention_buffer_capacity) {
        attention.reserve_device_buffer_capacity(reserve_capacity, device_);
        attention.rebuild_device_buffer(
            device_,
            extra_int_inputs,
            AttentionInput::BufferReusePolicy::FIXED_CAPACITY);
      }
    }
    spec_verify_attention_host_buffer_ = attention.attention_host_buffer;
    spec_verify_attention_device_buffer_ = attention.attention_device_buffer;
    spec_verify_attention_buffer_capacity_ =
        attention.attention_buffer_capacity;
    input_params.graph.expanded_block_tables = expanded_block_tables_flat.view(
        {expanded_block_rows, verify_block_table_width});
    layer::ExpandedDecodeMetadataBuilder::populate_expanded_layout(
        input_params,
        input_params.graph.expanded_kv_seq_lens,
        input_params.graph.expanded_block_tables,
        input_params.graph.expanded_kv_seq_lens_vec,
        options_.block_size());
    input_params.graph.input_tokens_override = validate_input.token_ids;
    input_params.graph.spec_verify_source_addresses_stable = true;
  } else {
    input_params.attention.rebuild_device_buffer(device_);
    if (supports_explicit_spec_verify_replay_update()) {
      build_expanded_spec_verify_graph_input(
          input_params, device_, options_.block_size());
    }
  }
#else
  input_params.attention.rebuild_device_buffer(device_);
#endif
  validate_input.device_tensors_ready = true;
  // This metadata is independent of the in-flight final draft. Keep it on the
  // auxiliary stream and hand it to the compute stream with a device event.
#if defined(USE_NPU)
  input_params.graph.spec_verify_static_graph_tasks_prepared =
      use_explicit_spec_verify_replay_update && static_graph_tasks_prepared;
#endif
  if (record_ready_event) {
    finish_metadata_prepare(*prepare_stream_, validate_input);
  }
}

bool MTPWorkerImpl::prepare_static_mtp_graph_tasks_before_final_draft(
    const ForwardInput& input) {
#if defined(USE_NPU)
  if (!should_use_explicit_spec_verify_replay_update(input) ||
      input.input_params.embedding.linear_state_ids.size() != 1 ||
      embedding_cache_ == nullptr ||
      input.input_params.embedding.embedding_ids.empty()) {
    return false;
  }
  const auto& block_tables = input.input_params.attention.host.block_tables;
  if (!block_tables.defined() || block_tables.dim() != 2 ||
      block_tables.size(0) != 1) {
    return false;
  }
  const std::vector<int32_t> accepted_prefix_lengths =
      embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  if (accepted_prefix_lengths.size() != 1) {
    return false;
  }
  const int64_t verify_block_table_width =
      spec_verify_block_table_width(block_tables);
  const auto& kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  if (kv_seq_lens.empty()) {
    return false;
  }
  const int64_t spec_verify_max_kv_seq_len =
      static_cast<int64_t>(
          *std::max_element(kv_seq_lens.begin(), kv_seq_lens.end())) +
      options_.num_speculative_tokens();
  const SpecVerifyGraphTaskSignal signal{
      .linear_state_id = input.input_params.embedding.linear_state_ids.front(),
      .num_accepted_tokens = accepted_prefix_lengths.front(),
      .spec_width = options_.num_speculative_tokens() + 1,
      .block_table_width = verify_block_table_width,
      .max_kv_seq_len = spec_verify_max_kv_seq_len,
  };
  return impl_->prepare_static_mtp_graph_tasks(signal, *compute_stream_);
#else
  (void)input;
  return false;
#endif
}

// Per-seq variable-length validate input construction.
// Each seq gets per_seq_val_tokens[i] tokens in the ATB kernel input,
// reducing attention/FFN computation for seqs with fewer speculative tokens.
void MTPWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input,
    const std::vector<int32_t>& per_seq_val_tokens) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input;
  clear_ready_events(validate_input);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_sequences = input_params.meta.num_sequences;
  CHECK_EQ(per_seq_val_tokens.size(), static_cast<size_t>(num_sequences))
      << "per_seq_val_tokens size mismatch with num_sequences";
  int32_t total_num_val_tokens = 0;
  int32_t max_val_tokens = 0;
  for (int32_t i = 0; i < num_sequences; ++i) {
    total_num_val_tokens += per_seq_val_tokens[static_cast<size_t>(i)];
    max_val_tokens =
        std::max(max_val_tokens, per_seq_val_tokens[static_cast<size_t>(i)]);
  }
  const int32_t block_size = options_.block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  Slice<int32_t> positions = {
      input.positions_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.positions_host.numel())};
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  const bool use_atb_spec_kernel =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      use_chunked_prefill_spec_verify_path();
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!use_atb_spec_kernel) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                                 row_ctx.block_table_stride);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec;
  std::vector<int32_t> atb_q_seq_lens_vec;
  std::vector<int32_t> atb_q_cu_seq_lens_vec;
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const int32_t seq_val_tokens =
        per_seq_val_tokens[static_cast<size_t>(seq_id)];
    const int32_t start_position = positions[seq_id];
    const int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    CHECK_EQ(start_position + 1, kv_len)
        << "validate position/kv_len mismatch, seq_id=" << seq_id
        << ", start_position=" << start_position << ", kv_len=" << kv_len;

    for (int32_t val_idx = 0; val_idx < seq_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = val_idx == 0 ? token_ids[seq_id] : -val_idx;
      row.position_offset = val_idx;
      row.append_kv_len = !use_atb_spec_kernel;
      row.append_q_len_one = !use_atb_spec_kernel;
      row.append_block_table = !use_atb_spec_kernel;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
    }

    if (use_atb_spec_kernel) {
      const int32_t kv_len_after_validation = kv_len + seq_val_tokens - 1;
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          atb_q_seq_lens_vec, atb_q_cu_seq_lens_vec, seq_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  specBuilder::set_token_position_tensors(validate_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          token_options,
                                          position_options);
  if (!use_atb_spec_kernel) {
    input_params.meta.num_sequences = total_num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (use_atb_spec_kernel) {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     max_val_tokens,
                                     std::move(atb_q_seq_lens_vec),
                                     std::move(atb_q_cu_seq_lens_vec),
                                     atb_kv_max_seq_len,
                                     std::move(atb_kv_seq_lens_vec));
  } else {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }

  update_sampling_params(
      validate_input.sampling_params, per_seq_val_tokens, total_num_val_tokens);

  for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
    token_num = total_num_val_tokens;
  }

  if (use_chunked_prefill_spec_verify_path()) {
    input_params.embedding.input_embedding = torch::Tensor();
    input_params.is_spec_verify = true;
    if (!input_params.attention.host.q_seq_lens.empty()) {
      std::vector<int32_t> q_cu_seq_lens_vec;
      q_cu_seq_lens_vec.reserve(num_sequences + 1);
      q_cu_seq_lens_vec.emplace_back(0);
      for (int32_t q_len : input_params.attention.host.q_seq_lens) {
        q_cu_seq_lens_vec.emplace_back(q_cu_seq_lens_vec.back() + q_len);
      }
      input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
    }
    std::vector<int32_t> accepted_prefix_lengths(num_sequences, 1);
    if (embedding_cache_ != nullptr &&
        !input.input_params.embedding.embedding_ids.empty()) {
      accepted_prefix_lengths = embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
    }
    // Clamp num_accepted_tokens to Qwen3.5 GDN conv_state history capacity
    // (kernel_conv_size - 1 = 3). Larger values describe history that has
    // already rolled out of conv_state; passing them to aclnnCausalConv1d
    // makes tiling fail when the current step's per_seq_val_tokens is small
    // (e.g. adaptive prunes down to 2 while nat=5).
    if (supports_explicit_spec_verify_replay_update()) {
      clamp_gdn_conv_history(accepted_prefix_lengths);
    }
    input_params.num_accepted_tokens =
        torch::tensor(accepted_prefix_lengths, token_options);
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
  }

  input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
  if (supports_explicit_spec_verify_replay_update()) {
    build_expanded_spec_verify_graph_input(
        input_params, device_, options_.block_size());
  }
#endif
  validate_input.device_tensors_ready = true;
  finish_metadata_prepare(*prepare_stream_, validate_input);
}

void MTPWorkerImpl::prepare_draft_extend_inputs(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& extend_input,
    bool force_two_rows,
    bool wait_for_compute_stream) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  // Regular draft preparation may consume tensors produced by the previous
  // compute. The placeholder-only first-draft prelaunch has no such dependency.
  if (wait_for_compute_stream) {
    prepare_stream_->wait_stream(*compute_stream_);
  }
  extend_input = base_input;
  // Adaptive pruning needs draft selected-probs; force return_probs on so the
  // sampler's greedy fast-path doesn't skip probs assignment. Skip Qwen3.5:
  // its SSM (CausalConv1d) path fails when return_probs changes sampler
  // routing; on Qwen3.5 the controller falls back to computing probs from
  // logits (see adaptive_pruning_helpers.cpp).
  extend_input.sampling_params.return_probs =
      !extend_input.sampling_params.all_greedy_sample ||
      (adaptive_enabled() && !supports_explicit_spec_verify_replay_update());
  clear_ready_events(extend_input);
  extend_input.device_tensors_ready = false;
  auto& input_params = extend_input.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;

  const bool dp_enabled = parallel_args_.dp_size() > 1;
  const bool use_chunked_prefill =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel();
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "draft extend state count mismatch";

  const int32_t block_size = options_.block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(base_input);
  torch::TensorOptions token_options = extend_input.token_ids.options();
  torch::TensorOptions position_options = extend_input.positions.options();
  Slice<int32_t> token_ids = {
      base_input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(base_input.token_ids_host.numel())};

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences * 2);
  buf.out_positions.reserve(num_sequences * 2);
  buf.out_new_cache_slots.reserve(num_sequences * 2);
  buf.out_kv_seq_lens.reserve(num_sequences * (use_chunked_prefill ? 1 : 2));
  buf.out_q_seq_lens.reserve(num_sequences * (use_chunked_prefill ? 1 : 2));
  buf.out_q_cu_seq_lens.reserve(num_sequences * 2);
  if (!use_chunked_prefill) {
    buf.out_block_tables.reserve(static_cast<size_t>(num_sequences) * 2 *
                                 row_ctx.block_table_stride);
  }
  std::vector<torch::Tensor> expanded_embeddings;
  std::vector<int32_t> selected_row_idx;
  expanded_embeddings.reserve(num_sequences * 2);
  selected_row_idx.reserve(num_sequences);

  auto to_worker_device = [this](const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.device() == device_) {
      return tensor;
    }
    return tensor.to(device_);
  };

  torch::Tensor placeholder = embedding_cache_->embedding_placeholder();
  CHECK(placeholder.defined())
      << "embedding placeholder must be initialized for fake draft context";
  placeholder = to_worker_device(placeholder);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    auto add_row = [&](int32_t token_id,
                       int32_t position_offset,
                       const torch::Tensor& embedding) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = token_id >= 0 ? token_id : 0;
      row.position_offset = position_offset;
      row.append_kv_len = !use_chunked_prefill;
      row.append_q_len_one = !use_chunked_prefill;
      row.append_block_table = !use_chunked_prefill;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      if (embedding.defined()) {
        expanded_embeddings.emplace_back(to_worker_device(embedding));
      } else {
        expanded_embeddings.emplace_back(placeholder);
      }
    };

    EmbeddingCache::DecodeState state = last_states[seq_id];
    const int32_t current_token_id = token_ids[seq_id];
    if (!state.valid || state.token_id != current_token_id) {
      state = EmbeddingCache::DecodeState();
      state.token_id = current_token_id >= 0 ? current_token_id : 0;
    }
    if (use_chunked_prefill) {
      int32_t prev_token_id = state.prev_token_id;
      torch::Tensor prev_embedding = state.prev_embedding;
      const bool prev_is_placeholder = prev_token_id < 0;
      if (prev_is_placeholder) {
        prev_token_id = current_token_id >= 0 ? current_token_id : 0;
        prev_embedding = torch::Tensor();
      }
      add_row(prev_token_id, /*position_offset=*/-1, prev_embedding);
      if (prev_is_placeholder) {
        // Redirect to padding block 0 to avoid overwriting correct KV cache.
        buf.out_new_cache_slots.back() = 0;
      }
      add_row(state.token_id, /*position_offset=*/0, state.embedding);
      specBuilder::append_seq_len_by_layout(buf.out_q_seq_lens, 2);
      const int32_t kv_len = specBuilder::calc_kv_len(
          base_input.input_params.attention.host.kv_seq_lens,
          seq_id,
          /*offset=*/0);
      specBuilder::update_kv_seq_lens_and_max(
          buf.out_kv_seq_lens, kv_len, buf.meta.kv_max_seq_len);
      selected_row_idx.emplace_back(2 * seq_id + 1);
      continue;
    }
    const bool use_two_rows =
        force_two_rows || dp_enabled || state.all_draft_accepted;
    if (use_two_rows) {
      int32_t prev_token_id = state.prev_token_id;
      int32_t prev_position_offset = -1;
      torch::Tensor prev_embedding = state.prev_embedding;
      const bool prev_is_placeholder = prev_token_id < 0;
      if (prev_is_placeholder) {
        prev_token_id = state.token_id;
        prev_embedding = torch::Tensor();
      }
      add_row(prev_token_id, prev_position_offset, prev_embedding);
      if (prev_is_placeholder) {
        // Redirect to padding block 0 to avoid overwriting correct KV cache.
        buf.out_new_cache_slots.back() = 0;
      }
    }

    selected_row_idx.emplace_back(
        static_cast<int32_t>(expanded_embeddings.size()));
    add_row(state.token_id, /*position_offset=*/0, state.embedding);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft extend slots/positions mismatch";
  CHECK_EQ(expanded_embeddings.size(), buf.out_positions.size())
      << "draft extend embeddings/positions mismatch";

  specBuilder::set_token_position_tensors(extend_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          token_options,
                                          position_options);
  if (use_chunked_prefill) {
    input_params.meta.num_sequences = num_sequences;
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    std::vector<int32_t> q_cu_seq_lens_vec;
    q_cu_seq_lens_vec.reserve(buf.out_q_seq_lens.size());
    int32_t cumulative_q_len = 0;
    for (int32_t q_len : buf.out_q_seq_lens) {
      cumulative_q_len += q_len;
      q_cu_seq_lens_vec.emplace_back(cumulative_q_len);
    }
    specBuilder::update_input_params(input_params,
                                     buf,
                                     /*q_max_seq_len=*/2,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(q_cu_seq_lens_vec),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/false);
  } else {
    input_params.meta.num_sequences =
        static_cast<int32_t>(buf.out_positions.size());
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }
  if (supports_explicit_spec_verify_replay_update()) {
    input_params.attention.host.q_cu_seq_lens.clear();
    input_params.attention.host.q_cu_seq_lens.reserve(
        input_params.meta.num_sequences + 1);
    input_params.attention.host.q_cu_seq_lens.emplace_back(0);
    for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
      input_params.attention.host.q_cu_seq_lens.emplace_back(
          input_params.attention.host.q_cu_seq_lens.back() +
          input_params.get_q_seq_len(i));
    }
  }
  input_params.attention.rebuild_device_buffer(device_);

  input_params.embedding.input_embedding = torch::stack(expanded_embeddings);

  if (!input_params.parallel.dp_global_token_nums.empty()) {
    if (use_chunked_prefill) {
      for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
        token_num *= 2;
      }
      for (int32_t& token_num :
           input_params.parallel.raw_dp_global_token_nums) {
        token_num *= 2;
      }
    } else if (dp_enabled) {
      constexpr int32_t num_extend_tokens = 2;
      for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
        token_num *= num_extend_tokens;
      }
      for (int32_t& token_num :
           input_params.parallel.raw_dp_global_token_nums) {
        token_num *= num_extend_tokens;
      }
    } else if (input_params.parallel.dp_global_token_nums.size() == 1) {
      input_params.parallel.dp_global_token_nums[0] =
          static_cast<int32_t>(buf.out_positions.size());
    }
  }

#if defined(USE_NPU)
  // The extend layout is the 2B cache variant during steady overlap decode.
  draft_impl_->prepare_dp_ep_padding_on_stream(input_params, *prepare_stream_);
#endif

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options =
      params.selected_token_idxes.defined()
          ? params.selected_token_idxes.options()
          : torch::dtype(torch::kInt).device(device_);
  if (use_chunked_prefill || dp_enabled || force_two_rows) {
    // These layouts always append two rows per sequence and select the second
    // row.  Build the tiny control tensor directly on device; copying a
    // temporary pinned CPU tensor forces its allocator to synchronize before
    // the asynchronous H2D has completed.
    params.selected_token_idxes = torch::arange(
        /*start=*/1,
        /*end=*/2 * num_sequences,
        /*step=*/2,
        idx_options);
  } else {
    params.selected_token_idxes =
        safe_to(specBuilder::make_cpu_int_tensor(selected_row_idx),
                idx_options,
                /*non_blocking=*/true);
  }
  if (!params.sample_idxes.defined()) {
    // This control tensor is always the identity mapping. Generate it directly
    // on device instead of allocating a short-lived pinned H2D source.
    params.sample_idxes = torch::arange(
        /*start=*/0, /*end=*/num_sequences, idx_options);
  }
  extend_input.device_tensors_ready = true;
  finish_metadata_prepare(*prepare_stream_, extend_input);
}

void MTPWorkerImpl::prepare_draft_inputs(const ForwardInput& input,
                                         ForwardInput& draft_input,
                                         int32_t position_offset) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input = input;
  // Adaptive pruning needs draft selected-probs; force return_probs on so the
  // sampler's greedy fast-path doesn't skip probs assignment. Skip Qwen3.5:
  // its SSM (CausalConv1d) path fails when return_probs changes sampler
  // routing; on Qwen3.5 the controller falls back to computing probs from
  // logits (see adaptive_pruning_helpers.cpp).
  draft_input.sampling_params.return_probs =
      !draft_input.sampling_params.all_greedy_sample ||
      (adaptive_enabled() && !supports_explicit_spec_verify_replay_update());
  clear_ready_events(draft_input);
  draft_input.device_tensors_ready = false;

  auto& input_params = draft_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t block_size = options_.block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  specBuilder::DecodeBuildBuffers buf;
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = position_offset;
    row.append_token = false;
    specBuilder::append_decode_row(row_ctx, row, block_size, buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "draft kv slots/positions mismatch";

  torch::TensorOptions position_options = input.positions.options();
  set_positions_tensor(draft_input, buf.out_positions, position_options);
  specBuilder::update_input_params(
      input_params,
      buf,
      input_params.meta.q_max_seq_len,
      std::move(input_params.attention.host.q_seq_lens),
      std::move(input_params.attention.host.q_cu_seq_lens),
      buf.meta.kv_max_seq_len,
      std::move(buf.out_kv_seq_lens));
  if (supports_explicit_spec_verify_replay_update()) {
    input_params.attention.host.q_cu_seq_lens.clear();
    input_params.attention.host.q_cu_seq_lens.reserve(
        input_params.meta.num_sequences + 1);
    input_params.attention.host.q_cu_seq_lens.emplace_back(0);
    for (int32_t i = 0; i < input_params.meta.num_sequences; ++i) {
      input_params.attention.host.q_cu_seq_lens.emplace_back(
          input_params.attention.host.q_cu_seq_lens.back() +
          input_params.get_q_seq_len(i));
    }
  }
  input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
  // Later draft steps share the B cache variant.
  draft_impl_->prepare_dp_ep_padding_on_stream(input_params, *prepare_stream_);
#endif
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.device_tensors_ready = false;

  // Positions/KV metadata do not depend on the in-flight draft result. Prepare
  // them concurrently; token ids and embeddings are filled on compute_stream.
  finish_metadata_prepare(*prepare_stream_, draft_input);
}

SampleOutput MTPWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  std::vector<torch::Tensor> draft_token_ids_steps;
  std::vector<torch::Tensor> draft_probs_steps;
  draft_token_ids_steps.reserve(draft_outputs.size());
  draft_probs_steps.reserve(draft_outputs.size());
  for (const ForwardOutput& draft_output : draft_outputs) {
    draft_token_ids_steps.emplace_back(draft_output.sample_output.next_tokens);
    draft_probs_steps.emplace_back(draft_output.sample_output.probs);
  }

  std::pair<torch::Tensor, torch::Tensor> validate_tensors =
      specBuilder::draftProbs::build_validate_tensors(
          draft_token_ids_steps,
          draft_probs_steps,
          batch_size,
          vocab_size,
          enable_opt_validate_probs_,
          /*draft_probs_required=*/!sampling_params.all_greedy_sample);
  return validate(sampling_params,
                  validate_tensors.first,
                  validate_tensors.second,
                  target_output,
                  num_speculative_tokens,
                  pruned_prefix_lengths);
}

SampleOutput MTPWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const ForwardOutput& target_output,
    int32_t num_speculative_tokens,
    const std::vector<int32_t>* pruned_prefix_lengths) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  torch::Tensor bonus_token_ids =
      target_output.sample_output.next_tokens
          .index({"...", ISlice(num_val_tokens - 1, None, num_val_tokens)})
          .view({-1, 1});

  SampleOutput sample_output;
  if (sampling_params.all_greedy_sample && !target_output.logprobs) {
    torch::Tensor target_token_ids =
        target_output.sample_output.next_tokens.view(
            {batch_size, num_val_tokens});
    torch::Tensor target_draft_token_ids = target_token_ids.slice(
        /*dim=*/1, /*start=*/0, /*end=*/num_val_tokens - 1);
    auto [accepted_token_ids, masked_accepted_token_ids] =
        RejectionSampler::greedy_sample_from_token_ids(
            draft_token_ids.to(target_draft_token_ids),
            target_draft_token_ids,
            bonus_token_ids,
            /*mask_out_rejected_tokens=*/true);
    (void)accepted_token_ids;

    sample_output.next_tokens = masked_accepted_token_ids;
    torch::Tensor embeddings = target_output.sample_output.embeddings;
    sample_output.embeddings =
        embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});
  } else {
    torch::Tensor target_logits =
        target_output.logits.view({batch_size, num_val_tokens, vocab_size});

    // prepare input for rejection sampling
    std::unique_ptr<RejectionSampler> rejection_sampler =
        std::make_unique<RejectionSampler>(sampling_params.do_sample,
                                           sampling_params.all_random_sample,
                                           sampling_params.all_greedy_sample,
                                           target_output.logprobs,
                                           target_output.max_top_logprobs,
                                           enable_fused_kernel_);

    // get the accepted tokens
    sample_output = rejection_sampler->forward(
        draft_token_ids.to(bonus_token_ids),
        draft_probs.defined() ? draft_probs.to(target_logits.device())
                              : torch::Tensor(),
        target_logits,
        bonus_token_ids,
        /*mask_out_rejected_tokens=*/true);

    // process embedding
    torch::Tensor embeddings = target_output.sample_output.embeddings;
    sample_output.embeddings =
        embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});
  }

  if (pruned_prefix_lengths != nullptr) {
    // Build cut/keep masks once from pruned_prefix_lengths and reuse across
    // both helpers below, so we avoid re-uploading prefix_lengths and
    // rebuilding identical arange+eq+logical_and masks per call.
    const adaptive_pruning::PrunedPrefixMasks pruning_masks =
        adaptive_pruning::build_pruned_prefix_masks(
            *pruned_prefix_lengths,
            num_speculative_tokens,
            sample_output.next_tokens.device());
    sync_pruned_boundary_outputs(sample_output,
                                 target_output,
                                 batch_size,
                                 num_val_tokens,
                                 pruning_masks);
    apply_pruned_prefix_lengths(sample_output,
                                target_output.sample_output.next_tokens,
                                num_speculative_tokens,
                                pruning_masks);
  }

  return sample_output;
}

}  // namespace xllm
