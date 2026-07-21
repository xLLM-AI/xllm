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

#include <algorithm>
#include <memory>

#include "common/metrics.h"
#if defined(USE_MLU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif
#include "core/framework/block/block_utils.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "spec_input_builder.h"
#include "util/pretty_print.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;

namespace {

ProcessGroup* spec_broadcast_group(const ParallelArgs& parallel_args) {
  return parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_
                                            : parallel_args.process_group_;
}

void broadcast_spec_tokens(torch::Tensor& tokens,
                           ProcessGroup* pg,
                           int32_t root_rank = 0) {
  if (pg == nullptr || pg->world_size() <= 1 || !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  pg->broadcast(tokens, root_rank);
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
  estimate_options.enable_rdma_scale_padding =
      options.instance_role() != InstanceRole::DEFAULT;
  return estimate_options;
}

void record_metadata_ready_event(Stream& stream, ForwardInput& input) {
  StreamEventPtr event = stream.record_event();
  if (event == nullptr) {
    stream.synchronize();
  }
  input.metadata_ready_event = event;
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
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
}

void build_expanded_spec_verify_graph_input(ModelInputParams& input_params,
                                            const torch::Device& device) {
  clear_expanded_spec_verify_graph_input(input_params);
  if (!input_params.is_spec_verify ||
      !input_params.meta.batch_forward_type.is_chunked_prefill()) {
    return;
  }

  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return;
  }
  CHECK_EQ(q_seq_lens.size(), kv_seq_lens.size())
      << "spec verify q/kv seq lens must both be sequence-scoped";
  CHECK(input_params.attention.device.block_tables.defined())
      << "spec verify block tables must be rebuilt before graph input";

  const int64_t batch_size = static_cast<int64_t>(q_seq_lens.size());
  CHECK_GE(input_params.attention.device.block_tables.size(0), batch_size)
      << "spec verify block table rows are fewer than sequences";

  std::vector<int32_t> expanded_kv_seq_lens;
  std::vector<torch::Tensor> expanded_block_rows;
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int32_t q_len = q_seq_lens[static_cast<size_t>(seq_idx)];
    const int32_t kv_len = kv_seq_lens[static_cast<size_t>(seq_idx)];
    CHECK_GE(q_len, 1) << "spec verify q_len must be positive";
    CHECK_GE(kv_len, q_len) << "kv_len must include validate query tokens";
    for (int32_t token_idx = 0; token_idx < q_len; ++token_idx) {
      expanded_kv_seq_lens.emplace_back(kv_len - q_len + token_idx + 1);
      expanded_block_rows.emplace_back(
          input_params.attention.device.block_tables.select(/*dim=*/0,
                                                            seq_idx));
    }
  }
  if (expanded_kv_seq_lens.empty()) {
    return;
  }

  torch::Tensor expanded_kv_seq_lens_host =
      torch::tensor(expanded_kv_seq_lens,
                    torch::TensorOptions()
                        .dtype(torch::kInt)
                        .device(torch::kCPU)
                        .pinned_memory(true));
  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
  input_params.graph.expanded_kv_seq_lens =
      expanded_kv_seq_lens_host.to(device, /*non_blocking=*/true);
  input_params.graph.expanded_block_tables =
      torch::stack(expanded_block_rows, 0);
  input_params.graph.expanded_kv_seq_lens_vec = std::move(expanded_kv_seq_lens);
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
      input, processed_input, prepare_stream);
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

bool has_mtp_prefill_placeholder_extra_token(
    const std::vector<int32_t>& extra_token_ids,
    int32_t placeholder) {
  return std::find(extra_token_ids.begin(),
                   extra_token_ids.end(),
                   placeholder) != extra_token_ids.end();
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

// CP partition keeps the final -1 placeholder on exactly one rank's shard.
// The owning rank injects the target next_token at that local placeholder; non-
// owning ranks have no placeholder and skip injection. selected_token_idxes is
// left in the CP all-gather global space for the draft LmHead.
void apply_cp_mtp_prefill_target_tokens(
    ForwardInput& input,
    const torch::Tensor& next_tokens,
    int32_t placeholder,
    const torch::TensorOptions& token_options) {
  auto& embedding = input.input_params.embedding;
  CHECK(embedding.mtp_shifted_token_ids.defined());
  CHECK(input.sampling_params.selected_token_idxes.defined())
      << "selected_token_idxes required for CP MTP final-chunk draft input";

  torch::Tensor shifted_cpu =
      embedding.mtp_shifted_token_ids.device().is_cpu()
          ? embedding.mtp_shifted_token_ids
          : embedding.mtp_shifted_token_ids.to(torch::kCPU);
  if (shifted_cpu.scalar_type() != torch::kInt32) {
    shifted_cpu = shifted_cpu.to(torch::kInt32);
  }
  shifted_cpu = shifted_cpu.contiguous();

  const auto& extra_token_ids = embedding.extra_token_ids;
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  torch::Tensor next_cpu = to_cpu_int_tensor_for_read(next_tokens);

  input.device_tensors_ready = false;
  input.token_ids_host = shifted_cpu.clone();
  int32_t* host_tokens = input.token_ids_host.data_ptr<int32_t>();
  const int64_t num_tokens = input.token_ids_host.numel();

  // Inject each sequence's target next_token into its -1 placeholder slot. The
  // global prediction position is token-level gathered into exactly one CP
  // rank's mtp_shifted shard (see cp_partition gather_token_level_tensor), so a
  // local scan for the placeholder is the authoritative owner check: the owning
  // rank replaces it, non-owning ranks legitimately have no placeholder and
  // skip injection. selected_token_idxes is intentionally left unchanged: the
  // draft LmHead CP all-gathers hidden before gathering selected rows (same as
  // the target path, which yields a valid selected_hidden_from_lm_head), so the
  // indices must stay in the CP all-gather global space, not local rows.
  int32_t next_seq_idx = 0;
  for (int32_t seq = 0; seq < num_sequences; ++seq) {
    if (seq >= static_cast<int32_t>(extra_token_ids.size()) ||
        extra_token_ids[seq] != placeholder) {
      continue;
    }
    CHECK_LT(next_seq_idx, next_cpu.numel());
    const int32_t next_token = next_cpu.data_ptr<int32_t>()[next_seq_idx];
    for (int64_t i = 0; i < num_tokens; ++i) {
      if (host_tokens[i] != placeholder) {
        continue;
      }
      host_tokens[i] = next_token;
      break;
    }
    ++next_seq_idx;
  }
  CHECK_EQ(next_seq_idx, next_cpu.numel())
      << "unused target next_tokens for CP MTP prefill";

  input.token_ids =
      safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
  embedding.mtp_shifted_token_ids = input.token_ids;
  input.device_tensors_ready = true;
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

bool is_qwen3_5_target_model_type(const std::string& model_type) {
  return model_type == "qwen3_5" || model_type == "qwen3_5_moe" ||
         model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text" ||
         model_type.rfind("qwen3_5_", 0) == 0;
}

bool is_qwen3_5_draft_model_type(const std::string& model_type) {
  return model_type == "qwen3_5_mtp" || model_type == "qwen3_5_moe_mtp";
}

bool is_mimo_target_model_type(const std::string& model_type) {
  return model_type == "mimo";
}

}  // namespace

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : MTPWorkerImpl(parallel_args,
                    device,
                    options,
                    MTPTargetOptions(options),
                    mtp_draft_options(options),
                    ::xllm::SpeculativeConfig::get_instance()
                        .enable_opt_validate_probs()) {}

MTPWorkerImpl::MTPWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options,
                             const runtime::Options& target_options,
                             const runtime::Options& draft_options,
                             bool enable_opt_validate_probs)
    : SpeculativeWorkerImpl(parallel_args, device, options, target_options),
      enable_opt_validate_probs_(enable_opt_validate_probs) {
  draft_impl_ = std::make_unique<LLMWorkerImpl>(
      MTPDraftParallelArgs(parallel_args, options),
      device,
      mtp_draft_options(draft_options));
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
  }

  if (draft_impl_ != nullptr &&
      draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const bool draft_owns_shared_weights =
        options_.enable_mtp_draft_body_tp1() &&
        is_qwen3_5_draft_model_type(
            draft_impl_->context_.get_model_args().model_type());
    // Qwen3.5 draft checkpoints contain complete embedding and LMHead weights.
    // Other MTP drafts retain their existing target-weight sharing contract;
    // only their transformer body is replicated with TP1 parallel arguments.
    if (!draft_owns_shared_weights) {
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
#if defined(USE_NPU)
  if (result && use_qwen3_5_spec_verify_path()) {
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
    if (util::is_deepseek_v4_model_type(args.model_type())) {
      return args.hc_mult() * args.hidden_size();
    }
  }
  return static_cast<int64_t>(embedding_size_);
}

bool MTPWorkerImpl::use_qwen3_5_spec_verify_path() const {
  return impl_ != nullptr &&
         impl_->get_status() != WorkerImpl::Status::UNINITIALIZED &&
         is_qwen3_5_target_model_type(
             impl_->context_.get_model_args().model_type());
}

// MiMo MTP validation requires chunked-prefill mode (same as Qwen3.5) to
// avoid the read-before-write race in FlashInfer batch-decode: validation
// token 1 (at position p+1) must attend to the KV written by token 0 (at
// position p) within the same batch call, but all-reads-then-all-writes
// ordering means it reads garbage.  Chunked prefill executes tokens causally
// so token 1 always sees token 0's committed KV.
bool MTPWorkerImpl::use_mimo_spec_verify_path() const {
  return impl_ != nullptr &&
         impl_->get_status() != WorkerImpl::Status::UNINITIALIZED &&
         is_mimo_target_model_type(
             impl_->context_.get_model_args().model_type());
}

bool MTPWorkerImpl::use_chunked_prefill_spec_verify_path() const {
  return use_qwen3_5_spec_verify_path() || use_mimo_spec_verify_path();
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

#if defined(USE_NPU) || defined(USE_MLU)
bool MTPWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  const int64_t num_blocks = kv_cache_shape.key_cache_shape()[0];
  CHECK(impl_ != nullptr);
  CHECK(draft_impl_ != nullptr);

  if (kv_cache_transfer_ == nullptr) {
#if defined(USE_NPU)
    kv_cache_transfer_ = std::make_shared<SpecKVCacheTransfer>(
        options_.transfer_listen_port(),
        options_.instance_role(),
        context_.get_model_args().index_n_heads() > 0);
#elif defined(USE_MLU)
    CHECK_EQ(::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type(),
             "Mooncake")
        << "MLU MTP push only supports Mooncake KV transfer.";
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
  SpeculativeWorkerImpl::prepare_work_before_execute(input, processed_input);
}

std::optional<ForwardOutput> MTPWorkerImpl::step_empty(
    const ForwardInput& input) {
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
    draft_outputs.emplace_back(run_llm_no_sync_impl(*draft_impl_,
                                                    new_input,
                                                    *prepare_stream_,
                                                    *compute_stream_,
                                                    draft_extend_prepared)
                                   .value());

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
    return output;
  }
}

std::optional<ForwardOutput> MTPWorkerImpl::step_prefill(
    const ForwardInput& input) {
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
    if (embeddings.defined()) {
      prefill_input.input_params.embedding.input_embedding = embeddings.clone();
    }
    if (output.sample_output.next_tokens.defined()) {
      const auto& extra_token_ids =
          prefill_input.input_params.embedding.extra_token_ids;
      if (options_.cp_size() > 1 &&
          has_mtp_prefill_placeholder_extra_token(extra_token_ids, -1)) {
        apply_cp_mtp_prefill_target_tokens(prefill_input,
                                           output.sample_output.next_tokens,
                                           -1,
                                           prefill_input.token_ids.options());
      } else {
        replace_host_token_placeholders(prefill_input,
                                        -1,
                                        output.sample_output.next_tokens,
                                        prefill_input.token_ids.options());
      }
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
    // Under CP the target prefill `embeddings` is the full local token shard,
    // which cannot be indexed by the CP all-gather-space selected indices.
    // Prefer the LmHead-gathered per-sequence hidden when present so the cache
    // stores it directly; fall back to the full hidden for the non-CP path,
    // where index_select on the complete local sequence is valid.
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
  if (options_.cp_size() > 1) {
    CHECK(input_params.embedding.mtp_shifted_token_ids.defined());
    CHECK_EQ(input_params.embedding.mtp_shifted_token_ids.numel(),
             prefill_input.token_ids.numel());
    prefill_input.token_ids = input_params.embedding.mtp_shifted_token_ids;
    torch::Tensor shifted_cpu = prefill_input.token_ids.device().is_cpu()
                                    ? prefill_input.token_ids
                                    : prefill_input.token_ids.to(torch::kCPU);
    if (shifted_cpu.scalar_type() != torch::kInt32) {
      shifted_cpu = shifted_cpu.to(torch::kInt32);
    }
    prefill_input.token_ids_host = shifted_cpu.contiguous();
    finish_metadata_prepare(*prepare_stream_, prefill_input);
    return;
  }

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

  // Get decode state of last step
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
  prepare_draft_extend_inputs(input, last_states, current_draft_input);
  draft_outputs.reserve(num_speculative_tokens);
  const bool reuse_mtp_topk_state = layer::is_mtp_dsa_topk_reuse_enabled(
      draft_impl_->context_.get_model_args());
  MtpTopkStatePtr mtp_topk_state;
  for (int32_t draft_idx = 0; draft_idx < num_speculative_tokens; ++draft_idx) {
    if (reuse_mtp_topk_state) {
      current_draft_input.input_params.mtp_topk_state = mtp_topk_state;
    }
    std::optional<ForwardOutput> draft_output_opt =
        run_llm_no_sync_impl(*draft_impl_,
                             current_draft_input,
                             *prepare_stream_,
                             *compute_stream_,
                             draft_prepared[draft_idx]);

    // Overlap next-step input preparation with async draft forward.
    if (draft_idx == num_speculative_tokens - 1) {
      prepare_validate_inputs(input, validate_input);
    } else {
      prepare_draft_inputs(input, next_step_input, draft_idx + 1);
    }

    CHECK(draft_output_opt.has_value())
        << "draft output is empty in speculative step";

    draft_outputs.emplace_back(std::move(draft_output_opt.value()));
    {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      if (reuse_mtp_topk_state) {
        mtp_topk_state = specBuilder::select_mtp_topk_state_for_next_step(
            draft_outputs.back().mtp_topk_state,
            current_draft_input.sampling_params);
      }
      // Unify this step's draft next_tokens across the consensus group before
      // process_draft_sample_output() compresses the still-full [batch, vocab]
      // probs into the cache: gathering the cached prob with a unified token
      // yields a unified prob, so we only broadcast the [batch] token tensor.
      if (get_optimization_config().enable_spec_token_broadcast) {
        SampleOutput& draft_sample = draft_outputs.back().sample_output;
        broadcast_spec_tokens(draft_sample.next_tokens,
                              spec_broadcast_group(parallel_args_));
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
    }
    record_current_metadata_ready_event(current_draft_input, *compute_stream_);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return run_validate(input, draft_outputs, validate_input);
}

void MTPWorkerImpl::fill_validate_input_from_draft_outputs(
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input,
    Stream& compute_stream) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  CHECK_EQ(draft_outputs.size(), static_cast<size_t>(num_speculative_tokens))
      << "draft output count mismatch";
  CHECK(validate_input.token_ids.defined())
      << "validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "validate token_ids must be flat";
  CHECK_EQ(validate_input.token_ids.numel() % num_val_tokens, 0)
      << "validate token_ids size must be divisible by validation width";

  const int64_t total_num_val_tokens = validate_input.token_ids.numel();
  const int64_t num_sequences = total_num_val_tokens / num_val_tokens;
  const auto token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_metadata_ready_event(validate_input, compute_stream);
  torch::Tensor validate_token_rows =
      validate_input.token_ids.view({num_sequences, num_val_tokens});

  validate_input.device_tensors_ready = false;
  for (int32_t i = 0; i < num_speculative_tokens; ++i) {
    const auto& draft_output = draft_outputs[i];
    const torch::Tensor& next_tokens = draft_output.sample_output.next_tokens;
    CHECK(next_tokens.defined())
        << "draft next_tokens must be defined for validate token fill";
    torch::Tensor draft_tokens =
        safe_to(next_tokens.flatten(), token_options, /*non_blocking=*/true);
    CHECK_EQ(draft_tokens.numel(), num_sequences)
        << "draft token count must match validate sequence count";
    validate_token_rows.select(/*dim=*/1, /*index=*/i + 1)
        .copy_(draft_tokens, /*non_blocking=*/true);
  }
  validate_input.device_tensors_ready = true;
  record_metadata_ready_event(compute_stream, validate_input);
}

std::optional<ForwardOutput> MTPWorkerImpl::run_validate(
    const ForwardInput& input,
    const std::vector<ForwardOutput>& draft_outputs,
    ForwardInput& validate_input) {
  // run the target model to get the verification scores
  Timer timer;
  ForwardInput target_prepared;
  fill_validate_input_from_draft_outputs(
      draft_outputs, validate_input, *compute_stream_);
  ForwardOutput target_output = run_llm_no_sync_impl(*impl_,
                                                     validate_input,
                                                     *prepare_stream_,
                                                     *compute_stream_,
                                                     target_prepared)
                                    .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  // verify the proposals with target and update the batch
  timer.reset();
  SampleOutput val_output;
  {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    val_output = validate(input.sampling_params, draft_outputs, target_output);
  }
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              timer.elapsed_seconds());

  // Catch-all for cross-rank RNG divergence: unify the accepted next_tokens to
  // the consensus group's rank 0 so all_draft_accepted and the next
  // draft-extend row layout agree across ranks.
  if (get_optimization_config().enable_spec_token_broadcast) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    broadcast_spec_tokens(val_output.next_tokens,
                          spec_broadcast_group(parallel_args_));
  }

  const int32_t ret = compute_stream_->synchronize();
  CHECK_EQ(ret, 0) << "failed to synchronize MTP compute stream, ret=" << ret;
  release_retained_inputs(target_output);
  val_output.next_tokens = val_output.next_tokens.to(torch::kCPU);
  write_target_context_to_cache(input, val_output);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  clear_all_output_embeddings(target_output);
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = val_output;
  return target_output;
}

void MTPWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output) {
  CHECK(embedding_cache_ != nullptr)
      << "embedding_cache_ must be initialized before target cache write";
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "target context cache write requires embedding ids";
  embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      options_.num_speculative_tokens());
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
                                            ForwardInput& validate_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  validate_input = input;
  clear_ready_events(validate_input);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
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
      const int32_t kv_len_after_validation = kv_len + num_speculative_tokens;
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

  update_sampling_params(
      validate_input.sampling_params, num_val_tokens, total_num_val_tokens);

  for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
    token_num *= num_val_tokens;
  }

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
    std::vector<int32_t> accepted_prefix_lengths(num_sequences, 1);
    if (embedding_cache_ != nullptr &&
        !input.input_params.embedding.embedding_ids.empty()) {
      accepted_prefix_lengths = embedding_cache_->read_accepted_prefix_lengths(
          input.input_params.embedding.embedding_ids);
    }
    input_params.num_accepted_tokens =
        torch::tensor(accepted_prefix_lengths, token_options);
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
  }

  input_params.attention.rebuild_device_buffer(device_);
#if defined(USE_NPU)
  if (use_qwen3_5_spec_verify_path()) {
    build_expanded_spec_verify_graph_input(input_params, device_);
  }
#endif
  validate_input.device_tensors_ready = true;
  finish_metadata_prepare(*prepare_stream_, validate_input);
}

void MTPWorkerImpl::prepare_draft_extend_inputs(
    const ForwardInput& base_input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& extend_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  extend_input = base_input;
  extend_input.sampling_params.return_probs =
      !extend_input.sampling_params.all_greedy_sample;
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
    const bool use_two_rows = dp_enabled || state.all_draft_accepted;
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
  if (use_qwen3_5_spec_verify_path()) {
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

  // Establish cross-stream ordering before stacking the draft-extend
  // embeddings. The stack runs on prepare_stream_, but the embedding rows it
  // reads are produced/finalized by work enqueued on compute_stream_. Make
  // prepare_stream_ wait on compute_stream_ via an event.
  prepare_stream_->wait_stream(*compute_stream_);

  input_params.embedding.input_embedding = torch::stack(expanded_embeddings);

  if (!input_params.parallel.dp_global_token_nums.empty()) {
    if (use_chunked_prefill) {
      for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
        token_num *= 2;
      }
    } else if (dp_enabled) {
      constexpr int32_t num_extend_tokens = 2;
      for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
        token_num *= num_extend_tokens;
      }
    } else if (input_params.parallel.dp_global_token_nums.size() == 1) {
      input_params.parallel.dp_global_token_nums[0] =
          static_cast<int32_t>(buf.out_positions.size());
    }
  }

  auto& params = extend_input.sampling_params;
  torch::TensorOptions idx_options =
      params.selected_token_idxes.defined()
          ? params.selected_token_idxes.options()
          : torch::dtype(torch::kInt).device(device_);
  params.selected_token_idxes =
      safe_to(specBuilder::make_cpu_int_tensor(selected_row_idx),
              idx_options,
              /*non_blocking=*/true);
  if (!params.sample_idxes.defined()) {
    std::vector<int32_t> sample_idxes_vec;
    sample_idxes_vec.reserve(num_sequences);
    for (int32_t i = 0; i < num_sequences; ++i) {
      sample_idxes_vec.emplace_back(i);
    }
    params.sample_idxes =
        safe_to(specBuilder::make_cpu_int_tensor(sample_idxes_vec),
                idx_options,
                /*non_blocking=*/true);
  }
  extend_input.device_tensors_ready = true;
  finish_metadata_prepare(*prepare_stream_, extend_input);
}

void MTPWorkerImpl::prepare_draft_inputs(const ForwardInput& input,
                                         ForwardInput& draft_input,
                                         int32_t position_offset) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  draft_input = input;
  draft_input.sampling_params.return_probs =
      !draft_input.sampling_params.all_greedy_sample;
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
  if (use_qwen3_5_spec_verify_path()) {
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
  // token_ids is intentionally filled later from the previous draft output.
  draft_input.device_tensors_ready = false;

  finish_metadata_prepare(*prepare_stream_, draft_input);
}

SampleOutput MTPWorkerImpl::validate(
    const SamplingParameters& sampling_params,
    const std::vector<ForwardOutput>& draft_outputs,
    const ForwardOutput& target_output) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  std::vector<torch::Tensor> draft_token_ids_steps;
  std::vector<torch::Tensor> draft_probs_steps;
  draft_token_ids_steps.reserve(draft_outputs.size());
  draft_probs_steps.reserve(draft_outputs.size());
  for (const auto& draft_output : draft_outputs) {
    draft_token_ids_steps.emplace_back(draft_output.sample_output.next_tokens);
    draft_probs_steps.emplace_back(draft_output.sample_output.probs);
  }

  auto [draft_token_ids, draft_probs] =
      specBuilder::draftProbs::build_validate_tensors(
          draft_token_ids_steps,
          draft_probs_steps,
          batch_size,
          vocab_size,
          enable_opt_validate_probs_,
          /*draft_probs_required=*/!sampling_params.all_greedy_sample);
  return validate(sampling_params, draft_token_ids, draft_probs, target_output);
}

SampleOutput MTPWorkerImpl::validate(const SamplingParameters& sampling_params,
                                     const torch::Tensor& draft_token_ids,
                                     const torch::Tensor& draft_probs,
                                     const ForwardOutput& target_output) {
  const int32_t num_target_tokens =
      target_output.sample_output.next_tokens.numel();
  const int32_t num_val_tokens = options_.num_speculative_tokens() + 1;
  CHECK_EQ(num_target_tokens % num_val_tokens, 0);
  const int32_t batch_size = num_target_tokens / num_val_tokens;
  const int32_t vocab_size = target_output.logits.size(/*dim=*/-1);

  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  auto bonus_token_ids =
      target_output.sample_output.next_tokens
          .index({"...", ISlice(num_val_tokens - 1, None, num_val_tokens)})
          .view({-1, 1});

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

    SampleOutput sample_output;
    sample_output.next_tokens = masked_accepted_token_ids;
    torch::Tensor embeddings = target_output.sample_output.embeddings;
    sample_output.embeddings =
        embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});
    return sample_output;
  }

  auto target_logits =
      target_output.logits.view({batch_size, num_val_tokens, vocab_size});

  // prepare input for rejection sampling
  auto rejection_sampler =
      std::make_unique<RejectionSampler>(sampling_params.do_sample,
                                         sampling_params.all_random_sample,
                                         sampling_params.all_greedy_sample,
                                         target_output.logprobs,
                                         target_output.max_top_logprobs,
                                         enable_fused_kernel_);

  // get the accepted tokens
  SampleOutput sample_output = rejection_sampler->forward(
      draft_token_ids.to(bonus_token_ids),
      draft_probs.defined() ? draft_probs.to(target_logits.device())
                            : torch::Tensor(),
      target_logits,
      bonus_token_ids,
      /*mask_out_rejected_tokens=*/true);

  // process embedding
  auto embeddings = target_output.sample_output.embeddings;
  sample_output.embeddings =
      embeddings.view({batch_size, num_val_tokens, embeddings.size(-1)});

  return sample_output;
}

}  // namespace xllm
