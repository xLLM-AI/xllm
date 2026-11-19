/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/model/model_input_types.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/speculative/spec_verify.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class JsonObjectGrammarState;

template <typename Input>
void prepare_dummy_input(Input& input,
                         const ModelContext& context,
                         const torch::Device& device) {
  auto& params = input.input_params;
  const auto& parallel = context.get_parallel_args();
  const int32_t num_speculative_tokens = context.num_speculative_tokens();
  if (params.meta.num_sequences != 0 ||
      params.meta.batch_forward_type.is_empty() ||
      (num_speculative_tokens <= 0 && input.token_ids.defined() &&
       input.token_ids.numel() != 0) ||
      (parallel.dp_size() <= 1 && parallel.ep_size() <= 1 &&
       parallel.mapping_data().empty())) {
    return;
  }
  build_dummy_forward_input(input, num_speculative_tokens, device);
  if (num_speculative_tokens > 0) {
    if (has_linear_attention_layers(context.get_model_args())) {
      params.num_accepted_tokens_host = {1};
      params.num_accepted_tokens =
          torch::ones({1}, input.token_ids.options().dtype(torch::kInt32));
    }
    const int32_t num_decoding_tokens = num_speculative_tokens + 1;
    params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    params.meta.q_max_seq_len = num_decoding_tokens;
    params.meta.kv_max_seq_len = num_decoding_tokens;
    params.embedding.input_embedding = torch::Tensor();
    params.attn_metadata.reset();
    params.attention = {};
    params.attention.host.q_seq_lens = {num_decoding_tokens};
    params.attention.host.q_cu_seq_lens = {0, num_decoding_tokens};
    params.attention.host.kv_seq_lens = {num_decoding_tokens};
    params.attention.host.kv_cache_tokens_nums = {0};
    params.attention.host.new_cache_slots.assign(num_decoding_tokens,
                                                 /*value=*/-1);
    params.attention.rebuild_device_buffer(device);
  }
}

template <typename Params>
bool should_run_speculative_decode(const Params& params) {
  if (!params.meta.batch_forward_type.is_decode()) {
    return false;
  }

  const auto& dp_token_nums = params.parallel.dp_global_token_nums;
  const auto& dp_is_decode = params.parallel.dp_is_decode;
  if (dp_is_decode.empty()) {
    return dp_token_nums.size() <= 1;
  }
  if (dp_is_decode.size() != dp_token_nums.size()) {
    return false;
  }

  bool any_active = false;
  for (size_t i = 0; i < dp_is_decode.size(); ++i) {
    if (dp_token_nums[i] == 0) {
      continue;
    }
    any_active = true;
    if (dp_is_decode[i] != 1) {
      return false;
    }
  }
  return any_active;
}

template <typename Params>
void scale_speculative_parallel_token_counts(Params& params,
                                             int32_t num_speculative_tokens) {
  const int32_t num_decoding_tokens = num_speculative_tokens + 1;
  scale_parallel_token_counts(params.parallel, num_decoding_tokens);
  params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      params.expert.eplb_decode_token_mask, num_decoding_tokens);
}

// Shared tail of run_validate: marks the output as a speculative decode
// verify result, attaches the row metadata, clears embeddings, and installs
// the validation sample output.
ForwardOutput finalize_verify_output(
    ForwardOutput target_output,
    SampleOutput val_output,
    const std::vector<JsonObjectGrammarState>& json_object_states);

template <typename BaseWorker, typename Input, typename PrepareInput>
class SpeculativeModelWorkerImpl final : public BaseWorker {
 public:
  template <typename... Args>
  explicit SpeculativeModelWorkerImpl(PrepareInput prepare_input,
                                      Args&&... args)
      : BaseWorker(std::forward<Args>(args)...),
        prepare_input_(std::move(prepare_input)) {}

 protected:
  using BaseWorker::prepare_model_input;

  void prepare_model_input(Input& input) override { prepare_input_(input); }

  bool owns_npu_parallel_input_prepare() const override { return false; }
  bool can_prepare_without_compute_stream_wait() const override {
    return false;
  }

 private:
  PrepareInput prepare_input_;
};

// Base class for all speculative decoding workers.
// Provides common logic: target model management, step dispatch, and
// sampling parameter updates. Subclasses implement algorithm-specific
// draft generation and validation (MTP, Eagle3, Suffix, DFlash, etc.).
template <typename TargetInput>
class SpeculativeWorkerImpl : public WorkerImpl {
 public:
  using TargetModelParams = std::remove_reference_t<
      decltype(std::declval<TargetInput&>().input_params)>;

  ~SpeculativeWorkerImpl() override;

  void drain_pending_execution() override;

 protected:
  // `options` is passed to WorkerImpl (preserves enable_schedule_overlap etc.),
  // `target_options` is used to create impl_ (target model worker).
  // Each algorithm subclass decides its own target_options.
  SpeculativeWorkerImpl(const ParallelArgs& parallel_args,
                        const torch::Device& device,
                        const runtime::Options& options,
                        const runtime::Options& target_options,
                        WorkerType worker_type = WorkerType::LLM);

 public:
  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override {
    // do nothing
    return true;
  };

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports) override {
    return impl_->link_cluster(cluster_ids, addrs, ports);
  };

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports) override {
    return impl_->unlink_cluster(cluster_ids, addrs, ports);
  };

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override {
    return impl_->estimate_kv_cache_capacity();
  };

  // allocate kv cache. blocking call
  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  void get_cache_info(uint64_t& cluster_id,
                      std::string& addr,
                      uint16_t& port) override {
    impl_->get_cache_info(cluster_id, addr, port);
  };

  std::shared_ptr<const JsonObjectGrammar> ensure_json_object_grammar(
      bool reasoning_enabled) override {
    return impl_->ensure_json_object_grammar(reasoning_enabled);
  }

  // Common step dispatch: prefill / decode / empty
  std::optional<ForwardOutput> step(const TargetInput& input) override;

  folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings) override {
    return impl_->pull_kv_blocks_async(src_cluster_id, src_addr, mappings);
  };

 protected:
#if defined(USE_NPU)
  template <typename Params>
  void prepare_target_padding(Params& params);
#endif

  bool can_prepare_without_compute_stream_wait() const override {
    return false;
  }

  bool prepare_model_decode_continuation(
      TargetInput& inputs,
      const ForwardOutput& prior_output) override;

  // Algorithm-specific virtual methods for subclasses to implement
  virtual std::optional<ForwardOutput> step_prefill(
      const TargetInput& input) = 0;
  virtual std::optional<ForwardOutput> step_decode(
      const TargetInput& inputs) = 0;
  virtual std::optional<ForwardOutput> step_empty(
      const TargetInput& inputs) = 0;

  void prepare_validate_inputs(const TargetInput& inputs,
                               TargetInput& validate_inputs,
                               bool expand_decode_metadata_per_token);

  bool use_orthogonal_cp_consensus() const {
    return parallel_args_.cp_size() > 1 &&
           parallel_args_.tp_group_ != nullptr &&
           parallel_args_.cp_group_ != nullptr &&
           parallel_args_.cp_group_ != parallel_args_.tp_group_;
  }

  // Speculative state is replicated across every model rank within one DP
  // replica. With orthogonal CP x TP, tp_group_ covers only one CP shard,
  // while cp_group_ connects the same TP rank across CP shards. Broadcast
  // along both axes so every replica caches and consumes the same sampled
  // token, without crossing into another DP replica through the world group.
  void broadcast_spec_tokens(torch::Tensor& tokens) const {
    ProcessGroup* tp_group = parallel_args_.tp_group_ != nullptr
                                 ? parallel_args_.tp_group_
                                 : parallel_args_.process_group_;
    if (!use_orthogonal_cp_consensus()) {
      sync_sampling_tensor(tokens, tp_group);
      return;
    }

    sync_sampling_tensor(tokens, parallel_args_.tp_group_);
    sync_sampling_tensor(tokens, parallel_args_.cp_group_);
  }

  void maybe_broadcast_spec_tokens(torch::Tensor& tokens,
                                   SamplingMode sampling_mode) const {
    if (use_orthogonal_cp_consensus() ||
        (get_optimization_config().enable_spec_token_broadcast &&
         sampling_mode != SamplingMode::GREEDY)) {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      broadcast_spec_tokens(tokens);
    }
  }

 protected:
  // Target model worker
  std::unique_ptr<WorkerImpl> impl_;
};
}  // namespace xllm
