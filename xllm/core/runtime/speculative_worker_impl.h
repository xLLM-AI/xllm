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
#include <vector>

#include "common/macros.h"
#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "framework/sampling/draft_sampling_mode.h"
#include "framework/sampling/rejection_sampler.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/options.h"

namespace xllm {

// Returns whether this rank may execute the multi-step speculative decode
// plan for the current global DP batch.
bool should_run_speculative_decode(const ModelInputParams& params);

// Keep padded and raw DP token-count views in the same speculative layout.
void scale_speculative_parallel_token_counts(ModelInputParams& params,
                                             int32_t multiplier);

struct SpeculativeOutputStats {
  std::vector<int64_t> accepted_per_position;
  int64_t committed_tokens = 0;
};

SpeculativeOutputStats calculate_speculative_output_stats(
    const torch::Tensor& tokens,
    int64_t num_speculative_tokens);

// Base class for all speculative decoding workers.
// Provides common logic: target model management, step dispatch, and
// sampling parameter updates. Subclasses implement algorithm-specific
// draft generation and validation (MTP, Eagle3, Suffix, DFlash, etc.).
class SpeculativeWorkerImpl : public WorkerImpl {
 public:
  ~SpeculativeWorkerImpl() override = default;

 protected:
  // `options` is passed to WorkerImpl (preserves enable_schedule_overlap etc.),
  // `target_options` is used to create impl_ (target model worker).
  // Each algorithm subclass decides its own target_options.
  SpeculativeWorkerImpl(const ParallelArgs& parallel_args,
                        const torch::Device& device,
                        const runtime::Options& options,
                        const runtime::Options& target_options);

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

  // prepare input for execution
  ForwardInput prepare_inputs(Batch& batch) override {
    return impl_->prepare_inputs(batch);
  };

  // prepare work before model execution
  void prepare_work_before_execute(const ForwardInput& input,
                                   ForwardInput& new_input) override;
  void restore_json_object_states(ForwardInput& input) override;

  // Common step dispatch: prefill / decode / empty
  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;

  folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings) override {
    return impl_->pull_kv_blocks_async(src_cluster_id, src_addr, mappings);
  };

 protected:
  // Algorithm-specific virtual methods for subclasses to implement
  virtual std::optional<ForwardOutput> step_prefill(
      const ForwardInput& input) = 0;
  virtual std::optional<ForwardOutput> step_decode(
      const ForwardInput& inputs) = 0;
  virtual std::optional<ForwardOutput> step_empty(
      const ForwardInput& inputs) = 0;

  // Common helper: update sampling params for validation
  void update_sampling_params(SamplingParameters& sampling_params,
                              const int32_t num_val_tokens,
                              const int32_t total_num_val_tokens);
  void update_sampling_params(SamplingParameters& sampling_params,
                              const std::vector<int32_t>& per_seq_val_tokens,
                              const int32_t total_num_val_tokens);

  static void force_greedy_draft_sampling(SamplingParameters& sampling_params);

  // prepare inputs for target model at Decode phase (validation).
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs);
  // Per-seq variant used by adaptive-speculative pruning: each sequence's
  // validate row width equals per_seq_val_tokens[i] (must be in [1, N+1]).
  // The dense meta/token/position/kv-slot buffers are rebuilt as varlen with
  // total_tokens = Σ per_seq_val_tokens.
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               const std::vector<int32_t>& per_seq_val_tokens);

  // Overwrite dp_global_token_nums / raw_dp_global_token_nums with the true
  // post-pruning validate token count of every DP peer, gathered over the DP
  // group. Adaptive pruning makes each rank's validate token count
  // data-dependent, so the engine-supplied global vector (which assumes a
  // uniform per-seq width) no longer matches; DpEpPadding needs the real
  // per-rank counts to compute matching MoE all-to-all pads. No-op when the DP
  // group spans a single rank. MUST be called on every DP rank each validate
  // step (both the pruned and the unpruned branch) so the collective stays in
  // lockstep and does not deadlock.
  void sync_dp_global_token_nums_after_prune(ModelInputParams& input_params,
                                             int32_t local_total_val_tokens);

  // Idle-rank counterpart: a DP rank whose shard is empty still runs the target
  // validate forward (fake input) while busy peers run the pruned forward.
  // Both must join the same DP allgather. This variant contributes the idle
  // rank's own current dp_global_token_nums entry (already scaled to the
  // uniform validate width) so it stays symmetric with the busy peers.
  void sync_dp_global_token_nums_for_idle_rank(ModelInputParams& input_params);

  // Target-side cache budget after reserving storage for a colocated draft.
  // DeepSeek-V4's fixed SWA pools require both geometries to participate.
  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity_with_draft(
      LLMWorkerImpl& draft_impl,
      const runtime::Options& target_options,
      const runtime::Options& draft_options);

 protected:
  // Target model worker
  std::unique_ptr<LLMWorkerImpl> impl_;

  // Optional adaptive pruning controller. Subclasses create it in their ctor
  // when the algorithm supports adaptive per-seq validate pruning (MTP,
  // DFlash, DSpark). Left null otherwise. Held in the base so shared plumbing
  // (predictor setter, profile hook) does not need per-subclass duplication.
  std::unique_ptr<AdaptiveSpeculativeController> adaptive_spec_controller_;

  bool enable_fused_kernel_ = false;
  int32_t embedding_size_ = 0;
  DraftSamplingMode draft_sampling_mode_ = DraftSamplingMode::GREEDY;
};
}  // namespace xllm
