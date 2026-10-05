/* Copyright 2026 The xLLM Authors.

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
#include <functional>
#include <memory>
#include <string>
#include <tuple>

#include "framework/sampling/draft_sampling_mode.h"
#include "runtime/speculative_worker_impl.h"

namespace xllm {

class AdaptiveSpeculativeController;
class EmbeddingCache;
class LLMWorkerImpl;

// Base for draft-model speculative workers (MTP, DFlash/DSpark); Suffix has no
// draft model and derives from SpeculativeWorkerImpl directly.
template <typename TargetInput>
class DraftModelSpecWorkerImpl : public SpeculativeWorkerImpl<TargetInput> {
 public:
  ~DraftModelSpecWorkerImpl() override;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;
  bool task_models_loaded() const override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  // Derived step_* paths build their own draft inputs.
  TargetInput update_input_by_last_step_output(TargetInput& inputs) override;

 protected:
  using TargetModelParams =
      typename SpeculativeWorkerImpl<TargetInput>::TargetModelParams;
  using SpeculativeWorkerImpl<TargetInput>::compute_stream_;
  using SpeculativeWorkerImpl<TargetInput>::context_;
  using SpeculativeWorkerImpl<TargetInput>::device_;
  using SpeculativeWorkerImpl<TargetInput>::dtype_;
  using SpeculativeWorkerImpl<TargetInput>::hierarchy_kv_cache_transfer_;
  using SpeculativeWorkerImpl<TargetInput>::impl_;
  using SpeculativeWorkerImpl<TargetInput>::kv_cache_transfer_;
  using SpeculativeWorkerImpl<TargetInput>::options_;
  using SpeculativeWorkerImpl<TargetInput>::parallel_args_;
  using SpeculativeWorkerImpl<TargetInput>::set_hierarchy_kv_cache_transfer;

  DraftModelSpecWorkerImpl(
      const ParallelArgs& parallel_args,
      const torch::Device& device,
      const runtime::Options& options,
      const runtime::Options& target_options,
      WorkerType worker_type,
      const std::function<std::unique_ptr<LLMWorkerImpl>()>& draft_factory);

  virtual KVCacheShape draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape) const;
  // Reuse grouped pool counts; otherwise use the draft's geometry and DP-local
  // TP size, borrowing only the target's block count.
  KVCacheShape build_draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape,
      int64_t draft_world_size = -1) const;

  // MTP-family drafts reserve a pre-head hidden-state placeholder per row.
  virtual int64_t get_embedding_placeholder_size() const { return 0; }

  void prepare_hierarchy_kv_cache_transfers();
  void finalize_hierarchy_kv_cache_transfers();
  void init_embedding_cache(int64_t num_blocks);
  using AllocateFn = std::function<bool(WorkerImpl&, const KVCacheShape&)>;
  bool allocate_pools(const KVCacheShape& kv_cache_shape,
                      const AllocateFn& allocate);

  // Target-side cache budget after reserving storage for the colocated draft.
  // DeepSeek-V4's fixed SWA pools require both geometries to participate.
  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity_with_draft(
      const runtime::Options& target_options,
      const runtime::Options& draft_options);

  void prepare_draft_input(const TargetInput& input);

  static void force_greedy_draft_sampling(SamplingParameters& sampling_params);

  // All DP ranks, including unpruned ranks, must join this collective each
  // validation step so MoE padding uses matching post-pruning token counts.
  void sync_dp_global_token_nums_after_prune(TargetModelParams& input_params,
                                             int32_t local_total_val_tokens);

  // Idle ranks join the same collective using their prepared dummy width.
  void sync_dp_global_token_nums_for_idle_rank(TargetModelParams& input_params);

  std::unique_ptr<LLMWorkerImpl> draft_impl_;
  std::unique_ptr<EmbeddingCache> embedding_cache_;
  std::unique_ptr<AdaptiveSpeculativeController> adaptive_spec_controller_;

  DraftSamplingMode draft_sampling_mode_ = DraftSamplingMode::GREEDY;
};

}  // namespace xllm
