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

#include <absl/functional/function_ref.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "core/framework/speculative/draft_vocab_mapping.h"
#include "core/framework/speculative/embedding_cache.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "runtime/speculative_worker_impl.h"

namespace xllm {

ParallelArgs configure_draft_parallel_args(ParallelArgs parallel_args);

class LLMWorkerImpl;
class LlmForwardInput;
class VlmForwardInput;

template <typename Input>
std::optional<ForwardOutput> run_worker_no_sync_impl(
    WorkerImpl& worker,
    const Input& input,
    Input& processed_input,
    Stream& prepare_stream,
    Stream& compute_stream,
    WorkerImpl::ForwardCompletion completion =
        WorkerImpl::ForwardCompletion::CALLER_RETAINS_INPUT);

// Base for draft-model speculative workers (MTP, DFlash/DSpark); Suffix has no
// draft model and derives from SpeculativeWorkerImpl directly.
template <typename TargetInput>
class DraftModelSpecWorkerImpl : public SpeculativeWorkerImpl<TargetInput> {
 public:
  ~DraftModelSpecWorkerImpl() override;

  void drain_pending_execution() override;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

  std::optional<ForwardOutput> step(const TargetInput& input) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

 protected:
  // Derived step_* paths build their own draft inputs.
  bool prepare_model_decode_continuation(
      TargetInput& inputs,
      const ForwardOutput& prior_output) override;

  using SpeculativeWorkerImpl<TargetInput>::compute_stream_;
  using SpeculativeWorkerImpl<TargetInput>::context_;
  using SpeculativeWorkerImpl<TargetInput>::device_;
  using SpeculativeWorkerImpl<TargetInput>::dtype_;
  using SpeculativeWorkerImpl<TargetInput>::hierarchy_kv_transfer_;
  using SpeculativeWorkerImpl<TargetInput>::impl_;
  using SpeculativeWorkerImpl<TargetInput>::kv_transfer_;
  using SpeculativeWorkerImpl<TargetInput>::options_;
  using SpeculativeWorkerImpl<TargetInput>::parallel_args_;
  using SpeculativeWorkerImpl<TargetInput>::set_hierarchy_kv_transfer;

  DraftModelSpecWorkerImpl(const ParallelArgs& parallel_args,
                           const torch::Device& device,
                           const runtime::Options& options,
                           WorkerType worker_type,
                           const ParallelArgs& draft_parallel_args,
                           const runtime::Options& draft_opts);

  virtual void apply_draft_model_overrides(ModelArgs& args) = 0;

  virtual KVCacheShape draft_kv_cache_shape(
      const KVCacheShape& target_kv_cache_shape) const;

  // MTP-family drafts reserve a pre-head hidden-state placeholder per row.
  virtual int64_t get_embedding_placeholder_size() const { return 0; }

  // Some drafts replay every verified KV position, so retain the full
  // accepted span when allocating their embedding cache.
  virtual bool requires_full_target_replay() const { return false; }

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  int64_t get_draft_kv_pages_per_block() const;

  void prepare_hierarchy_kv_transfers();
  void finalize_hierarchy_kv_transfers();
  bool allocate_pools(
      const KVCacheShape& kv_cache_shape,
      absl::FunctionRef<bool(WorkerImpl&, const KVCacheShape&)> allocate);

  void update_decode_step_input(
      TargetInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states) const;

  std::optional<ForwardOutput> finalize_step_output(
      ForwardOutput output,
      std::optional<SampleOutput> verify_output = std::nullopt,
      const std::vector<JsonObjectGrammarState>& json_object_states = {}) const;

  void broadcast_spec_tokens(torch::Tensor& tokens) const;
  void maybe_broadcast_spec_tokens(torch::Tensor& tokens,
                                   SamplingMode sampling_mode) const;

  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  ModelContext load_model_config(ModelLoader& loader);
  KVCacheCreateOptions build_draft_kv_cache_layout(
      const KVCacheShape& shape) const;
#if defined(USE_NPU)
  void prepare_draft_padding(LlmModelParams& params,
                             bool acquire_graph_capture_lock = false);
  DpEpPaddingCache dp_ep_padding_cache_{2};
#endif
  std::unique_ptr<EmbeddingCache> embedding_cache_;

  DraftVocabMapping draft_vocab_;
  mtp_async::TargetSpecVerifyMode target_verify_mode_ =
      mtp_async::TargetSpecVerifyMode::GENERIC;

 private:
  bool use_orthogonal_cp_consensus() const;
};

}  // namespace xllm
