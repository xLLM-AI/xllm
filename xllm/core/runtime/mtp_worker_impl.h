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

#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache/embedding_cache.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#if defined(USE_NPU)
#include "framework/kv_cache_transfer/spec_kv_cache_transfer.h"
#endif
#include "runtime/speculative_worker_impl.h"

namespace xllm {

#if defined(USE_NPU)
using namespace llm_datadist;
#endif

// MTP (Multi-Token Prediction) speculative worker.
// Uses a draft model to generate proposals, then validates with target model.
// Eagle3WorkerImpl inherits from this class.
class MTPWorkerImpl : public SpeculativeWorkerImpl {
 public:
  MTPWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~MTPWorkerImpl() override = default;

 protected:
  // For derived classes (e.g. Eagle3WorkerImpl) that need custom options for
  // target and draft models. `options` is passed to WorkerImpl (preserves
  // enable_schedule_overlap etc.), `target_options` / `draft_options` are used
  // to create the respective workers.
  MTPWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options,
                const runtime::Options& target_options,
                const runtime::Options& draft_options,
                bool enable_opt_validate_probs = false);

 public:
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;
  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs) override;

  void fill_validate_input_from_draft_outputs(
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      Stream& compute_stream);
  std::optional<ForwardOutput> run_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input);

  virtual SampleOutput validate(const SamplingParameters& sampling_params,
                                const std::vector<ForwardOutput>& draft_outputs,
                                const ForwardOutput& target_output);

  // Hook for algorithm-specific draft output post-processing during decode.
  // Default MTP behavior always compresses probs for cache storage.
  virtual void process_draft_sample_output(SampleOutput& sample_output);

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        const ForwardOutput& target_output);

  // PD separation: placeholder size for empty embedding slot. Default: 1x
  // hidden_size. Eagle3 overrides to 3 * target_hidden_size.
  virtual int64_t get_embedding_placeholder_size();

  // prepare inputs for draft model at Prefill phase.
  void prepare_prefill_inputs(const ForwardInput& inputs,
                              ForwardInput& prefill_inputs);
  SpeculativeVerifyCapabilities speculative_verify_capabilities() const;
  bool supports_spec_verify_graph_input_update() const;
  // Returns true when validation must use chunked-prefill to avoid the
  // FlashInfer batch-decode read-before-write race on the bonus token.
  bool use_chunked_prefill_spec_verify_path() const;

  // Prepare target validate input from cached target context.
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               bool static_graph_tasks_prepared = false);
  bool prepare_static_mtp_graph_tasks_before_final_draft(
      const ForwardInput& input);

  // prepare inputs for draft model at Decode phase.
  void prepare_draft_inputs(const ForwardInput& inputs,
                            ForwardInput& draft_inputs,
                            int32_t position_offset);
  void update_decode_step_input(
      ForwardInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states) const;

  // Build draft-side input from cached target context at decode step start.
  void prepare_draft_extend_inputs(
      const ForwardInput& base_input,
      const std::vector<EmbeddingCache::DecodeState>& last_states,
      ForwardInput& extend_input);

  void write_target_context_to_cache(const ForwardInput& input,
                                     const SampleOutput& validate_output);

 protected:
  // Draft model worker
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  // Embedding cache for speculative decoding
  std::shared_ptr<EmbeddingCache> embedding_cache_;

  // Whether validation directly uses selected-only draft_probs [B, S].
  // If false, selected-only cache values are restored to dense [B, S, V].
  bool enable_opt_validate_probs_ = false;

  // Cached once when the target model is loaded. Decode-path decisions must
  // not repeatedly traverse the model's virtual capability interface.
  SpeculativeVerifyCapabilities target_spec_verify_capabilities_;

  // Immutable single-request draft row selectors. The steady workload
  // alternates between the legal one-row (index 0) and two-row (index 1)
  // layouts; keeping both indices on device avoids rebuilding/H2D each cycle.
  torch::Tensor draft_selected_row_zero_;
  torch::Tensor draft_selected_row_one_;
#if defined(USE_NPU)
  // Stable-address sources consumed by the target ACL graph's leading input
  // update. The existing H2D preparation overlaps with the final draft, so no
  // extra graph-external D2D launch is introduced.
  torch::Tensor spec_verify_tokens_host_;
  torch::Tensor spec_verify_tokens_device_;
  torch::Tensor spec_verify_positions_host_;
  torch::Tensor spec_verify_positions_device_;
  torch::Tensor spec_verify_attention_host_buffer_;
  torch::Tensor spec_verify_attention_device_buffer_;
  uint64_t spec_verify_attention_buffer_capacity_ = 0;

  // Stable validate-sampling controls for the common single-sequence greedy
  // path.  Their values depend on speculative width, not tensor-parallel
  // topology, and are rebuilt only when that width changes.
  torch::Tensor mtp_validate_greedy_indices_;
  torch::Tensor mtp_validate_greedy_do_sample_;
#endif

#if defined(USE_NPU) || defined(USE_MLU)
  std::shared_ptr<KVCacheTransfer> kv_cache_transfer_;
#endif
};
}  // namespace xllm
