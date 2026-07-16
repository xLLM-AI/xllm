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
  bool use_qwen3_5_spec_verify_path() const;
  bool use_mimo_spec_verify_path() const;
  // Returns true when validation must use chunked-prefill to avoid the
  // FlashInfer batch-decode read-before-write race on the bonus token.
  bool use_chunked_prefill_spec_verify_path() const;

  // Prepare target validate input from cached target context.
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs);

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
      ForwardInput& extend_input,
      Stream& preparation_stream);

  struct PendingTargetContext {
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    // Both tensors stay on device.  A steady-state overlap step consumes them
    // by queueing gather/update ops behind rejection sampling on the same
    // stream.  They are materialized on CPU only when the batch shape/order
    // changes and the host cache fallback is required.
    torch::Tensor accepted_tokens;
    torch::Tensor accepted_tokens_host;
    torch::Tensor accepted_embeddings;
    StreamEventPtr ready_event;
  };

  struct PendingDraftContext {
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    std::optional<ForwardOutput> output;
    ForwardInput prepared_input;
  };

  void stage_target_context_write(const ForwardInput& input,
                                  const SampleOutput& validate_output,
                                  StreamEventPtr ready_event,
                                  torch::Tensor accepted_tokens_host);
  torch::Tensor acquire_accepted_tokens_host_buffer(
      const torch::Tensor& accepted_tokens);
  bool pending_target_context_matches(const ForwardInput& input) const;
  void flush_pending_target_context();
  void prepare_next_first_draft_template(const ForwardInput& input,
                                         ForwardInput& draft_input);
  void enqueue_next_first_draft(const ForwardInput& input,
                                const SampleOutput& validate_output,
                                ForwardInput draft_input);
  bool pending_draft_context_matches(const ForwardInput& input) const;

 protected:
  // Draft model worker
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  // Embedding cache for speculative decoding
  std::shared_ptr<EmbeddingCache> embedding_cache_;

  // Rejection sampling produces accepted state on the compute stream.  Keep
  // that state device-resident so the next overlap task can be fully enqueued
  // without waiting for target verification to finish.
  PendingTargetContext pending_target_context_;
  // A single persistent pinned destination is sufficient for accepted-token
  // D2H: the preceding pending target context is always flushed before the
  // next validation can submit another copy. The pending context holds a view
  // into this storage until the copy event is synchronized and CPU consumers
  // have finished reading it.
  torch::Tensor accepted_tokens_host_buffer_;
  // Draft step 0 is submitted at the tail of the preceding target validation,
  // before control returns to the scheduler.  The following scheduler turn
  // consumes this output and only submits draft steps 1..N-1.
  PendingDraftContext pending_draft_context_;

  // Whether validation directly uses selected-only draft_probs [B, S].
  // If false, selected-only cache values are restored to dense [B, S, V].
  bool enable_opt_validate_probs_ = false;

#if defined(USE_NPU) || defined(USE_MLU)
  std::shared_ptr<KVCacheTransfer> kv_cache_transfer_;
#endif
};
}  // namespace xllm
