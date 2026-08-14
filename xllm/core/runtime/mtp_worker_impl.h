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
#include <optional>
#include <string>
#include <vector>

#include "core/framework/speculative/embedding_cache.h"
#include "core/framework/speculative/mtp_async_state.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#if defined(USE_NPU)
#include "framework/kv_cache_transfer/spec_kv_cache_transfer.h"
#endif
#include "core/framework/speculative/adaptive_speculative_controller.h"
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
                bool enable_opt_validate_probs = false,
                bool enable_adaptive_speculative_decode = false);

 public:
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override;

  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info) override;

  uint32_t transfer_kv_blocks(
      uint64_t batch_id,
      Slice<BlockTransferInfo>& block_transfer_info) override;

#if defined(USE_NPU) || defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  ForwardInput update_input_by_last_step_output(ForwardInput& inputs) override;
  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

 protected:
  // MTP composite: leaves own model-specific NPU input preparation.
  bool owns_npu_parallel_input_prepare() const override;

  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_decode(const ForwardInput& inputs) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& inputs) override;

  void fill_validate_input_from_draft_outputs(
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      const std::vector<int32_t>& per_seq_val_tokens,
      Stream& compute_stream);
  // Adaptive pruning path: compute per-seq prefix lengths, truncate draft
  // outputs, and run variable-length validate.
  std::optional<ForwardOutput> run_adaptive_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      int32_t num_speculative_tokens);
  std::optional<ForwardOutput> run_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>* pruned_prefix_lengths = nullptr);
  std::optional<ForwardOutput> run_validate(
      const ForwardInput& input,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>& per_seq_val_tokens,
      const std::vector<int32_t>* pruned_prefix_lengths = nullptr);

  virtual SampleOutput validate(
      const SamplingParameters& sampling_params,
      const std::vector<ForwardOutput>& draft_outputs,
      const ForwardOutput& target_output,
      int32_t num_speculative_tokens,
      // No default: Google Style bans default args on virtuals — they resolve
      // statically from the declared base type, so an override changing the
      // default would silently diverge when called through a base reference.
      // Callers must pass nullptr explicitly for the static path.
      const std::vector<int32_t>* pruned_prefix_lengths);

  // Hook for algorithm-specific draft output post-processing during decode.
  // Default MTP behavior always compresses probs for cache storage.
  virtual void process_draft_sample_output(SampleOutput& sample_output);

  SampleOutput validate(
      const SamplingParameters& sampling_params,
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& draft_probs,
      const ForwardOutput& target_output,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>* pruned_prefix_lengths = nullptr);

  // PD separation: placeholder size for empty embedding slot. Default: 1x
  // hidden_size. Eagle3 overrides to 3 * target_hidden_size.
  virtual int64_t get_embedding_placeholder_size();

  // prepare inputs for draft model at Prefill phase.
  void prepare_prefill_inputs(const ForwardInput& inputs,
                              ForwardInput& prefill_inputs);
  bool supports_explicit_spec_verify_replay_update() const;
  bool should_use_explicit_spec_verify_replay_update(
      const ForwardInput& input) const;
  // Returns true when the target model's spec-verify kernel requires the
  // validate width (val_tokens) to be identical across every sequence in the
  // batch. Currently Qwen3.5 GDN's FusedRecurrentGatedDeltaRule spec-verify
  // path has this constraint; other paths accept per-seq variable widths.
  // Kept separate from supports_explicit_spec_verify_replay_update() so the
  // two capabilities can diverge for future targets.
  bool requires_uniform_validate_width() const;
  int64_t spec_verify_block_table_width(
      const torch::Tensor& block_tables) const;
  // Returns true when validation must use chunked-prefill to avoid the
  // FlashInfer batch-decode read-before-write race on the bonus token.
  bool use_chunked_prefill_spec_verify_path() const;

  // Prepare target validate input from cached target context.
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               bool static_graph_tasks_prepared = false,
                               bool record_ready_event = true);
  void prepare_validate_inputs(const ForwardInput& inputs,
                               ForwardInput& validate_inputs,
                               const std::vector<int32_t>& per_seq_val_tokens);
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
      ForwardInput& extend_input,
      bool force_two_rows = false,
      bool wait_for_compute_stream = true);

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
    torch::Tensor base_positions;
    torch::Tensor base_kv_seq_lens;
    StreamEventPtr ready_event;
  };

  struct PendingDraftContext {
    std::vector<int32_t> embedding_ids;
    std::vector<std::string> request_ids;
    std::vector<int32_t> dp_global_token_nums;
    std::vector<int32_t> raw_dp_global_token_nums;
    std::vector<uint64_t> dp_global_batch_generations;
    std::optional<ForwardOutput> output;
    ForwardInput prepared_input;
  };

  void stage_target_context_write(const ForwardInput& input,
                                  const SampleOutput& validate_output,
                                  torch::Tensor base_positions,
                                  torch::Tensor base_kv_seq_lens,
                                  StreamEventPtr ready_event,
                                  torch::Tensor accepted_tokens_host);
  torch::Tensor acquire_accepted_tokens_host_buffer(
      const torch::Tensor& accepted_tokens);
  bool pending_target_context_matches(const ForwardInput& input) const;
  bool device_target_context_ready_for_batch(const ForwardInput& input) const;
  void flush_pending_target_context();
  bool supports_combined_first_draft_execution() const;
  bool can_use_combined_first_draft() const;
  bool can_prelaunch_next_first_draft(const ForwardInput& input) const;
  void prepare_next_first_draft_template(const ForwardInput& input,
                                         ForwardInput& combined_input);
  void enqueue_next_first_draft(const ForwardInput& input,
                                const SampleOutput& validate_output,
                                const torch::Tensor& base_positions,
                                const torch::Tensor& base_kv_seq_lens,
                                ForwardInput combined_input);
  void submit_pending_first_draft(const ForwardInput& batch_identity_input,
                                  ForwardInput draft_input);
  bool pending_draft_context_matches(const ForwardInput& input) const;

  void write_target_context_to_cache(const ForwardInput& input,
                                     const SampleOutput& validate_output,
                                     int32_t num_speculative_tokens);
  void record_validate_metrics(
      const SampleOutput& validate_output,
      int32_t num_speculative_tokens,
      const std::vector<int32_t>* pruned_prefix_lengths = nullptr) const;
  bool adaptive_enabled() const;

 protected:
  // Draft model worker
  std::unique_ptr<LLMWorkerImpl> draft_impl_;

  // Embedding cache for speculative decoding
  std::shared_ptr<EmbeddingCache> embedding_cache_;

  // Rejection sampling produces accepted state on the compute stream.  Keep
  // that state device-resident so the next overlap task can be fully enqueued
  // without waiting for target verification to finish.
  PendingTargetContext pending_target_context_;
  std::vector<int32_t> device_context_ready_embedding_ids_;
  std::vector<std::string> device_context_ready_request_ids_;
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
  // adaptive_spec_controller_ now lives on SpeculativeWorkerImpl (base class).

  // Classified once when the corresponding models are loaded. Decode-path
  // decisions only read these closed policies.
  mtp_async::TargetSpecVerifyMode target_spec_verify_mode_ =
      mtp_async::TargetSpecVerifyMode::GENERIC;
  mtp_async::CombinedDraftExecutionPath combined_draft_execution_path_ =
      mtp_async::CombinedDraftExecutionPath::UNSUPPORTED;

#if defined(USE_NPU)
  // Stable-address sources consumed by the target ACL graph's leading input
  // update. The existing H2D preparation overlaps with the final draft, so no
  // extra graph-external D2D launch is introduced.
  torch::Tensor spec_verify_attention_host_buffer_;
  torch::Tensor spec_verify_attention_device_buffer_;
  uint64_t spec_verify_attention_buffer_capacity_ = 0;
  std::shared_ptr<int> spec_verify_attention_buffer_owner_ =
      std::make_shared<int>(0);

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
