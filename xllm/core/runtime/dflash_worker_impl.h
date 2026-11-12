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
#include <vector>

#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/speculative/draft_proposal.h"
#include "core/framework/speculative/embedding_cache.h"
#include "core/kv_cache/transfer/kv_cache_transfer.h"
#include "core/util/tensor_helper.h"
#include "framework/model/model_args.h"
#include "runtime/draft_model_spec_worker_impl.h"

namespace xllm {

namespace dflash_detail {

inline void invalidate_draft_model_geometry(LlmModelParams& input_params) {
  // Attention metadata is model-owned: DeepSeek-V4 bakes DSA group layout and
  // sparse tiling values such as ori_win_left into opaque tensors. A draft
  // input copied from the target must rebuild those tensors for draft geometry.
  input_params.attn_metadata.reset();
  // Block drafters own no recurrent layers. Target slots and validity masks
  // must not reach the draft metadata builder, including during graph warmup.
  input_params.clear_linear_attention_state();
}

}  // namespace dflash_detail

class DFlashWorkerImpl : public DraftModelSpecWorkerImpl<LlmForwardInput> {
 public:
  DFlashWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~DFlashWorkerImpl() override;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

 protected:
  DFlashWorkerImpl(
      const ParallelArgs& parallel_args,
      const torch::Device& device,
      const runtime::Options& options,
      absl::FunctionRef<std::unique_ptr<LLMWorkerImpl>()> draft_factory,
      int32_t (*configure_draft_config)(ModelArgs&, const runtime::Options&));

  static runtime::Options draft_options(const runtime::Options& options,
                                        int32_t num_speculative_tokens = 0);
  std::optional<ForwardOutput> step_prefill(
      const LlmForwardInput& input) override;

  std::optional<ForwardOutput> step_decode(
      const LlmForwardInput& input) override;

  std::optional<ForwardOutput> step_empty(
      const LlmForwardInput& input) override;

  // Draft produces all speculative tokens of a block in one forward, so its
  // output is a single [batch, num_speculative_tokens] block rather than the
  // per-step outputs an autoregressive drafter (e.g. MTP) yields; the
  // proposal carries the draft forward's retained no-sync handles.
  //
  // virtual: DSpark overrides the draft sampling (parallel block sample ->
  // one forward + sequential Markov-head sampling loop).
  virtual DraftProposal run_decode_draft(const LlmForwardInput& input,
                                         LlmForwardInput& validate_input);

  // Shared with subclasses (DSpark): build the N/N+1-wide draft query block and
  // the target validate input. A DSpark override of run_decode_draft calls both
  // before its draft forward.
  void prepare_query_inputs(const LlmForwardInput& input,
                            LlmForwardInput& query_input);
  void prepare_validate_inputs(const LlmForwardInput& input,
                               LlmForwardInput& validate_input);

  int64_t draft_replicated_block_pages() const override {
    return SpeculativeConfig::draft_replicated_block_pages(
        options_.speculative_algorithm(),
        parallel_args_.kv_split_size_effective());
  }

  void expand_draft_block_tables(LlmModelParams& input_params) const;

  bool draft_input_per_request_ = true;
  BatchForwardType draft_forward_type_ = BatchForwardType::CHUNKED_PREFILL;
  bool draft_needs_dummy_tokens_ = false;
  bool sample_from_anchor_ = false;

 private:
  DecodeExpansionOptions prepare_for_draft() const;

  bool prepare_parallel_draft_metadata(
      LlmForwardInput& query_input,
      const DecodeExpansionOptions& decode_options);

  void fill_validate_input_from_draft_outputs(
      const DraftProposal& draft_proposal,
      LlmForwardInput& validate_input,
      Stream& compute_stream);
  std::optional<ForwardOutput> run_validate(const LlmForwardInput& input,
                                            const DraftProposal& draft_proposal,
                                            LlmForwardInput& validate_input);

  std::vector<std::string> write_context_kv(
      const LlmForwardInput& input,
      const torch::Tensor& context_hidden,
      const torch::Tensor& positions_device,
      const torch::Tensor& new_cache_slots_device);

  std::vector<std::string> write_target_context_to_cache(
      const LlmForwardInput& input,
      const SampleOutput& validate_output);

 protected:
  torch::Tensor anchor_token_ids(const LlmForwardInput& input) const;

  int32_t mask_token_id_ = -1;
  int64_t expected_context_hidden_size_ = 0;
  // Whether the target model owns recurrent (linear attention) layers.
  bool target_is_hybrid_recurrent_ = false;

 private:
  CachedDeviceIndices draft_logits_indices_;
};

}  // namespace xllm
