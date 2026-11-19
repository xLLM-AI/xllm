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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/speculative/embedding_cache.h"
#include "core/kv_cache/transfer/kv_transfer.h"
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
  void apply_target_model_overrides(ModelArgs& args) const override;
  std::optional<ForwardOutput> step_prefill(
      const LlmForwardInput& input) override;

  std::optional<ForwardOutput> step_decode(
      const LlmForwardInput& input) override;

  std::optional<ForwardOutput> step_empty(
      const LlmForwardInput& input) override;

  virtual std::pair<torch::Tensor, torch::Tensor> run_decode_draft(
      const LlmForwardInput& input,
      LlmForwardInput& validate_input,
      LlmForwardInput& draft_forward_input);

  virtual DecodeExpansionOptions prepare_for_draft() const;

  void prepare_query_inputs(const LlmForwardInput& input,
                            LlmForwardInput& query_input);
  void prepare_validate_inputs(const LlmForwardInput& input,
                               LlmForwardInput& validate_input);

  bool draft_needs_dummy_tokens_ = false;

 private:
  bool prepare_parallel_draft_metadata(
      LlmForwardInput& query_input,
      const DecodeExpansionOptions& decode_options);

  std::optional<ForwardOutput> run_validate(
      const LlmForwardInput& input,
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& draft_logits,
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
