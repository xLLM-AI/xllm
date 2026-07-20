/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "framework/kv_cache/embedding_cache.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "runtime/speculative_worker_impl.h"

namespace xllm {

class DFlashWorkerImpl final : public SpeculativeWorkerImpl {
 public:
  DFlashWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~DFlashWorkerImpl() override = default;

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

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_empty(const ForwardInput& input) override;

  // Draft produces all speculative tokens of a block in one forward, so its
  // output is a single [batch, num_speculative_tokens] block rather than the
  // per-step outputs an autoregressive drafter (e.g. MTP) yields.
  struct DraftBlock {
    torch::Tensor token_ids;
    torch::Tensor probs;
    // The draft runs no-sync, so execute_no_sync_on_stream stashes its still
    // in-flight input in ForwardOutput::retained_input. Anchor it here so the
    // draft input outlives run_validate's unconditional compute-stream sync;
    // otherwise the buffer frees when run_decode_draft returns and the validate
    // stage could reuse memory the draft kernel is still reading.
    std::shared_ptr<ForwardInput> draft_retained_input;
  };

  DraftBlock run_decode_draft(const ForwardInput& input,
                              ForwardInput& validate_input);

 private:
  void fill_validate_input_from_draft_outputs(const DraftBlock& draft_block,
                                              ForwardInput& validate_input,
                                              Stream& compute_stream);

  std::optional<ForwardOutput> run_validate(const ForwardInput& input,
                                            const DraftBlock& draft_block,
                                            ForwardInput& validate_input);

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const DraftBlock& draft_block,
                        const ForwardOutput& target_output);

  SampleOutput validate(const SamplingParameters& sampling_params,
                        const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        const ForwardOutput& target_output);

  void process_draft_sample_output(SampleOutput& sample_output);

  // Mirrors sampled tokens to rank 0 under schedule-overlap so every rank
  // commits the same prefix. No-op for a single rank.
  void maybe_broadcast_spec_tokens(torch::Tensor& tokens);

  void update_decode_step_input(
      ForwardInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states) const;

  void prepare_validate_inputs(const ForwardInput& input,
                               ForwardInput& validate_input);

  void prepare_query_inputs(const ForwardInput& input,
                            ForwardInput& query_input);

  void write_context_kv(const ForwardInput& input,
                        const torch::Tensor& context_hidden,
                        const torch::Tensor& positions_device,
                        const torch::Tensor& new_cache_slots_device);

  void write_target_context_to_cache(const ForwardInput& input,
                                     const SampleOutput& validate_output);

  std::unique_ptr<LLMWorkerImpl> draft_impl_;
  std::shared_ptr<EmbeddingCache> embedding_cache_;
#if defined(USE_NPU) || defined(USE_MLU)
  std::shared_ptr<KVCacheTransfer> kv_cache_transfer_;
#endif
  int32_t mask_token_id_ = -1;
  int64_t expected_context_hidden_size_ = 0;
};

}  // namespace xllm
