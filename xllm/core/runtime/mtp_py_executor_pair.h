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

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>

namespace xllm {

class PyExecutorImpl;

namespace detail {

struct MtpPyGraphOutput {
  // Borrowed final graph input, not an output snapshot or API field.
  torch::Tensor draft_embedding_destination;
  torch::Tensor token_state;
  torch::Tensor committed_tokens;
  torch::Tensor target_embeddings;
  torch::Tensor committed_log_probs;
  torch::Tensor target_top_log_probs;
  torch::Tensor target_top_tokens;
};

class __attribute__((visibility("hidden"))) MtpPyGraphVariantRegistry final {
 public:
  static std::unique_ptr<MtpPyGraphVariantRegistry> create(
      PyExecutorImpl& target_executor,
      PyExecutorImpl& draft_executor,
      int32_t max_variants);

  ~MtpPyGraphVariantRegistry();

  MtpPyGraphOutput execute_sparse(const torch::Tensor& block_table,
                                  const torch::Tensor& first_kv_seq_lens,
                                  const torch::Tensor& first_slots,
                                  const torch::Tensor& repair_token_ids,
                                  const torch::Tensor& seed_token_ids,
                                  const torch::Tensor& base_positions,
                                  const torch::Tensor& kv_seq_lens,
                                  const torch::Tensor& draft_input_embedding,
                                  int32_t batch_size,
                                  int32_t speculative_tokens,
                                  int64_t vocab_size,
                                  int32_t block_size,
                                  bool target_step_major_layout,
                                  bool logprobs,
                                  int32_t max_top_logprobs);

 private:
  explicit MtpPyGraphVariantRegistry(pybind11::object registry);

  pybind11::object registry_;
  pybind11::object execute_sparse_;
};

}  // namespace detail
}  // namespace xllm
