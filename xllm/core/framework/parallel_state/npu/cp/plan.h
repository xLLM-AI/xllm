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

#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "framework/parallel_state/npu_cp_ep_padding.h"
#include "framework/parallel_state/npu_cp_prepare.h"
#include "util/json_reader.h"

namespace xllm {

class ProcessGroup;
struct ModelInputParams;

namespace npu::cp {

// Canonical, global-real input layout before CP partitioning.
struct SourceLayout {
  std::vector<int32_t> q_seq_lens;
  torch::Tensor positions;
  bool have_prefix_slots = false;
  std::vector<int32_t> kv_cache_tokens_per_seq;
};

// Parallel and model settings needed to derive every CP auxiliary tensor.
struct ParallelConfig {
  int32_t size = 1;
  int32_t rank = 0;
  int32_t block_size = 0;
  int32_t kv_split_size = 1;
  int32_t num_experts_per_tok = 1;
  nlohmann::json mapping_data;
  torch::Device device = torch::kCPU;
  torch::ScalarType dtype = torch::kFloat;
  bool is_prefill = true;
};

// Owns the complete set of layout and auxiliary tensors for one CP forward.
class Plan final {
 public:
  Plan() = default;

  static Plan build(const SourceLayout& source, const ParallelConfig& config);

  Plan to(const torch::Device& device) const;

  bool enabled() const { return layout_.enabled; }
  int32_t size() const { return layout_.cp_size; }
  int32_t rank() const { return layout_.cp_rank; }
  int64_t global_real_token_num() const {
    return layout_.global_real_token_num;
  }
  int64_t local_padded_token_num() const {
    return layout_.local_padded_token_num;
  }
  int64_t recovered_token_num() const {
    return static_cast<int64_t>(size()) * local_padded_token_num();
  }

  const CpPrefillInputs& prefill_inputs() const { return prefill_inputs_; }
  CpPrefillInputs& mutable_prefill_inputs() { return prefill_inputs_; }

  const CpEpPaddingData& ep_padding_data() const { return ep_padding_data_; }
  CpEpPaddingData& mutable_ep_padding_data() { return ep_padding_data_; }

  void preprocess(torch::Tensor& hidden,
                  torch::Tensor& positions,
                  ModelInputParams& input_params) const;
  torch::Tensor postprocess(const torch::Tensor& hidden,
                            ProcessGroup* process_group) const;

  torch::Tensor localize_slots_recovered(
      const torch::Tensor& global_slots) const;

 private:
  NpuCpPrefillPlan layout_;
  CpPrefillInputs prefill_inputs_;
  CpEpPaddingData ep_padding_data_;
};

}  // namespace npu::cp
}  // namespace xllm
