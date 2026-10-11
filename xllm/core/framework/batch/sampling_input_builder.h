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

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

class Sequence;

// Common sampling-row layout for sequence and Rec inputs. Request parameters
// are non-owning; row indices and token statistics belong to this build.
class SamplingInputBuilder final {
 public:
  using TokenCounts = std::unordered_map<int32_t, int32_t>;

  void reserve(size_t rows);
  void append(const RequestSamplingParam* params,
              int32_t token_index,
              const TokenCounts* counts = nullptr,
              const TokenCounts* excluded_counts = nullptr,
              bool sample = true,
              const Sequence* sequence = nullptr);
  void merge(SamplingInputBuilder other, int32_t token_offset);
  SamplingParameters build();

  size_t size() const { return selected_token_indices_.size(); }
  bool empty() const { return selected_token_indices_.empty(); }
  const std::vector<int32_t>& sample_indices() const { return sample_indices_; }
  const std::vector<int32_t>& selected_token_indices() const {
    return selected_token_indices_;
  }

 private:
  std::vector<const RequestSamplingParam*> params_;
  std::vector<const Sequence*> sequences_;
  std::vector<int32_t> selected_token_indices_;
  std::vector<int32_t> sample_indices_;
  std::vector<std::vector<int64_t>> token_ids_;
  std::vector<std::vector<int32_t>> token_counts_;
  std::vector<int32_t> token_lengths_;
};

}  // namespace xllm
