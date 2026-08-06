/* Copyright 2026 The xLLM Authors.

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

#include "layers/common/attention_metadata.h"

namespace xllm {
struct ModelInputParams;

namespace layer {

class ExpandedDecodeMetadataBuilder final {
 public:
  static void populate(ModelInputParams& target,
                       const ModelInputParams& source,
                       const torch::Tensor& kv_seq_lens,
                       int32_t block_size);

  static void populate_expanded_layout(
      ModelInputParams& target,
      const torch::Tensor& expanded_kv_seq_lens,
      const torch::Tensor& expanded_block_tables,
      std::vector<int32_t> expanded_host_kv_seq_lens,
      int32_t block_size);

  static std::vector<int32_t> build_tokenwise_kv_seq_lens(
      const std::vector<int32_t>& q_seq_lens,
      const std::vector<int32_t>& kv_seq_lens);

  static ExpandedDecodeMetadata build(const ModelInputParams& params);

  static void validate(const ExpandedDecodeMetadata& metadata,
                       int64_t expected_sequence_count = -1,
                       int32_t block_size = 0);
};

}  // namespace layer
}  // namespace xllm
