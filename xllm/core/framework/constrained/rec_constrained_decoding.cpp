/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/constrained/rec_constrained_decoding.h"

#include <glog/logging.h>

#include "core/framework/state_dict/rec_vocab_dict.h"
#include "core/util/slice.h"
#include "core/util/tensor_helper.h"

namespace xllm {
RecConstrainedDecoding::RecConstrainedDecoding(RecVocabDict* vocab_dict,
                                               const int32_t vocab_size,
                                               torch::ScalarType dtype,
                                               torch::Device device)
    : vocab_size_(vocab_size),
      vocab_dict_(vocab_dict),
      device_(device),
      dtype_(dtype) {}

void RecConstrainedDecoding::build_mask_cache() {
  CHECK(vocab_dict_ != nullptr)
      << "RecVocabDict must be initialized before constrained decoding.";
  first_token_mask_ = torch::full({vocab_size_}, kPreMaskFactor, dtype_);

  std::vector<int32_t> empty_token_ids;
  Slice<int32_t> prefix_token_ids = {empty_token_ids.data(),
                                     empty_token_ids.size()};

  const std::unordered_set<int32_t>& first_token_ids =
      vocab_dict_->get_next_tokens_by_prefix_tokens(prefix_token_ids);

  for (auto token_id : first_token_ids) {
    first_token_mask_[token_id] = 0;
  }

  first_token_mask_ = safe_to(first_token_mask_, device_, true);

  build_mask_cache_ = true;

  LOG(INFO) << "Build mask cache, first token ids size:"
            << first_token_ids.size();
}

torch::Tensor RecConstrainedDecoding::generate_mask(
    const std::vector<std::vector<int32_t>>& generated_token_list) {
  if (!build_mask_cache_ || 0 == generated_token_list.size()) {
    return torch::Tensor();
  }

  size_t token_size = generated_token_list[0].size();

  if (0 == token_size) {
    size_t sequence_num = generated_token_list.size();
    auto mask = first_token_mask_.unsqueeze(0);
    return mask.repeat({static_cast<int64_t>(sequence_num), 1});
  }

  return generate_decode_mask(generated_token_list);
}

torch::Tensor RecConstrainedDecoding::generate_decode_mask(
    const std::vector<std::vector<int32_t>>& generated_token_list) {
  size_t sequence_num = generated_token_list.size();
  torch::TensorOptions options = torch::dtype(dtype_).device(device_);
  auto mask = torch::full({static_cast<int64_t>(sequence_num), vocab_size_},
                          kPreMaskFactor,
                          options);

  std::vector<int64_t> token_indices;
  std::vector<int64_t> vocab_indices;

  constexpr int32_t kMaxIndexNumPerToken = 8192;
  token_indices.reserve(kMaxIndexNumPerToken * sequence_num);
  vocab_indices.reserve(kMaxIndexNumPerToken * sequence_num);

  for (size_t token_idx = 0; token_idx < sequence_num; ++token_idx) {
    Slice<int32_t> tokens_slice(generated_token_list[token_idx]);

    const std::unordered_set<int32_t>& next_token_ids =
        vocab_dict_->get_next_tokens_by_prefix_tokens(tokens_slice);

    if (next_token_ids.empty()) {
      LOG(ERROR) << "Fail to generate mask for tokens:"
                 << generated_token_list[token_idx];
      continue;
    }
    for (int32_t vocab_idx : next_token_ids) {
      token_indices.push_back(static_cast<int64_t>(token_idx));
      vocab_indices.push_back(static_cast<int64_t>(vocab_idx));
    }
  }

  if (!token_indices.empty()) {
    auto token_index_tensor = torch::tensor(token_indices, torch::kInt64);
    auto vocab_index_tensor = torch::tensor(vocab_indices, torch::kInt64);
    token_index_tensor = safe_to(token_index_tensor, device_, true);
    vocab_index_tensor = safe_to(vocab_index_tensor, device_, true);
    mask.index_put_({token_index_tensor, vocab_index_tensor}, 0.0f);
  }

  return mask;
}
}  // namespace xllm
