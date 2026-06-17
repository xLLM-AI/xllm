/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "rec_constrained_decoding.h"

#include <c10/core/TensorOptions.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>

#include "common/global_flags.h"
#include "framework/state_dict/rec_vocab_dict.h"
#include "util/slice.h"
#include "util/tensor_helper.h"

namespace xllm {
RecConstrainedDecoding::RecConstrainedDecoding(RecVocabDict* vocab_dict,
                                               const int32_t vocab_size,
                                               torch::ScalarType dtype,
                                               torch::Device device,
                                               bool use_gen_threadpool)
    : use_gen_threadpool_(use_gen_threadpool),
      vocab_size_(vocab_size),
      vocab_dict_(vocab_dict),
      device_(device),
      dtype_(dtype) {
  if (use_gen_threadpool_) {
    gen_threadpool_ = std::make_unique<ThreadPool>(
        /*num_threads=*/GEN_MASK_THREAD_NUM,
        /*cpu_binding=*/false,
        /*pool_name=*/"RecConstrainedDecoding.generate_mask");
  }

  build_mask_cache_ = false;
}

bool RecConstrainedDecoding::build_mask_cache() {
  CHECK(vocab_dict_ != nullptr)
      << "RecVocabDict must be initialized before constrained decoding.";
  first_token_mask_ = torch::full({vocab_size_}, PRE_MASK_FACTOR, dtype_);

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

  return true;
}

torch::Tensor RecConstrainedDecoding::generate_mask(
    const std::vector<std::vector<int32_t>>& generated_token_list) {
  if (!build_mask_cache_ || 0 == generated_token_list.size()) {
    return torch::Tensor();
  }

  size_t token_size = generated_token_list[0].size();

  // Generate mask for first token
  if (0 == token_size) {
    size_t sequence_num = generated_token_list.size();
    auto mask = first_token_mask_.unsqueeze(0);
    return mask.repeat({static_cast<int64_t>(sequence_num), 1});
  }

  // Generate mask for non-first token
  return generate_decode_mask(generated_token_list);
}

torch::Tensor RecConstrainedDecoding::generate_decode_mask(
    const std::vector<std::vector<int32_t>>& generated_token_list) {
  size_t sequence_num = generated_token_list.size();
  torch::TensorOptions options = torch::dtype(dtype_).device(device_);
  auto mask = torch::full({static_cast<int64_t>(sequence_num), vocab_size_},
                          PRE_MASK_FACTOR,
                          options);

  std::mutex global_batch_mutex;
  std::vector<int64_t> global_batch_token_indices;
  std::vector<int64_t> global_batch_vocab_indices;

  int max_index_num_per_token = 8192;
  global_batch_token_indices.reserve(max_index_num_per_token * sequence_num);
  global_batch_vocab_indices.reserve(max_index_num_per_token * sequence_num);

  auto update_mask = [&](size_t start_idx, size_t end_idx) {
    std::vector<int64_t> local_token_indices;
    std::vector<int64_t> local_vocab_indices;
    local_token_indices.reserve(max_index_num_per_token *
                                (end_idx - start_idx));
    local_vocab_indices.reserve(max_index_num_per_token *
                                (end_idx - start_idx));

    for (size_t token_idx = start_idx; token_idx < end_idx; ++token_idx) {
      Slice<int32_t> tokens_slice(generated_token_list[token_idx]);

      const std::unordered_set<int32_t>& next_token_ids =
          vocab_dict_->get_next_tokens_by_prefix_tokens(tokens_slice);

      if (next_token_ids.size() > 0) {
        for (int32_t vocab_idx : next_token_ids) {
          local_token_indices.push_back(static_cast<int64_t>(token_idx));
          local_vocab_indices.push_back(static_cast<int64_t>(vocab_idx));
        }
      } else {
        LOG(ERROR) << "Fail to generate mask for tokens:"
                   << generated_token_list[token_idx];
      }
    }

    // Merge local results to global batch (thread-safe)
    if (!local_token_indices.empty()) {
      std::lock_guard<std::mutex> lock(global_batch_mutex);
      global_batch_token_indices.insert(global_batch_token_indices.end(),
                                        local_token_indices.begin(),
                                        local_token_indices.end());
      global_batch_vocab_indices.insert(global_batch_vocab_indices.end(),
                                        local_vocab_indices.begin(),
                                        local_vocab_indices.end());
    }
  };

  if (use_gen_threadpool_) {
    const size_t batch_size = std::max(
        1UL, (sequence_num + GEN_MASK_THREAD_NUM - 1) / GEN_MASK_THREAD_NUM);
    const size_t num_batches = (sequence_num + batch_size - 1) / batch_size;

    std::vector<std::future<void>> futures;
    std::vector<std::shared_ptr<std::promise<void>>> promises;

    promises.reserve(num_batches);
    futures.reserve(num_batches);

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      auto promise = std::make_shared<std::promise<void>>();
      futures.push_back(promise->get_future());
      promises.push_back(promise);

      size_t start_idx = batch_idx * batch_size;
      size_t end_idx = std::min(start_idx + batch_size, sequence_num);

      gen_threadpool_->schedule(
          [update_mask, start_idx, end_idx, promise]() mutable {
            update_mask(start_idx, end_idx);
            promise->set_value();
          });
    }

    for (auto& future : futures) {
      future.get();
    }
  } else {
    update_mask(0, sequence_num);
  }

  if (!global_batch_token_indices.empty()) {
    auto token_indices =
        torch::tensor(global_batch_token_indices, torch::kInt64);
    auto vocab_indices =
        torch::tensor(global_batch_vocab_indices, torch::kInt64);
    token_indices = safe_to(token_indices, device_, true);
    vocab_indices = safe_to(vocab_indices, device_, true);
    mask.index_put_({token_indices, vocab_indices}, 0.0f);
  }

  return mask;
}
}  // namespace xllm
