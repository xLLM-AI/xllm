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

#pragma once
#include <torch/torch.h>
#include <torch/types.h>

#include "constrained_decoding.h"
#include "util/threadpool.h"

namespace xllm {

class RecVocabDict;

class RecConstrainedDecoding : public ConstrainedDecoding {
 public:
  RecConstrainedDecoding(RecVocabDict* vocab_dict,
                         const int32_t vocab_size,
                         torch::ScalarType dtype,
                         torch::Device device,
                         bool use_gen_threadpool_ = true);
  virtual ~RecConstrainedDecoding() = default;

  bool build_mask_cache() override;

  torch::Tensor generate_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list) override;

 private:
  torch::Tensor generate_decode_mask(
      const std::vector<std::vector<int32_t>>& generated_token_list);

 private:
  constexpr static float PRE_MASK_FACTOR = -10000.0f;
  constexpr static int GEN_MASK_THREAD_NUM = 16;

 private:
  bool build_mask_cache_;
  bool use_gen_threadpool_;
  int32_t vocab_size_;
  RecVocabDict* vocab_dict_ = nullptr;
  torch::Device device_;
  torch::ScalarType dtype_;
  torch::Tensor first_token_mask_;
  std::unique_ptr<ThreadPool> gen_threadpool_;
};

}  // namespace xllm
