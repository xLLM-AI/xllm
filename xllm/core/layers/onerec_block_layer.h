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

#include <atomic>
#include <cstdint>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#else
using aclrtEvent = void*;
#endif

namespace xllm {
namespace layer {

class OneRecBlockLayer : public torch::nn::Module {
 public:
  virtual ~OneRecBlockLayer() = default;

  virtual torch::Tensor forward(torch::Tensor& hidden_states,
                                torch::Tensor& attn_mask,
                                KVCache& kv_cache,
                                ModelInputParams& input_params,
                                torch::Tensor* encoder_output = nullptr,
                                int32_t node_id = 0,
                                aclrtEvent* event = nullptr,
                                std::atomic<bool>* event_flag = nullptr) = 0;
};

}  // namespace layer
}  // namespace xllm
