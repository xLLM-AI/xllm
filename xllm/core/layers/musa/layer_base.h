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

#include "MTTOplib/Attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"

namespace xllm::layer {

struct ForwardParams {
  torch::Tensor& positions;
  AttentionMetadata const& attn_meta;
  KVCache& kv_cache;
  ModelInputParams const& input_params;
};

class MUSALayerBaseImpl : public torch::nn::Module {
 public:
  MUSALayerBaseImpl() = default;
  explicit MUSALayerBaseImpl(torch::TensorOptions const& options)
      : options_(options) {};
  virtual ~MUSALayerBaseImpl() = default;

  bool load_weight_common(StateDict const& state_dict,
                          std::vector<int64_t> const& shape,
                          int32_t idx) {
    if (state_dict.size() == 0) {
      return false;
    }
    DEFINE_WEIGHT(weight);
    weight_ = torch::empty(shape, options_);
    LOAD_WEIGHT(weight);
    if (weight_is_loaded_) {
      weights_[idx] = weight_.clone();
    }
    return weight_is_loaded_;
  }

  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual torch::Tensor forward(torch::Tensor& input,
                                ForwardParams& fwd_params) = 0;

 protected:
  std::vector<torch::Tensor> weights_;
  torch::TensorOptions options_;
};
TORCH_MODULE(MUSALayerBase);

}  // namespace xllm::layer