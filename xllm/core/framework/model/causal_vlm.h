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

#include <c10/core/Device.h>
#include <torch/types.h>

#include "causal_lm.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/quantization/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "model_args.h"
#include "model_params_view.h"

namespace xllm {

class CausalVLM {
 public:
  virtual ~CausalVLM() = default;
  virtual MMDict encode(const ModelParamsView& parameters) = 0;
  virtual torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) = 0;
};

template <typename ModelType>
class CausalVLMImpl : public CausalLMImpl<ModelType>, public CausalVLM {
 public:
  CausalVLMImpl(ModelType model, const ModelContext& context)
      : CausalLMImpl<ModelType>(std::move(model), context) {}

  MMDict encode(const ModelParamsView& parameters) override {
    return this->model_->get_multimodal_embeddings(parameters);
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelParamsView& input_params) override {
    return this->model_->get_input_embeddings(input_ids, input_params);
  }
};

}  // namespace xllm
