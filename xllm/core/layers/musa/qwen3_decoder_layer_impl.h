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

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention_metadata.h"
#include "layers/musa/layer_base.h"

namespace xllm {
namespace layer {
class Qwen3DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit Qwen3DecoderLayerImpl(const ModelContext& context);

  ~Qwen3DecoderLayerImpl() {};

  void load_state_dict(const StateDict& state_dict);

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  std::vector<std::shared_ptr<MUSALayerBaseImpl>> layers_;
  // ParallelArgs parallel_args_;
};
TORCH_MODULE(Qwen3DecoderLayer);

}  // namespace layer
}  // namespace xllm