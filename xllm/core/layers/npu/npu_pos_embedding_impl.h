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

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/operations/fusion/embedding/positional_embedding.h"

namespace xllm {
namespace layer {

class NpuRotaryEmbeddingImpl : public BaseLayer {
 public:
  explicit NpuRotaryEmbeddingImpl(const ModelContext& context);

  ~NpuRotaryEmbeddingImpl() override = default;

  torch::Tensor forward(const torch::Tensor& cos_sin_pos,
                        const torch::Tensor& position,
                        int nodeId);

 private:
  int64_t init_layer() override;

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& cos_sin_pos,
                               const torch::Tensor& position);

  int64_t init_node(atb_speed::Model::Node& node);

  atb_speed::Model::Node embedding_node_;
  std::string modelName_;
  std::vector<at::Tensor> atOutTensors_;
  torch::Tensor inv_freq;
  atb::Tensor internal_cos_sin_pos;
  atb::Tensor internal_position;
};
TORCH_MODULE(NpuRotaryEmbedding);

class NpuPosEmbedding : public torch::nn::ModuleHolder<NpuRotaryEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<NpuRotaryEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuRotaryEmbeddingImpl;

  NpuPosEmbedding(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuRotaryEmbeddingImpl>(context)) {}
};

}  // namespace layer
}  // namespace xllm
