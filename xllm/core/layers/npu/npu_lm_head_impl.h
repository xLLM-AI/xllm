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
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "loader/lm_head_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/operations/fusion/lmhead/lmhead.h"

namespace xllm {
namespace layer {

class NpuLmHeadImpl : public BaseLayer {
 public:
  explicit NpuLmHeadImpl(const ModelContext& context);

  ~NpuLmHeadImpl() override = default;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& seleted_idxes,
                        int nodeId);

  torch::Tensor forward_with_hidden(const torch::Tensor& hidden_states,
                                    const torch::Tensor& seleted_idxes,
                                    torch::Tensor& out_hidden,
                                    int nodeId);

 private:
  void param_from_args(atb_speed::common::LmHeadParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  int64_t init_layer() override;

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::common::LmHeadParam& param);

  // void build_node_variant_pack(atb_speed::Model::Node& node, torch::Tensor&
  // hidden_states,torch::Tensor&
  // seleted_idxes,std::vector<std::shared_ptr<at::Tensor>>& tensor_storage);
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes);

  atb_speed::Model::Node lm_head_node_prefill_;
  atb_speed::Model::Node lm_head_node_decode_;

  std::string model_name_;
  torch::Tensor torch_placeholder_;
  atb::Tensor placeholder_;
  std::vector<at::Tensor> atOutTensors_;

  atb_speed::common::LmHeadParam lm_head_param_prefill_;
  atb_speed::common::LmHeadParam lm_head_param_decode_;

  std::vector<std::shared_ptr<at::Tensor>> prefill_tensor_storage_;
  std::vector<std::shared_ptr<at::Tensor>> decode_tensor_storage_;
  atb::Tensor hidden_states_atb_;
  atb::Tensor seleted_idxes_atb_;

  int64_t vocab_size_ = -1;
  int64_t padded_vocab_size_ = -1;
};
TORCH_MODULE(NpuLmHead);

}  // namespace layer
}  // namespace xllm
