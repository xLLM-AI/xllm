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
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "loader/glm4_decoder_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/models/glm/layer/decoder_layer.h"

namespace xllm {
namespace layer {

class NpuGlm4DecoderLayerImpl : public BaseLayer {
 public:
  explicit NpuGlm4DecoderLayerImpl(const ModelContext& context);

  ~NpuGlm4DecoderLayerImpl() override = default;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

 private:
  void param_from_args(atb_speed::chatglm::ChatglmLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               ModelInputParams& input_params,
                               bool is_prefill,
                               bool use_graph_decode_input);

  void initialize_quantization_parameters(
      atb_speed::chatglm::ChatglmLayerParam& param);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::chatglm::ChatglmLayerParam& param);

  int64_t init_attn_mask();

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_graph_node_;
  atb_speed::Model::Node decode_eager_node_;
  std::string model_name_;
  atb_speed::chatglm::ChatglmLayerParam prefill_param_;
  atb_speed::chatglm::ChatglmLayerParam decode_graph_param_;
  atb_speed::chatglm::ChatglmLayerParam decode_eager_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;

  at::Tensor decode_attn_mask_;

  at::Tensor at_placeholder_;
};
TORCH_MODULE(NpuGlm4DecoderLayer);

}  // namespace layer
}  // namespace xllm
