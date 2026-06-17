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
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/base/model.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/model_factory.h"
#include "core/framework/model/model_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "loader/qwen2_vision_encoder_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_atb_layers/models/qwen3_vl/qwen3_vl_encoder.h"

namespace xllm {
namespace layer {

class NpuQwen2VisionEncoderLayerImpl : public BaseLayer {
 public:
  explicit NpuQwen2VisionEncoderLayerImpl(const ModelContext& context);

  ~NpuQwen2VisionEncoderLayerImpl() override = default;

  // void load_state_dict(const StateDict& state_dict) override;

  // void verify_loaded_weights() const override;

  void merge_loaded_weights() override;

  int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& cu_seqlen,
                        std::vector<int>& cu_seqlen_vec,
                        int node_id = 0,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr);

 private:
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& cu_seqlen,
                               std::vector<int>& cu_seqlen_vec,
                               bool is_prefill);

  void get_weights_col_packed_qkv();

  void param_from_args(atb_speed::qwen::VisionEncoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::VisionEncoderLayerParam& param);

  void pad_qkv_weights();

  void pad_mlp_weights();

  torch::Tensor pad_tensor(const torch::Tensor& tensor,
                           int64_t target_shape,
                           int64_t dim = 0) {
    int64_t pad_size = target_shape - tensor.size(dim);
    if (tensor.dim() == 1) {
      return torch::nn::functional::pad(
          tensor, torch::nn::functional::PadFuncOptions({0, pad_size}));
    } else if (tensor.dim() == 2) {
      if (1 == dim)
        return torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({0, pad_size, 0, 0}));
      else
        return torch::nn::functional::pad(
            tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
    }
    return tensor;
  }

  atb_speed::Model::Node encode_node_;
  std::string model_name_;

  atb_speed::qwen::VisionEncoderLayerParam encode_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;
  at::Tensor cu_seqlen_;
  at::Tensor at_placeholder_;
  std::vector<torch::Tensor> qkv_weight;
  std::vector<torch::Tensor> qkv_bias;
  int device_id_;
};
TORCH_MODULE(NpuQwen2VisionEncoderLayer);

}  // namespace layer
}  // namespace xllm
