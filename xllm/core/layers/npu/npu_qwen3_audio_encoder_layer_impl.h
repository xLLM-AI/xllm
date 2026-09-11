/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>
#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/model.h"
#include "core/framework/model/model_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/npu/loader/qwen3_audio_encoder_loader.h"
#include "core/layers/npu/npu_base_layer.h"
#include "xllm_atb_layers/models/qwen3_audio/qwen3_audio.h"

namespace xllm::layer {

class NpuQwen3AudioEncoderLayerImpl final : public BaseLayer {
 public:
  explicit NpuQwen3AudioEncoderLayerImpl(const ModelContext& context);

  ~NpuQwen3AudioEncoderLayerImpl() override = default;

  void merge_loaded_weights() override;

  int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cu_seqlen,
                        std::vector<int32_t>& cu_seqlen_vec,
                        int32_t node_id = 0);

 private:
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cu_seqlen,
                               std::vector<int32_t>& cu_seqlen_vec);

  void param_from_args(atb_speed::qwen::AudioEncoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::AudioEncoderLayerParam& param);

  atb_speed::Model::Node encode_node_;
  std::string model_name_;

  atb_speed::qwen::AudioEncoderLayerParam encode_param_;

  atb::Tensor internal_tensors_;
};
TORCH_MODULE(NpuQwen3AudioEncoderLayer);

}  // namespace xllm::layer
