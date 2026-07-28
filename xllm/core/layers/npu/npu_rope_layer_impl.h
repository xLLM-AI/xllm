/* Copyright 2026 The xLLM Authors.

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

#include <cstdint>
#include <tuple>
#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/model.h"
#include "npu_base_layer.h"

namespace xllm {
namespace layer {

class NpuRopeLayerImpl : public BaseLayer {
 public:
  explicit NpuRopeLayerImpl(const ModelContext& context);

  ~NpuRopeLayerImpl() override = default;

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& cos,
      const torch::Tensor& sin,
      const torch::Tensor& seq_len,
      std::vector<int32_t>& seq_len_host,
      int32_t node_id = 0);

 private:
  int64_t init_layer() override;

  int64_t init_node(atb_speed::Model::Node& node);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& query,
                               const torch::Tensor& key,
                               const torch::Tensor& cos,
                               const torch::Tensor& sin,
                               const torch::Tensor& seq_len,
                               std::vector<int32_t>& seq_len_host);

  atb_speed::Model::Node rope_node_;
  std::vector<at::Tensor> output_tensors_;
  atb::Tensor internal_query_;
  atb::Tensor internal_key_;
  atb::Tensor internal_cos_;
  atb::Tensor internal_sin_;
  atb::Tensor internal_seq_len_;
  int64_t head_dim_ = 0;
};

TORCH_MODULE(NpuRopeLayer);

}  // namespace layer
}  // namespace xllm
