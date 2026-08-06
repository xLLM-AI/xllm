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

#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "loader/siglip_encoder_loader.h"
#include "npu_base_layer.h"
#include "xllm_atb_layers/pytorch/atb_torch/core/include/base_operation.h"
#include "xllm_atb_layers/pytorch/atb_torch/core/include/graph_operation.h"

namespace xllm {
namespace layer {

class NpuSiglipEncoderLayerUpImpl : public BaseLayer {
 public:
  NpuSiglipEncoderLayerUpImpl(const ModelContext& context,
                              const std::string& prefix = "");

  ~NpuSiglipEncoderLayerUpImpl() override = default;

  void load_state_dict(const StateDict& state_dict) override;

  torch::Tensor forward(const torch::Tensor& x);

 private:
  void build_graph(const std::string& prefix = "");

  atb_torch::GraphOperation graph_;
  std::vector<std::shared_ptr<atb_torch::BaseOperation>> ops_;
  std::vector<torch::Tensor> weights_;

  ModelArgs model_args_;
  torch::TensorOptions options_;

  std::string prefix_;
};
TORCH_MODULE(NpuSiglipEncoderLayerUp);

class NpuSiglipEncoderLayerDownImpl : public BaseLayer {
 public:
  NpuSiglipEncoderLayerDownImpl(const ModelContext& context,
                                const std::string& prefix = "");

  ~NpuSiglipEncoderLayerDownImpl() override = default;

  void load_state_dict(const StateDict& state_dict) override;

  torch::Tensor forward(torch::Tensor& x, torch::Tensor& y);

 private:
  void build_graph(const std::string& prefix = "");

  std::string prefix_;

  atb_torch::GraphOperation graph_;
  std::vector<std::shared_ptr<atb_torch::BaseOperation>> ops_;
  std::vector<torch::Tensor> weights_;

  ModelArgs model_args_;
  torch::TensorOptions options_;
};
TORCH_MODULE(NpuSiglipEncoderLayerDown);

class NpuSiglipEncoderLayerImpl : public BaseLayer {
 public:
  NpuSiglipEncoderLayerImpl(const ModelContext& context,
                            const std::string& prefix = "");

  ~NpuSiglipEncoderLayerImpl() override = default;

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& weight_str) const {};

  torch::Tensor forward(const torch::Tensor& x);

 private:
  std::string prefix_;

  ModelArgs model_args_;
  torch::TensorOptions options_;

  NpuSiglipEncoderLayerUp up_{nullptr};
  NpuSiglipEncoderLayerDown down_{nullptr};
};
TORCH_MODULE(NpuSiglipEncoderLayer);

}  // namespace layer
}  // namespace xllm
