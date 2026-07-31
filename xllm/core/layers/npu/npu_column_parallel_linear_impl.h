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
#include "framework/state_dict/state_dict.h"
#include "loader/column_parallel_linear_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/operations/fusion/linear/linear_parallel.h"

namespace xllm {
namespace layer {
// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class NpuColumnParallelLinearImpl : public BaseLayer {
 public:
  NpuColumnParallelLinearImpl(const ModelContext& context);

  ~NpuColumnParallelLinearImpl() override = default;

  virtual void merge_loaded_weights() override;

  void fuse_eagle3_quarot_input_rotation(torch::Tensor global_rotation);

  virtual int64_t init_layer() override;

  virtual torch::Tensor forward(const torch::Tensor& input, int nodeId);

 protected:
  void build_node_variant_pack(atb_speed::Model::Node& node,
                               const torch::Tensor& input);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::common::LinearParallelParam& linearParam);

  void param_from_args(atb_speed::common::LinearParallelParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args);

  atb_speed::Model::Node linear_node_;
  std::string model_name_;

  std::vector<at::Tensor> at_out_tensors_;
  atb::Tensor internal_input;
  torch::Tensor tensor_placeholder_;
  atb::Tensor placeholder_;

  atb_speed::common::LinearParallelParam linear_param_;
};
TORCH_MODULE(NpuColumnParallelLinear);

}  // namespace layer
}  // namespace xllm
