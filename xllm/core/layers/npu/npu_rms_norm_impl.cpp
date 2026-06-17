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

#include "npu_rms_norm_impl.h"

#include <glog/logging.h>

#include "core/framework/config/load_config.h"

namespace xllm {
namespace layer {

void NpuRMSNormImpl::param_from_args(atb::infer::RmsNormParam& param,
                                     const ModelArgs& args) {
  param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  param.normParam.epsilon = args.rms_norm_eps();
}

NpuRMSNormImpl::NpuRMSNormImpl(const ModelContext& context)
    : BaseLayer(context) {
  param_from_args(norm_param_, context.get_model_args());

  atb_weight_tensors_.resize(1);

  loader_ = std::make_unique<RMSNormLoader>(
      1,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

int64_t NpuRMSNormImpl::init_layer() {
  name_ = "rms_norm_layer";
  model_name_ = "llm";
  CHECK_OPERATION_STATUS_RETURN(init_node(norm_node_, norm_param_));

  return atb::NO_ERROR;
}

int64_t NpuRMSNormImpl::init_node(atb_speed::Model::Node& node,
                                  atb::infer::RmsNormParam& param) {
  atb::Operation* operation = nullptr;
  atb::Status atbStatus = atb::CreateOperation(param, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);

  node.inTensors.at(1) = &atb_weight_tensors_[0];
  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor NpuRMSNormImpl::forward(torch::Tensor& x, int nodeId) {
  atb::Status st;

  build_node_variant_pack(norm_node_, x);
  st = execute_node(norm_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;

  return x;
}

void NpuRMSNormImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                             torch::Tensor& x) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  node.variantPack.inTensors.at(0) = internal_tensors_;
  node.variantPack.inTensors.at(1) = *node.inTensors.at(1);
  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm
