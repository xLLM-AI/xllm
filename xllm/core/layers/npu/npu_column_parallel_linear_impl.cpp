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

#include "layers/npu/npu_column_parallel_linear_impl.h"

#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
namespace xllm {
namespace layer {

void NpuColumnParallelLinearImpl::param_from_args(
    atb_speed::common::LinearParallelParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.fusionLinearParam.isBF16 = args.dtype() == "bfloat16";
  param.unpadInputs = true;
  param.fusionLinearParam.transposeType = 1;

  if (parallel_args.world_size() > 1) {
    if (parallel_args.mapping_data().empty()) {
      const bool use_local_tp = (dp_size_ > 1) || (cp_size_ > 1);
      if (use_local_tp) {
        param.tensorParallelInfo.rank = dp_local_tp_rank_;
        param.tensorParallelInfo.worldSize = dp_local_tp_size_;
      } else {
        param.tensorParallelInfo.rank = parallel_args.rank();
        param.tensorParallelInfo.worldSize = parallel_args.world_size();
      }
      param.parallelType = atb_speed::common::COLUMN_PARALLEL;
      const int32_t tp_group_id =
          use_local_tp ? (parallel_args.rank() / dp_local_tp_size_) : 0;
      param.tensorParallelInfo.commDomain = std::to_string(tp_group_id);
      param.tensorParallelInfo.backend =
          ParallelConfig::get_instance().communication_backend();
    } else {
      param.parallelType = atb_speed::common::COLUMN_PARALLEL;
      atb_speed::common::ParallelInfo parallelInfo =
          parallel_args.mapping().Get(atb_speed::base::ATTN_TP);
      param.tensorParallelInfo.rank = parallelInfo.rank;
      param.tensorParallelInfo.worldSize = parallelInfo.rankIds.size();
      param.tensorParallelInfo.backend =
          ::xllm::ParallelConfig::get_instance().communication_backend();
      parallelInfo.InitCommDomain(param.tensorParallelInfo.hcommInfo,
                                  param.tensorParallelInfo.commDomain);
    }
  }
}

NpuColumnParallelLinearImpl::NpuColumnParallelLinearImpl(
    const ModelContext& context)
    : BaseLayer(context) {
  param_from_args(
      linear_param_, context.get_model_args(), context.get_parallel_args());
  atb_weight_tensors_.resize(1);
  at_out_tensors_.resize(1);

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  tensor_placeholder_ = torch::zeros({1}).to(options);
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
    loader_ = std::make_unique<ColumParallelLinearLoader>(
        1, context, LoadMode::kManual);
  } else {
    loader_ = std::make_unique<ColumParallelLinearLoader>(
        1, context, LoadMode::kEager);
  }
}

void NpuColumnParallelLinearImpl::merge_loaded_weights() {
  auto& at_weight_tensors = loader_->get_at_weight_tensors();

  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[0]);
  init_layer();
}

int64_t NpuColumnParallelLinearImpl::init_layer() {
  name_ = "atb_parallel_linear_layer";
  model_name_ = "Atb Parallel Linear";
  CHECK_OPERATION_STATUS_RETURN(init_node(linear_node_, linear_param_));

  return atb::NO_ERROR;
}

int64_t NpuColumnParallelLinearImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::common::LinearParallelParam& linearParam) {
  atb::Operation* operation = nullptr;
  atb::Status atbStatus =
      atb_speed::common::LinearParallel(linearParam, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Get unexpected input num: " << node.operation->GetInputNum();
    return -1;
  }
  if (node.operation->GetOutputNum() < 1) {
    LOG(ERROR) << "Get unexpected output num: "
               << node.operation->GetOutputNum();
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);

  node.inTensors.at(1) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  ATB_SPEED_LOG_DEBUG("AddLinear");

  return atb::NO_ERROR;
}

torch::Tensor NpuColumnParallelLinearImpl::forward(const torch::Tensor& input,
                                                   int nodeId) {
  atb::Status st;
  build_node_variant_pack(linear_node_, input);
  st = execute_node(linear_node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;

  return at_out_tensors_.at(0);
}

void NpuColumnParallelLinearImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    const torch::Tensor& input) {
  internal_input = atb_speed::Utils::AtTensor2Tensor(input);

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.operation->GetInputNum());
  inTensorDescs.resize(node.operation->GetInputNum());

  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = internal_input;
  inTensorDescs.at(0) = internal_input.desc;
  // weight
  node.variantPack.inTensors.at(1) = *node.inTensors.at(1);
  inTensorDescs.at(1) = node.inTensors.at(1)->desc;

  for (int i = 2; i < 7; i++) {
    node.variantPack.inTensors.at(i) = placeholder_;
    inTensorDescs.at(i) = placeholder_.desc;
  }

  node.operation->InferShape(inTensorDescs, outTensorDescs);
  at::Tensor output =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));
  at_out_tensors_.at(0) = output;
  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(at_out_tensors_.at(0));
}

}  // namespace layer
}  // namespace xllm
