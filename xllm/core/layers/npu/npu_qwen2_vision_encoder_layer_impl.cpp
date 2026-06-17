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

#include "npu_qwen2_vision_encoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <iostream>
#include <map>

#include "common/global_flags.h"
#include "core/framework/config/load_config.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"
#include "xllm_atb_layers/models/qwen3_vl/qwen3_vl_encoder.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 18;

void NpuQwen2VisionEncoderLayerImpl::param_from_args(
    atb_speed::qwen::VisionEncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.rmsNormEps = args.rms_norm_eps();
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank =
      args.mm_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_hidden_size() / args.mm_num_attention_heads();
  std::optional<long int> optionalValue = args.mm_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.enableLogN = false;
  param.MLPActivationType = atb::infer::ActivationType::ACTIVATION_SWISH;
}

NpuQwen2VisionEncoderLayerImpl::NpuQwen2VisionEncoderLayerImpl(
    const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  // at_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  loader_ = std::make_unique<Qwen2VisionEncoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

void NpuQwen2VisionEncoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  Device::empty_cache(device_.index());
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }

  init_layer();
}

int64_t NpuQwen2VisionEncoderLayerImpl::init_layer() {
  name_ = "qwen2_encoder_layer";
  model_name_ = "qwen2_vl";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t NpuQwen2VisionEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::VisionEncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::Qwen3VL_EncoderLayer(param, &operation);
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
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  return atb::NO_ERROR;
}

torch::Tensor NpuQwen2VisionEncoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    int node_id,
    aclrtEvent* event,
    std::atomic<bool>* event_flag) {
  atb::Status st;

  build_node_variant_pack(
      encode_node_, x, cos_pos, sin_pos, cu_seqlen, cu_seqlen_vec, true);
  // mstxRangeEnd(id);
  st = execute_node(encode_node_, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "excute encode layer fail, error code: " << st;
  return x;
}

void NpuQwen2VisionEncoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3).hostData =
      cu_seqlen_vec.data();

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm
