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

// copy from qwen3 vl, please follow its modifications
#include "npu_glm4_vision_encoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include "common/global_flags.h"
#include "core/framework/config/load_config.h"
#include "xllm_atb_layers/models/glm4v/glm4v_encoder.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 8;

void NpuGlm4VisionEncoderLayerImpl::param_from_args(
    atb_speed::glm::VisionEncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.supportLcoc = false;
  param.supportLora = false;
  param.loraEnableGMM = false;
  param.enableLogN = false;
  param.backend = "hccl";
  param.rank = parallel_args.rank();
  param.worldSize = parallel_args.world_size();

  param.quantType = 0;
  param.quantGroupSize = 64;

  param.numAttentionHeadsPerRank =
      args.mm_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_hidden_size() / args.mm_num_attention_heads();
  std::optional<long int> optionalValue = args.mm_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;

  param.rmsNormEps = args.rms_norm_eps();
}

NpuGlm4VisionEncoderLayerImpl::NpuGlm4VisionEncoderLayerImpl(
    const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  loader_ = std::make_unique<Glm4VisionEncoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

void NpuGlm4VisionEncoderLayerImpl::merge_loaded_weights() {
  CHECK(loader_ != nullptr) << "glm4 vision encoder loader is not initialized";
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }

  Device::empty_cache(device_.index());
  init_layer();
}

int64_t NpuGlm4VisionEncoderLayerImpl::init_layer() {
  name_ = "glm4_vision_encoder_layer";
  model_name_ = "glm4v";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t NpuGlm4VisionEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::glm::VisionEncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::glm::Glm4v_EncoderLayer(param, &operation);
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

torch::Tensor NpuGlm4VisionEncoderLayerImpl::forward(
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

void NpuGlm4VisionEncoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  auto actual_weight_num = WEIGHT_COUNT_PER_LAYER - 2;
  for (size_t i = 0; i < actual_weight_num; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
    // LOG(INFO) << model_name_ << "inTensors[" << i << "]:"
    //               << atb_speed::TensorUtil::TensorToString(
    //                      node.variantPack.inTensors.at(i));
  }
  node.variantPack.inTensors.at(actual_weight_num) = internal_tensors_;
  node.variantPack.inTensors.at(actual_weight_num + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(actual_weight_num + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(actual_weight_num + 3) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(actual_weight_num + 3).hostData =
      cu_seqlen_vec.data();

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm
