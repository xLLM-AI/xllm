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

#include "core/layers/npu/npu_qwen3_audio_encoder_layer_impl.h"

#include <glog/logging.h>

#include <cstdint>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::layer {

namespace {
constexpr int32_t kWeightCountPerLayer = 18;
constexpr int32_t kActivationInputCount = 2;
constexpr int32_t kInputCount = kWeightCountPerLayer + kActivationInputCount;
}  // namespace

void NpuQwen3AudioEncoderLayerImpl::param_from_args(
    atb_speed::qwen::AudioEncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.rmsNormEps = args.mm_audio_layer_norm_eps();
  param.worldSize = parallel_args.world_size();
  const bool padding_heads =
      args.mm_audio_num_attention_heads() % param.worldSize > 0;
  if (padding_heads) {
    LOG(FATAL) << "You are running qwen3 audio encoder with " << param.worldSize
               << " cards, but got attention heads num "
               << args.mm_audio_num_attention_heads()
               << ". The attention head count must be divisible by the "
                  "tensor parallel world size.";
  }
  param.numAttentionHeadsPerRank =
      args.mm_audio_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_audio_hidden_size() / args.mm_audio_num_attention_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<uint32_t>(args.mm_audio_num_attention_heads()) /
      param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.enableLogN = false;
}

NpuQwen3AudioEncoderLayerImpl::NpuQwen3AudioEncoderLayerImpl(
    const ModelContext& context)
    : BaseLayer(context) {
  const ModelArgs& model_args = context.get_model_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  atb_weight_tensors_.resize(kWeightCountPerLayer);
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  loader_ =
      std::make_unique<Qwen3AudioEncoderLoader>(kWeightCountPerLayer, context);
}

void NpuQwen3AudioEncoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  std::vector<torch::Tensor>& at_weight_tensors =
      loader_->get_at_weight_tensors();
  c10_npu::NPUCachingAllocator::emptyCache();
  for (int32_t index = 0; index < kWeightCountPerLayer; ++index) {
    atb_weight_tensors_[index] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[index]);
  }

  init_layer();
}

int64_t NpuQwen3AudioEncoderLayerImpl::init_layer() {
  name_ = "qwen3_audio_encoder_layer";
  model_name_ = "qwen3_audio";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t NpuQwen3AudioEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::AudioEncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::Qwen3_Audio_EncoderLayer(param, &operation);
  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < kInputCount) {
    LOG(ERROR) << "Qwen3 audio encoder operation expects at least "
               << kInputCount << " inputs, got "
               << node.operation->GetInputNum();
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(1);
  for (int32_t weight_tensor_id = 0; weight_tensor_id < kWeightCountPerLayer;
       ++weight_tensor_id) {
    node.inTensors.at(weight_tensor_id) =
        &atb_weight_tensors_[weight_tensor_id];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  return atb::NO_ERROR;
}

torch::Tensor NpuQwen3AudioEncoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cu_seqlen,
    std::vector<int32_t>& cu_seqlen_vec,
    int32_t node_id) {
  build_node_variant_pack(encode_node_, x, cu_seqlen, cu_seqlen_vec);
  const atb::Status status = execute_node(encode_node_, node_id);
  LOG_IF(FATAL, status != 0)
      << model_name_ << " execute encode layer failed, error code: " << status;
  return x;
}

void NpuQwen3AudioEncoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cu_seqlen,
    std::vector<int32_t>& cu_seqlen_vec) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  node.variantPack.inTensors.at(kWeightCountPerLayer) = internal_tensors_;
  node.variantPack.inTensors.at(kWeightCountPerLayer + 1) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(kWeightCountPerLayer + 1).hostData =
      cu_seqlen_vec.data();

  for (int32_t index = 0; index < kWeightCountPerLayer; ++index) {
    CHECK_THROW(node.inTensors.at(index) == nullptr,
                model_name_ << " inTensor " << index << " is NULL");
    node.variantPack.inTensors.at(index) = *node.inTensors.at(index);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace xllm::layer
