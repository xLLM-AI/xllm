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

#include "npu_mistral3_vision_encoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <map>

#include "common/global_flags.h"
#include "core/framework/config/load_config.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

// The ATB graph expects 6 weight tensors:
//   in_input_norm_weight, in_qkv_weight, in_attn_proj_weight,
//   in_post_norm_weight, in_mlp_gate_weight (gate+up combined),
//   in_mlp_down_weight
// We use a larger count to accommodate the loader's separate gate/up slots.
// The ATB graph only references the first kAtbWeightCount tensors.
const uint64_t WEIGHT_COUNT_PER_LAYER =
    36;  // match Mistral3VisionTensorId enum size

void NpuMistral3VisionEncoderLayerImpl::param_from_args(
    atb_speed::mistral::MistralVisionEncoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.isBF16 = args.dtype() == "bfloat16";
  param.rmsNormEps = args.rms_norm_eps();
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank =
      args.mm_num_attention_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead =
      args.mm_hidden_size() / args.mm_num_attention_heads();
  // Pixtral ViT: kv heads = q heads (full attention, no GQA)
  param.numKeyValueHeadsPerRank = param.numAttentionHeadsPerRank;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
}

NpuMistral3VisionEncoderLayerImpl::NpuMistral3VisionEncoderLayerImpl(
    const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  param_from_args(encode_param_, model_args, parallel_args);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  loader_ = std::make_unique<Mistral3VisionEncoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
}

void NpuMistral3VisionEncoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  Device::empty_cache(device_.index());

  // Map loader weights to ATB graph's expected weight layout.
  // The ATB graph uses 6 weight slots:
  //   [0] input_norm_weight
  //   [1] qkv_weight
  //   [2] attn_out_proj_weight
  //   [3] post_norm_weight
  //   [4] mlp_gate_weight (gate+up combined after merge)
  //   [5] mlp_down_weight
  atb_weight_tensors_[0] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[kInputNormWeight]);
  atb_weight_tensors_[1] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[kQWeight]);
  atb_weight_tensors_[2] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[kAttentionOutWeight]);
  atb_weight_tensors_[3] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[kPostNormWeight]);
  atb_weight_tensors_[4] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[kMlpGateWeight]);
  atb_weight_tensors_[5] =
      atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[kMlpDownWeight]);

  init_layer();
}

int64_t NpuMistral3VisionEncoderLayerImpl::init_layer() {
  name_ = "mistral3_vision_encoder_layer";
  model_name_ = "mistral3";
  CHECK_OPERATION_STATUS_RETURN(init_node(encode_node_, encode_param_));
  return atb::NO_ERROR;
}

int64_t NpuMistral3VisionEncoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::mistral::MistralVisionEncoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::mistral::EncoderLayer(param, &operation);
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

  // Wire the first 6 slots to the ATB weight tensors
  const uint64_t kAtbWeightCount = 6;
  for (size_t weightTensorId = 0; weightTensorId < kAtbWeightCount;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);
  return atb::NO_ERROR;
}

torch::Tensor NpuMistral3VisionEncoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params,
    int node_id) {
  atb::Status st;

  build_node_variant_pack(encode_node_,
                          x,
                          cos_pos,
                          sin_pos,
                          cu_seqlen,
                          cu_seqlen_vec,
                          input_params);
  st = execute_node(encode_node_, node_id);
  LOG_IF(FATAL, st != 0) << model_name_
                         << " execute vision encode layer fail, error code: "
                         << st;
  return x;
}

void NpuMistral3VisionEncoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlen,
    std::vector<int>& cu_seqlen_vec,
    ModelInputParams& input_params) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);

  // ATB graph weight count = 6, runtime inputs start at index 6
  const uint64_t kAtbWeightCount = 6;
  node.variantPack.inTensors.at(kAtbWeightCount) = internal_tensors_;
  node.variantPack.inTensors.at(kAtbWeightCount + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(kAtbWeightCount + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(kAtbWeightCount + 3) =
      atb_speed::Utils::AtTensor2Tensor(cu_seqlen);
  node.variantPack.inTensors.at(kAtbWeightCount + 3).hostData =
      cu_seqlen_vec.data();

  for (size_t i = 0; i < kAtbWeightCount; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm
