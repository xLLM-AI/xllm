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

#include "npu_llama_decoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include <map>

#include "common/global_flags.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/layers/common/attention_mask.h"
#include "loader/llama_decoder_loader.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 50;

NpuLlamaDecoderLayerImpl::NpuLlamaDecoderLayerImpl(const ModelContext& context)
    : BaseLayer(context) {
  param_from_args(prefill_param_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  true);
  param_from_args(decode_param_,
                  context.get_model_args(),
                  context.get_parallel_args(),
                  false);

  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};

  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));

  loader_ = std::make_unique<LlamaDecoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
}

// fix param
void NpuLlamaDecoderLayerImpl::param_from_args(
    atb_speed::llama::LlamaLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool isPrefill) {
  param.isFA = false;
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = isPrefill;
  param.enableSpeculate = false;
  param.enableSplitFuse =
      ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
      isPrefill;
  param.enableLora = false;
  param.loraEnableGMM = false;
  param.packQuantType = {1, 1};
  param.linearQuantType = {0, -1, -1, 0, 0, -1, 0};
  param.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};
  param.enableKvQuant = false;
  param.quantGroupSize = 0;
  param.normEps = args.rms_norm_eps();
  param.normEps = 0.00001;
  // param.normType = 0;
  param.enableFA3 = false;
  param.worldSize = parallel_args.world_size();
  param.numAttentionHeadsPerRank = args.n_heads() / param.worldSize;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend = "lccl";
  param.tensorParallelInfo = {
      parallel_args.rank(), parallel_args.world_size(), "lccl"};
  // param.enableLogN = false;
}

void NpuLlamaDecoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();

  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  Device::empty_cache(device_.index());
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }

  init_layer();
}

int64_t NpuLlamaDecoderLayerImpl::init_layer() {
  init_attn_mask();
  name_ = "llama_decoder_layer";
  model_name_ = "llama";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t NpuLlamaDecoderLayerImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t NpuLlamaDecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::llama::LlamaLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::llama::LlamaDecoderLayer decoder_layer(param);
  decoder_layer.BuildGraph(&operation);
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

torch::Tensor NpuLlamaDecoderLayerImpl::forward(torch::Tensor& x,
                                                torch::Tensor& cos_pos,
                                                torch::Tensor& sin_pos,
                                                torch::Tensor& attn_mask,
                                                KVCache& kv_cache,
                                                ModelInputParams& input_params,
                                                int node_id) {
  atb::Status st;

  if (!input_params.meta.batch_forward_type.is_decode()) {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            true);
    // mstxRangeEnd(id);
    st = execute_node(prefill_node_, node_id);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute prefill layer fail, error code: " << st;
  } else {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            decode_attn_mask_,
                            kv_cache,
                            input_params,
                            false);
    st = execute_node(decode_node_, node_id + 1000);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void NpuLlamaDecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensors_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.kv_seq_lens);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6).hostData =
      input_params.attention.host.kv_seq_lens.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7).hostData =
      placeholder_vec_.data();
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) = placeholder_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.block_tables);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.new_cache_slots);
  if (is_prefill &&
      ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        input_params.attention.host.q_seq_lens.data();
  }
  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << "inTensor " << i << "is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensors_;
}

}  // namespace layer
}  // namespace xllm
