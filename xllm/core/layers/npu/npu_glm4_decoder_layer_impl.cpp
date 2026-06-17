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

#include "npu_glm4_decoder_layer_impl.h"

#include <glog/logging.h>
#include <mstx/ms_tools_ext.h>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "loader/glm4_decoder_loader.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

const uint64_t WEIGHT_COUNT_PER_LAYER = 52;

void NpuGlm4DecoderLayerImpl::param_from_args(
    atb_speed::chatglm::ChatglmLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool isPrefill) {
  param.isFA = false;
  param.enableSwiGLU = true;

  param.enableLcoc = false;
  param.rmsnormQKNorm = false;
  param.isPrefill = isPrefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSplitFuse =
      ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() &&
      isPrefill;
  param.loraEnableGMM = false;

  param.linearTransposeType = {1, -1, -1, 1, 1, -1, 1};  // TODO
  param.quantGroupSize = 0;
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / parallel_args.world_size();
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank =
      static_cast<int>(optionalValue.value()) / parallel_args.world_size();
  param.backend =
      ::xllm::ParallelConfig::get_instance().communication_backend();
  param.tensorParallelInfo = {
      parallel_args.rank(),
      parallel_args.world_size(),
      ::xllm::ParallelConfig::get_instance().communication_backend()};
  param.linearHasBias = {true, false, false, false};
  param.useQKNorm = false;
  param.enableAclGraphPagedAttention =
      ::xllm::ExecutionConfig::get_instance().enable_graph() && !isPrefill;
  param.numHiddenLayers = args.n_layers();
  param.usePostSelfAttnLayerNorm = true;
  param.usePostMlpLayerNorm = true;
  initialize_quantization_parameters(param);
}

void NpuGlm4DecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::chatglm::ChatglmLayerParam& param) {
  param.linearDescs = {static_cast<int>(LinearTypeV2::INVALID),
                       static_cast<int>(LinearTypeV2::INVALID),
                       static_cast<int>(LinearTypeV2::INVALID),
                       static_cast<int>(LinearTypeV2::INVALID),
                       static_cast<int>(LinearTypeV2::INVALID),
                       static_cast<int>(LinearTypeV2::INVALID),
                       static_cast<int>(LinearTypeV2::INVALID)};
  param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                         static_cast<int>(PackType::ALL_FP)};
  param.linearQuantType = {static_cast<int>(LinearType::FP),
                           static_cast<int>(LinearType::INVALID),
                           static_cast<int>(LinearType::INVALID),
                           static_cast<int>(LinearType::FP),
                           static_cast<int>(LinearType::FP),
                           static_cast<int>(LinearType::INVALID),
                           static_cast<int>(LinearType::FP)};
}

NpuGlm4DecoderLayerImpl::NpuGlm4DecoderLayerImpl(const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_graph_param_, model_args, parallel_args, false);
  decode_eager_param_ = decode_graph_param_;
  decode_eager_param_.enableAclGraphPagedAttention = false;
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  placeholder_ = atb_speed::Utils::AtTensor2Tensor(
      torch::zeros({1}).to(device_).to(dtype_));
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);

  loader_ = std::make_unique<Glm4DecoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

int64_t NpuGlm4DecoderLayerImpl::init_layer() {
  init_attn_mask();
  name_ = "glm4_decoder_layer";
  model_name_ = "glm4";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_graph_node_, decode_graph_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_eager_node_, decode_eager_param_));

  return atb::NO_ERROR;
}

int64_t NpuGlm4DecoderLayerImpl::init_attn_mask() {
  torch::Dtype dtype =
      prefill_param_.isBF16 ? torch::kBFloat16 : torch::kFloat16;
  decode_attn_mask_ = torch::zeros({1}).to(device_).to(dtype);

  return atb::NO_ERROR;
}

int64_t NpuGlm4DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::chatglm::ChatglmLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::chatglm::ChatglmDecoderLayer decoder_layer(param);
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

torch::Tensor NpuGlm4DecoderLayerImpl::forward(torch::Tensor& x,
                                               torch::Tensor& cos_pos,
                                               torch::Tensor& sin_pos,
                                               torch::Tensor& attn_mask,
                                               KVCache& kv_cache,
                                               ModelInputParams& input_params,
                                               aclrtEvent* event,
                                               std::atomic<bool>* event_flag,
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
                            true,
                            false);
    // mstxRangeEnd(id);
    st = execute_node(prefill_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute prefill layer fail, error code: " << st;
  } else {
    const bool use_graph_decode_input =
        ::xllm::ExecutionConfig::get_instance().enable_graph() &&
        input_params.graph.tiling_data.defined();
    auto& decode_node =
        use_graph_decode_input ? decode_graph_node_ : decode_eager_node_;
    build_node_variant_pack(decode_node,
                            x,
                            cos_pos,
                            sin_pos,
                            decode_attn_mask_,
                            kv_cache,
                            input_params,
                            false,
                            use_graph_decode_input);
    st = execute_node(decode_node, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return at_placeholder_;
}

void NpuGlm4DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    at::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    bool use_graph_decode_input) {
  internal_tensors_ = atb_speed::Utils::AtTensor2Tensor(x);
  // std::cout<<"node.variantPack.inTensors.size:"<<node.variantPack.inTensors.size()<<std::endl;
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
  size_t input_idx = WEIGHT_COUNT_PER_LAYER + 11;
  if (is_prefill &&
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache())) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_seq_lens);
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        input_params.attention.host.q_seq_lens.data();
  }

  if (!is_prefill && use_graph_decode_input &&
      input_params.graph.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(input_params.graph.tiling_data);
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
