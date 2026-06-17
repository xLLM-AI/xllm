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

#include "npu_glm4_moe_lite_decoder_layer.h"

#include <boost/algorithm/string.hpp>
#include <utility>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "layers/common/rotary_embedding_util.h"

namespace xllm {
namespace layer {

NpuGlm4MoeDecoderLiteImpl::NpuGlm4MoeDecoderLiteImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : BaseLayer(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  auto model_args = context.get_model_args();

  if (boost::iequals(model_args.rope_scaling_rope_type(), "deepseek_yarn")) {
    const float attn_scale =
        model_args.attn_scalar().value_or(static_cast<float>(
            model_args.qk_nope_head_dim() + model_args.qk_rope_head_dim()));
    sm_scale_ = 1.0f / std::sqrt(attn_scale);
    float mscale = layer::rotary::yarn_get_mscale(
        model_args.rope_scaling_factor(),
        model_args.rope_scaling_mscale_all_dim());
    sm_scale_ = sm_scale_ * mscale * mscale;
  } else if (boost::iequals(model_args.rope_scaling_rope_type(), "mrope")) {
    sm_scale_ = std::pow(model_args.head_dim(), -0.5);
  } else {
    const float attn_scale = model_args.attn_scalar().value_or(
        static_cast<float>(model_args.head_dim()));
    sm_scale_ = 1.0f / std::sqrt(attn_scale);
  }
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  // TODO PREFIX CACHE
  // param_from_args(
  //     prefill_param_prefixcache_, model_args, parallel_args, true, true);
  param_from_args(prefill_param_, model_args, parallel_args, true, false);
  param_from_args(decode_param_, model_args, parallel_args, false, false);
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  device_id_ = options.device().index();

  loader_ = std::make_unique<Glm4MoeDecoderLiteLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      layer_id_,
      prefill_param_.firstKDenseReplace,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);

  initialize_tensors(options);
}

void NpuGlm4MoeDecoderLiteImpl::initialize_tensors(
    const torch::TensorOptions& options) {
  // initializ placeholder

  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  int_tensor_placeholder_ = torch::ones({1}).to(torch::kInt32).to(device_);
  slot_tensor_placeholder_ = torch::full({1}, 0).to(torch::kInt32).to(device_);
  block_tables_placeholder_ =
      torch::zeros({1, 1}).to(torch::kInt32).to(device_);
  tensor_placeholder_ = torch::zeros({1}).to(options);
  loader_->resize_experts_weights(num_experts_per_partition_);
  expert_group_ = torch::arange(1024, torch::kInt32).to(device_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);
  initialize_weight_tensors(options);
}

void NpuGlm4MoeDecoderLiteImpl::param_from_args(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill,
    bool is_prefixcache) {
  initialize_basic_parameters(
      param, args, parallel_args, is_prefill, is_prefixcache);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
  param.useMLA = true;
  param.actual_headNum = args.actual_n_heads() / dp_local_tp_size_;
}

void NpuGlm4MoeDecoderLiteImpl::initialize_weight_tensors(
    const torch::TensorOptions& options) {
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    at_weight_tensors[i] = torch::zeros({1}).to(options);
  }
}

void NpuGlm4MoeDecoderLiteImpl::initialize_basic_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill,
    bool is_prefixcache) {
  param.isFA = false;
  param.enableFusedMLA =
      ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enablePrefixCache =
      is_prefill &&
      ::xllm::KVCacheConfig::get_instance().enable_prefix_cache() &&
      is_prefixcache;
  param.isNzCache = ::xllm::KVCacheConfig::get_instance().enable_prefix_cache();
  param.enableSwiGLU = true;
  param.enableLcoc = is_prefill;  // false;

  param.attnLinearTransposeType = {1, 1, 1, 1, 1, 1};
  param.mlpLinearTransposeType = {1, -1, 1, -1};

  param.enableSplitFuse =
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache()) &&
      is_prefill;

  param.enableAclGraphPagedAttention = false;

  param.moeLinearTransposeType = (layer_id_ < args.first_k_dense_replace())
                                     ? std::vector<int>{-1, -1, -1, -1}
                                     : std::vector<int>{1, 1, -1, 1};

  param.normEps = args.rms_norm_eps();
  // param.rank = parallel_args.rank();
  param.backend =
      ::xllm::ParallelConfig::get_instance().communication_backend();
  // param.rankTableFile =
  // ::xllm::EPLBConfig::get_instance().rank_tablefile();

  param.layerId = layer_id_;
  param.numHiddenLayers = args.n_layers();
  if (quantize_type_.empty()) {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }

  param.enableSpeculate = false;                    // MTP
  param.maskfree = true;                            // TODO
  param.enableSwiGLUQuantForSharedExperts = false;  // TODO

  param.useQKNorm = true;
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = std::max(
      1, static_cast<int>(optionalValue.value()) / parallel_args.world_size());

  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;

  param.linearTransposeType = {1, -1, -1, 1, -1, -1, -1};
  // param.worldSize = parallel_args.world_size();
}

void NpuGlm4MoeDecoderLiteImpl::initialize_attention_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.qLoraRank = args.q_lora_rank();
  // NOTE: The operation in this conditional is theoretically compatible with
  // DeepSeek, but we add this specific check to ensure DeepSeek behavior
  // remains unchanged
  if (args.model_type() != "kimi_k2") {
    param.headNum = args.n_heads();
  }
  param.qkNopeHeadDim = args.qk_nope_head_dim();
  param.qkRopeHeadDim = args.qk_rope_head_dim();
  param.kvLoraRank = args.kv_lora_rank();
  param.softmaxScale = sm_scale_;
  if (quantize_type_ == "w8a8_dynamic" && num_speculative_tokens_ == 0) {
    param.enableMlaPreprocess = param.isBF16 ? false : true;
  } else {
    param.enableMlaPreprocess = false;
  }

  param.enableFA3 = false;           // TODO
  param.enableKvQuantLayer = false;  // TODO

  param.linearHasBias = {true, false, false, false};
  // param.enableFA3 = false;           // TODO
  // param.enableKvQuantLayer = false;  // TODO
}

void NpuGlm4MoeDecoderLiteImpl::initialize_mlp_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normScaling";
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  param.expertParallelDegree = 0;
  param.enableFusedRouting = true;
  param.numOfSharedExperts = args.n_shared_experts();
  param.numOfExperts = args.num_experts();
  param.numOfDeviceExperts = args.num_experts();
  param.routedScalingFactor = args.routed_scaling_factor();
  param.deviceExpert.resize(num_experts_per_partition_);
  param.firstKDenseReplace = args.first_k_dense_replace();
  param.numOfGroups = args.n_group();
  param.topkGroups = atb::SVector<int>{args.topk_group()};
  param.isDenseLayer = param.layerId < param.firstKDenseReplace;
  param.enableDispatchCombineV2 = true;
  // param.deviceExpert.resize(args.n_routed_experts());
  std::iota(
      param.deviceExpert.begin(), param.deviceExpert.end(), start_expert_id_);
  // param.maskStartIdx = 0;
  param.routingMethod = "noAuxTc";

  // param.quantGroupSize = 0;
  param.enableInitQuant = false;
  param.enableSwigluQuant = false;
  param.enableFusedTopk = true;

  param.enableCVOverlap = false;  // TODO
}

void NpuGlm4MoeDecoderLiteImpl::initialize_parallel_parameters(
    atb_speed::glm::MoeLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = dp_local_tp_size_;
  param.mapping = parallel_args.mapping();
  param.tensorParallelInfo = {
      parallel_args.rank(),
      parallel_args.world_size(),
      ::xllm::ParallelConfig::get_instance().communication_backend(),
      ::xllm::EPLBConfig::get_instance().rank_tablefile(),
      nullptr,
      ""};

  param.PrintParam();
  param.maxDecodeDpTokenSize = 0;  // TODO
}

void NpuGlm4MoeDecoderLiteImpl::initialize_quantization_parameters(
    atb_speed::glm::MoeLayerParam& param) {
  if (quantize_type_.empty()) {
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP)};
    param.linearQuantType = {static_cast<int>(LinearType::FP),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::FP),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID)};

    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::FP)};
    }
  } else {
    param.kvQuantHasOffset = 1;
    param.enableGMMSwigluQuant = 1;
    param.enableInitQuant = 1;
    param.moePackQuantType = static_cast<int>(PackType::PACK_QUANT_UNDEFINED);
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8_ANTI),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INT)};
    param.linearQuantType = {static_cast<int>(LinearType::INT),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::FP),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID),
                             static_cast<int>(LinearType::INVALID)};

    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};

      param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID)};

      param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID)};
    }
  }
}

void NpuGlm4MoeDecoderLiteImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  Device::empty_cache(device_.index());
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }
  init_layer();
}

int64_t NpuGlm4MoeDecoderLiteImpl::init_layer() {
  BaseLayer::name_ = "glm4_moe_lite_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "Glm4_Moe_lite";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));

  return atb::NO_ERROR;
}

int64_t NpuGlm4MoeDecoderLiteImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::glm::MoeLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::glm::MoeDecoderLayer<atb::infer::RmsNormParam> decoder_layer(
      param);
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

torch::Tensor NpuGlm4MoeDecoderLiteImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
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
                            true);
    st = execute_node(prefill_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << " excute prefill layer fail, error code: " << st;
  } else {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            /*attn_mask*/ tensor_placeholder_,
                            kv_cache,
                            input_params,
                            false);
    st = execute_node(decode_node_, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << " excute decode layer fail, error code: " << st;
  }

  return tensor_placeholder_;
}

void NpuGlm4MoeDecoderLiteImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    bool is_prefill) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  auto& dp_ep_padding = input_params.parallel.dp_ep_padding_data;

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
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
      const_cast<int32_t*>(input_params.attention.host.kv_seq_lens.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.block_tables);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
      atb_speed::Utils::AtTensor2Tensor(
          input_params.attention.device.new_cache_slots);

  // ADD in_q_len
  if (input_params.meta.batch_forward_type.is_chunked_prefill()) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.kv_cache_tokens_nums);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(
            input_params.attention.host.kv_cache_tokens_nums.data());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(placeholder_vec_zero_.data());
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(input_params.expert.expert_array);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);

  int32_t input_idx = WEIGHT_COUNT_PER_LAYER + 16;

  // ADD prefix
  if (input_params.meta.batch_forward_type.is_chunked_prefill()) {
    node.variantPack.inTensors.at(input_idx) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.history_compressed_kv);
    node.variantPack.inTensors.at(input_idx + 1) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.history_k_rope);
    node.variantPack.inTensors.at(input_idx + 2) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.ring_cur_seqlen);
    node.variantPack.inTensors.at(input_idx + 2).hostData =
        const_cast<int32_t*>(
            input_params.attention.host.ring_cur_seqlen.data());
    node.variantPack.inTensors.at(input_idx + 3) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.ring_cache_seqlen);
    node.variantPack.inTensors.at(input_idx + 3).hostData =
        const_cast<int32_t*>(
            input_params.attention.host.ring_cache_seqlen.data());
  }

  // if (is_prefill &&
  //     (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
  //     ::xllm::KVCacheConfig::get_instance().enable_prefix_cache())) {
  //   node.variantPack.inTensors.at(input_idx) =
  //       atb_speed::Utils::AtTensor2Tensor(input_params.attention.device.q_seq_lens);
  //   node.variantPack.inTensors.at(input_idx).hostData =
  //       const_cast<int32_t*>(input_params.attention.host.q_seq_lens.data());
  //   input_idx++;
  // }

  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(at_start_expert_id_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(at_in_device_expert_count_);

  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(input_idx++) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);

  if (::xllm::ExecutionConfig::get_instance().enable_graph() && !is_prefill &&
      input_params.graph.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(input_params.graph.tiling_data);
  }

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;
}

}  // namespace layer
}  // namespace xllm
