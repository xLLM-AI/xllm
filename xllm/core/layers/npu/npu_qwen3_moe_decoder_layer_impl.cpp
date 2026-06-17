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

#include "npu_qwen3_moe_decoder_layer_impl.h"

#include <gflags/gflags.h>

#include <unordered_set>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
namespace xllm {
namespace layer {

static const uint64_t WEIGHT_COUNT_PER_LAYER = 55;

NpuQwen3MoeDecoderLayerImpl::NpuQwen3MoeDecoderLayerImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : BaseLayer(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  num_experts_ = model_args.num_experts();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.num_experts() / ep_size_;
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  // int ep_rank = prefill_param_.rank /  ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  param_from_args(prefill_param_, model_args, parallel_args, true);
  param_from_args(decode_graph_param_, model_args, parallel_args, false);
  decode_eager_param_ = decode_graph_param_;
  decode_eager_param_.enableAclGraphPagedAttention = false;
  loader_ = std::make_unique<Qwen3MoeDecoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
  initialize_tensors(options);
}

void NpuQwen3MoeDecoderLayerImpl::initialize_tensors(
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
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  expert_group_ = torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
  quant_add_norm_scaling_ =
      torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
  quant_add_norm_offset_ =
      torch::tensor({1}, torch::dtype(torch::kInt32)).to(device_);
}

void NpuQwen3MoeDecoderLayerImpl::param_from_args(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
}

void NpuQwen3MoeDecoderLayerImpl::initialize_basic_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.isPrefill = is_prefill;

  // prefill only feature
  param.enableLcoc = is_prefill;  // false;
  param.enableSplitFuse =
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache()) &&
      is_prefill;

  // decode only feature
  param.enableAclGraphPagedAttention =
      ::xllm::ExecutionConfig::get_instance().enable_graph() && !is_prefill;
  param.enableInitRoutingV3 = !is_prefill;

  // Can be applied to prefill, but has not been tested yet
  param.enableFusedReducesumDiv = !is_prefill;
  param.enableAclnnExternelAddRmsNorm =
      ::xllm::KernelConfig::get_instance().enable_intralayer_addnorm() &&
      !is_prefill;
  param.enableAclnnAddRmsNorm = !is_prefill;

  param.swigluBackend = atb_speed::common::OpBackend::ACLNN;
  param.mlpLinearTransposeType = {static_cast<int>(TransposeType::INVALID),
                                  static_cast<int>(TransposeType::INVALID),
                                  static_cast<int>(TransposeType::INVALID),
                                  static_cast<int>(TransposeType::INVALID)};
  if (quantize_type_.empty()) {
    param.moeLinearTransposeType = {static_cast<int>(TransposeType::TRANSPOSE),
                                    static_cast<int>(TransposeType::TRANSPOSE),
                                    static_cast<int>(TransposeType::INVALID),
                                    static_cast<int>(TransposeType::TRANSPOSE)};
  } else {
    param.moeLinearTransposeType = {
        static_cast<int>(TransposeType::TRANSPOSE),
        static_cast<int>(TransposeType::NOT_TRANSPOSE),
        static_cast<int>(TransposeType::INVALID),
        static_cast<int>(TransposeType::TRANSPOSE)};
  }
  param.normEps = args.rms_norm_eps();
  param.rank = parallel_args.rank();
  param.backend =
      ::xllm::ParallelConfig::get_instance().communication_backend();
  // param.rankTableFile =
  // ::xllm::EPLBConfig::get_instance().rank_tablefile();

  param.layerId = layer_id_;
  param.numHiddenLayers = 0;
  param.enableIntraLayerAddNorm = false;
  param.enableInterLayerAddNorm = false;
  if (quantize_type_.empty()) {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }

  param.enableSpeculate = false;                    // MTP
  param.enableSwiGLUQuantForSharedExperts = false;  // TODO

  param.useQKNorm = true;
  param.rmsnormQKNorm = true;
  param.hiddenSizePerAttentionHead = args.head_dim();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = std::max(
      1, static_cast<int>(optionalValue.value()) / parallel_args.world_size());
  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;

  param.attnLinearTransposeType = {static_cast<int>(TransposeType::TRANSPOSE),
                                   static_cast<int>(TransposeType::INVALID),
                                   static_cast<int>(TransposeType::INVALID),
                                   static_cast<int>(TransposeType::TRANSPOSE),
                                   static_cast<int>(TransposeType::INVALID),
                                   static_cast<int>(TransposeType::INVALID)};
  param.worldSize = parallel_args.world_size();

  if (is_prefill) {
    param.enableAclnnRmsNorm = quantize_type_.empty();
  }
}

void NpuQwen3MoeDecoderLayerImpl::initialize_attention_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.enableFA3 = false;           // TODO
  param.enableKvQuantLayer = false;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_mlp_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normalization";
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  param.expertParallelDegree = 1;
  param.enableFusedRouting = 1;
  param.numOfExperts = args.num_experts();
  param.maskStartIdx = 0;
  param.routingMethod = "integratedSoftmaxTopK";
  param.quantGroupSize = 0;

  param.enableInitQuant = false;
  param.enableSwigluQuant = false;
  param.enableCVOverlap = false;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param,
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

  param.maxDecodeDpTokenSize = 0;  // TODO
}

void NpuQwen3MoeDecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::qwen::MoeDecoderLayerParam& param) {
  if (quantize_type_.empty()) {
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID)};

    param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::FP)};
  } else {
    param.packQuantType = {static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC_ANTI)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INT),
                                 static_cast<int>(LinearType::INVALID),
                                 static_cast<int>(LinearType::INVALID)};
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INVALID)};
    param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INT)};
  }
}

void NpuQwen3MoeDecoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  Device::empty_cache(device_.index());
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }
  init_layer();
}

int64_t NpuQwen3MoeDecoderLayerImpl::init_layer() {
  name_ = "qwen3_moe_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "Qwen3_Moe";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_graph_node_, decode_graph_param_));
  CHECK_OPERATION_STATUS_RETURN(
      init_node(decode_eager_node_, decode_eager_param_));

  return atb::NO_ERROR;
}

int64_t NpuQwen3MoeDecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::qwen::MoeDecoderLayerParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::qwen::MoeDecoderLayer(param, &operation);
  node.operation.reset(operation);
  CHECK_NOTNULL(node.operation);
  CHECK_GT(node.operation->GetInputNum(), 0);
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());
  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(node.outTensors.size());
  node.variantPack.outTensors.resize(node.outTensors.size());

  return atb::NO_ERROR;
}

torch::Tensor NpuQwen3MoeDecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
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
                            residual,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params,
                            true,
                            false);
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
                            residual,
                            cos_pos,
                            sin_pos,
                            /*attn_mask*/ tensor_placeholder_,
                            kv_cache,
                            input_params,
                            false,
                            use_graph_decode_input);
    st = execute_node(decode_node, node_id + 1000, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "excute decode layer fail, error code: " << st;
  }

  return tensor_placeholder_;
}

void NpuQwen3MoeDecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    bool is_prefill,
    bool use_graph_decode_input) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  int32_t input_idx = 0;
  auto& dp_ep_padding = input_params.parallel.dp_ep_padding_data;

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(input_params.expert.expert_array);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());

  if (!input_params.attention.device.block_tables.defined() ||
      input_params.attention.device.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10).hostData =
        const_cast<int32_t*>(placeholder_vec_.data());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.kv_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10).hostData =
        const_cast<int32_t*>(input_params.attention.host.kv_seq_lens.data());
  }
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  if (!input_params.attention.device.block_tables.defined() ||
      input_params.attention.device.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(block_tables_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.block_tables);
  }
  if (!input_params.attention.device.block_tables.defined() ||
      input_params.attention.device.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(slot_tensor_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.new_cache_slots);
  }

  input_idx = WEIGHT_COUNT_PER_LAYER + 16;
  if (is_prefill &&
      (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ||
       ::xllm::KVCacheConfig::get_instance().enable_prefix_cache())) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_seq_lens);
    node.variantPack.inTensors.at(input_idx - 1).hostData =
        const_cast<int32_t*>(input_params.attention.host.q_seq_lens.data());
  }

  if (!is_prefill && use_graph_decode_input &&
      input_params.graph.tiling_data.defined()) {
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(input_params.graph.tiling_data);
  }

  if (input_params.meta.batch_forward_type.is_decode() &&
      ::xllm::KernelConfig::get_instance().enable_intralayer_addnorm() &&
      residual.has_value()) {
    // input
    auto& residual_tensor = residual.value();
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(residual_tensor);
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(quant_add_norm_scaling_);
    node.variantPack.inTensors.at(input_idx++) =
        atb_speed::Utils::AtTensor2Tensor(quant_add_norm_offset_);

    // output
    auto residual_tensor_ = atb_speed::Utils::AtTensor2Tensor(residual_tensor);
    node.variantPack.outTensors.at(1) = residual_tensor_;
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
