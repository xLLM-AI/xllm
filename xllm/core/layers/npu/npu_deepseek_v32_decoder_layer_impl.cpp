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

#include "npu_deepseek_v32_decoder_layer_impl.h"

#include <gflags/gflags.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cctype>
#include <iostream>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "framework/parallel_state/npu_cp_prepare.h"
#include "framework/quant_args.h"
#include "layers/common/rotary_embedding_util.h"
#include "loader/deepseek_v32_decoder_loader.h"

namespace xllm {
namespace layer {

namespace {
constexpr int32_t kQProjBLinearIndex = 1;
constexpr int32_t kIndexerWqBLinearIndex = 6;
constexpr int32_t kIndexerProjLinearIndex = 8;
constexpr int32_t kTranspose = 1;
constexpr int32_t kNotTranspose = 0;

std::string normalize_quant_type(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

std::optional<std::string> get_layer_quant_desc(
    const QuantArgs& quant_args,
    int32_t layer_id,
    const std::vector<std::string>& local_prefixes) {
  const std::string layer_prefix =
      "model.layers." + std::to_string(layer_id) + ".";
  std::optional<std::pair<std::string, std::string>> resolved;
  std::optional<std::string> unresolved_prefix;
  for (const std::string& local_prefix : local_prefixes) {
    const std::string prefix = layer_prefix + local_prefix;
    auto quant_desc = quant_args.get_quant_method(prefix, "weight");
    if (!quant_desc.has_value()) {
      if (resolved.has_value()) {
        LOG(FATAL) << "Inconsistent quant_desc for prefix '" << prefix
                   << "': unresolved, but prefix '" << resolved->first
                   << "' resolved to '" << resolved->second << "'.";
      }
      unresolved_prefix = prefix;
      continue;
    }
    const std::string quant_type = normalize_quant_type(quant_desc.value());
    if (unresolved_prefix.has_value()) {
      LOG(FATAL) << "Inconsistent quant_desc for prefix '"
                 << unresolved_prefix.value() << "': unresolved, but prefix '"
                 << prefix << "' resolved to '" << quant_type << "'.";
    }
    if (!resolved.has_value()) {
      resolved = std::make_pair(prefix, quant_type);
      continue;
    }
    CHECK_EQ(resolved->second, quant_type)
        << "Inconsistent quant_desc for prefix '" << prefix << "' resolved to '"
        << quant_type << "', but prefix '" << resolved->first
        << "' resolved to '" << resolved->second << "'.";
  }
  if (!resolved.has_value()) {
    CHECK(quant_args.quant_descs().empty())
        << "Missing quant_desc for layer " << layer_id
        << " attention prefixes: "
        << boost::algorithm::join(local_prefixes, ", ");
    return std::nullopt;
  }
  return resolved->second;
}

std::optional<std::string> get_optional_layer_quant_desc(
    const QuantArgs& quant_args,
    int32_t layer_id,
    const std::string& local_prefix) {
  const std::string prefix =
      "model.layers." + std::to_string(layer_id) + "." + local_prefix;
  auto quant_desc = quant_args.get_quant_method(prefix, "weight");
  if (!quant_desc.has_value()) {
    return std::nullopt;
  }
  return normalize_quant_type(quant_desc.value());
}

int32_t quant_desc_to_linear_desc(const std::optional<std::string>& quant_desc,
                                  int32_t default_desc) {
  if (!quant_desc.has_value()) {
    return default_desc;
  }
  const std::string quant_type = normalize_quant_type(quant_desc.value());
  if (quant_type == "float") {
    return static_cast<int32_t>(LinearTypeV2::FLOAT16);
  }
  if (quant_type == "w8a8_dynamic") {
    return static_cast<int32_t>(LinearTypeV2::W8A8_DYNAMIC);
  }
  if (quant_type == "w8a8") {
    return static_cast<int32_t>(LinearTypeV2::W8A8);
  }
  LOG(FATAL) << "Unsupported DeepSeek V32 attention quant_desc: "
             << quant_desc.value();
  return default_desc;
}

std::vector<int32_t> resolve_attn_linear_quant_types(
    const QuantArgs& quant_args,
    int32_t layer_id) {
  const int32_t w8a8_desc = static_cast<int32_t>(LinearTypeV2::W8A8);
  const int32_t fp16_desc = static_cast<int32_t>(LinearTypeV2::FLOAT16);
  const int32_t q_a_desc = quant_desc_to_linear_desc(
      get_layer_quant_desc(quant_args, layer_id, {"self_attn.q_a_proj"}),
      w8a8_desc);
  const int32_t kv_a_desc = quant_desc_to_linear_desc(
      get_layer_quant_desc(
          quant_args, layer_id, {"self_attn.kv_a_proj_with_mqa"}),
      w8a8_desc);
  const int32_t q_b_desc = quant_desc_to_linear_desc(
      get_layer_quant_desc(quant_args, layer_id, {"self_attn.q_b_proj"}),
      w8a8_desc);
  const int32_t o_desc = quant_desc_to_linear_desc(
      get_layer_quant_desc(quant_args, layer_id, {"self_attn.o_proj"}),
      w8a8_desc);
  const int32_t indexer_wq_b_desc = quant_desc_to_linear_desc(
      get_optional_layer_quant_desc(
          quant_args, layer_id, "self_attn.indexer.wq_b"),
      fp16_desc);
  CHECK_EQ(q_a_desc, kv_a_desc)
      << "DeepSeek V32 currently requires self_attn.q_a_proj and "
      << "self_attn.kv_a_proj_with_mqa to share the same quant_desc.";
  return {q_a_desc,
          q_b_desc,
          kv_a_desc,
          fp16_desc,
          fp16_desc,
          o_desc,
          indexer_wq_b_desc,
          fp16_desc,
          fp16_desc};
}

std::vector<int> to_atb_linear_quant_types(
    const std::vector<int32_t>& linear_quant_types) {
  std::vector<int> atb_linear_quant_types;
  atb_linear_quant_types.reserve(linear_quant_types.size());
  for (int32_t linear_quant_type : linear_quant_types) {
    atb_linear_quant_types.emplace_back(static_cast<int>(linear_quant_type));
  }
  return atb_linear_quant_types;
}
}  // namespace

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_Q_PROJ_A_WEIGHT = 4,
  IN_Q_PROJ_A_BIAS = 5,
  IN_Q_PROJ_A_DESCALE = 6,
  IN_Q_PROJ_A_OFFSET = 7,
  IN_Q_PROJ_A_SCALE = 8,
  IN_Q_PROJ_A_COMPRESS_IDX = 9,
  IN_Q_PROJ_A_LAYERNORM_WEIGHT = 10,
  IN_Q_PROJ_A_LAYERNORM_BIAS = 11,

  IN_Q_PROJ_B_WEIGHT = 12,
  IN_Q_PROJ_B_BIAS = 13,
  IN_Q_PROJ_B_DESCALE = 14,
  IN_Q_PROJ_B_OFFSET = 15,
  IN_Q_PROJ_B_SCALE = 16,
  IN_Q_PROJ_B_COMPRESS_IDX = 17,

  IN_KV_PROJ_WITH_MQA_WEIGHT = 18,
  IN_KV_PROJ_WITH_MQA_BIAS = 19,
  IN_KV_PROJ_WITH_MQA_DESCALE = 20,
  IN_KV_PROJ_WITH_MQA_OFFSET = 21,
  IN_KV_PROJ_WITH_MQA_SCALE = 22,
  IN_KV_PROJ_WITH_MQA_COMPRESS_IDX = 23,

  IN_KV_PROJ_A_LAYERNORM_WEIGHT = 24,
  IN_KV_PROJ_A_LAYERNORM_BIAS = 25,

  IN_K_PROJ_B_FOR_Q_WEIGHT = 26,
  IN_K_PROJ_B_FOR_Q_BIAS = 27,
  IN_K_PROJ_B_FOR_Q_DESCALE = 28,
  IN_K_PROJ_B_FOR_Q_OFFSET = 29,
  IN_K_PROJ_B_FOR_Q_SCALE = 30,
  IN_K_PROJ_B_FOR_Q_COMPRESS_IDX = 31,

  IN_V_PROJ_B_FOR_O_WEIGHT = 32,
  IN_V_PROJ_B_FOR_O_BIAS = 33,
  IN_V_PROJ_B_FOR_O_DESCALE = 34,
  IN_V_PROJ_B_FOR_O_OFFSET = 35,
  IN_V_PROJ_B_FOR_O_SCALE = 36,
  IN_V_PROJ_B_FOR_O_COMPRESS_IDX = 37,

  IN_ATTENTION_OUT_WEIGHT = 38,
  IN_ATTENTION_OUT_BIAS = 39,
  IN_ATTENTION_OUT_DESCALE = 40,
  IN_ATTENTION_OUT_OFFSET = 41,
  IN_ATTENTION_OUT_SCALE = 42,
  IN_ATTENTION_OUT_COMPRESS_IDX = 43,

  IN_SELFATTENTION_OUT_NORM_WEIGHT = 44,
  IN_SELFATTENTION_OUT_NORM_BIAS = 45,
  IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT = 46,
  IN_SELFATTENTION_OUT_NEW_NORM_BIAS = 47,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 48,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 49,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 50,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 51,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 52,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 53,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 54,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 55,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 56,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 57,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 58,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 59,

  IN_SHARED_EXPERT_GATE_WEIGHT = 60,
  IN_SHARED_EXPERT_GATE_BIAS = 61,
  IN_SHARED_EXPERT_GATE_DESCALE = 62,
  IN_SHARED_EXPERT_GATE_OFFSET = 63,
  IN_SHARED_EXPERT_GATE_SCALE = 64,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 65,

  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 66,
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 67,
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE = 68,
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET = 69,
  IN_BLOCK_SPARSE_MOE_GATE_SCALE = 70,
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 71,

  IN_MLP_GATEUP_WEIGHT_EXPERT = 72,
  IN_MLP_GATEUP_BIAS_EXPERT = 73,
  IN_MLP_GATEUP_DESCALE_EXPERT = 74,
  IN_MLP_GATEUP_OFFSET_EXPERT = 75,
  IN_MLP_GATEUP_SCALE_EXPERT = 76,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = 77,

  IN_MLP_DOWN_WEIGHT_EXPERT = 78,
  IN_MLP_DOWN_BIAS_EXPERT = 79,
  IN_MLP_DOWN_DESCALE_EXPERT = 80,
  IN_MLP_DOWN_OFFSET_EXPERT = 81,
  IN_MLP_DOWN_SCALE_EXPERT = 82,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 83,

  IN_INDEXER_WQ_B_WEIGHT = 84,
  IN_INDEXER_WQ_B_BIAS = 85,
  IN_INDEXER_WQ_B_DESCALE = 86,
  IN_INDEXER_WQ_B_OFFSET = 87,
  IN_INDEXER_WQ_B_SCALE = 88,
  IN_INDEXER_WQ_B_COMPRESS_IDX = 89,

  IN_INDEXER_WK_WEIGHT = 90,
  IN_INDEXER_WK_BIAS = 91,
  IN_INDEXER_WK_DESCALE = 92,
  IN_INDEXER_WK_OFFSET = 93,
  IN_INDEXER_WK_SCALE = 94,
  IN_INDEXER_WK_COMPRESS_IDX = 95,

  IN_INDEXER_K_NORM_WEIGHT = 96,
  IN_INDEXER_K_NORM_BIAS = 97,

  IN_INDEXER_PROJ_WEIGHT = 98,
  IN_INDEXER_PROJ_BIAS = 99,
  IN_INDEXER_PROJ_DESCALE = 100,
  IN_INDEXER_PROJ_OFFSET = 101,
  IN_INDEXER_PROJ_SCALE = 102,
  IN_INDEXER_PROJ_COMPRESS_IDX = 103,
  IN_Q_PROJ_A_RECOMPUTE_WEIGHT = 104,
  IN_Q_PROJ_A_RECOMPUTE_BIAS = 105,
  IN_Q_PROJ_A_RECOMPUTE_DESCALE = 106,
  IN_Q_PROJ_A_RECOMPUTE_OFFSET = 107,
  IN_Q_PROJ_A_RECOMPUTE_SCALE = 108,
  IN_Q_PROJ_A_RECOMPUTE_COMPRESS_IDX = 109,
};

static const uint64_t WEIGHT_COUNT_PER_LAYER = 110;

NpuDeepseekV32DecoderLayerImpl::NpuDeepseekV32DecoderLayerImpl(
    const ModelContext& context,
    const int32_t layer_id)
    : BaseLayer(context),
      device_id_(context.get_tensor_options().device().index()),
      layer_id_(layer_id),
      num_speculative_tokens_(
          context.get_model_args().num_speculative_tokens()) {
  // compute sm_scale
  // TODO: refactor this code
  auto args = context.get_model_args();
  if (boost::iequals(args.rope_scaling_rope_type(), "deepseek_yarn")) {
    const float attn_scale = args.attn_scalar().value_or(
        static_cast<float>(args.qk_nope_head_dim() + args.qk_rope_head_dim()));
    sm_scale_ = 1.0f / std::sqrt(attn_scale);
    float mscale = layer::rotary::yarn_get_mscale(
        args.rope_scaling_factor(), args.rope_scaling_mscale_all_dim());
    sm_scale_ = sm_scale_ * mscale * mscale;
  } else if (boost::iequals(args.rope_scaling_rope_type(), "mrope")) {
    sm_scale_ = std::pow(args.head_dim(), -0.5);
  } else {
    const float attn_scale =
        args.attn_scalar().value_or(static_cast<float>(args.head_dim()));
    sm_scale_ = 1.0f / std::sqrt(attn_scale);
  }

  auto parallel_args = context.get_parallel_args();
  auto model_args = context.get_model_args();
  auto quant_args = context.get_quant_args();
  auto options = context.get_tensor_options();
  const DsaTopkShareDecision topk_decision =
      get_dsa_topk_share_decision(model_args, layer_id_);
  skip_topk_ = topk_decision.reuse_topk;
  output_topk_ = topk_decision.output_topk;

  rank_ = parallel_args.rank();
  first_k_dense_replace_ = model_args.first_k_dense_replace();
  n_layers_ = model_args.n_layers();
  num_experts_ = model_args.n_routed_experts();
  quant_group_size_ = static_cast<int32_t>(quant_args.group_size());
  attn_linear_quant_types_ =
      resolve_attn_linear_quant_types(quant_args, layer_id_);
  if (quantize_type_ == "w4a8_dynamic") {
    CHECK_GE(quant_group_size_, 0)
        << "W4A8_DYNAMIC group_size must be >= 0, got " << quant_group_size_;
    CHECK_EQ(quant_args.quant_version(), "1.0.0")
        << "W4A8_DYNAMIC only supports quant_version 1.0.0, got "
        << (quant_args.quant_version().empty() ? "<empty>"
                                               : quant_args.quant_version());
  }
  localWorldSize_ = parallel_args.mapping().localWorldSize();
  ep_size_ = parallel_args.ep_size();
  ep_local_tp_size_ = parallel_args.world_size() / ep_size_;
  CHECK_EQ(parallel_args.world_size(), ep_size_ * ep_local_tp_size_);
  ep_local_tp_rank_ = parallel_args.rank() % ep_local_tp_size_;
  num_experts_per_partition_ = model_args.n_routed_experts() / ep_size_;
  redundant_experts_num_ =
      ::xllm::EPLBConfig::get_instance().redundant_experts_num();
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    num_experts_per_partition_ += redundant_experts_num_;
  }
  ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
  start_expert_id_ = ep_rank_ * num_experts_per_partition_;
  end_expert_id_ = start_expert_id_ + num_experts_per_partition_ - 1;

  dp_size_ = parallel_args.dp_size();
  cp_size_ = parallel_args.cp_size();
  dp_local_tp_size_ = parallel_args.world_size() / (dp_size_ * cp_size_);
  CHECK_EQ(parallel_args.world_size(), dp_size_ * dp_local_tp_size_ * cp_size_);
  dp_local_tp_rank_ = parallel_args.rank() % dp_local_tp_size_;

  param_from_args(
      prefill_param_, model_args, parallel_args, /*is_prefill=*/true);
  param_from_args(
      decode_param_, model_args, parallel_args, /*is_prefill=*/false);
  has_mtp_topk_fallback_ =
      skip_topk_ && model_args.index_share_for_mtp_iteration() &&
      model_args.model_type().find("_mtp") != std::string::npos;
  if (has_mtp_topk_fallback_) {
    mtp_prefill_fallback_param_ = prefill_param_;
    mtp_prefill_fallback_param_.skipTopk = false;
    mtp_prefill_fallback_param_.outputTopk = true;
    mtp_decode_fallback_param_ = decode_param_;
    mtp_decode_fallback_param_.skipTopk = false;
    mtp_decode_fallback_param_.outputTopk = true;
  }

  loader_ = std::make_unique<DeekseekV32DecoderLoader>(
      WEIGHT_COUNT_PER_LAYER,
      context,
      layer_id_,
      prefill_param_.firstKDenseReplace,
      prefill_param_.numOfDeviceExperts,
      prefill_param_.qkRopeHeadDim,
      prefill_param_.numAttentionHeadsPerRank,
      decode_param_.worldSize,
      qk_nope_head_dim_,
      kv_lora_rank_,
      num_key_value_heads_,
      v_head_dim_,
      prefill_param_.isBF16,
      decode_param_.isBF16,
      attn_linear_quant_types_,
      skip_topk_ && !has_mtp_topk_fallback_,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
  initialize_tensors(options);
}

void NpuDeepseekV32DecoderLayerImpl::initialize_tensors(
    const torch::TensorOptions& options) {
  // initializ placeholder
  atb_weight_tensors_.resize(WEIGHT_COUNT_PER_LAYER);
  placeholder_vec_ = {1};
  int_tensor_placeholder_ = torch::ones({1}).to(torch::kInt32).to(device_);
  slot_tensor_placeholder_ = torch::full({1}, 0).to(torch::kInt32).to(device_);
  block_tables_placeholder_ =
      torch::zeros({1, 1}).to(torch::kInt32).to(device_);
  tensor_placeholder_ = torch::zeros({1}).to(options);

  expert_group_ = torch::arange(1024, torch::kInt32).to(device_);
  one_hot_ = torch::tensor({1}, torch::kInt32).to(device_);
  zero_hot_ = torch::tensor({0}, torch::kInt32).to(device_);
  at_start_expert_id_ =
      torch::tensor({start_expert_id_}, torch::kInt64).to(device_);
  at_in_device_expert_count_ =
      torch::tensor({num_experts_per_partition_ - 1}, torch::kInt64)
          .to(device_);

  auto& device_expert_list = loader_->get_device_expert_list();
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    auto layer_expert_routing_map_ =
        build_expert_routing_map(device_expert_list);
    std::vector<torch::Tensor> tensors_vec;
    for (int i = 0; i < n_layers_ - first_k_dense_replace_; i++) {
      tensors_vec.emplace_back(layer_expert_routing_map_);
    }
    expert_routing_map_ = torch::stack(tensors_vec, 0);
  }
}

void NpuDeepseekV32DecoderLayerImpl::param_from_args(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  initialize_basic_parameters(param, args, parallel_args, is_prefill);
  initialize_attention_parameters(param, args, parallel_args);
  initialize_mlp_parameters(param, args, parallel_args);
  initialize_parallel_parameters(param, parallel_args);
  initialize_quantization_parameters(param);
}

void NpuDeepseekV32DecoderLayerImpl::initialize_basic_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args,
    bool is_prefill) {
  param.isFA = false;
  param.isPrefill = is_prefill;
  param.isBF16 = args.dtype() == "bfloat16";
  param.enableSwiGLU = true;
  param.enableLcoc = true;
  // TODO: modify xllm_atb_layers

  param.attnLinearTransposeType = {
      kTranspose, kTranspose, kTranspose, kTranspose, kTranspose, kTranspose};
  if (attn_linear_quant_types_.size() > kQProjBLinearIndex &&
      attn_linear_quant_types_[kQProjBLinearIndex] ==
          static_cast<int>(LinearTypeV2::W8A8_DYNAMIC)) {
    param.attnLinearTransposeType[kQProjBLinearIndex] = kNotTranspose;
  }
  if (attn_linear_quant_types_.size() > kIndexerWqBLinearIndex &&
      param.attnLinearTransposeType.size() <= kIndexerProjLinearIndex) {
    param.attnLinearTransposeType.insert(param.attnLinearTransposeType.end(),
                                         {kTranspose, kTranspose, kTranspose});
  }
  param.mlpLinearTransposeType = {1, -1, 1, -1};

  if (layer_id_ < args.first_k_dense_replace()) {
    param.moeLinearTransposeType = {-1, -1, -1, -1};
  } else if (quantize_type_ == "w4a8_dynamic") {
    param.moeLinearTransposeType = {1, 0, -1, 0};
  } else {
    param.moeLinearTransposeType = {1, 0, -1, 1};
  }

  param.worldSize = parallel_args.world_size();
  param.normEps = args.rms_norm_eps();
  param.numAttentionHeadsPerRank = args.n_heads() / dp_local_tp_size_;
  param.hiddenSizePerAttentionHead = args.hidden_size() / args.n_heads();
  std::optional<long int> optionalValue = args.n_kv_heads();
  param.numKeyValueHeadsPerRank = 1;
  // static_cast<int>(optionalValue.value()) / param.worldSize;
  param.rank = parallel_args.rank();
  param.backend =
      ::xllm::ParallelConfig::get_instance().communication_backend();
  param.rankTableFile = ::xllm::EPLBConfig::get_instance().rank_tablefile();

  param.layerId = layer_id_;
  param.numHiddenLayers = args.n_layers();
  param.enableIntraLayerAddNorm = true;
  param.enableInterLayerAddNorm = false;
  if (quantize_type_ == "" || quantize_type_ == "w4a8_dynamic") {
    param.enableGMMSwigluQuant = false;
  } else {
    param.enableGMMSwigluQuant =
        (is_prefill && parallel_args.world_size() > 16) || !is_prefill;
  }
  param.enableDpOut = false;  // TODO
  if (num_speculative_tokens_ == 0) {
    param.enableSpeculate = false;  // MTP
  } else {
    param.enableSpeculate = true;
  }
  param.maskfree = true;  // TODO
  const bool is_shared_expert_layer =
      layer_id_ >= args.first_k_dense_replace() && args.n_shared_experts() > 0;
  param.enableSwiGLUQuantForSharedExperts =
      (quantize_type_ == "w8a8_dynamic" || quantize_type_ == "w4a8_dynamic") &&
      is_shared_expert_layer;
  if ((::xllm::KVCacheConfig::get_instance().enable_prefix_cache() ||
       ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) &&
      ::xllm::ParallelConfig::get_instance().cp_size() > 1 && is_prefill) {
    param.enablePrefixCacheCP = true;
    // When kv_split_size collapses to 1 (each CP rank owns a full KV replica)
    // the ATB prefix AllGather is replaced by an identity reshape so the
    // downstream SparseFlashAttention input contract stays unchanged.
    // See `sparse_latent_attention.cpp::AddPrefixConcatCpNode` and S6 in the
    // KV-split / CP decoupling plan. The default (kv_split_size == cp_size)
    // keeps this false, preserving legacy behavior byte-for-byte.
    param.enablePrefixCacheLocal =
        (parallel_args.kv_split_size_effective() == 1);
  }
  num_key_value_heads_ = static_cast<int>(args.n_kv_heads().value());
  qk_nope_head_dim_ = args.qk_nope_head_dim();
  v_head_dim_ = args.v_head_dim();
  kv_lora_rank_ = args.kv_lora_rank();
  qk_rope_head_dim_ = args.qk_rope_head_dim();
}

void NpuDeepseekV32DecoderLayerImpl::initialize_attention_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
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
  // not support in glm_moe_dsa
  bool is_glm_moe_dsa =
      args.model_type().find("glm_moe_dsa") != std::string::npos;
  if (quantize_type_ == "w8a8_dynamic" && !is_glm_moe_dsa) {
    param.enableMlaPreprocess = true;
  } else {
    param.enableMlaPreprocess = false;
  }

  param.enableFA3 = false;           // TODO
  param.isNzCache = false;           // TODO
  param.enableKvQuantLayer = false;  // TODO
  param.index_head_dim = args.index_head_dim();
  param.index_n_heads = args.index_n_heads();
  param.index_topk = args.index_topk();
  param.skipTopk = skip_topk_;
  param.outputTopk = output_topk_;
}

void NpuDeepseekV32DecoderLayerImpl::initialize_mlp_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ModelArgs& args,
    const ParallelArgs& parallel_args) {
  param.hasSharedExpert = (args.n_shared_experts() > 0);
  param.hasSharedExpertGate = false;
  param.processLogits = "normScaling";
  param.routedScalingFactor = args.routed_scaling_factor();
  param.numOfSelectedExperts = {args.num_experts_per_tok()};

  if (ep_size_ > 1) {
    param.expertParallelDegree = std::max(
        ::xllm::EPLBConfig::get_instance().expert_parallel_degree(), 1);
  } else {
    param.expertParallelDegree = 0;
  }

  param.deviceExpert.resize(num_experts_per_partition_);
  // param.deviceExpert.resize(args.n_routed_experts());
  std::iota(
      param.deviceExpert.begin(), param.deviceExpert.end(), start_expert_id_);
  param.numOfExperts = args.n_routed_experts();
  param.numOfDeviceExperts = num_experts_per_partition_;
  param.maskStartIdx = 0;
  param.firstKDenseReplace = args.first_k_dense_replace();
  param.numOfSharedExperts = args.n_shared_experts();
  param.routingMethod = "noAuxTc";
  param.numOfGroups = args.n_group();
  param.topkGroups = atb::SVector<int>{args.topk_group()};
  param.isDynamicEp = param.expertParallelDegree == 2 ? true : false;

  param.quantGroupSize =
      quantize_type_ == "w4a8_dynamic" ? quant_group_size_ : 0;
  if (quantize_type_ == "") {
    param.enableInitQuant = false;
    param.enableSwigluQuant = false;
  } else {
    param.enableInitQuant = true;
    param.enableSwigluQuant = param.isPrefill && !param.enableGMMSwigluQuant;
  }
  param.enableFusedTopk = true;

  param.enableCVOverlap = false;           // TODO
  param.enableExpertCumSumOutput = false;  // TODO
  param.enableLoadBalance = false;         // TODO
  param.enableEPWB = false;                // TODO
  param.numOfRedundantExpert = 0;          // TODO
  param.enableInfNan = param.isPrefill;    // TODO

  param.dispatchAndCombineHcclComm = parallel_args.dispatchAndCombineHcclComm();
  param.dispatchAndCombinecommDomain =
      parallel_args.dispatchAndCombinecommDomain();

  param.scaledTopk = -1;
  param.enableATBGateMatmul = true;

  param.enableIndexGmm = false;
  // LCOC fused all2all path is unstable under ACL graph launch in current
  // runtime; keep it for eager mode and fall back to the standard dynamic-ep
  // path when graph is enabled.
  param.enableLcocAll2All =
      param.isPrefill && cp_size_ == 1 && dp_size_ == 1 &&
      !::xllm::ExecutionConfig::get_instance().enable_graph();

  if (layer_id_ >= param.firstKDenseReplace) {
    param.enableQkvdownDp = false;
    param.enableSharedExpertDp = false;
    param.enableGatingDp = false;
    if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
      param.enableExpertCumSumOutput = param.isPrefill ? false : true;
      param.enableEPWB = true;
      param.numOfRedundantExpert = ep_size_ * redundant_experts_num_;
    }
  }
  if (layer_id_ < param.firstKDenseReplace) {
    param.isDenseLayer = true;
  }
}

void NpuDeepseekV32DecoderLayerImpl::initialize_parallel_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param,
    const ParallelArgs& parallel_args) {
  param.lmHeadLocalTp = dp_local_tp_size_;
  param.enableSharedExpertOverlap = false;  // TODO

  param.enableAllToAllMC2 = (param.expertParallelDegree == 2);
  param.enableGatherPreNorm = true;
  param.enableExtraOprojTp = false;  // TODO
  param.isMlpFullTP = false;         // TODO
  param.mapping = parallel_args.mapping();
  // Decode node should not use CP-specific graph branches.
  if (!param.isPrefill) {
    auto cp_info = param.mapping.Get(atb_speed::base::ATTN_CP);
    cp_info.rank = 0;
    cp_info.rankIds = {static_cast<uint32_t>(parallel_args.rank())};
    param.mapping.Register(atb_speed::base::ATTN_CP, cp_info);
  }
  param.maxDecodeDpTokenSize = 0;  // TODO
}

void NpuDeepseekV32DecoderLayerImpl::initialize_quantization_parameters(
    atb_speed::deepseekV2::DecoderLayerParam& param) {
  if (quantize_type_ == "") {
    param.moePackQuantType = static_cast<int>(PackType::ALL_FP);
    param.packQuantType = {static_cast<int>(PackType::ALL_FP),
                           static_cast<int>(PackType::ALL_FP)};
    param.attnLinearQuantType = {static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP),
                                 static_cast<int>(LinearType::FP)};
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
  } else if (quantize_type_ == "w8a8_dynamic") {
    param.moePackQuantType = static_cast<int>(PackType::ALL_W8A8_DYNAMIC);
    param.packQuantType = {static_cast<int>(PackType::MIX_W8A8),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC)};
    param.attnLinearQuantType =
        to_atb_linear_quant_types(attn_linear_quant_types_);
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID)};
    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT)};
    }
  } else if (quantize_type_ == "w4a8_dynamic") {
    param.moePackQuantType = static_cast<int>(PackType::ALL_W4A8);
    // The ATB W4A8 path keeps dense/shared MLP linears on W8A8 dynamic;
    // only routed experts use W4A8 GMM.
    param.packQuantType = {static_cast<int>(PackType::MIX_W8A8),
                           static_cast<int>(PackType::ALL_W8A8_DYNAMIC)};
    param.attnLinearQuantType =
        to_atb_linear_quant_types(attn_linear_quant_types_);
    param.mlpLinearQuantType = {static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID),
                                static_cast<int>(LinearType::INT),
                                static_cast<int>(LinearType::INVALID)};
    if (layer_id_ < param.firstKDenseReplace) {
      param.moeLinearQuantType = {static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INVALID)};
    } else {
      param.moeLinearQuantType = {static_cast<int>(LinearType::FP),
                                  static_cast<int>(LinearType::INT),
                                  static_cast<int>(LinearType::INVALID),
                                  static_cast<int>(LinearType::INT)};
    }
  }
}

void NpuDeepseekV32DecoderLayerImpl::merge_loaded_weights() {
  loader_->merge_loaded_weights();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  Device::empty_cache(device_.index());
  for (int i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    atb_weight_tensors_[i] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
  }
  init_layer();
}

torch::Tensor NpuDeepseekV32DecoderLayerImpl::build_expert_routing_map(
    std::vector<int32_t> expert_lists) {
  std::unordered_map<int64_t, std::vector<int64_t>> expert_routing_map;

  for (int64_t i = 0; i < expert_lists.size(); ++i) {
    int64_t v = expert_lists[i];
    expert_routing_map[v].emplace_back(i);
  }

  std::vector<int64_t> keys;
  std::vector<int32_t> values;
  for (auto& [key, indices] : expert_routing_map) {
    int num_of_duplications = indices.size();
    int selected_index = ep_rank_ % num_of_duplications;
    indices = {indices[selected_index]};

    keys.emplace_back(key);
    values.emplace_back(static_cast<int32_t>(indices[0]));
  }

  int64_t map_size = expert_routing_map.size();
  auto options = torch::TensorOptions().dtype(torch::kInt32);
  auto input = torch::zeros({map_size}, options);

  auto index_tensor = torch::tensor(keys, torch::kInt64);
  auto value_tensor = torch::tensor(values, torch::kInt32);
  auto result = input.scatter(0, index_tensor, value_tensor).to(device_);
  // result = result.reshape({ep_size_,result.size(0)/ep_size_}).contiguous();
  return result;
}

std::string NpuDeepseekV32DecoderLayerImpl::get_expert_shm_key(
    int32_t layer_id,
    int32_t expert_index,
    const std::string& suffix) {
  std::string shm_key =
      "layer_" + std::to_string(layer_id - first_k_dense_replace_) + "_" +
      "expert_" + std::to_string(expert_index) + "_" + suffix;
  return shm_key;
}

void NpuDeepseekV32DecoderLayerImpl::prepare_expert_weight(
    const std::vector<int32_t>& expert_list) {
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  auto& experts_weights = loader_->get_experts_weight_tensors();
  expert_routing_map_buffer_ = build_expert_routing_map(expert_list);
  auto& expert_buffer = ExpertBuffer::Instance();

  const int32_t num_local_experts = num_experts_per_partition_;
  const int32_t hidden_dim =
      at_weight_tensors[IN_MLP_GATEUP_WEIGHT_EXPERT].size(1);
  const int32_t combined_dim =
      at_weight_tensors[IN_MLP_GATEUP_WEIGHT_EXPERT].size(2);
  const int32_t gate_dim = combined_dim / 2;

  expert_buffer.initialize_or_reuse(
      /*gateup_weight_shape*/ {num_experts_per_partition_,
                               hidden_dim,
                               combined_dim},
      /*gateup_offset_shape*/ {num_experts_per_partition_, combined_dim, 1},
      /*gateup_scale_shape*/ {num_experts_per_partition_, combined_dim, 1},
      /*down_weight_shape*/
      {num_experts_per_partition_, hidden_dim, gate_dim},
      /*down_offset_shape*/ {num_experts_per_partition_, hidden_dim, 1},
      /*down_scale_shape*/ {num_experts_per_partition_, hidden_dim, 1},
      at_weight_tensors[IN_MLP_GATEUP_WEIGHT_EXPERT].options(),
      at_weight_tensors[IN_MLP_GATEUP_OFFSET_EXPERT].options(),
      at_weight_tensors[IN_MLP_GATEUP_SCALE_EXPERT].options()

  );

  const int start_expert_idx = num_experts_per_partition_ * ep_rank_;
  const int end_expert_idx = start_expert_idx + num_experts_per_partition_ - 1;

  auto& shared_buffer = loader_->get_expert_shared_buffer();
  for (const auto& pair : experts_weights) {
    for (int expert_idx = start_expert_idx; expert_idx <= end_expert_idx;
         ++expert_idx) {
      std::string shm_key =
          get_expert_shm_key(layer_id_, expert_list[expert_idx], pair.first);
      experts_weights[pair.first][expert_idx - start_expert_idx] =
          shared_buffer->get_tensor(expert_list[expert_idx],
                                    layer_id_ - first_k_dense_replace_,
                                    shm_key);
      // experts_weights_[pair.first][expert_idx] =
      // shared_buffer_->get_tensors(shm_key);
    }
  }

  merge_and_copy_gate_up_weights(expert_buffer.gateup_weight,
                                 experts_weights["gate_proj.weight"],
                                 experts_weights["up_proj.weight"],
                                 /*do_transpose=*/true);

  merge_and_copy_gate_up_weights(expert_buffer.gateup_offset,
                                 experts_weights["gate_proj.weight_offset"],
                                 experts_weights["up_proj.weight_offset"]);

  merge_and_copy_gate_up_weights(expert_buffer.gateup_scale,
                                 experts_weights["gate_proj.weight_scale"],
                                 experts_weights["up_proj.weight_scale"]);

  merge_and_copy_down_weights(expert_buffer.down_weight,
                              experts_weights["down_proj.weight"]);

  merge_and_copy_down_weights(expert_buffer.down_offset,
                              experts_weights["down_proj.weight_offset"]);

  merge_and_copy_down_weights(expert_buffer.down_scale,
                              experts_weights["down_proj.weight_scale"]);

  expert_buffer.gateup_weight =
      at_npu::native::npu_format_cast(expert_buffer.gateup_weight, 29);
}

void NpuDeepseekV32DecoderLayerImpl::merge_and_copy_gate_up_weights(
    torch::Tensor&
        target_buffer,  // [num_experts, hidden_dim, gate_dim + up_dim]
    const std::vector<torch::Tensor>& experts_gate,  // [gate_dim, hidden_dim]
    const std::vector<torch::Tensor>& experts_up,    // [up_dim, hidden_dim]
    bool do_transpose) {
  const int64_t num_experts = experts_gate.size();
  const int64_t gate_dim = experts_gate[0].size(0);
  const int64_t up_dim = experts_up[0].size(0);
  const int64_t hidden_dim = experts_gate[0].size(1);

  target_buffer = at_npu::native::npu_format_cast(target_buffer.contiguous(), 2)
                      .reshape({num_experts, gate_dim + up_dim, hidden_dim});

  for (int64_t index = 0; index < num_experts; ++index) {
    target_buffer[index].slice(0, 0, gate_dim).copy_(experts_gate[index]);

    target_buffer[index]
        .slice(0, gate_dim, gate_dim + up_dim)
        .copy_(experts_up[index]);
  }

  if (do_transpose) {
    target_buffer = target_buffer.transpose(1, 2).contiguous();
  }
}

void NpuDeepseekV32DecoderLayerImpl::merge_and_copy_down_weights(
    torch::Tensor& target_buffer,
    const std::vector<torch::Tensor>& experts_down) {
  const int64_t num_experts = experts_down.size();

  for (int64_t index = 0; index < num_experts; ++index) {
    target_buffer[index].copy_(experts_down[index]);
  }
}

void NpuDeepseekV32DecoderLayerImpl::update_expert_weight() {
  auto& expert_buffer = ExpertBuffer::Instance();
  auto& at_weight_tensors = loader_->get_at_weight_tensors();
  const auto tensor_pairs = {
      std::make_pair(IN_MLP_GATEUP_WEIGHT_EXPERT,
                     std::ref(expert_buffer.gateup_weight)),
      std::make_pair(IN_MLP_GATEUP_OFFSET_EXPERT,
                     std::ref(expert_buffer.gateup_offset)),
      std::make_pair(IN_MLP_GATEUP_SCALE_EXPERT,
                     std::ref(expert_buffer.gateup_scale)),
      std::make_pair(IN_MLP_DOWN_WEIGHT_EXPERT,
                     std::ref(expert_buffer.down_weight)),
      std::make_pair(IN_MLP_DOWN_OFFSET_EXPERT,
                     std::ref(expert_buffer.down_offset)),
      std::make_pair(IN_MLP_DOWN_SCALE_EXPERT,
                     std::ref(expert_buffer.down_scale))};
  for (auto& [index, buffer_tensor] : tensor_pairs) {
    std::swap(at_weight_tensors[index], buffer_tensor);
    atb_weight_tensors_[index] =
        atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[index]);
    prefill_node_.inTensors.at(index) = &atb_weight_tensors_[index];
    decode_node_.inTensors.at(index) = &atb_weight_tensors_[index];
    if (has_mtp_topk_fallback_) {
      mtp_prefill_fallback_node_.inTensors.at(index) =
          &atb_weight_tensors_[index];
      mtp_decode_fallback_node_.inTensors.at(index) =
          &atb_weight_tensors_[index];
    }
  }
  expert_routing_map_[layer_id_ - first_k_dense_replace_] =
      expert_routing_map_buffer_;
  expert_routing_map_ = expert_routing_map_.contiguous();
}

int64_t NpuDeepseekV32DecoderLayerImpl::init_layer() {
  name_ = "deepseek_v2_decoder_layer " + std::to_string(layer_id_);
  model_name_ = "DeepSeek_V2";
  CHECK_OPERATION_STATUS_RETURN(init_node(prefill_node_, prefill_param_));
  CHECK_OPERATION_STATUS_RETURN(init_node(decode_node_, decode_param_));
  if (has_mtp_topk_fallback_) {
    CHECK_OPERATION_STATUS_RETURN(
        init_node(mtp_prefill_fallback_node_, mtp_prefill_fallback_param_));
    CHECK_OPERATION_STATUS_RETURN(
        init_node(mtp_decode_fallback_node_, mtp_decode_fallback_param_));
  }
  return atb::NO_ERROR;
}

int64_t NpuDeepseekV32DecoderLayerImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::deepseekV2::DecoderLayerParam& param) {
  bool eplb_enabled = ::xllm::EPLBConfig::get_instance().enable_eplb() &&
                      layer_id_ >= decode_param_.firstKDenseReplace &&
                      !decode_param_.isPrefill;
  atb::Operation* operation = nullptr;
  atb_speed::deepseekV2::DecoderLayer(param, &operation);
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

  size_t out_tensor_num = 1;
  if (eplb_enabled) {
    ++out_tensor_num;
  }
  if (param.outputTopk) {
    ++out_tensor_num;
  }
  node.outTensors.resize(out_tensor_num);

  size_t inTensorId = 1;

  for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER;
       ++weightTensorId) {
    node.inTensors.at(weightTensorId) = &atb_weight_tensors_[weightTensorId];
  }

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());

  node.variantPack.outTensors.reserve(out_tensor_num);
  node.variantPack.outTensors.resize(out_tensor_num);
  return atb::NO_ERROR;
}

torch::Tensor NpuDeepseekV32DecoderLayerImpl::forward(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    int node_id) {
  CHECK(!is_topk_sharing_enabled())
      << "DSA top-k sharing layer requires forward_with_topk.";
  return forward_with_topk(x,
                           cos_pos,
                           sin_pos,
                           attn_mask,
                           kv_cache,
                           input_params,
                           torch::Tensor(),
                           nullptr,
                           event,
                           event_flag,
                           node_id);
}

torch::Tensor NpuDeepseekV32DecoderLayerImpl::forward_with_topk(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const torch::Tensor& shared_topk_indices,
    torch::Tensor* output_topk_indices,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    int node_id) {
  atb::Status st;
  ModelInputParams& input_params_new =
      const_cast<ModelInputParams&>(input_params);
  if (input_params_new.meta.batch_forward_type.is_decode()) {
    build_node_variant_pack(decode_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params_new,
                            false,
                            shared_topk_indices,
                            output_topk_indices,
                            skip_topk_,
                            output_topk_);
    st = execute_node(decode_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "execute prefill layer fail, error code: " << st;
  } else {
    build_node_variant_pack(prefill_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params_new,
                            true,
                            shared_topk_indices,
                            output_topk_indices,
                            skip_topk_,
                            output_topk_);
    st = execute_node(prefill_node_, node_id, event, event_flag);
    LOG_IF(FATAL, st != 0) << model_name_
                           << "execute prefill layer fail, error code: " << st;
  }
  return tensor_placeholder_;
}

torch::Tensor NpuDeepseekV32DecoderLayerImpl::forward_with_mtp_topk_fallback(
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    torch::Tensor* output_topk_indices,
    aclrtEvent* event,
    std::atomic<bool>* event_flag,
    int node_id) {
  CHECK(has_mtp_topk_fallback_)
      << "MTP top-k fallback is only valid for NextN sharing layers.";
  CHECK(output_topk_indices != nullptr && output_topk_indices->defined())
      << "MTP top-k fallback requires an output top-k tensor.";

  atb::Status st;
  ModelInputParams& input_params_new =
      const_cast<ModelInputParams&>(input_params);
  if (input_params_new.meta.batch_forward_type.is_decode()) {
    build_node_variant_pack(mtp_decode_fallback_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params_new,
                            false,
                            torch::Tensor(),
                            output_topk_indices,
                            false,
                            true);
    st = execute_node(mtp_decode_fallback_node_, node_id, event, event_flag);
  } else {
    build_node_variant_pack(mtp_prefill_fallback_node_,
                            x,
                            cos_pos,
                            sin_pos,
                            attn_mask,
                            kv_cache,
                            input_params_new,
                            true,
                            torch::Tensor(),
                            output_topk_indices,
                            false,
                            true);
    st = execute_node(mtp_prefill_fallback_node_, node_id, event, event_flag);
  }
  LOG_IF(FATAL, st != 0)
      << model_name_
      << "execute MTP top-k fallback layer fail, error code: " << st;
  return tensor_placeholder_;
}

void NpuDeepseekV32DecoderLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    torch::Tensor& x,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& attn_mask,
    KVCache& kv_cache,
    ModelInputParams& input_params,
    bool is_prefill,
    const torch::Tensor& shared_topk_indices,
    torch::Tensor* output_topk_indices,
    bool skip_topk,
    bool output_topk) {
  internal_tensor_ = atb_speed::Utils::AtTensor2Tensor(x);
  // final_hidden_states_ = torch::zeros_like(x);
  int32_t input_idx = 0;
  auto& dp_ep_padding = input_params.parallel.dp_ep_padding_data;
  auto& cp_ep_padding = input_params.parallel.cp_ep_padding_data;
  auto& cp_inputs = input_params.parallel.cp_prefill_inputs;
  const bool use_cp_ep_padding = (cp_size_ > 1 && is_prefill);

  if (dp_size_ <= 1 && ep_size_ <= 1 || cp_size_ > 1) {
    dp_ep_padding.set_placeholder(tensor_placeholder_);
  }
  if (!use_cp_ep_padding) {
    cp_ep_padding.set_placeholder(tensor_placeholder_);
  }
  // set micro batch 0 input part
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER) = internal_tensor_;
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 1) =
      atb_speed::Utils::AtTensor2Tensor(use_cp_ep_padding
                                            ? cp_ep_padding.expert_array()
                                            : dp_ep_padding.expert_array());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 2) =
      atb_speed::Utils::AtTensor2Tensor(expert_group_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 3) =
      atb_speed::Utils::AtTensor2Tensor(one_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 4) =
      atb_speed::Utils::AtTensor2Tensor(zero_hot_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 5) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 6) =
      atb_speed::Utils::AtTensor2Tensor(cos_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 7) =
      atb_speed::Utils::AtTensor2Tensor(sin_pos);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 8) =
      atb_speed::Utils::AtTensor2Tensor(attn_mask);

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 9) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_k_cache());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 10) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_v_cache());

  if ((!input_params.attention.device.block_tables.defined() ||
       input_params.attention.device.block_tables.storage().data() ==
           nullptr)) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(placeholder_vec_.data());
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.kv_seq_lens);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 11).hostData =
        const_cast<int32_t*>(input_params.attention.host.kv_seq_lens.data());
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 12) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 13).hostData =
      const_cast<int32_t*>(placeholder_vec_.data());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 14) =
      atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);

  if (!input_params.attention.device.block_tables.defined() ||
      input_params.attention.device.block_tables.storage().data() == nullptr) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(block_tables_placeholder_);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 16) =
        atb_speed::Utils::AtTensor2Tensor(slot_tensor_placeholder_);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 15) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.block_tables);
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 16) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.new_cache_slots);
  }

  if (num_speculative_tokens_ > 0 && !is_prefill) {
    if ((!input_params.attention.device.block_tables.defined() ||
         input_params.attention.device.block_tables.storage().data() ==
             nullptr)) {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
          atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17).hostData =
          const_cast<int32_t*>(placeholder_vec_.data());
    } else {
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
          atb_speed::Utils::AtTensor2Tensor(
              input_params.attention.device.q_seq_lens);
      node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17).hostData =
          const_cast<int32_t*>(input_params.attention.host.q_seq_lens.data());
    }
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 17) =
        atb_speed::Utils::AtTensor2Tensor(tensor_placeholder_);
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 18) =
      atb_speed::Utils::AtTensor2Tensor(use_cp_ep_padding
                                            ? cp_ep_padding.attn_padding_idx()
                                            : dp_ep_padding.attn_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 19) =
      atb_speed::Utils::AtTensor2Tensor(
          use_cp_ep_padding ? cp_ep_padding.attn_unpadding_idx()
                            : dp_ep_padding.attn_unpadding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 20) =
      atb_speed::Utils::AtTensor2Tensor(use_cp_ep_padding
                                            ? cp_ep_padding.ffn_padding_idx()
                                            : dp_ep_padding.ffn_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 21) =
      atb_speed::Utils::AtTensor2Tensor(
          use_cp_ep_padding ? cp_ep_padding.ffn_unpadding_idx()
                            : dp_ep_padding.ffn_unpadding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 22) =
      atb_speed::Utils::AtTensor2Tensor(
          use_cp_ep_padding
              ? cp_ep_padding.lm_head_skip_padding_token_indices()
              : dp_ep_padding.lm_head_skip_padding_token_indices());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 23) =
      atb_speed::Utils::AtTensor2Tensor(
          use_cp_ep_padding ? cp_ep_padding.gather_prenorm_idx()
                            : dp_ep_padding.gather_prenorm_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 24) =
      atb_speed::Utils::AtTensor2Tensor(at_start_expert_id_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 25) =
      atb_speed::Utils::AtTensor2Tensor(at_in_device_expert_count_);
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 26) =
      atb_speed::Utils::AtTensor2Tensor(use_cp_ep_padding
                                            ? cp_ep_padding.padding_idx()
                                            : dp_ep_padding.padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 27) =
      atb_speed::Utils::AtTensor2Tensor(use_cp_ep_padding
                                            ? cp_ep_padding.un_padding_idx()
                                            : dp_ep_padding.un_padding_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 28) =
      atb_speed::Utils::AtTensor2Tensor(use_cp_ep_padding
                                            ? cp_ep_padding.dynamic_ep_idx()
                                            : dp_ep_padding.dynamic_ep_idx());
  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 29) =
      atb_speed::Utils::AtTensor2Tensor(use_cp_ep_padding
                                            ? cp_ep_padding.moe_idx()
                                            : dp_ep_padding.moe_idx());
  if (::xllm::EPLBConfig::get_instance().enable_eplb() &&
      layer_id_ >= decode_param_.firstKDenseReplace) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 30) =
        atb_speed::Utils::AtTensor2Tensor(expert_routing_map_);
    if (!is_prefill) {
      node.variantPack.outTensors.at(1) = atb_speed::Utils::AtTensor2Tensor(
          input_params.expert
              .expert_load_data[layer_id_ - decode_param_.firstKDenseReplace]);
    }
  }

  for (size_t i = 0; i < WEIGHT_COUNT_PER_LAYER; ++i) {
    CHECK_THROW(node.inTensors.at(i) == nullptr,
                model_name_ << " inTensor " << i << " is NULL");
    node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
  }

  node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 30) =
      atb_speed::Utils::AtTensor2Tensor(kv_cache.get_index_cache());

  const bool empty_eager_batch =
      !input_params.enable_graph && input_params.meta.num_sequences == 0;
  const bool empty_graph_batch =
      input_params.enable_graph && input_params.meta.actual_num_sequences == 0;
  const bool empty_batch = empty_eager_batch || empty_graph_batch;
  const bool use_q_cu_seq_lens =
      !empty_batch && input_params.attention.device.q_cu_seq_lens.defined() &&
      input_params.attention.device.q_cu_seq_lens.numel() != 0;
  if (use_q_cu_seq_lens) {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 31) =
        atb_speed::Utils::AtTensor2Tensor(
            input_params.attention.device.q_cu_seq_lens);
  } else {
    node.variantPack.inTensors.at(WEIGHT_COUNT_PER_LAYER + 31) =
        atb_speed::Utils::AtTensor2Tensor(int_tensor_placeholder_);
  }

  int32_t cp_input_index = WEIGHT_COUNT_PER_LAYER + 32;
  if (skip_topk) {
    CHECK(shared_topk_indices.defined())
        << "DSA top-k sharing requires previous top-k indices.";
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(shared_topk_indices);
  }

  if (cp_size_ > 1 && is_prefill) {
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(cp_inputs.cp_o_recover_idx);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(cp_inputs.cp_kv_recover_idx);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(cp_inputs.cp_load_balance_idx);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(cp_inputs.k_gather_index_prev);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(cp_inputs.k_gather_index_next);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(
            cp_inputs.actual_seq_lengths_query_prev);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(
            cp_inputs.actual_seq_lengths_query_next);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(
            cp_inputs.actual_seq_lengths_key_prev);
    node.variantPack.inTensors.at(cp_input_index++) =
        atb_speed::Utils::AtTensor2Tensor(
            cp_inputs.actual_seq_lengths_key_next);
    if (::xllm::KVCacheConfig::get_instance().enable_prefix_cache() ||
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      node.variantPack.inTensors.at(cp_input_index++) =
          atb_speed::Utils::AtTensor2Tensor(
              input_params.attention.device.in_prefix_slots);
    }
  }

  node.variantPack.outTensors.at(0) = internal_tensor_;
  if (output_topk) {
    CHECK(output_topk_indices != nullptr && output_topk_indices->defined())
        << "DSA top-k sharing output tensor is not initialized.";
    node.variantPack.outTensors.at(node.variantPack.outTensors.size() - 1) =
        atb_speed::Utils::AtTensor2Tensor(*output_topk_indices);
  }
}

}  // namespace layer
}  // namespace xllm
