/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/mlu/glm5_next/glm5_next_kda.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include "core/common/constants.h"
#include "core/kernels/param.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "framework/state_dict/utils.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/mlu/linear_attention_utils.h"

namespace xllm {
namespace layer {

namespace {

constexpr std::array<const char*, 6> kInputProjectionNames = {
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "b_proj.weight",
    "f_a_proj.weight",
    "g_a_proj.weight"};

void load_column_linear(ColumnParallelLinear& linear,
                        const StateDict& state_dict,
                        const std::string& prefix) {
  StateDict linear_state = state_dict.get_dict_with_prefix(prefix);
  if (linear_state.size() > 0 && !linear->is_weight_loaded()) {
    linear->load_state_dict(linear_state);
  }
}

void load_conv_linear(ColumnParallelLinear& linear,
                      const StateDict& state_dict,
                      const std::string& name) {
  torch::Tensor weight = state_dict.get_tensor(name + ".weight");
  if (weight.defined() && !linear->is_weight_loaded()) {
    CHECK_EQ(weight.dim(), 3) << name << ".weight must be [channels, 1, K]";
    CHECK_EQ(weight.size(1), 1) << name << ".weight middle dimension must be 1";
    linear->load_state_dict(StateDict({{"weight", weight.squeeze(1)}}));
  }
}

}  // namespace

Glm5NextKDAInputProjectionImpl::Glm5NextKDAInputProjectionImpl(
    const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto* tp_group = context.get_parallel_args().tp_group_;
  rank_ = tp_group->rank();
  tp_size_ = tp_group->world_size();
  const int64_t num_local_heads = args.linear_num_key_heads() / tp_size_;
  const int64_t head_dim = args.linear_key_head_dim();
  const int64_t local_size = num_local_heads * head_dim;
  shard_sizes_ = {
      local_size, local_size, local_size, num_local_heads, head_dim, head_dim};
  const int64_t output_size =
      std::accumulate(shard_sizes_.begin(), shard_sizes_.end(), int64_t{0});
  weight_ = register_parameter(/*name=*/"weight",
                               torch::empty({output_size, args.hidden_size()},
                                            context.get_tensor_options()),
                               /*requires_grad=*/false);
}

void Glm5NextKDAInputProjectionImpl::load_state_dict(
    const StateDict& state_dict) {
  int64_t offset = 0;
  for (size_t index = 0; index < shard_sizes_.size(); ++index) {
    torch::Tensor destination = weight_.narrow(0, offset, shard_sizes_[index]);
    offset += shard_sizes_[index];
    if (loaded_[index]) {
      continue;
    }
    weight::load_sharded_weight(state_dict,
                                kInputProjectionNames[index],
                                /*dim=*/0,
                                index < 4 ? rank_ : 0,
                                index < 4 ? tp_size_ : 1,
                                destination,
                                loaded_[index]);
  }
}

void Glm5NextKDAInputProjectionImpl::verify_loaded_weights(
    const std::string& prefix) const {
  for (size_t index = 0; index < loaded_.size(); ++index) {
    CHECK(loaded_[index]) << "Missing required weight after all shards loaded: "
                          << prefix << kInputProjectionNames[index];
  }
}

torch::Tensor Glm5NextKDAInputProjectionImpl::forward(
    const torch::Tensor& hidden_states) {
  kernel::MatmulParams params;
  params.a = hidden_states;
  params.b = weight_;
  return kernel::matmul(params);
}

Glm5NextKDAImpl::Glm5NextKDAImpl(const ModelContext& context)
    : gate_lower_bound_(context.get_model_args().linear_lower_bound()) {
  const auto& args = context.get_model_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  CHECK(parallel_args.tp_group_ != nullptr);
  tp_size_ = parallel_args.tp_group_->world_size();
  rank_ = parallel_args.tp_group_->rank();
  const int64_t num_heads = args.linear_num_key_heads();
  head_dim_ = args.linear_key_head_dim();
  conv_kernel_size_ = args.linear_conv_kernel_dim();

  CHECK_GT(num_heads, 0) << "GLM5-Next KDA requires positive head count";
  CHECK_EQ(num_heads, args.linear_num_value_heads())
      << "GLM5-Next KDA currently requires equal q/k/v head counts";
  CHECK_EQ(head_dim_, args.linear_value_head_dim())
      << "GLM5-Next KDA currently requires equal key/value head dimensions";
  CHECK_EQ(num_heads % tp_size_, 0)
      << "GLM5-Next KDA head count must be divisible by TP size";
  CHECK_GT(conv_kernel_size_, 0)
      << "GLM5-Next KDA requires positive causal-conv width";
  CHECK_LT(gate_lower_bound_, 0.0f)
      << "GLM5-Next KDA gate lower bound must be negative";

  local_num_heads_ = num_heads / tp_size_;
  const int64_t projection_size = num_heads * head_dim_;
  local_projection_size_ = local_num_heads_ * head_dim_;
  const QuantArgs no_quant_args{};

  in_proj_qkvbfg_a_ = register_module(/*name=*/"in_proj_qkvbfg_a",
                                      Glm5NextKDAInputProjection(context));
  f_b_proj_ = register_module("f_b_proj",
                              ColumnParallelLinear(head_dim_,
                                                   projection_size,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  g_b_proj_ = register_module("g_b_proj",
                              ColumnParallelLinear(head_dim_,
                                                   projection_size,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  // torch_mlu's causal-conv kernels require activations, weights, and the
  // convolution cache to use the same dtype. The checkpoint loader casts the
  // source FP32 convolution weights into the model dtype (normally BF16), as
  // done by vLLM-MLU's GLM5-Next adapter.
  q_conv1d_ = register_module("q_conv1d",
                              ColumnParallelLinear(conv_kernel_size_,
                                                   projection_size,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  k_conv1d_ = register_module("k_conv1d",
                              ColumnParallelLinear(conv_kernel_size_,
                                                   projection_size,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));
  v_conv1d_ = register_module("v_conv1d",
                              ColumnParallelLinear(conv_kernel_size_,
                                                   projection_size,
                                                   /*bias=*/false,
                                                   /*gather_output=*/false,
                                                   no_quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  const torch::TensorOptions fp32_options = options.dtype(torch::kFloat32);
  dt_bias_ =
      register_parameter("dt_bias",
                         torch::empty({local_projection_size_}, fp32_options),
                         /*requires_grad=*/false);
  A_log_ = register_parameter("A_log",
                              torch::empty({local_num_heads_}, fp32_options),
                              /*requires_grad=*/false);

  o_norm_ = register_module(
      "o_norm", RmsNormGated(head_dim_, args.rms_norm_eps(), options));
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(projection_size,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              no_quant_args,
                                              parallel_args.tp_group_,
                                              options));
  chunk_kda_ =
      register_module("chunk_kda", kernel::mlu::ChunkKDA(local_num_heads_));
}

void Glm5NextKDAImpl::load_state_dict(const StateDict& state_dict) {
  in_proj_qkvbfg_a_->load_state_dict(state_dict);
  load_column_linear(f_b_proj_, state_dict, "f_b_proj.");
  load_column_linear(g_b_proj_, state_dict, "g_b_proj.");

  load_conv_linear(q_conv1d_, state_dict, "q_conv1d");
  load_conv_linear(k_conv1d_, state_dict, "k_conv1d");
  load_conv_linear(v_conv1d_, state_dict, "v_conv1d");

  StateDict o_norm_state = state_dict.get_dict_with_prefix("o_norm.");
  if (o_norm_state.size() > 0 && !o_norm_is_loaded_) {
    o_norm_->load_state_dict(o_norm_state);
    o_norm_is_loaded_ = o_norm_state.get_tensor("weight").defined();
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));

  weight::load_sharded_weight(state_dict,
                              "dt_bias",
                              /*dim=*/0,
                              static_cast<int32_t>(rank_),
                              static_cast<int32_t>(tp_size_),
                              dt_bias_,
                              dt_bias_is_loaded_);
  torch::Tensor a_log = state_dict.get_tensor("A_log");
  if (a_log.defined()) {
    // HF KDA checkpoints exist in both [H] and [1,1,H,1] forms. Normalize the
    // latter before applying the head shard so the runtime/kernel contract is
    // always the compact FP32 [H_local] form.
    CHECK(a_log.dim() == 1 || (a_log.dim() == 4 && a_log.size(0) == 1 &&
                               a_log.size(1) == 1 && a_log.size(3) == 1))
        << "GLM5-Next KDA A_log must be [H] or [1, 1, H, 1]";
    StateDict a_log_state =
        a_log.dim() == 1
            ? state_dict
            : StateDict({{"A_log", a_log.reshape({a_log.numel()})}});
    weight::load_sharded_weight(a_log_state,
                                "A_log",
                                /*dim=*/0,
                                static_cast<int32_t>(rank_),
                                static_cast<int32_t>(tp_size_),
                                A_log_,
                                A_log_is_loaded_);
  }
}

void Glm5NextKDAImpl::verify_loaded_weights(const std::string& prefix) const {
  in_proj_qkvbfg_a_->verify_loaded_weights(prefix);
  CHECK(f_b_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "f_b_proj.weight";
  CHECK(g_b_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "g_b_proj.weight";
  CHECK(q_conv1d_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "q_conv1d.weight";
  CHECK(k_conv1d_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "k_conv1d.weight";
  CHECK(v_conv1d_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "v_conv1d.weight";
  CHECK(o_norm_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "o_norm.weight";
  CHECK(o_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "o_proj.weight";
  CHECK(dt_bias_is_loaded_)
      << "Missing required weight after all shards loaded: " << prefix
      << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight after all shards loaded: "
                          << prefix << "A_log";
}

torch::Tensor Glm5NextKDAImpl::get_linear_state_indices(
    const ModelParamsView& input_params,
    const torch::Device& device) const {
  CHECK(!input_params.embedding.linear_state_ids.empty())
      << "linear_state_ids must be populated for GLM5-Next KDA";
  if (input_params.embedding.linear_state_indices.defined()) {
    return input_params.embedding.linear_state_indices;
  }
  return torch::tensor(
      input_params.embedding.linear_state_ids,
      torch::TensorOptions().dtype(torch::kInt).device(device));
}

torch::Tensor Glm5NextKDAImpl::forward(const torch::Tensor& hidden_states,
                                       const AttentionMetadata& attn_metadata,
                                       KVCache& kv_cache,
                                       const ModelParamsView& input_params) {
  const int64_t num_tokens = hidden_states.size(0);
  const bool is_target_verify =
      input_params.meta.batch_forward_type.is_decode() &&
      input_params.meta.q_max_seq_len > 1;
  if (is_target_verify) {
    CHECK(attn_metadata.is_chunked_prefill)
        << "GLM5-Next KDA Spec Verify requires chunked-prefill Dense Validate "
           "Span";
  }

  torch::Tensor projected = in_proj_qkvbfg_a_->forward(hidden_states);
  const auto projections = projected.split_with_sizes(
      {3 * local_projection_size_, local_num_heads_, head_dim_, head_dim_}, -1);
  torch::Tensor mixed_qkv = projections[0].contiguous();
  torch::Tensor beta = projections[1].contiguous();
  torch::Tensor raw_gate = f_b_proj_->forward(projections[2].contiguous())
                               .view({num_tokens, local_num_heads_, head_dim_})
                               .contiguous();
  torch::Tensor output_gate =
      g_b_proj_->forward(projections[3].contiguous())
          .view({num_tokens, local_num_heads_, head_dim_})
          .contiguous();
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;

  torch::Tensor conv_cache = kv_cache.get_conv_cache().transpose(-1, -2);
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  const int64_t checkpoint_stride = KVCacheShape::linear_ssm_checkpoint_stride(
      kv_cache.get_conv_cache().size(0), ssm_cache.size(0));
  if (is_target_verify) {
    CHECK_EQ(checkpoint_stride, attn_metadata.max_query_len)
        << "GLM5-Next KDA Spec Verify checkpoint stride mismatch";
    const int64_t expected_conv_state_len =
        (attn_metadata.max_query_len - 1) + (conv_kernel_size_ - 1);
    CHECK_EQ(conv_cache.size(2), expected_conv_state_len)
        << "GLM5-Next KDA Spec Verify conv state length mismatch";
  }

  torch::Tensor conv_weight = torch::cat(
      {q_conv1d_->weight(), k_conv1d_->weight(), v_conv1d_->weight()},
      /*dim=*/0);
  torch::Tensor logical_state_indices =
      get_linear_state_indices(input_params, mixed_qkv.device());
  torch::Tensor state_base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);
  torch::Tensor core_output;

  if (!is_target_verify &&
      (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill)) {
    mixed_qkv = mixed_qkv.transpose(/*dim0=*/0, /*dim1=*/1);
    mixed_qkv = xllm::kernel::mlu::causal_conv1d_fn(
        mixed_qkv,
        conv_weight,
        conv_cache,
        attn_metadata.q_cu_seq_lens,
        attn_metadata.batch,
        attn_metadata.token_block_offset,
        attn_metadata.tot,
        /*bias=*/std::nullopt,
        logical_state_indices,
        attn_metadata.has_initial_states,
        /*initial_state_idx=*/std::nullopt,
        /*num_accepted_tokens=*/std::nullopt,
        /*inplace_final_state=*/true);
    mixed_qkv = mixed_qkv.transpose(/*dim0=*/0, /*dim1=*/1);
    std::tie(q, k, v) = split_mixed_qkv(
        mixed_qkv, local_num_heads_, local_num_heads_, head_dim_, head_dim_);

    torch::Tensor initial_state = ssm_cache.index({state_base_indices});
    initial_state.index_put_(
        {~attn_metadata.has_initial_states, torch::indexing::Ellipsis}, 0.0f);
    torch::Tensor cu_seqlens = attn_metadata.q_cu_seq_lens.contiguous();
    torch::Tensor final_state;
    std::tie(core_output, final_state) =
        chunk_kda_->forward_raw_gate(q,
                                     k,
                                     v,
                                     raw_gate,
                                     A_log_,
                                     dt_bias_,
                                     gate_lower_bound_,
                                     beta,
                                     initial_state,
                                     cu_seqlens,
                                     attn_metadata.chunk_indices,
                                     /*output_final_state=*/true,
                                     /*use_qk_l2norm=*/true);
    core_output = core_output.squeeze(/*dim=*/0).contiguous();
    ssm_cache.index_put_({state_base_indices},
                         final_state.to(ssm_cache.scalar_type()));
  } else {
    if (is_target_verify) {
      mixed_qkv = xllm::kernel::mlu::causal_conv1d_update_decode(
          mixed_qkv,
          conv_cache,
          conv_weight,
          /*bias_opt=*/std::nullopt,
          logical_state_indices,
          /*activation=*/true,
          /*pad_slot_id=*/kPaddingLinearStateId,
          attn_metadata.q_cu_seq_lens,
          static_cast<int32_t>(attn_metadata.max_query_len),
          input_params.num_accepted_tokens);
    } else {
      mixed_qkv = xllm::kernel::mlu::causal_conv1d_update_decode(
          mixed_qkv,
          conv_cache,
          conv_weight,
          /*bias_opt=*/std::nullopt,
          logical_state_indices,
          /*activation=*/true,
          /*pad_slot_id=*/kPaddingLinearStateId);
    }
    std::tie(q, k, v) = split_mixed_qkv(
        mixed_qkv, local_num_heads_, local_num_heads_, head_dim_, head_dim_);

    xllm::kernel::FusedSigmoidGatingDeltaRuleUpdateParams params;
    params.A_log = A_log_;
    params.a = raw_gate.view({num_tokens, local_projection_size_});
    params.dt_bias = dt_bias_;
    params.q = q;
    params.k = k;
    params.v = v;
    params.b = beta;
    params.initial_state_source = ssm_cache;
    params.initial_state_indices =
        is_target_verify ? build_ssm_checkpoint_indices(state_base_indices,
                                                        checkpoint_stride)
                         : state_base_indices;
    if (is_target_verify) {
      // Slot zero is reserved; every checkpoint of a virtual request skips
      // state instead of overwriting the reserved slot.
      params.initial_state_indices.masked_fill_(
          logical_state_indices.eq(kPaddingLinearStateId).unsqueeze(1), 0);
    }
    params.cu_seqlens = attn_metadata.q_cu_seq_lens;
    params.scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));
    if (is_target_verify) {
      params.num_accepted_tokens = input_params.num_accepted_tokens;
    }
    params.use_qk_l2norm_in_kernel = true;
    params.is_kda = true;
    params.kda_use_safe_gate = true;
    params.kda_gate_lower_bound = gate_lower_bound_;
    core_output = xllm::kernel::fused_sigmoid_gating_delta_rule_update(params)
                      .squeeze(/*dim=*/0)
                      .contiguous();
  }

  core_output = core_output.view({-1, head_dim_});
  output_gate = output_gate.view({-1, head_dim_});
  torch::Tensor normalized = o_norm_->forward(core_output);
  normalized = (normalized.to(torch::kFloat32) *
                torch::sigmoid(output_gate.to(torch::kFloat32)))
                   .to(normalized.scalar_type());
  normalized = normalized.view({num_tokens, local_projection_size_});
  return o_proj_->forward(normalized);
}

}  // namespace layer
}  // namespace xllm
