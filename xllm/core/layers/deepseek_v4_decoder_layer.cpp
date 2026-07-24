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

#include "deepseek_v4_decoder_layer.h"

#include <glog/logging.h>

#include "framework/config/parallel_config.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  int64_t hidden_size = args.hidden_size();

  hc_mult_ = args.hc_mult();
  hc_sinkhorn_iters_ = args.hc_sinkhorn_iters();
  hc_eps_ = static_cast<double>(args.hc_eps());
  norm_eps_ = static_cast<double>(args.rms_norm_eps());

  attention_ = register_module("attn", DSAttention(context, layer_id));
  attn_norm_ = register_module(
      "attn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  ffn_norm_ = register_module(
      "ffn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  FusedMoEArgs moe_args;
  moe_args.is_gated = true;
  // DeepseekV4 drives expert routing through its own DeepseekV4Gate and only
  // calls forward_with_selected_experts().  The FusedMoE internal gate_ is
  // therefore never used; skip loading its weights to avoid redundant memory
  // allocation and a duplicate copy of the router weight matrix.
  moe_args.skip_gate_load = true;
  moe_mlp_ = register_module(
      "ffn", FusedMoE(args, moe_args, quant_args, parallel_args, options));
  // Register as "gate" to match Python's mlp.gate module path.
  gate_ = register_module("gate", DeepseekV4Gate(context, layer_id));

  const int64_t mix_hc = (2 + hc_mult_) * hc_mult_;
  const int64_t hc_dim = hc_mult_ * hidden_size;
  auto hc_options = options.dtype(torch::kFloat32);
  hc_attn_fn_ = register_parameter("hc_attn_fn",
                                   torch::empty({mix_hc, hc_dim}, hc_options),
                                   /*requires_grad=*/false);
  hc_ffn_fn_ = register_parameter("hc_ffn_fn",
                                  torch::empty({mix_hc, hc_dim}, hc_options),
                                  /*requires_grad=*/false);
  hc_attn_base_ = register_parameter("hc_attn_base",
                                     torch::empty({mix_hc}, hc_options),
                                     /*requires_grad=*/false);
  hc_ffn_base_ = register_parameter("hc_ffn_base",
                                    torch::empty({mix_hc}, hc_options),
                                    /*requires_grad=*/false);
  hc_attn_scale_ = register_parameter("hc_attn_scale",
                                      torch::empty({3}, hc_options),
                                      /*requires_grad=*/false);
  hc_ffn_scale_ = register_parameter("hc_ffn_scale",
                                     torch::empty({3}, hc_options),
                                     /*requires_grad=*/false);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attn_state = state_dict.get_dict_with_prefix("attn.");
  if (attn_state.size() == 0) {
    attn_state = state_dict.get_dict_with_prefix("self_attn.");
  }
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
  }

  auto attn_norm_state = state_dict.get_dict_with_prefix("attn_norm.");
  if (attn_norm_state.size() == 0) {
    attn_norm_state = state_dict.get_dict_with_prefix("input_layernorm.");
  }
  if (attn_norm_state.size() > 0) {
    attn_norm_->load_state_dict(attn_norm_state);
  }

  auto ffn_norm_state = state_dict.get_dict_with_prefix("ffn_norm.");
  if (ffn_norm_state.size() == 0) {
    ffn_norm_state =
        state_dict.get_dict_with_prefix("post_attention_layernorm.");
  }
  if (ffn_norm_state.size() > 0) {
    ffn_norm_->load_state_dict(ffn_norm_state);
  }

  auto ffn_state = state_dict.get_dict_with_prefix("ffn.");
  if (ffn_state.size() == 0) {
    ffn_state = state_dict.get_dict_with_prefix("mlp.");
  }
  if (ffn_state.size() > 0) {
    auto gate_state = ffn_state.get_dict_with_prefix("gate.");
    if (gate_state.size() == 0) {
      gate_state = state_dict.get_dict_with_prefix("gate.");
    }
    if (gate_state.size() > 0) {
      gate_->load_state_dict(gate_state);
    }
    moe_mlp_->load_state_dict(ffn_state);
  }

  LOAD_WEIGHT(hc_attn_fn);
  LOAD_WEIGHT(hc_ffn_fn);
  LOAD_WEIGHT(hc_attn_base);
  LOAD_WEIGHT(hc_ffn_base);
  LOAD_WEIGHT(hc_attn_scale);
  LOAD_WEIGHT(hc_ffn_scale);
}

void DeepseekV4DecoderLayerImpl::verify_loaded_weights() const {}

torch::Tensor DeepseekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& input_ids) {
  (void)positions;

  residual = std::nullopt;

  CHECK(attn_metadata.dsa_metadata)
      << "DeepseekV4DecoderLayer requires DSA metadata for DSAttention path.";

  // FlashComm1 (sequence parallel): x arrives token-sharded across the TP
  // group. hc_pre / norm run on the local shard; the full token dimension is
  // restored right before attention so the DSA kernels and q/kv projections see
  // all tokens. Attention's o_b_proj reduce-scatters the output back to a shard
  // (see RowParallelLinearImpl::forward), so residual and attn output stay
  // shard-aligned. A no-op when no FlashComm1 context is active.
  const auto& fc1 = parallel_state::current_flash_comm1_context();

  // Gather the token shard back to full tokens (lossless bf16) at an SP
  // boundary.
  auto fc1_lossless_gather = [&](const torch::Tensor& t) -> torch::Tensor {
    return parallel_state::all_gather_dim0_unpad(
        t, fc1.tp_group, fc1.num_tokens);
  };

  auto residual_attn = x;
  auto [attn_input, post_attn, comb_attn] =
      hc_pre(x, hc_attn_fn_, hc_attn_scale_, hc_attn_base_);
  attn_input = std::get<0>(attn_norm_->forward(attn_input));

  if (fc1.enabled) {
    attn_input = fc1_lossless_gather(attn_input);
  }

  auto& dsa = *(attn_metadata.dsa_metadata);
  const auto compress_metadata = std::make_tuple(
      dsa.c1_metadata, dsa.c4_metadata, dsa.c128_metadata, dsa.qli_metadata);
  KVState kv_state{kv_cache.get_swa_cache(),
                   kv_cache.get_compress_kv_state(),
                   kv_cache.get_compress_score_state(),
                   kv_cache.get_compress_index_kv_state(),
                   kv_cache.get_compress_index_score_state()};
  auto [attn_output, attn_lse] =
      attention_->forward(dsa,
                          attn_input,
                          kv_cache,
                          kv_state,
                          attn_metadata.is_prefill,
                          attn_metadata.is_chunked_prefill,
                          compress_metadata);
  (void)attn_lse;
  attn_input = attn_output;
  x = hc_post(attn_input, residual_attn, post_attn, comb_attn);

  auto residual_ffn = x;
  auto [ffn_input, post_ffn, comb_ffn] =
      hc_pre(x, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_);
  ffn_input = std::get<0>(ffn_norm_->forward(ffn_input));

  // MoE boundary: all-gather the shard back to full tokens so gate + MoE run
  // with baseline routing semantics; the output is sharded back below.
  if (fc1.enabled) {
    ffn_input = fc1_lossless_gather(ffn_input);
  }

  auto ffn_input_2d = ffn_input.reshape({-1, ffn_input.size(-1)});
  std::optional<torch::Tensor> gate_input_ids = std::nullopt;
  if (input_ids.has_value() && input_ids.value().defined()) {
    auto flat_input_ids =
        input_ids.value().reshape({-1}).to(ffn_input.device());
    const int64_t token_count = flat_input_ids.size(0);
    const int64_t hidden_rows = ffn_input_2d.size(0);
    if (token_count == hidden_rows) {
      gate_input_ids = flat_input_ids;
    } else if (token_count > 0 && hidden_rows % token_count == 0) {
      const int64_t repeat_factor = hidden_rows / token_count;
      gate_input_ids = flat_input_ids.unsqueeze(1)
                           .repeat({1, repeat_factor})
                           .reshape({hidden_rows});
    }
  }
  if (gate_->is_hash_layer()) {
    CHECK(gate_input_ids.has_value())
        << "DeepseekV4 hash gate requires input_ids for routing";
  }
  auto [topk_weights, topk_ids] = gate_->forward(ffn_input_2d, gate_input_ids);
  ffn_input = moe_mlp_->forward_with_selected_experts(
      ffn_input, topk_weights, topk_ids, input_params);

  if (fc1.enabled) {
    // MoE returns full tokens (expert parallelism shards experts, not tokens);
    // shard back so it stays aligned with the sharded residual before hc_post.
    const int64_t moe_rows = ffn_input.defined() ? ffn_input.size(0) : 0;
    CHECK_EQ(moe_rows, fc1.num_tokens)
        << "FlashComm1: MoE output must own this DP replica's full token "
           "count before TP token sharding (expected "
        << fc1.num_tokens << ", got " << moe_rows << ")";
    ffn_input = parallel_state::shard_dim0_padded(
        ffn_input, fc1.tp_rank, fc1.tp_world_size);
  }
  x = hc_post(ffn_input, residual_ffn, post_ffn, comb_ffn);

  return x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4DecoderLayerImpl::hc_pre(const torch::Tensor& x,
                                   const torch::Tensor& hc_fn,
                                   const torch::Tensor& hc_scale,
                                   const torch::Tensor& hc_base) {
  kernel::HcPreParams params;
  params.x = x;
  params.hc_fn = hc_fn;
  params.hc_scale = hc_scale;
  params.hc_base = hc_base;
  params.hc_mult = hc_mult_;
  params.hc_sinkhorn_iters = hc_sinkhorn_iters_;
  params.norm_eps = norm_eps_;
  params.hc_eps = hc_eps_;
  return kernel::hc_pre(params);
}

torch::Tensor DeepseekV4DecoderLayerImpl::hc_post(const torch::Tensor& x,
                                                  const torch::Tensor& residual,
                                                  const torch::Tensor& post,
                                                  const torch::Tensor& comb) {
  kernel::HcPostParams params;
  if (x.dim() == 2 && residual.dim() == 3 && post.dim() == 2 &&
      comb.dim() == 3) {
    params.x = x.unsqueeze(0);
    params.residual = residual.unsqueeze(0);
    params.post = post.unsqueeze(0);
    params.comb = comb.unsqueeze(0);
    return kernel::hc_post(params).squeeze(0);
  }

  params.x = x;
  params.residual = residual;
  params.post = post;
  params.comb = comb;
  return kernel::hc_post(params);
}

}  // namespace layer
}  // namespace xllm
