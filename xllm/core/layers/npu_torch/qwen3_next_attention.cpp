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

#include "qwen3_next_attention.h"

#include <glog/logging.h>

#include <tuple>
#include <vector>

#include "common/flash_comm1_context.h"

namespace xllm {
namespace layer {

Qwen3NextAttentionImpl::Qwen3NextAttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    int32_t layer_id) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  layer_id_ = layer_id;
  rank_ = parallel_args.tp_group_->rank();
  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;

  if (total_num_kv_heads >= tp_size) {
    CHECK(total_num_kv_heads % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK(tp_size % total_num_kv_heads == 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_size / total_num_kv_heads;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  attn_output_gate_ = args.attn_output_gate();
  // 1. QKV linear
  qkv_proj_ = register_module(
      "qkv_proj",
      QKVParallelLinear(args.hidden_size(),
                        attn_output_gate_ ? num_heads_ * 2 : num_heads_,
                        num_kv_heads_,
                        args.head_dim(),
                        num_kv_head_replicas_,
                        /*bias=*/args.attention_bias(),
                        /*gather_output=*/false,
                        parallel_args,
                        options,
                        quant_args));

  // 2. O proj
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  // 3. Q norm
  q_norm_ = register_module(
      "q_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 4. K norm
  k_norm_ = register_module(
      "k_norm", Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 5. Rotary embedding
  const int rotary_dim =
      static_cast<int>(head_dim_ * args.partial_rotary_factor());
  rotary_emb_ =
      register_module("rotary_emb",
                      PartialRotaryEmbedding(rotary_dim,
                                             args.max_position_embeddings(),
                                             args.rope_theta(),
                                             head_dim_,
                                             true,
                                             false,
                                             options));

  // 6. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));

  // 7. Fused split_qkv_rmsnorm_mrope kernel setup
  rotary_dim_ = static_cast<int64_t>(head_dim_ * args.partial_rotary_factor());
  rms_norm_eps_ = args.rms_norm_eps();
  mrope_section_ = args.rope_scaling_mrope_section();
  is_interleaved_ = args.rope_scaling_mrope_interleaved();
  use_fused_qkv_ = false;
  if (attn_output_gate_ && !mrope_section_.empty() &&
      mrope_section_.size() == 3 && rotary_dim_ > 0 &&
      xllm::kernel::has_split_qkv_rmsnorm_mrope_specialization(
          num_heads_, num_kv_heads_, head_dim_)) {
    mrope_gather_pattern_ =
        xllm::kernel::build_split_qkv_rmsnorm_mrope_gather_pattern(
            rotary_dim_, mrope_section_, is_interleaved_, options.device());
    use_fused_qkv_ = true;
    LOG(INFO) << "Qwen3NextAttention layer " << layer_id_
              << ": using fused split_qkv_rmsnorm_mrope kernel";
  }
}

torch::Tensor Qwen3NextAttentionImpl::build_mrope_cos_sin(
    const torch::Tensor& positions) const {
  auto cos_sin_cache = rotary_emb_->get_cos_sin_cache();
  if (positions.dim() == 1) {
    return cos_sin_cache.index_select(0, positions).repeat({1, 3});
  }
  // positions is [3, T] for mRoPE (graph mode or VL)
  // transpose from [3, T] to [T, 3]
  auto positions_t = positions.permute({1, 0}).contiguous();
  auto gathered = cos_sin_cache.index_select(0, positions_t.view({-1}));
  // [T, 3, rope_dim]
  return gathered.view({positions.size(1), -1});
}

torch::Tensor Qwen3NextAttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const torch::Tensor& mrope_cos_sin) {
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  torch::Tensor h = hidden_states;

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    h = gather_sequence(hidden_states, *fc1_ctx);
  }

  auto qkv = qkv_proj_->forward(h);

  if (use_fused_qkv_) {
    const int64_t T = qkv.size(0);
    xllm::kernel::SplitQkvRmsnormMropeParams params;
    params.qkvg = qkv;
    params.q_weight = q_norm_->weight();
    params.k_weight = k_norm_->weight();
    params.cos_sin = mrope_cos_sin;
    params.gather_pattern = mrope_gather_pattern_;
    params.eps = rms_norm_eps_;
    params.num_q_heads = num_heads_;
    params.num_kv_heads = num_kv_heads_;
    params.head_size = head_dim_;

    auto [q, k, v, gate] = xllm::kernel::split_qkv_rmsnorm_mrope(params);

    auto q_flat = q.view({T, q_size_});
    auto k_flat = k.view({T, kv_size_});
    auto v_flat = v.view({T, kv_size_});

    auto out = std::get<0>(
        attn_->forward(attn_metadata, q_flat, k_flat, v_flat, kv_cache));
    out = out * torch::sigmoid(gate.view({T, q_size_}));

    if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
      return o_proj_->forward(out, row_parallel_reduce_mode_for_fc1(*fc1_ctx));
    }
    return o_proj_->forward(out);
  }

  // Fallback path: weight-reordered layout [Q | G | K | V]
  torch::Tensor q, k, v;
  torch::Tensor gate;

  if (attn_output_gate_) {
    q = qkv.slice(-1, 0, q_size_);
    gate = qkv.slice(-1, q_size_, q_size_ * 2);
    k = qkv.slice(-1, q_size_ * 2, q_size_ * 2 + kv_size_);
    v = qkv.slice(-1, q_size_ * 2 + kv_size_, q_size_ * 2 + kv_size_ * 2);
  } else {
    q = qkv.slice(-1, 0, q_size_);
    k = qkv.slice(-1, q_size_, q_size_ + kv_size_);
    v = qkv.slice(-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
  }

  const int64_t T = q.size(0);
  auto q_3d = q.view({T, num_heads_, head_dim_});
  q = std::get<0>(q_norm_->forward(q_3d)).view({T, q_size_});
  auto k_3d = k.view({T, num_kv_heads_, head_dim_});
  k = std::get<0>(k_norm_->forward(k_3d)).view({T, kv_size_});

  rotary_emb_->forward(positions, q, k);
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));

  if (attn_output_gate_) {
    out = out * torch::sigmoid(gate);
  }

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    return o_proj_->forward(out, row_parallel_reduce_mode_for_fc1(*fc1_ctx));
  }
  return o_proj_->forward(out);
}

void Qwen3NextAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});

  if (attn_output_gate_ && qkv_proj_->is_weight_loaded() &&
      !qkv_weight_reordered_) {
    // Rearrange q_proj rows from per-head interleaved [q0,g0,q1,g1,...]
    // to grouped [q0,q1,...,g0,g1,...] so forward output is [Q|G|K|V].
    auto w = qkv_proj_->weight();
    auto qg_rows = w.slice(0, 0, q_size_ * 2);
    const int64_t hidden = w.size(1);
    auto qg_3d = qg_rows.view({num_heads_, 2 * head_dim_, hidden});
    auto q_part = qg_3d.slice(1, 0, head_dim_);
    auto g_part = qg_3d.slice(1, head_dim_, 2 * head_dim_);
    auto reordered = torch::cat(
        {q_part.reshape({q_size_, hidden}), g_part.reshape({q_size_, hidden})},
        0);
    qg_rows.copy_(reordered);

    // Reorder weight_scale and weight_offset for W8A8 dynamic quantization.
    // These are per-channel (per output row) tensors that must match the
    // reordered weight layout for correct dequantization.
    const int64_t qg_size = q_size_ * 2;
    auto reorder_per_channel = [this, qg_size](torch::Tensor tensor) {
      if (!tensor.defined() || tensor.numel() == 0) {
        return;
      }
      auto qg_part = tensor.slice(0, 0, qg_size);
      auto qg_2d = qg_part.view({num_heads_, 2 * head_dim_});
      auto q_scale = qg_2d.slice(1, 0, head_dim_);
      auto g_scale = qg_2d.slice(1, head_dim_, 2 * head_dim_);
      auto reordered_scale = torch::cat(
          {q_scale.reshape({q_size_}), g_scale.reshape({q_size_})}, 0);
      qg_part.copy_(reordered_scale);
    };

    if (qkv_proj_->is_weight_scale_loaded()) {
      reorder_per_channel(qkv_proj_->weight_scale());
    }
    if (qkv_proj_->is_weight_offset_loaded()) {
      reorder_per_channel(qkv_proj_->weight_offset());
    }

    qkv_weight_reordered_ = true;
  }

  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
    q_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
    k_norm_->load_state_dict(StateDict({{"weight", w}}));
  }

  // Gemma RMSNorm uses (1 + w) as the scale factor, but the fused kernel
  // uses standard RMSNorm (w only). Pre-add 1 so the fused kernel produces
  // the same result as Qwen3NextRMSNorm (gemma_rms_norm).
  if (use_fused_qkv_) {
    if (q_norm_->is_weight_loaded() && !q_norm_weight_adjusted_) {
      q_norm_->weight().add_(1.0);
      q_norm_weight_adjusted_ = true;
    }
    if (k_norm_->is_weight_loaded() && !k_norm_weight_adjusted_) {
      k_norm_->weight().add_(1.0);
      k_norm_weight_adjusted_ = true;
    }
  }
}

}  // namespace layer
}  // namespace xllm
