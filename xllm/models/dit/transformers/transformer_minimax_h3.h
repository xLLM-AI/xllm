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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/model_context/dit_model_context.h"
#include "core/framework/model_loader/dit_model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/linear.h"
#include "models/dit/utils/dit_parallel_mixin.h"
#if defined(USE_NPU)
#include "models/dit/utils/dit_block_weight_manager.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#endif
#include "models/dit/utils/timestep_embedding.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"

namespace xllm {
namespace minimax_h3 {

constexpr char kPackedSequenceTensorName[] = "packed";
constexpr int64_t kPackedSequenceAlignment = 64;

// Loads and verifies the weight/bias pair of a plain nn::Linear; shared by
// the transformer's projection layers and the smaller sub-modules.
inline void load_linear(const StateDict& state_dict,
                        const std::string& name,
                        torch::nn::Linear& linear,
                        bool& weight_loaded,
                        bool& bias_loaded) {
  weight::load_weight(
      state_dict, name + ".weight", linear->weight, weight_loaded);
  weight::load_weight(state_dict, name + ".bias", linear->bias, bias_loaded);
}

inline void verify_linear(const std::string& prefix,
                          const std::string& name,
                          bool weight_loaded,
                          bool bias_loaded) {
  CHECK(weight_loaded) << "weight is not loaded for "
                       << prefix + name + ".weight";
  CHECK(bias_loaded) << "bias is not loaded for " << prefix + name + ".bias";
}

// RMSNorm module owning its scale weight; the fp32 math lives in
// dit::rms_norm so non-module callers can share it.
class H3RMSNormImpl final : public torch::nn::Module {
 public:
  H3RMSNormImpl(int64_t dim, double eps) : eps_(eps) {
    weight_ = register_parameter("weight", torch::ones({dim}));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return dit::rms_norm(input, weight_, eps_);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight_, weight_is_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

 private:
  double eps_;
  torch::Tensor weight_;
  bool weight_is_loaded_ = false;
};
TORCH_MODULE(H3RMSNorm);

class H3SwiGLUFFNImpl final : public torch::nn::Module {
 public:
  H3SwiGLUFFNImpl(int64_t hidden_size,
                  int64_t ffn_dim,
                  bool bias,
                  const ModelContext& context) {
    CHECK(!bias) << "MiniMax-H3 SwiGLU checkpoint does not use bias";
    const QuantArgs& quant_args = context.get_quant_args();
    const auto& options = context.get_tensor_options();
    ProcessGroup* tp_group = context.get_parallel_args().dit_tp_group_;
    proj_in_ =
        register_module("proj_in",
                        layer::ColumnParallelLinear(hidden_size,
                                                    2 * ffn_dim,
                                                    bias,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    tp_group,
                                                    options));
    proj_out_ = register_module(
        "proj_out",
        layer::RowParallelLinear(ffn_dim,
                                 hidden_size,
                                 bias,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 tp_group,
                                 options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor x =
        proj_in_->forward(hidden_states.reshape({-1, hidden_states.size(-1)}));
    std::vector<torch::Tensor> chunks = x.chunk(2, -1);
    torch::Tensor activation = torch::silu(chunks[0]) * chunks[1];
    return proj_out_->forward(activation).view(hidden_states.sizes());
  }

  void load_state_dict(const StateDict& state_dict) {
    const StateDict proj_in_state =
        state_dict.get_dict_with_prefix("net.0.proj.");
    const torch::Tensor weight = proj_in_state.get_tensor("weight");
    if (weight.defined()) {
      // Converted checkpoints store [up, gate]. Match the native [gate, up]
      // projected-column order; the fused loader shards each half separately.
      const auto parts = weight.chunk(2, /*dim=*/0);
      proj_in_->load_state_dict(
          StateDict({{"gate.weight", parts[1]}, {"up.weight", parts[0]}}),
          {"gate.", "up."});
    }
    proj_out_->load_state_dict(state_dict.get_dict_with_prefix("net.2."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(proj_in_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "net.0.proj.weight";
    CHECK(proj_out_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "net.2.weight";
  }

 private:
  layer::ColumnParallelLinear proj_in_{nullptr};
  layer::RowParallelLinear proj_out_{nullptr};
};
TORCH_MODULE(H3SwiGLUFFN);

class H3TimestepEmbeddingImpl final : public torch::nn::Module {
 public:
  H3TimestepEmbeddingImpl(int64_t in_dim,
                          int64_t hidden_dim,
                          int64_t out_dim,
                          const ModelContext& context) {
    const auto options = context.get_tensor_options().dtype(torch::kFloat32);
    ProcessGroup* tp_group = context.get_parallel_args().dit_tp_group_;
    linear_1_ =
        register_module("linear_1",
                        layer::ColumnParallelLinear(in_dim,
                                                    hidden_dim,
                                                    /*bias=*/true,
                                                    /*gather_output=*/false,
                                                    QuantArgs(),
                                                    tp_group,
                                                    options));
    linear_2_ = register_module(
        "linear_2",
        layer::RowParallelLinear(hidden_dim,
                                 out_dim,
                                 /*bias=*/true,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 QuantArgs(),
                                 tp_group,
                                 options));
  }

  torch::Tensor forward(const torch::Tensor& sample) {
    return linear_2_->forward(torch::silu(linear_1_->forward(sample)));
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    if (state_dict.get_tensor("linear_1.bias").defined()) {
      linear_1_bias_is_loaded_ = true;
    }
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
    if (state_dict.get_tensor("linear_2.bias").defined()) {
      linear_2_bias_is_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(linear_1_->is_weight_loaded()) << prefix + "linear_1.weight";
    CHECK(linear_1_bias_is_loaded_) << prefix + "linear_1.bias";
    CHECK(linear_2_->is_weight_loaded()) << prefix + "linear_2.weight";
    CHECK(linear_2_bias_is_loaded_) << prefix + "linear_2.bias";
  }

 private:
  layer::ColumnParallelLinear linear_1_{nullptr};
  layer::RowParallelLinear linear_2_{nullptr};
  bool linear_1_bias_is_loaded_ = false;
  bool linear_2_bias_is_loaded_ = false;
};
TORCH_MODULE(H3TimestepEmbedding);

class H3RotaryPosEmbedImpl final : public torch::nn::Module {
 public:
  explicit H3RotaryPosEmbedImpl(int64_t rope_freq_dim = 16,
                                double rope_theta = 10000.0)
      : rope_freq_dim_(rope_freq_dim), rope_theta_(rope_theta) {
    CHECK_GT(rope_freq_dim_, 0);
    CHECK_GT(rope_theta_, 0.0);
    torch::Tensor arange =
        torch::arange(0,
                      2 * rope_freq_dim_,
                      2,
                      torch::TensorOptions().dtype(torch::kFloat32));
    inv_freq_ = register_buffer(
        "inv_freq",
        1.0 / torch::pow(rope_theta_,
                         arange / static_cast<double>(2 * rope_freq_dim_)));
  }

  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& position_ids) const {
    torch::Tensor pos = position_ids.to(torch::kFloat32);
    torch::Tensor freqs =
        pos.unsqueeze(-1) * inv_freq_.to(pos.device()).view({1, 1, -1});
    torch::Tensor freqs_t = freqs.select(1, 0);
    torch::Tensor freqs_h = freqs.select(1, 1);
    torch::Tensor freqs_w = freqs.select(1, 2);
    torch::Tensor cat = torch::cat({freqs_t, freqs_h, freqs_w}, -1);
    cat = torch::cat({cat, cat}, -1);
    return {torch::cos(cat), torch::sin(cat)};
  }

 private:
  int64_t rope_freq_dim_;
  double rope_theta_;
  torch::Tensor inv_freq_;
};
TORCH_MODULE(H3RotaryPosEmbed);

class H3AttentionImpl final : public torch::nn::Module,
                              public dit::SequenceParallelMixin {
 public:
  H3AttentionImpl(int64_t hidden_size,
                  int64_t heads,
                  int64_t dim_head,
                  double qk_norm_eps,
                  const ModelContext& context,
                  ProcessGroup* sp_group)
      : dit::SequenceParallelMixin(/*process_group=*/sp_group),
        dim_head_(dim_head),
        sp_size_(sp_group == nullptr ? 1 : sp_group->world_size()) {
    norm_q_ = register_module("norm_q", H3RMSNorm(dim_head, qk_norm_eps));
    norm_k_ = register_module("norm_k", H3RMSNorm(dim_head, qk_norm_eps));
    ProcessGroup* tp_group = context.get_parallel_args().dit_tp_group_;
    const int64_t tp_size = tp_group == nullptr ? 1 : tp_group->world_size();
    CHECK_EQ(heads % tp_size, 0)
        << "MiniMax-H3 attention heads must be divisible by TP size";
    local_heads_ = heads / tp_size;
    CHECK_EQ(local_heads_ % sp_size_, 0)
        << "MiniMax-H3 local attention heads must be divisible by SP size";
    const int64_t inner_dim = heads * dim_head;
    const QuantArgs& quant_args = context.get_quant_args();
    const torch::TensorOptions& options = context.get_tensor_options();
    to_qkv_ =
        register_module("to_qkv",
                        layer::ColumnParallelLinear(hidden_size,
                                                    3 * inner_dim,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    tp_group,
                                                    options));
    to_out_ = register_module(
        "to_out",
        layer::RowParallelLinear(inner_dim,
                                 hidden_size,
                                 /*bias=*/false,
                                 /*input_is_parallelized=*/true,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 tp_group,
                                 options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& cos = torch::Tensor(),
                        const torch::Tensor& sin = torch::Tensor()) {
    const bool use_communication_overlap =
        DiTConfig::get_instance().dit_sp_communication_overlap();
    const int64_t batch_size = hidden_states.size(0);
    const int64_t local_sequence_length = hidden_states.size(1);
    const std::vector<torch::Tensor> projections =
        to_qkv_->forward(hidden_states.reshape({-1, hidden_states.size(-1)}))
            .view({batch_size, local_sequence_length, -1})
            .chunk(3, -1);
    std::array<torch::Tensor, 3> qkv_tensor{};
    std::array<std::function<torch::Tensor()>, 3> qkv_work{};
    for (size_t index = 0; index < projections.size(); ++index) {
      qkv_tensor[index] = projections[index].contiguous().view(
          {batch_size, local_sequence_length, local_heads_, dim_head_});
      if (sp_size_ > 1) {
        qkv_work[index] = parallel_state::all_to_all_4D(
            qkv_tensor[index],
            /*scatter_idx=*/2,
            /*gather_idx=*/1,
            /*async_ops=*/use_communication_overlap,
            process_group_);
      }
    }

    torch::Tensor& q = qkv_tensor[0];
    torch::Tensor& k = qkv_tensor[1];
    torch::Tensor& v = qkv_tensor[2];
    if (sp_size_ > 1) {
      q = qkv_work[0]();
      k = qkv_work[1]();
      v = qkv_work[2]();
    }
    q = norm_q_->forward(q);
    k = norm_k_->forward(k);
    if (cos.defined() && sin.defined()) {
      // Broadcast [S,D] cos/sin over the head dimension of [B,S,H,D] q/k.
      torch::Tensor cos_b = cos.unsqueeze(1);
      torch::Tensor sin_b = sin.unsqueeze(1);
      q = dit::apply_rotary(q, cos_b, sin_b);
      k = dit::apply_rotary(k, cos_b, sin_b);
    }
#if defined(USE_NPU)
    const int64_t sequence_length = q.size(1);
    const int64_t live_sequence_length =
        sp_size_ > 1
            ? unpad_tensor(q, kPackedSequenceTensorName, /*dim=*/1).size(1)
            : sequence_length;
    const int64_t attention_heads = q.size(2);
    std::vector<c10::SymInt> sequence_ends;
    sequence_ends.reserve(2 * batch_size);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      sequence_ends.emplace_back(batch * sequence_length +
                                 live_sequence_length);
      if (live_sequence_length < sequence_length) {
        sequence_ends.emplace_back((batch + 1) * sequence_length);
      }
    }
    const auto results =
        at_npu::native::custom_ops::npu_fused_infer_attention_score(
            q.reshape({-1, attention_heads, dim_head_}),
            k.reshape({-1, attention_heads, dim_head_}),
            v.reshape({-1, attention_heads, dim_head_}),
            /*pse_shift=*/std::nullopt,
            /*atten_mask=*/std::nullopt,
            /*actual_seq_lengths=*/c10::SymIntArrayRef(sequence_ends),
            /*actual_seq_lengths_kv=*/c10::SymIntArrayRef(sequence_ends),
            /*dequant_scale1=*/std::nullopt,
            /*quant_scale1=*/std::nullopt,
            /*dequant_scale2=*/std::nullopt,
            /*quant_scale2=*/std::nullopt,
            /*quant_offset2=*/std::nullopt,
            /*antiquant_scale=*/std::nullopt,
            /*antiquant_offset=*/std::nullopt,
            /*key_antiquant_scale=*/std::nullopt,
            /*key_antiquant_offset=*/std::nullopt,
            /*value_antiquant_scale=*/std::nullopt,
            /*value_antiquant_offset=*/std::nullopt,
            /*block_table=*/std::nullopt,
            /*query_padding_size=*/std::nullopt,
            /*kv_padding_size=*/std::nullopt,
            /*key_shared_prefix=*/std::nullopt,
            /*value_shared_prefix=*/std::nullopt,
            /*actual_shared_prefix_len=*/std::nullopt,
            /*query_rope=*/std::nullopt,
            /*key_rope=*/std::nullopt,
            /*key_rope_antiquant_scale=*/std::nullopt,
            /*num_heads=*/attention_heads,
            /*scale=*/std::pow(dim_head_, -0.5),
            /*pre_tokens=*/2147483647,
            /*next_tokens=*/2147483647,
            /*input_layout=*/"TND",
            /*num_key_value_heads=*/attention_heads);
    torch::Tensor out = std::get<0>(results).view(
        {batch_size, sequence_length, attention_heads, dim_head_});
#else
    // SDPA has no packed-segment metadata; exclude SP padding explicitly.
    if (sp_size_ > 1) {
      q = unpad_tensor(q, kPackedSequenceTensorName, /*dim=*/1);
      k = unpad_tensor(k, kPackedSequenceTensorName, /*dim=*/1);
      v = unpad_tensor(v, kPackedSequenceTensorName, /*dim=*/1);
    }
    q = q.permute({0, 2, 1, 3}).contiguous();
    k = k.permute({0, 2, 1, 3}).contiguous();
    v = v.permute({0, 2, 1, 3}).contiguous();
    torch::Tensor out = torch::scaled_dot_product_attention(
        q, k, v, std::nullopt, /*dropout_p=*/0.0, /*is_causal=*/false);
    out = out.permute({0, 2, 1, 3}).contiguous();
    if (sp_size_ > 1) {
      out = pad_tensor(out, kPackedSequenceTensorName, /*dim=*/1);
    }
#endif
    if (sp_size_ > 1) {
      auto out_work = parallel_state::all_to_all_4D(out,
                                                    /*scatter_idx=*/1,
                                                    /*gather_idx=*/2,
                                                    /*async_ops=*/false,
                                                    process_group_);
      out = out_work();
    }
    out =
        out.view({batch_size, local_sequence_length, local_heads_ * dim_head_});
    return to_out_->forward(out.reshape({-1, local_heads_ * dim_head_}))
        .view({batch_size, local_sequence_length, -1});
  }

  void load_state_dict(const StateDict& state_dict) {
    to_qkv_->load_state_dict(state_dict, {"to_q.", "to_k.", "to_v."});
    to_out_->load_state_dict(state_dict.get_dict_with_prefix("to_out.0."));
    norm_q_->load_state_dict(state_dict.get_dict_with_prefix("norm_q."));
    norm_k_->load_state_dict(state_dict.get_dict_with_prefix("norm_k."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(to_qkv_->is_weight_loaded())
        << "Q/K/V weights are not loaded for " << prefix;
    CHECK(to_out_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "to_out.0.weight";
    norm_q_->verify_loaded_weights(prefix + "norm_q.");
    norm_k_->verify_loaded_weights(prefix + "norm_k.");
  }

 private:
  int64_t local_heads_;
  int64_t dim_head_;
  int64_t sp_size_;
  layer::ColumnParallelLinear to_qkv_{nullptr};
  layer::RowParallelLinear to_out_{nullptr};
  H3RMSNorm norm_q_{nullptr};
  H3RMSNorm norm_k_{nullptr};
};
TORCH_MODULE(H3Attention);

class H3TokenRefinerBlockImpl final : public torch::nn::Module {
 public:
  H3TokenRefinerBlockImpl(int64_t hidden_size,
                          int64_t heads,
                          int64_t dim_head,
                          int64_t ffn_dim,
                          double norm_eps,
                          double qk_norm_eps,
                          const ModelContext& context) {
    norm1_ = register_module("norm1", H3RMSNorm(hidden_size, norm_eps));
    attn_ = register_module("attn",
                            H3Attention(hidden_size,
                                        heads,
                                        dim_head,
                                        qk_norm_eps,
                                        context,
                                        /*sp_group=*/nullptr));
    norm2_ = register_module("norm2", H3RMSNorm(hidden_size, norm_eps));
    ff_ = register_module("ff",
                          H3SwiGLUFFN(hidden_size, ffn_dim, false, context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor norm1 = norm1_->forward(hidden_states);
    torch::Tensor attention = attn_->forward(norm1);
    torch::Tensor h = hidden_states + attention;
    torch::Tensor norm2 = norm2_->forward(h);
    h = h + ff_->forward(norm2);
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    attn_->verify_loaded_weights(prefix + "attn.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    ff_->verify_loaded_weights(prefix + "ff.");
  }

 private:
  H3RMSNorm norm1_{nullptr};
  H3Attention attn_{nullptr};
  H3RMSNorm norm2_{nullptr};
  H3SwiGLUFFN ff_{nullptr};
};
TORCH_MODULE(H3TokenRefinerBlock);

class H3TokenRefinerImpl final : public torch::nn::Module {
 public:
  H3TokenRefinerImpl(int64_t hidden_size,
                     int64_t heads,
                     int64_t dim_head,
                     int64_t ffn_dim,
                     int64_t layers,
                     double norm_eps,
                     double qk_norm_eps,
                     double final_norm_eps,
                     const ModelContext& context) {
    blocks_ = register_module("refiner_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < layers; ++i) {
      blocks_->push_back(H3TokenRefinerBlock(hidden_size,
                                             heads,
                                             dim_head,
                                             ffn_dim,
                                             norm_eps,
                                             qk_norm_eps,
                                             context));
    }
    final_norm_ =
        register_module("final_norm", H3RMSNorm(hidden_size, final_norm_eps));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    for (auto& block : *blocks_) {
      hidden_states = block->as<H3TokenRefinerBlock>()->forward(hidden_states);
    }
    hidden_states = final_norm_->forward(hidden_states);
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    dit::load_module_list<H3TokenRefinerBlock>(
        *blocks_, state_dict, "refiner_blocks.");
    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("final_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    dit::verify_module_list<H3TokenRefinerBlock>(*blocks_,
                                                 prefix + "refiner_blocks.");
    final_norm_->verify_loaded_weights(prefix + "final_norm.");
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  H3RMSNorm final_norm_{nullptr};
};
TORCH_MODULE(H3TokenRefiner);

class H3AdaLayerNormModulationImpl final : public torch::nn::Module {
 public:
  H3AdaLayerNormModulationImpl(int64_t time_embed_dim,
                               int64_t hidden_size,
                               const ModelContext& context)
      : hidden_size_(hidden_size) {
    linear_ = register_module(
        "linear",
        layer::ColumnParallelLinear(time_embed_dim,
                                    6 * hidden_size * 3,
                                    /*bias=*/true,
                                    /*gather_output=*/true,
                                    context.get_quant_args(),
                                    context.get_parallel_args().dit_tp_group_,
                                    context.get_tensor_options()));
  }

  torch::Tensor forward(const torch::Tensor& temb) {
    torch::Tensor x =
        linear_->forward(torch::silu(temb).to(linear_->weight().scalar_type()));
    return x.view({-1, 6 * hidden_size_});
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(linear_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "linear.weight";
  }

 private:
  int64_t hidden_size_;
  layer::ColumnParallelLinear linear_{nullptr};
};
TORCH_MODULE(H3AdaLayerNormModulation);

class H3TransformerBlockImpl final : public torch::nn::Module {
 public:
  H3TransformerBlockImpl(int64_t hidden_size,
                         int64_t heads,
                         int64_t dim_head,
                         int64_t ffn_dim,
                         int64_t time_embed_dim,
                         double norm_eps,
                         double qk_norm_eps,
                         const ModelContext& context,
                         ProcessGroup* sp_group) {
    norm1_ = register_module("norm1", H3RMSNorm(hidden_size, norm_eps));
    attn_ = register_module(
        "attn",
        H3Attention(
            hidden_size, heads, dim_head, qk_norm_eps, context, sp_group));
    norm2_ = register_module("norm2", H3RMSNorm(hidden_size, norm_eps));
    ff_ = register_module("ff",
                          H3SwiGLUFFN(hidden_size, ffn_dim, false, context));
    adaln_ = register_module(
        "adaln_proj",
        H3AdaLayerNormModulation(time_embed_dim, hidden_size, context));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb,
                        const torch::Tensor& adaln_indices,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin) {
    torch::Tensor modulation =
        adaln_->forward(temb).index_select(0, adaln_indices).unsqueeze(0);
    auto chunks = modulation.chunk(6, -1);
    const torch::Tensor& shift_msa = chunks[0];
    const torch::Tensor& scale_msa = chunks[1];
    const torch::Tensor& gate_msa = chunks[2];
    const torch::Tensor& shift_mlp = chunks[3];
    const torch::Tensor& scale_mlp = chunks[4];
    const torch::Tensor& gate_mlp = chunks[5];

    torch::Tensor norm_h = norm1_->forward(hidden_states);
    norm_h = norm_h * (1.0 + scale_msa) + shift_msa;
    torch::Tensor h =
        hidden_states + gate_msa * attn_->forward(norm_h, cos, sin);
    norm_h = norm2_->forward(h);
    norm_h = norm_h * (1.0 + scale_mlp) + shift_mlp;
    h = h + gate_mlp * ff_->forward(norm_h);
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    ff_->load_state_dict(state_dict.get_dict_with_prefix("ff."));
    adaln_->load_state_dict(state_dict.get_dict_with_prefix("adaln_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm1_->verify_loaded_weights(prefix + "norm1.");
    attn_->verify_loaded_weights(prefix + "attn.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
    ff_->verify_loaded_weights(prefix + "ff.");
    adaln_->verify_loaded_weights(prefix + "adaln_proj.");
  }

#if defined(USE_NPU)
  void build_weight_loader() { weight_loader_.build_from_module(*this); }

  dit::BlockWeightLoader& weight_loader() { return weight_loader_; }
#endif

 private:
  H3RMSNorm norm1_{nullptr};
  H3Attention attn_{nullptr};
  H3RMSNorm norm2_{nullptr};
  H3SwiGLUFFN ff_{nullptr};
  H3AdaLayerNormModulation adaln_{nullptr};
#if defined(USE_NPU)
  dit::BlockWeightLoader weight_loader_;
#endif
};
TORCH_MODULE(H3TransformerBlock);

class H3AdaLayerNormOutImpl final : public torch::nn::Module {
 public:
  H3AdaLayerNormOutImpl(int64_t hidden_size,
                        int64_t time_embed_dim,
                        double eps,
                        const ModelContext& context) {
    norm_ = register_module("norm", H3RMSNorm(hidden_size, eps));
    linear_ = register_module(
        "linear",
        layer::ColumnParallelLinear(time_embed_dim,
                                    2 * hidden_size,
                                    /*bias=*/true,
                                    /*gather_output=*/true,
                                    context.get_quant_args(),
                                    context.get_parallel_args().dit_tp_group_,
                                    context.get_tensor_options()));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& temb,
                        const torch::Tensor& timestep_indices) {
    torch::Tensor x =
        linear_->forward(torch::silu(temb).to(linear_->weight().scalar_type()));
    auto chunks = x.index_select(0, timestep_indices).unsqueeze(0).chunk(2, -1);
    const torch::Tensor& shift = chunks[0];
    const torch::Tensor& scale = chunks[1];
    torch::Tensor h = norm_->forward(hidden_states);
    return h * (1.0 + scale) + shift;
  }

  void load_state_dict(const StateDict& state_dict) {
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
    if (state_dict.get_tensor("linear.bias").defined()) {
      linear_bias_is_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) {
    norm_->verify_loaded_weights(prefix + "norm.");
    CHECK(linear_->is_weight_loaded()) << prefix + "linear.weight";
    CHECK(linear_bias_is_loaded_) << prefix + "linear.bias";
  }

 private:
  H3RMSNorm norm_{nullptr};
  layer::ColumnParallelLinear linear_{nullptr};
  bool linear_bias_is_loaded_ = false;
};
TORCH_MODULE(H3AdaLayerNormOut);

}  // namespace minimax_h3

struct MiniMaxH3TransformerOutput {
  torch::Tensor sample;
  torch::Tensor audio_sample;
};

struct MiniMaxH3TransformerConditioning {
  torch::Tensor text_embeds;
  std::pair<torch::Tensor, torch::Tensor> rotary;
};

class MiniMaxH3Transformer3DModelImpl final
    : public torch::nn::Module,
      public dit::SequenceParallelMixin {
 public:
  explicit MiniMaxH3Transformer3DModelImpl(const DiTModelContext& context)
      : dit::SequenceParallelMixin(
            /*process_group=*/context.get_parallel_args().dit_sp_group_),
        options_(context.get_tensor_options()) {
    using namespace minimax_h3;
    const ParallelArgs& parallel_args = context.get_parallel_args();
    CHECK(parallel_args.tp_size() == 1 ||
          parallel_args.dit_tp_group_ != nullptr)
        << "MiniMax-H3 TP requires a DiT TP process group";
    CHECK(parallel_args.sp_size() == 1 ||
          parallel_args.dit_sp_group_ != nullptr)
        << "MiniMax-H3 SP requires a DiT SP process group";
    use_rolling_load_ = dit_rolling_load_enabled(parallel_args);
    const ModelArgs& model_args = context.get_model_args("transformer");
    const int64_t heads = model_args.n_heads();
    const int64_t head_dim = model_args.head_dim();
    hidden_size_ = model_args.hidden_size();
    const int64_t num_layers = model_args.num_layers();
    const int64_t num_refiner_layers = model_args.num_refiner_layers();
    const int64_t ffn_dim = model_args.ffn_dim();
    const int64_t in_channels = model_args.in_channels();
    audio_in_channels_ = model_args.audio_in_channels();
    const std::vector<int64_t> patch_size = model_args.wan_patch_size();
    const int64_t text_dim = model_args.text_dim();
    freq_dim_ = model_args.time_freq_dim();
    const int64_t time_embed_hidden_dim = model_args.time_embed_hidden_dim();
    const int64_t time_embed_dim = model_args.time_embed_dim();
    const int64_t rope_freq_dim = model_args.rope_freq_dim();
    const double rope_theta = model_args.rope_theta();
    const double norm_eps = model_args.norm_eps();
    const double qk_norm_eps = model_args.qk_norm_eps();
    const double final_norm_eps = model_args.final_norm_eps();
    CHECK_EQ(patch_size.size(), static_cast<size_t>(3))
        << "MiniMax-H3 transformer patch_size must have 3 dimensions";
    CHECK_GT(heads, 0);
    CHECK_GT(head_dim, 0);
    CHECK_GT(hidden_size_, 0);
    CHECK_GT(num_layers, 0);
    CHECK_GT(num_refiner_layers, 0);
    CHECK_GT(ffn_dim, 0);
    CHECK_GT(in_channels, 0);
    CHECK_GT(audio_in_channels_, 0);
    CHECK_GT(text_dim, 0);
    CHECK_GT(freq_dim_, 0);
    CHECK_GT(time_embed_hidden_dim, 0);
    CHECK_GT(time_embed_dim, 0);
    CHECK_GT(rope_freq_dim, 0);
    CHECK_EQ(heads % (parallel_args.tp_size() * parallel_args.sp_size()), 0)
        << "MiniMax-H3 attention heads must be divisible by TP size times SP "
           "size";
    video_patch_dim_ =
        in_channels * patch_size[0] * patch_size[1] * patch_size[2];
    CHECK_EQ(video_patch_dim_ % parallel_args.tp_size(), 0)
        << "MiniMax-H3 video projection dimension must be divisible by TP size";
    CHECK_EQ(audio_in_channels_ % parallel_args.tp_size(), 0)
        << "MiniMax-H3 audio projection dimension must be divisible by TP size";
    ModelContext transformer_context(parallel_args,
                                     model_args,
                                     context.get_quant_args("transformer"),
                                     options_);
    const auto register_linear =
        [this](const char* name, int64_t in_dim, int64_t out_dim) {
          return register_module(
              name,
              torch::nn::Linear(
                  torch::nn::LinearOptions(in_dim, out_dim).bias(true)));
        };
    const auto register_projection = [this,
                                      &transformer_context,
                                      &parallel_args](const char* name,
                                                      int64_t in_dim,
                                                      int64_t out_dim,
                                                      bool gather_output) {
      return register_module(
          name,
          layer::ColumnParallelLinear(
              in_dim,
              out_dim,
              /*bias=*/true,
              gather_output,
              QuantArgs(),
              parallel_args.dit_tp_group_,
              transformer_context.get_tensor_options().dtype(torch::kFloat32)));
    };
    proj_in_ = register_projection("proj_in",
                                   video_patch_dim_,
                                   hidden_size_,
                                   /*gather_output=*/true);
    audio_proj_in_ = register_projection("audio_proj_in",
                                         audio_in_channels_,
                                         hidden_size_,
                                         /*gather_output=*/true);
    context_embedder_ =
        register_linear("context_embedder", text_dim, hidden_size_);
    time_embedder_ = register_module("time_embedder",
                                     H3TimestepEmbedding(freq_dim_,
                                                         time_embed_hidden_dim,
                                                         time_embed_dim,
                                                         transformer_context));
    rope_ =
        register_module("rope", H3RotaryPosEmbed(rope_freq_dim, rope_theta));
    token_refiner_ = register_module("token_refiner",
                                     H3TokenRefiner(hidden_size_,
                                                    heads,
                                                    head_dim,
                                                    ffn_dim,
                                                    num_refiner_layers,
                                                    norm_eps,
                                                    qk_norm_eps,
                                                    final_norm_eps,
                                                    transformer_context));
    blocks_ = register_module("transformer_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers; ++i) {
      H3TransformerBlock block(hidden_size_,
                               heads,
                               head_dim,
                               ffn_dim,
                               time_embed_dim,
                               norm_eps,
                               qk_norm_eps,
                               transformer_context,
                               parallel_args.dit_sp_group_);
      if (use_rolling_load_) {
        // Keep only the packed host weights until the rolling manager loads a
        // block into its device slot.
        block->to(torch::kCPU);
      }
      blocks_->push_back(block);
    }
    norm_out_ = register_module(
        "norm_out",
        H3AdaLayerNormOut(
            hidden_size_, time_embed_dim, final_norm_eps, transformer_context));
    proj_out_ = register_projection("proj_out",
                                    hidden_size_,
                                    video_patch_dim_,
                                    /*gather_output=*/false);
    audio_proj_out_ = register_projection("audio_proj_out",
                                          hidden_size_,
                                          audio_in_channels_,
                                          /*gather_output=*/false);
  }

  static bool dit_rolling_load_enabled(const ParallelArgs& parallel_args) {
#if defined(USE_NPU)
    return LoadConfig::get_instance().enable_rolling_load() &&
           parallel_args.tp_size() == 1 &&
           !LoadConfig::get_instance().enable_vmm_weights();
#else
    (void)parallel_args;
    return false;
#endif
  }

  bool rolling_load_enabled() const { return use_rolling_load_; }

#if defined(USE_NPU)
  // Pack only after apply_reference_precision has established the final dtype.
  void prepare_for_rolling_load() {
    for (auto& block : *blocks_) {
      block->as<minimax_h3::H3TransformerBlock>()->build_weight_loader();
    }
  }

  // Non-owning pointers remain valid while this transformer's blocks exist.
  std::vector<dit::BlockWeightLoader*> get_block_weight_loaders() {
    std::vector<dit::BlockWeightLoader*> loaders;
    loaders.reserve(blocks_->size());
    for (auto& block : *blocks_) {
      loaders.push_back(
          &block->as<minimax_h3::H3TransformerBlock>()->weight_loader());
    }
    return loaders;
  }
#endif

  void apply_reference_precision(const torch::Device& device) {
    proj_in_->to(device, torch::kFloat32);
    audio_proj_in_->to(device, torch::kFloat32);
    time_embedder_->to(device, torch::kFloat32);
    rope_->to(device, torch::kFloat32);
    proj_out_->to(device, torch::kFloat32);
    audio_proj_out_->to(device, torch::kFloat32);

    context_embedder_->to(device, torch::kBFloat16);
    token_refiner_->to(device, torch::kBFloat16);
    blocks_->to(use_rolling_load_ ? torch::Device(torch::kCPU) : device,
                torch::kBFloat16);
    norm_out_->to(device, torch::kBFloat16);
    attention_dtype_ = torch::kBFloat16;

    validate_module_dtype("proj_in", *proj_in_, torch::kFloat32);
    validate_module_dtype("audio_proj_in", *audio_proj_in_, torch::kFloat32);
    validate_module_dtype("time_embedder", *time_embedder_, torch::kFloat32);
    validate_module_dtype("rope", *rope_, torch::kFloat32);
    validate_module_dtype("proj_out", *proj_out_, torch::kFloat32);
    validate_module_dtype("audio_proj_out", *audio_proj_out_, torch::kFloat32);
    validate_module_dtype(
        "context_embedder", *context_embedder_, torch::kBFloat16);
    validate_module_dtype("token_refiner", *token_refiner_, torch::kBFloat16);
    validate_module_dtype("transformer_blocks", *blocks_, torch::kBFloat16);
    validate_module_dtype("norm_out", *norm_out_, torch::kBFloat16);
  }

  MiniMaxH3TransformerConditioning prepare_conditioning(
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& position_ids) {
    // The ATen operator fuses bias for packed 3-D inputs, matching the
    // reference BF16 rounding instead of adding bias after a rounded matmul.
    torch::Tensor text_embeds = torch::linear(
        encoder_hidden_states.to(context_embedder_->weight.scalar_type()),
        context_embedder_->weight,
        context_embedder_->bias);
    text_embeds = token_refiner_->forward(text_embeds);
    torch::Tensor rotary_positions = position_ids;
    if (process_group_ != nullptr && process_group_->world_size() > 1) {
      const int64_t alignment =
          std::lcm(static_cast<int64_t>(process_group_->world_size()),
                   minimax_h3::kPackedSequenceAlignment);
      const int64_t padding =
          (alignment - position_ids.size(0) % alignment) % alignment;
      rotary_positions =
          torch::cat({position_ids,
                      torch::zeros({padding, position_ids.size(1)},
                                   position_ids.options())},
                     /*dim=*/0);
    }
    std::pair<torch::Tensor, torch::Tensor> rotary =
        rope_->forward(rotary_positions);
    // Attention blocks keep a single dtype for the whole denoising pass.
    // Convert the cache once here instead of once for Q and K in every block.
    rotary.first = rotary.first.to(attention_dtype_);
    rotary.second = rotary.second.to(attention_dtype_);
    return {std::move(text_embeds), std::move(rotary)};
  }

  MiniMaxH3TransformerOutput forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& audio_hidden_states,
      const MiniMaxH3TransformerConditioning& conditioning,
      const torch::Tensor& timestep,
      const torch::Tensor& timestep_indices,
      const torch::Tensor& position_ids,
      const torch::Tensor& token_tags,
      const torch::Tensor& video_indices,
      const torch::Tensor& audio_indices,
      const torch::Tensor& text_indices,
      std::function<void(int32_t)> before_layer_cb = nullptr,
      std::function<void(int32_t)> after_layer_cb = nullptr) {
    const torch::Tensor& text_embeds = conditioning.text_embeds;
    const int64_t batch = text_embeds.size(0);
    const int64_t sequence_length = position_ids.size(0);
    CHECK_EQ(
        sequence_length,
        video_indices.numel() + audio_indices.numel() + text_indices.numel());
    torch::Tensor packed = torch::zeros({batch, sequence_length, hidden_size_},
                                        text_embeds.options());
    packed.index_copy_(1, text_indices, text_embeds);
    const auto embed_media = [this,
                              batch,
                              &hidden_states,
                              &audio_hidden_states,
                              &video_indices,
                              &audio_indices](torch::Tensor local_packed,
                                              int64_t row_start) {
      const int64_t row_stop = row_start + local_packed.size(1);
      const torch::Tensor video_sources =
          torch::nonzero((video_indices >= row_start) &
                         (video_indices < row_stop))
              .flatten();
      const torch::Tensor audio_sources =
          torch::nonzero((audio_indices >= row_start) &
                         (audio_indices < row_stop))
              .flatten();
      if (video_sources.numel() > 0) {
        const torch::Tensor rows = hidden_states.index_select(1, video_sources)
                                       .reshape({-1, video_patch_dim_})
                                       .to(torch::kFloat32);
        const torch::Tensor embedded = proj_in_->forward(rows).view(
            {batch, video_sources.numel(), hidden_size_});
        local_packed.index_copy_(
            1,
            video_indices.index_select(0, video_sources) - row_start,
            embedded.to(local_packed.scalar_type()));
      }
      if (audio_sources.numel() > 0) {
        const torch::Tensor rows =
            audio_hidden_states.index_select(1, audio_sources)
                .reshape({-1, audio_in_channels_})
                .to(torch::kFloat32);
        const torch::Tensor embedded = audio_proj_in_->forward(rows).view(
            {batch, audio_sources.numel(), hidden_size_});
        local_packed.index_copy_(
            1,
            audio_indices.index_select(0, audio_sources) - row_start,
            embedded.to(local_packed.scalar_type()));
      }
      return local_packed;
    };
    torch::Tensor timestep_proj =
        dit::get_timestep_embedding(timestep.flatten(),
                                    freq_dim_,
                                    /*flip_sin_to_cos=*/true,
                                    /*downscale_freq_shift=*/0.0);
    torch::Tensor temb = time_embedder_->forward(timestep_proj);
    torch::Tensor timestep_indices_long = timestep_indices.to(torch::kLong);
    torch::Tensor adaln_indices =
        timestep_indices_long * 3 + token_tags.to(torch::kLong);
    auto run_blocks =
        [this, &temb, &conditioning, &before_layer_cb, &after_layer_cb](
            torch::Tensor local_packed,
            const torch::Tensor& local_adaln_indices,
            const torch::Tensor& local_timestep_indices) {
          int32_t block_index = 0;
          for (auto& block : *blocks_) {
            if (before_layer_cb) {
              before_layer_cb(block_index);
            }
            local_packed = block->as<minimax_h3::H3TransformerBlock>()->forward(
                local_packed,
                temb,
                local_adaln_indices,
                conditioning.rotary.first,
                conditioning.rotary.second);
            if (after_layer_cb) {
              after_layer_cb(block_index);
            }
            ++block_index;
          }
          local_packed =
              norm_out_->forward(local_packed, temb, local_timestep_indices);
          return local_packed;
        };
    // Preserve the reference's TP-local full-row FP32 GEMM geometry before
    // selecting media rows; its rounding affects the denoising trajectory.
    const auto project_heads = [this,
                                batch](const torch::Tensor& local_packed) {
      const torch::Tensor head_input =
          local_packed.to(torch::kFloat32).reshape({-1, hidden_size_});
      const torch::Tensor video = proj_out_->forward(head_input)
                                      .view({batch, local_packed.size(1), -1});
      const torch::Tensor audio = audio_proj_out_->forward(head_input)
                                      .view({batch, local_packed.size(1), -1});
      return torch::cat({video, audio}, /*dim=*/-1);
    };
    const bool use_sequence_parallel =
        process_group_ != nullptr && process_group_->world_size() > 1;
    if (use_sequence_parallel) {
      dit::SequenceParallelTensorMap outputs = sequence_parallel_forward(
          {{minimax_h3::kPackedSequenceTensorName, {packed, /*dim=*/1}},
           {"adaln_indices", {adaln_indices, /*dim=*/0}},
           {"timestep_indices", {timestep_indices_long, /*dim=*/0}}},
          [this, &embed_media, &run_blocks, &project_heads](
              const dit::SequenceParallelTensorMap& inputs) {
            torch::Tensor local =
                inputs.at(minimax_h3::kPackedSequenceTensorName).first;
            const int64_t row_start = process_group_->rank() * local.size(1);
            local = embed_media(local, row_start);
            local = run_blocks(local,
                               inputs.at("adaln_indices").first,
                               inputs.at("timestep_indices").first);
            return dit::SequenceParallelTensorMap{
                {minimax_h3::kPackedSequenceTensorName,
                 {project_heads(local), /*dim=*/1}}};
          },
          minimax_h3::kPackedSequenceAlignment);
      packed = outputs.at(minimax_h3::kPackedSequenceTensorName).first;
    } else {
      packed = embed_media(packed, /*row_start=*/0);
      packed = project_heads(
          run_blocks(packed, adaln_indices, timestep_indices_long));
    }
    const int64_t video_width = proj_out_->weight().size(0);
    torch::Tensor video =
        packed.slice(-1, 0, video_width).index_select(1, video_indices);
    torch::Tensor audio =
        packed.slice(-1, video_width).index_select(1, audio_indices);
    ProcessGroup* tp_group = proj_out_->process_group();
    if (tp_group != nullptr && tp_group->world_size() > 1) {
      video = parallel_state::gather(video.contiguous(), tp_group);
      audio = parallel_state::gather(audio.contiguous(), tp_group);
    }
    return {video, audio};
  }

  void load_state_dict(const StateDict& state_dict) {
    const auto load_projection = [&state_dict](
                                     layer::ColumnParallelLinear& projection,
                                     const std::string& name,
                                     bool& bias_loaded) {
      projection->load_state_dict(state_dict.get_dict_with_prefix(name + "."));
      if (state_dict.get_tensor(name + ".bias").defined()) {
        bias_loaded = true;
      }
    };
    load_projection(proj_in_, "proj_in", proj_in_bias_loaded_);
    load_projection(
        audio_proj_in_, "audio_proj_in", audio_proj_in_bias_loaded_);
    minimax_h3::load_linear(state_dict,
                            "context_embedder",
                            context_embedder_,
                            context_embedder_weight_loaded_,
                            context_embedder_bias_loaded_);
    time_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("time_embedder."));
    token_refiner_->load_state_dict(
        state_dict.get_dict_with_prefix("token_refiner."));
    dit::load_module_list<minimax_h3::H3TransformerBlock>(
        *blocks_, state_dict, "transformer_blocks.");
    norm_out_->load_state_dict(state_dict.get_dict_with_prefix("norm_out."));
    load_projection(proj_out_, "proj_out", proj_out_bias_loaded_);
    load_projection(
        audio_proj_out_, "audio_proj_out", audio_proj_out_bias_loaded_);
  }

  void verify_loaded_weights(const std::string& prefix) {
    CHECK(proj_in_->is_weight_loaded()) << prefix + "proj_in.weight";
    CHECK(proj_in_bias_loaded_) << prefix + "proj_in.bias";
    CHECK(audio_proj_in_->is_weight_loaded())
        << prefix + "audio_proj_in.weight";
    CHECK(audio_proj_in_bias_loaded_) << prefix + "audio_proj_in.bias";
    minimax_h3::verify_linear(prefix,
                              "context_embedder",
                              context_embedder_weight_loaded_,
                              context_embedder_bias_loaded_);
    time_embedder_->verify_loaded_weights(prefix + "time_embedder.");
    token_refiner_->verify_loaded_weights(prefix + "token_refiner.");
    dit::verify_module_list<minimax_h3::H3TransformerBlock>(
        *blocks_, prefix + "transformer_blocks.");
    norm_out_->verify_loaded_weights(prefix + "norm_out.");
    CHECK(proj_out_->is_weight_loaded()) << prefix + "proj_out.weight";
    CHECK(proj_out_bias_loaded_) << prefix + "proj_out.bias";
    CHECK(audio_proj_out_->is_weight_loaded())
        << prefix + "audio_proj_out.weight";
    CHECK(audio_proj_out_bias_loaded_) << prefix + "audio_proj_out.bias";
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    CHECK(loader != nullptr)
        << "MiniMax-H3 transformer loader must not be null";
    for (const auto& state_dict_ptr : loader->get_state_dicts()) {
      load_state_dict(*state_dict_ptr);
    }
    verify_loaded_weights("");
  }

 private:
  static void validate_module_dtype(const std::string& module_name,
                                    const torch::nn::Module& module,
                                    torch::ScalarType expected_dtype) {
    for (const auto& parameter : module.named_parameters(/*recurse=*/true)) {
      CHECK_EQ(parameter.value().scalar_type(), expected_dtype)
          << "MiniMax-H3 parameter dtype mismatch: " << module_name << '.'
          << parameter.key();
    }
    for (const auto& buffer : module.named_buffers(/*recurse=*/true)) {
      if (!buffer.value().is_floating_point()) {
        continue;
      }
      CHECK_EQ(buffer.value().scalar_type(), expected_dtype)
          << "MiniMax-H3 buffer dtype mismatch: " << module_name << '.'
          << buffer.key();
    }
  }

  torch::TensorOptions options_;
  torch::ScalarType attention_dtype_ = torch::kFloat32;
  bool use_rolling_load_ = false;
  int64_t hidden_size_;
  int64_t freq_dim_;
  int64_t audio_in_channels_;
  int64_t video_patch_dim_;
  layer::ColumnParallelLinear proj_in_{nullptr};
  bool proj_in_bias_loaded_ = false;
  layer::ColumnParallelLinear audio_proj_in_{nullptr};
  bool audio_proj_in_bias_loaded_ = false;
  torch::nn::Linear context_embedder_{nullptr};
  minimax_h3::H3TimestepEmbedding time_embedder_{nullptr};
  minimax_h3::H3RotaryPosEmbed rope_{nullptr};
  minimax_h3::H3TokenRefiner token_refiner_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  minimax_h3::H3AdaLayerNormOut norm_out_{nullptr};
  layer::ColumnParallelLinear proj_out_{nullptr};
  bool proj_out_bias_loaded_ = false;
  layer::ColumnParallelLinear audio_proj_out_{nullptr};
  bool audio_proj_out_bias_loaded_ = false;
  bool context_embedder_weight_loaded_ = false;
  bool context_embedder_bias_loaded_ = false;
};
TORCH_MODULE(MiniMaxH3Transformer3DModel);

REGISTER_MODEL_ARGS(MiniMaxH3Transformer3DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(n_heads, "num_attention_heads", 56);
  LOAD_ARG_OR(head_dim, "attention_head_dim", 128);
  LOAD_ARG_OR(hidden_size, "hidden_size", 5376);
  LOAD_ARG_OR(num_layers, "num_layers", 50);
  LOAD_ARG_OR(num_refiner_layers, "num_refiner_layers", 2);
  LOAD_ARG_OR(ffn_dim, "ffn_dim", 14336);
  LOAD_ARG_OR(in_channels, "in_channels", 24);
  LOAD_ARG_OR(audio_in_channels, "audio_in_channels", 32);
  LOAD_ARG_OR(wan_patch_size, "patch_size", (std::vector<int64_t>{1, 2, 2}));
  LOAD_ARG_OR(text_dim, "text_dim", 5120);
  LOAD_ARG_OR(time_freq_dim, "freq_dim", 256);
  LOAD_ARG_OR(time_embed_hidden_dim, "time_embed_hidden_dim", 5376);
  LOAD_ARG_OR(time_embed_dim, "time_embed_dim", 2688);
  LOAD_ARG_OR(rope_freq_dim, "rope_freq_dim", 16);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(norm_eps, "norm_eps", 1e-5f);
  LOAD_ARG_OR(qk_norm_eps, "qk_norm_eps", 1e-5f);
  LOAD_ARG_OR(final_norm_eps, "final_norm_eps", 1e-5f);
});

}  // namespace xllm
