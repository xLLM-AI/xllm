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
#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/csrc/aten/CustomFunctions.h>
#endif

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "core/layers/common/add_matmul.h"
#include "models/dit/transformers/transformer_qwen_image.h"
#include "models/model_registry.h"

namespace xllm {
namespace joyimage {

// ---------------------------------------------------------------------------
// Rotary position embedding (batched [B, S, D] cos/sin)
// ---------------------------------------------------------------------------
// Mirrors diffusers `_apply_rotary_emb_batched`:
//   x_out = x * cos + rotate_half(x) * sin
// where rotate_half interleaves pairs: [-x2, x1] over the last dim.
// x:        [B, S, H, D]
// cos/sin:  [B, S, D]  (broadcast over heads -> [B, S, 1, D])
inline torch::Tensor apply_rotary_emb_batched(const torch::Tensor& x,
                                              const torch::Tensor& cos,
                                              const torch::Tensor& sin) {
  auto cos_b = cos.unsqueeze(2).to(torch::kFloat32);  // [B, S, 1, D]
  auto sin_b = sin.unsqueeze(2).to(torch::kFloat32);  // [B, S, 1, D]

  auto x_float = x.to(torch::kFloat32);
  // rotate_half: view last dim as pairs, produce [-x_imag, x_real]
  auto x_pairs = x_float.reshape(
      {x_float.size(0), x_float.size(1), x_float.size(2), -1, 2});
  auto x_real = x_pairs.select(-1, 0);
  auto x_imag = x_pairs.select(-1, 1);
  auto rotated =
      torch::stack({-x_imag, x_real}, -1).flatten(3);  // [B, S, H, D]

  auto out = x_float * cos_b + rotated * sin_b;
  return out.to(x.dtype());
}

// Joint attention over [B, S, H, D] with an optional bool mask
// (True == attend). NPU uses npu_fusion_attention (BSND layout); the fallback
// uses scaled_dot_product_attention.
inline torch::Tensor joyimage_joint_attention(
    const torch::Tensor& query,  // [B, S, H, D]
    const torch::Tensor& key,    // [B, S, H, D]
    const torch::Tensor& value,  // [B, S, H, D]
    int64_t num_heads,
    const torch::Tensor& attn_mask = torch::Tensor()) {
#if defined(USE_NPU)
  // JoyImage uses full bidirectional attention. Do not pass the pipeline-side
  // padding mask to npu_fusion_attention: that mask is [B,1,1,Skv], while the
  // NPU kernel requires an explicit Sq dimension ([B,1,Sq,Skv], [Sq,Skv], ...).
  // This mirrors the QwenImageEditPlus/Wan DiT NPU paths, which run full
  // attention with atten_mask=nullopt.
  auto results = at_npu::native::custom_ops::npu_fusion_attention(
      query,
      key,
      value,
      num_heads,
      /*input_layout=*/"BSND",
      /*pse=*/torch::nullopt,
      /*padding_mask=*/torch::nullopt,
      /*atten_mask=*/torch::nullopt,
      /*scale=*/std::pow(static_cast<double>(query.size(3)), -0.5),
      /*keep_prob=*/1.0,
      /*pre_tockens=*/65535,
      /*next_tockens=*/65535);
  return std::get<0>(results);  // [B, S, H, D]
#else
  auto q = query.transpose(1, 2);  // [B, H, S, D]
  auto k = key.transpose(1, 2);
  auto v = value.transpose(1, 2);
  auto out_dtype = q.dtype();
  c10::optional<torch::Tensor> mask = c10::nullopt;
  if (attn_mask.defined()) {
    mask = attn_mask;  // [B, 1, 1, S] bool broadcasts over heads/query
  }
  auto output = torch::scaled_dot_product_attention(q,
                                                    k,
                                                    v,
                                                    mask,
                                                    /*dropout_p=*/0.0,
                                                    /*is_causal=*/false);
  if (output.dtype() != out_dtype) output = output.to(out_dtype);
  return output.transpose(1, 2).contiguous();  // [B, S, H, D]
#endif
}

// ---------------------------------------------------------------------------
// Wan-style learnable modulation table
// ---------------------------------------------------------------------------
// modulate_table: [1, factor, hidden] learnable parameter.
// forward(x[B, hidden]) -> factor tensors of [B, hidden], from
// (modulate_table + x).chunk(factor).
class ModulateImpl final : public torch::nn::Module {
 public:
  ModulateImpl(int64_t hidden_size, int64_t factor) : factor_(factor) {
    modulate_table_ = register_parameter(
        "modulate_table", torch::zeros({1, factor, hidden_size}));
  }

  std::vector<torch::Tensor> forward(const torch::Tensor& x) {
    // x: [B, hidden] -> [B, 1, hidden]
    auto xin = (x.dim() != 3) ? x.unsqueeze(1) : x;
    auto summed = modulate_table_.to(xin.dtype()) + xin;  // [B, factor, hidden]
    auto chunks = summed.chunk(factor_, /*dim=*/1);
    std::vector<torch::Tensor> out;
    out.reserve(factor_);
    for (auto& c : chunks) {
      out.emplace_back(c.squeeze(1));  // [B, hidden]
    }
    return out;
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "modulate_table", modulate_table_, loaded_);
  }
  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(loaded_) << "weight not loaded for " << prefix + "modulate_table";
  }

 private:
  int64_t factor_;
  torch::Tensor modulate_table_;
  bool loaded_{false};
};
TORCH_MODULE(Modulate);

// PixArt-style text projection: Linear -> gelu_tanh -> Linear.
class TextProjectionImpl final : public torch::nn::Module {
 public:
  TextProjectionImpl(const ModelContext& context,
                     int64_t in_features,
                     int64_t hidden_size)
      : options_(context.get_tensor_options()) {
    linear_1_ = register_module("linear_1",
                                layer::AddMatmulWeightTransposed(
                                    in_features, hidden_size, true, options_));
    linear_2_ = register_module("linear_2",
                                layer::AddMatmulWeightTransposed(
                                    hidden_size, hidden_size, true, options_));
  }

  torch::Tensor forward(const torch::Tensor& caption) {
    auto x = linear_1_->forward(caption);
    x = torch::gelu(x, "tanh");
    x = linear_2_->forward(x);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    linear_1_->load_state_dict(state_dict.get_dict_with_prefix("linear_1."));
    linear_2_->load_state_dict(state_dict.get_dict_with_prefix("linear_2."));
  }
  void verify_loaded_weights(const std::string& prefix) {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  layer::AddMatmulWeightTransposed linear_1_{nullptr};
  layer::AddMatmulWeightTransposed linear_2_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(TextProjection);

// condition_embedder: timesteps -> time_embedder -> (temb, time_proj);
// text_embedder(encoder_hidden_states).
class TimeTextEmbeddingImpl final : public torch::nn::Module {
 public:
  TimeTextEmbeddingImpl(const ModelContext& context,
                        int64_t hidden_size,
                        int64_t time_freq_dim,
                        int64_t time_proj_dim,
                        int64_t text_embed_dim)
      : options_(context.get_tensor_options()) {
    timesteps_proj_ =
        register_module("timesteps_proj",
                        qwenimage::Timesteps(context,
                                             time_freq_dim,
                                             /*flip_sin_to_cos=*/true,
                                             /*downscale_freq_shift=*/0.0,
                                             /*scale=*/1.0));
    time_embedder_ = register_module(
        "time_embedder",
        qwenimage::TimestepEmbedding(context, time_freq_dim, hidden_size));
    time_embedder_->to(torch::kFloat32);
    time_proj_ =
        register_module("time_proj",
                        layer::AddMatmulWeightTransposed(
                            hidden_size, time_proj_dim, true, options_));
    text_embedder_ = register_module(
        "text_embedder", TextProjection(context, text_embed_dim, hidden_size));
  }

  // Returns {temb[B, hidden], timestep_proj[B, time_proj_dim],
  //          text[B, L, hidden]}
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& timestep,
      const torch::Tensor& encoder_hidden_states) {
    auto t = timesteps_proj_->forward(timestep);
    auto temb = time_embedder_->forward(t);  // [B, hidden], computed in FP32
    temb = temb.to(encoder_hidden_states.dtype());
    auto act = torch::silu(temb);
    auto timestep_proj = time_proj_->forward(act);  // [B, time_proj_dim]
    auto text = text_embedder_->forward(encoder_hidden_states);
    return std::make_tuple(temb, timestep_proj, text);
  }

  void load_state_dict(const StateDict& state_dict) {
    time_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("time_embedder."));
    time_proj_->load_state_dict(state_dict.get_dict_with_prefix("time_proj."));
    text_embedder_->load_state_dict(
        state_dict.get_dict_with_prefix("text_embedder."));
  }
  void verify_loaded_weights(const std::string& prefix) {
    time_embedder_->verify_loaded_weights(prefix + "time_embedder.");
    time_proj_->verify_loaded_weights(prefix + "time_proj.");
    text_embedder_->verify_loaded_weights(prefix + "text_embedder.");
  }

  void keep_fp32_modules() { time_embedder_->to(torch::kFloat32); }

 private:
  qwenimage::Timesteps timesteps_proj_{nullptr};
  qwenimage::TimestepEmbedding time_embedder_{nullptr};
  layer::AddMatmulWeightTransposed time_proj_{nullptr};
  TextProjection text_embedder_{nullptr};
  torch::TensorOptions options_;
};
TORCH_MODULE(TimeTextEmbedding);

// Joint attention for a double-stream block. Separate fused QKV for image
// and text streams; RMSNorm on q/k per stream; batched RoPE on image q/k;
// joint SDPA over [image, text]; separate output projections.
class JoyAttentionImpl final : public torch::nn::Module {
 public:
  JoyAttentionImpl(const ModelContext& context,
                   int64_t dim,
                   int64_t num_heads,
                   int64_t head_dim,
                   double eps = 1e-6)
      : options_(context.get_tensor_options()),
        heads_(num_heads),
        head_dim_(head_dim) {
    int64_t inner = num_heads * head_dim;

    img_attn_qkv_ = register_module(
        "img_attn_qkv",
        layer::AddMatmulWeightTransposed(dim, inner * 3, true, options_));
    img_attn_q_norm_ = register_module(
        "img_attn_q_norm", qwenimage::RMSNorm(head_dim, eps, true, false));
    img_attn_k_norm_ = register_module(
        "img_attn_k_norm", qwenimage::RMSNorm(head_dim, eps, true, false));
    img_attn_proj_ = register_module(
        "img_attn_proj",
        layer::AddMatmulWeightTransposed(inner, dim, true, options_));

    txt_attn_qkv_ = register_module(
        "txt_attn_qkv",
        layer::AddMatmulWeightTransposed(dim, inner * 3, true, options_));
    txt_attn_q_norm_ = register_module(
        "txt_attn_q_norm", qwenimage::RMSNorm(head_dim, eps, true, false));
    txt_attn_k_norm_ = register_module(
        "txt_attn_k_norm", qwenimage::RMSNorm(head_dim, eps, true, false));
    txt_attn_proj_ = register_module(
        "txt_attn_proj",
        layer::AddMatmulWeightTransposed(inner, dim, true, options_));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,          // image [B, Si, D]
      const torch::Tensor& encoder_hidden_states,  // text  [B, St, D]
      const torch::Tensor& rope_cos,               // [B, Si, head_dim]
      const torch::Tensor& rope_sin,
      const torch::Tensor& attn_mask = torch::Tensor()) {
    auto img_qkv = img_attn_qkv_->forward(hidden_states);
    auto img_chunks = img_qkv.chunk(3, -1);
    auto txt_qkv = txt_attn_qkv_->forward(encoder_hidden_states);
    auto txt_chunks = txt_qkv.chunk(3, -1);

    auto reshape = std::vector<int64_t>{heads_, -1};
    auto img_q = img_chunks[0].unflatten(-1, reshape);  // [B, Si, H, Dh]
    auto img_k = img_chunks[1].unflatten(-1, reshape);
    auto img_v = img_chunks[2].unflatten(-1, reshape);
    auto txt_q = txt_chunks[0].unflatten(-1, reshape);
    auto txt_k = txt_chunks[1].unflatten(-1, reshape);
    auto txt_v = txt_chunks[2].unflatten(-1, reshape);

    img_q = img_attn_q_norm_->forward(img_q);
    img_k = img_attn_k_norm_->forward(img_k);
    txt_q = txt_attn_q_norm_->forward(txt_q);
    txt_k = txt_attn_k_norm_->forward(txt_k);

    if (rope_cos.defined()) {
      img_q = apply_rotary_emb_batched(img_q, rope_cos, rope_sin);
      img_k = apply_rotary_emb_batched(img_k, rope_cos, rope_sin);
    }

    // Joint order: [image, text]
    auto joint_q = torch::cat({img_q, txt_q}, 1);
    auto joint_k = torch::cat({img_k, txt_k}, 1);
    auto joint_v = torch::cat({img_v, txt_v}, 1);

    auto joint =
        joyimage_joint_attention(joint_q, joint_k, joint_v, heads_, attn_mask);
    joint = joint.flatten(2, 3).to(joint_q.dtype());  // [B, S, inner]

    int64_t si = hidden_states.size(1);
    auto img_out = joint.slice(1, 0, si);
    auto txt_out = joint.slice(1, si, joint.size(1));

    img_out = img_attn_proj_->forward(img_out);
    txt_out = txt_attn_proj_->forward(txt_out);
    return std::make_tuple(img_out, txt_out);
  }

  void load_state_dict(const StateDict& state_dict) {
    img_attn_qkv_->load_state_dict(
        state_dict.get_dict_with_prefix("img_attn_qkv."));
    img_attn_q_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("img_attn_q_norm."));
    img_attn_k_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("img_attn_k_norm."));
    img_attn_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("img_attn_proj."));
    txt_attn_qkv_->load_state_dict(
        state_dict.get_dict_with_prefix("txt_attn_qkv."));
    txt_attn_q_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("txt_attn_q_norm."));
    txt_attn_k_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("txt_attn_k_norm."));
    txt_attn_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("txt_attn_proj."));
  }
  void verify_loaded_weights(const std::string& prefix) {
    img_attn_qkv_->verify_loaded_weights(prefix + "img_attn_qkv.");
    img_attn_q_norm_->verify_loaded_weights(prefix + "img_attn_q_norm.");
    img_attn_k_norm_->verify_loaded_weights(prefix + "img_attn_k_norm.");
    img_attn_proj_->verify_loaded_weights(prefix + "img_attn_proj.");
    txt_attn_qkv_->verify_loaded_weights(prefix + "txt_attn_qkv.");
    txt_attn_q_norm_->verify_loaded_weights(prefix + "txt_attn_q_norm.");
    txt_attn_k_norm_->verify_loaded_weights(prefix + "txt_attn_k_norm.");
    txt_attn_proj_->verify_loaded_weights(prefix + "txt_attn_proj.");
  }

 private:
  torch::TensorOptions options_;
  int64_t heads_;
  int64_t head_dim_;
  layer::AddMatmulWeightTransposed img_attn_qkv_{nullptr};
  qwenimage::RMSNorm img_attn_q_norm_{nullptr};
  qwenimage::RMSNorm img_attn_k_norm_{nullptr};
  layer::AddMatmulWeightTransposed img_attn_proj_{nullptr};
  layer::AddMatmulWeightTransposed txt_attn_qkv_{nullptr};
  qwenimage::RMSNorm txt_attn_q_norm_{nullptr};
  qwenimage::RMSNorm txt_attn_k_norm_{nullptr};
  layer::AddMatmulWeightTransposed txt_attn_proj_{nullptr};
};
TORCH_MODULE(JoyAttention);

// Double-stream transformer block.
class TransformerBlockImpl final : public torch::nn::Module {
 public:
  TransformerBlockImpl(const ModelContext& context,
                       int64_t dim,
                       int64_t num_heads,
                       int64_t head_dim,
                       double mlp_width_ratio,
                       double eps = 1e-6)
      : eps_(eps) {
    int64_t mlp_hidden = static_cast<int64_t>(dim * mlp_width_ratio);

    img_mod_ = register_module("img_mod", Modulate(dim, 6));
    img_mlp_ = register_module(
        "img_mlp", qwenimage::FeedForward(context, dim, dim, /*mult=*/4));
    txt_mod_ = register_module("txt_mod", Modulate(dim, 6));
    txt_mlp_ = register_module(
        "txt_mlp", qwenimage::FeedForward(context, dim, dim, /*mult=*/4));
    attn_ = register_module(
        "attn", JoyAttention(context, dim, num_heads, head_dim, eps));
    (void)mlp_hidden;  // FeedForward uses dim*4 internally (== dim*ratio)
  }

  // FP32 layernorm without affine.
  static torch::Tensor fp32_norm(const torch::Tensor& x, double eps) {
    auto xf = x.to(torch::kFloat32);
    auto out =
        torch::layer_norm(xf, {xf.size(-1)}, /*weight=*/{}, /*bias=*/{}, eps);
    return out.to(x.dtype());
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& temb,  // [B, hidden]
      const torch::Tensor& rope_cos,
      const torch::Tensor& rope_sin,
      const torch::Tensor& attn_mask) {
    auto img_mod = img_mod_->forward(temb);  // 6 x [B, hidden]
    auto txt_mod = txt_mod_->forward(temb);

    auto& img_s1 = img_mod[0];
    auto& img_c1 = img_mod[1];
    auto& img_g1 = img_mod[2];
    auto& img_s2 = img_mod[3];
    auto& img_c2 = img_mod[4];
    auto& img_g2 = img_mod[5];
    auto& txt_s1 = txt_mod[0];
    auto& txt_c1 = txt_mod[1];
    auto& txt_g1 = txt_mod[2];
    auto& txt_s2 = txt_mod[3];
    auto& txt_c2 = txt_mod[4];
    auto& txt_g2 = txt_mod[5];

    auto hs = hidden_states;
    auto ehs = encoder_hidden_states;

    // --- attention ---
    auto img_normed = fp32_norm(hs, eps_);
    auto txt_normed = fp32_norm(ehs, eps_);
    auto img_modulated =
        img_normed * (1 + img_c1.unsqueeze(1)) + img_s1.unsqueeze(1);
    auto txt_modulated =
        txt_normed * (1 + txt_c1.unsqueeze(1)) + txt_s1.unsqueeze(1);

    auto attn_out = attn_->forward(
        img_modulated, txt_modulated, rope_cos, rope_sin, attn_mask);
    auto img_attn = std::get<0>(attn_out);
    auto txt_attn = std::get<1>(attn_out);

    hs = hs + img_attn * img_g1.unsqueeze(1);
    ehs = ehs + txt_attn * txt_g1.unsqueeze(1);

    // --- FFN ---
    auto img_ffn_normed = fp32_norm(hs, eps_);
    auto txt_ffn_normed = fp32_norm(ehs, eps_);
    auto img_ffn_in =
        img_ffn_normed * (1 + img_c2.unsqueeze(1)) + img_s2.unsqueeze(1);
    auto txt_ffn_in =
        txt_ffn_normed * (1 + txt_c2.unsqueeze(1)) + txt_s2.unsqueeze(1);
    auto img_ffn = img_mlp_->forward(img_ffn_in);
    auto txt_ffn = txt_mlp_->forward(txt_ffn_in);
    hs = hs + img_ffn * img_g2.unsqueeze(1);
    ehs = ehs + txt_ffn * txt_g2.unsqueeze(1);

    return std::make_tuple(hs, ehs);
  }

  void load_state_dict(const StateDict& state_dict) {
    img_mod_->load_state_dict(state_dict.get_dict_with_prefix("img_mod."));
    txt_mod_->load_state_dict(state_dict.get_dict_with_prefix("txt_mod."));
    img_mlp_->load_state_dict(state_dict.get_dict_with_prefix("img_mlp."));
    txt_mlp_->load_state_dict(state_dict.get_dict_with_prefix("txt_mlp."));
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
  }
  void verify_loaded_weights(const std::string& prefix) {
    img_mod_->verify_loaded_weights(prefix + "img_mod.");
    txt_mod_->verify_loaded_weights(prefix + "txt_mod.");
    img_mlp_->verify_loaded_weights(prefix + "img_mlp.");
    txt_mlp_->verify_loaded_weights(prefix + "txt_mlp.");
    attn_->verify_loaded_weights(prefix + "attn.");
  }

 private:
  double eps_;
  Modulate img_mod_{nullptr};
  Modulate txt_mod_{nullptr};
  qwenimage::FeedForward img_mlp_{nullptr};
  qwenimage::FeedForward txt_mlp_{nullptr};
  JoyAttention attn_{nullptr};
};
TORCH_MODULE(TransformerBlock);

class JoyImageEditPlusTransformer3DModelImpl : public torch::nn::Module {
 public:
  JoyImageEditPlusTransformer3DModelImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    auto model_args = context.get_model_args();
    hidden_size_ = model_args.hidden_size();
    num_heads_ = model_args.num_attention_heads();
    int64_t num_layers = model_args.num_layers();
    in_channels_ = model_args.in_channels();
    int64_t out_channels = model_args.out_channels();
    out_channels_ = (out_channels > 0) ? out_channels : in_channels_;
    patch_size_ = model_args.wan_patch_size();  // {pt, ph, pw}
    rope_dim_list_ = model_args.rope_dim_list();
    theta_ = model_args.rope_theta_dit();
    double mlp_width_ratio = model_args.mlp_width_ratio();
    int64_t text_dim = model_args.text_dim();

    head_dim_ = hidden_size_ / num_heads_;
    CHECK_EQ(hidden_size_ % num_heads_, 0)
        << "hidden_size must be divisible by num_attention_heads";

    // Conv3d patchifier: kernel == stride == patch_size.
    img_in_ = register_module(
        "img_in",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(
                in_channels_,
                hidden_size_,
                {patch_size_[0], patch_size_[1], patch_size_[2]})
                .stride({patch_size_[0], patch_size_[1], patch_size_[2]})
                .bias(true)));

    condition_embedder_ =
        register_module("condition_embedder",
                        TimeTextEmbedding(context,
                                          hidden_size_,
                                          /*time_freq_dim=*/256,
                                          /*time_proj_dim=*/hidden_size_ * 6,
                                          text_dim));

    double_blocks_ = register_module("double_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers; ++i) {
      auto block = TransformerBlock(
          context, hidden_size_, num_heads_, head_dim_, mlp_width_ratio);
      double_blocks_->push_back(block);
      block_layers_.push_back(block);
    }

    int64_t patch_prod = patch_size_[0] * patch_size_[1] * patch_size_[2];
    proj_out_ = register_module(
        "proj_out",
        layer::AddMatmulWeightTransposed(
            hidden_size_, out_channels_ * patch_prod, true, options_));
  }

  // 3D RoPE cos/sin for a spatial range [start, stop) over (t, h, w).
  // Returns {cos, sin} each [prod(stop-start), head_dim].
  std::pair<torch::Tensor, torch::Tensor> rope_for_range(
      const std::array<int64_t, 3>& start,
      const std::array<int64_t, 3>& stop,
      const torch::Device& device) {
    std::vector<int64_t> dims = rope_dim_list_;
    if (dims.empty()) {
      int64_t d = head_dim_ / 3;
      dims = {d, d, d};
    }
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    std::vector<torch::Tensor> grids;
    for (int i = 0; i < 3; ++i) {
      grids.push_back(torch::arange(start[i], stop[i], fopt));
    }
    auto mesh = torch::meshgrid({grids[0], grids[1], grids[2]}, "ij");

    std::vector<torch::Tensor> cos_parts, sin_parts;
    for (int i = 0; i < 3; ++i) {
      auto pos = mesh[i].reshape({-1});  // [N]
      int64_t dim = dims[i];
      auto idx = torch::arange(0, dim, 2, fopt).slice(0, 0, dim / 2);
      auto freqs = 1.0 / torch::pow(static_cast<double>(theta_), idx / dim);
      auto angles = torch::outer(pos, freqs);  // [N, dim/2]
      cos_parts.push_back(angles.cos().repeat_interleave(2, 1));
      sin_parts.push_back(angles.sin().repeat_interleave(2, 1));
    }
    return {torch::cat(cos_parts, 1), torch::cat(sin_parts, 1)};
  }

  // hidden_states: [B, N, C, pt, ph, pw]
  // timestep:      [B]
  // encoder_hidden_states: [B, L, text_dim]
  // encoder_hidden_states_mask: [B, L] (may be undefined)
  // shape_list: per-sample vector of (t, h, w) tuples (target + refs)
  torch::Tensor forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& timestep,
      const torch::Tensor& encoder_hidden_states,
      const torch::Tensor& encoder_hidden_states_mask,
      const std::vector<std::vector<std::array<int64_t, 3>>>& shape_list) {
    int64_t B = hidden_states.size(0);
    int64_t N = hidden_states.size(1);
    int64_t C = hidden_states.size(2);
    int64_t pt = hidden_states.size(3);
    int64_t ph = hidden_states.size(4);
    int64_t pw = hidden_states.size(5);
    auto device = hidden_states.device();

    // 1. Condition embeddings.
    auto cond = condition_embedder_->forward(timestep, encoder_hidden_states);
    auto vec = std::get<1>(cond);  // [B, hidden*6] -> but we use per-block temb
    auto txt = std::get<2>(cond);  // [B, L, hidden]
    // vec is the projected timestep [B, 6*hidden]; blocks re-add modulate_table
    // to a [B, hidden] signal. Diffusers passes vec.unflatten(1,(6,-1)) then
    // the Modulate table adds to it. To match, feed each block the [B, 6,
    // hidden] signal; our Modulate expects [B, hidden] or [B, 1, hidden]. We
    // therefore pass timestep_proj reshaped to [B, 6, hidden] and let Modulate
    // add table.
    auto temb6 = vec.unflatten(1, std::vector<int64_t>{6, -1});  // [B,6,hidden]

    // 2. Patchify via Conv3d.
    auto x = hidden_states.reshape({B * N, C, pt, ph, pw});
    x = img_in_->forward(x);  // [B*N, hidden, 1, 1, 1]
    auto img = x.reshape({B, N, hidden_size_});

    // 3. Per-sample 3D RoPE with temporal offsets, padded to N.
    std::vector<torch::Tensor> cos_list, sin_list;
    cos_list.reserve(B);
    sin_list.reserve(B);
    for (int64_t b = 0; b < B; ++b) {
      std::vector<torch::Tensor> cparts, sparts;
      int64_t t_off = 0;
      for (const auto& thw : shape_list[b]) {
        std::array<int64_t, 3> start = {t_off, 0, 0};
        std::array<int64_t, 3> stop = {t_off + thw[0], thw[1], thw[2]};
        auto cs = rope_for_range(start, stop, device);
        cparts.push_back(cs.first);
        sparts.push_back(cs.second);
        t_off += thw[0];
      }
      auto c = torch::cat(cparts, 0);
      auto s = torch::cat(sparts, 0);
      int64_t actual = c.size(0);
      if (actual < N) {
        c = torch::constant_pad_nd(c, {0, 0, 0, N - actual}, 1.0);
        s = torch::constant_pad_nd(s, {0, 0, 0, N - actual}, 0.0);
      }
      cos_list.push_back(c);
      sin_list.push_back(s);
    }
    auto rope_cos = torch::stack(cos_list, 0);  // [B, N, head_dim]
    auto rope_sin = torch::stack(sin_list, 0);

    // 4. Attention mask. NPU path uses full attention with no mask (same as
    //    QwenImageEditPlus/Wan DiT); fallback SDPA can still consume the
    //    compact padding mask if needed.
    torch::Tensor attn_mask;
#if !defined(USE_NPU)
    if (encoder_hidden_states_mask.defined()) {
      auto img_mask = torch::zeros(
          {B, N}, torch::TensorOptions().device(device).dtype(torch::kBool));
      for (int64_t b = 0; b < B; ++b) {
        int64_t actual = 0;
        for (const auto& thw : shape_list[b]) {
          actual += thw[0] * thw[1] * thw[2];
        }
        img_mask.index_put_({b, torch::indexing::Slice(0, actual)}, true);
      }
      auto full = torch::cat(
          {img_mask, encoder_hidden_states_mask.to(torch::kBool)}, 1);
      attn_mask = full.unsqueeze(1).unsqueeze(1);  // [B,1,1,N+L]
    }
#endif

    // 5. Blocks. Per-block modulation table added to temb6 -> Modulate handles
    //    it internally, so we feed temb6 (the [B,6,hidden] proj) as the "x".
    for (auto& block : block_layers_) {
      // temb6 is [B, 6, hidden]; Modulate adds its table then chunks by 6,
      // matching diffusers exactly.
      std::tie(img, txt) =
          block->forward(img, txt, temb6, rope_cos, rope_sin, attn_mask);
    }

    // 6. Output projection + reshape to [B, N, C_out, pt, ph, pw].
    img = proj_out_->forward(fp32_norm_out(img));
    img = img.reshape({B, N, pt, ph, pw, out_channels_})
              .permute({0, 1, 5, 2, 3, 4});
    return img;
  }

  static torch::Tensor fp32_norm_out(const torch::Tensor& x) {
    auto xf = x.to(torch::kFloat32);
    auto out =
        torch::layer_norm(xf, {xf.size(-1)}, /*weight=*/{}, /*bias=*/{}, 1e-6);
    return out.to(x.dtype());
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      // Conv3d img_in: load raw weight/bias.
      auto w = state_dict->get_tensor("img_in.weight");
      auto b = state_dict->get_tensor("img_in.bias");
      if (w.defined()) {
        img_in_->weight.data().copy_(w.to(options_));
        img_in_weight_loaded_ = true;
      }
      if (b.defined()) {
        img_in_->bias.data().copy_(b.to(options_));
        img_in_bias_loaded_ = true;
      }
      condition_embedder_->load_state_dict(
          state_dict->get_dict_with_prefix("condition_embedder."));
      proj_out_->load_state_dict(state_dict->get_dict_with_prefix("proj_out."));
      for (size_t i = 0; i < block_layers_.size(); ++i) {
        auto prefix = "double_blocks." + std::to_string(i) + ".";
        block_layers_[i]->load_state_dict(
            state_dict->get_dict_with_prefix(prefix));
      }
    }
    verify_loaded_weights();
    LOG(INFO) << "JoyImageEditPlus transformer loaded successfully.";
  }

  void verify_loaded_weights() {
    CHECK(img_in_weight_loaded_) << "img_in.weight not loaded";
    CHECK(img_in_bias_loaded_) << "img_in.bias not loaded";
    condition_embedder_->verify_loaded_weights("condition_embedder.");
    proj_out_->verify_loaded_weights("proj_out.");
    for (size_t i = 0; i < block_layers_.size(); ++i) {
      block_layers_[i]->verify_loaded_weights("double_blocks." +
                                              std::to_string(i) + ".");
    }
  }

  void keep_fp32_modules() { condition_embedder_->keep_fp32_modules(); }

 private:
  torch::TensorOptions options_;
  int64_t hidden_size_;
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t in_channels_;
  int64_t out_channels_;
  int64_t theta_;
  std::vector<int64_t> patch_size_;
  std::vector<int64_t> rope_dim_list_;

  torch::nn::Conv3d img_in_{nullptr};
  TimeTextEmbedding condition_embedder_{nullptr};
  torch::nn::ModuleList double_blocks_{nullptr};
  std::vector<TransformerBlock> block_layers_;
  layer::AddMatmulWeightTransposed proj_out_{nullptr};
  bool img_in_weight_loaded_{false};
  bool img_in_bias_loaded_{false};
};
TORCH_MODULE(JoyImageEditPlusTransformer3DModel);

REGISTER_MODEL_ARGS(JoyImageEditPlusTransformer3DModel, [&] {
  LOAD_ARG_OR(dtype, "dtype", "bfloat16");
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(num_attention_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(num_layers, "num_layers", 40);
  LOAD_ARG_OR(in_channels, "in_channels", 16);
  LOAD_ARG_OR(out_channels, "out_channels", 16);
  LOAD_ARG_OR(wan_patch_size, "patch_size", (std::vector<int64_t>{1, 2, 2}));
  LOAD_ARG_OR(mlp_width_ratio, "mlp_width_ratio", 4.0);
  LOAD_ARG_OR(text_dim, "text_dim", 4096);
  LOAD_ARG_OR(
      rope_dim_list, "rope_dim_list", (std::vector<int64_t>{16, 56, 56}));
  LOAD_ARG_OR(rope_theta_dit, "theta", 10000);
});

}  // namespace joyimage
}  // namespace xllm
