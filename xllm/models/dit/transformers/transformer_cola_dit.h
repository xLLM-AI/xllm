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

#include <torch/torch.h>

#if defined(USE_CUDA)
#include <ATen/autocast_mode.h>
#endif

#include <cmath>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "models/dit/utils/cola_block_causal_mask.h"
#include "models/dit/utils/cola_weight_loader.h"
#include "models/model_registry.h"

namespace xllm {

// ---------------------------------------------------------------------------
// Sinusoidal Timestep Embedding
// ---------------------------------------------------------------------------

// Matches diffusers convention: flip_sin_to_cos=False, downscale_freq_shift=0.
// Denominator is half_dim, NOT half_dim-1.
inline torch::Tensor get_sinusoidal_embedding(const torch::Tensor& timesteps,
                                              int64_t embedding_dim) {
  int64_t half_dim = embedding_dim / 2;
  // exponent[j] = -log(10000) * j / half_dim  (log of frequency)
  auto exponent = -std::log(10000.0f) *
                  torch::arange(0,
                                half_dim,
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(timesteps.device())) /
                  static_cast<float>(half_dim);
  // freq[j] = exp(exponent[j]) = 10000^(-j/half_dim)
  // emb[t, j] = t * freq[j]  — matches Python: emb = torch.exp(exponent); emb =
  // timesteps * emb
  auto emb = timesteps.to(torch::kFloat32).unsqueeze(1) *
             torch::exp(exponent).unsqueeze(0);
  return torch::cat({torch::sin(emb), torch::cos(emb)}, /*dim=*/-1);
}

// ---------------------------------------------------------------------------
// TimestepEmbedding: sinusoidal -> 3-layer MLP (SiLU)
// ---------------------------------------------------------------------------

class ColaTimestepEmbeddingImpl final : public torch::nn::Module {
 public:
  ColaTimestepEmbeddingImpl(int64_t sinusoidal_dim,
                            int64_t hidden_dim,
                            int64_t output_dim) {
    proj_in_ = register_module("proj_in",
                               torch::nn::Linear(sinusoidal_dim, hidden_dim));
    proj_hid_ =
        register_module("proj_hid", torch::nn::Linear(hidden_dim, hidden_dim));
    proj_out_ =
        register_module("proj_out", torch::nn::Linear(hidden_dim, output_dim));
    act_ = register_module("act", torch::nn::SiLU());
  }

  torch::Tensor forward(const torch::Tensor& timestep) {
    auto emb =
        get_sinusoidal_embedding(timestep, proj_in_->options.in_features());
    emb = act_->forward(proj_in_->forward(emb));
    emb = act_->forward(proj_hid_->forward(emb));
    emb = proj_out_->forward(emb);
    return emb;
  }

 private:
  torch::nn::Linear proj_in_{nullptr};
  torch::nn::Linear proj_hid_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  torch::nn::SiLU act_{nullptr};
};
TORCH_MODULE(ColaTimestepEmbedding);

// ---------------------------------------------------------------------------
// Rotary Embedding for ColaDiT (theta=10000, lang mode)
// ---------------------------------------------------------------------------

class ColaDiTRotaryEmbeddingImpl final : public torch::nn::Module {
 public:
  explicit ColaDiTRotaryEmbeddingImpl(int64_t dim) : dim_(dim) {
    // Compute inverse frequencies: inv_freq[i] = 1 / (10000^(2i/dim))
    auto inv_freq =
        1.0 /
        torch::pow(10000.0, torch::arange(0, dim, 2, torch::kFloat32) / dim);
    register_buffer("inv_freq", inv_freq);
  }

  // Compute cos/sin for positions [offset, offset+length).
  // Returns tensors of shape (length, dim).
  //
  // The official Cola-DLM TextRotaryEmbedding uses rotary_embedding_torch
  // which calls ``repeat(freqs, '... n -> ... (n r)', r=2)`` internally.
  // This REPEATS each frequency element twice in-place:
  //   [f0, f1, ..., f_{d/2-1}]  →  [f0, f0, f1, f1, ..., f_{d/2-1}, f_{d/2-1}]
  //
  // This is different from torch::cat({freqs, freqs}) which appends:
  //   [f0, f1, ..., f_{d/2-1}, f0, f1, ..., f_{d/2-1}]  (wrong!)
  //
  // With CONSECUTIVE-pair rotate_half (pairs (2k, 2k+1)):
  //   - repeat: both elements of each pair use the SAME angle f_k  ✓
  //   - cat:    elements 2k and 2k+1 use DIFFERENT angles f_k and f_{k+d/2}  ✗
  std::pair<torch::Tensor, torch::Tensor> get_cos_sin(int64_t length,
                                                      int64_t offset,
                                                      torch::Device device) {
    if (!inv_freq_.defined()) {
      inv_freq_ =
          1.0 / torch::pow(10000.0,
                           torch::arange(0, dim_, 2, torch::kFloat32) / dim_);
      inv_freq_ = inv_freq_.to(device);
    }
    auto positions = torch::arange(
        offset,
        offset + length,
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto freqs = positions.unsqueeze(1) * inv_freq_.unsqueeze(0);
    // Repeat each frequency element twice to match rotary_embedding_torch's
    // ``repeat(freqs, '... n -> ... (n r)', r=2)``:
    //   [f0, f1, ..., f_{d/2-1}] → [f0, f0, f1, f1, ..., f_{d/2-1}, f_{d/2-1}]
    auto emb = freqs.repeat_interleave(2, /*dim=*/-1);
    return {torch::cos(emb), torch::sin(emb)};
  }

  // Apply rotary embedding to q and k tensors of shape (L, heads, head_dim).
  // cos/sin have shape (L, rope_dim) — DOUBLED frequencies from get_cos_sin.
  // Uses CONSECUTIVE-pair rotate_half matching rotary_embedding_torch:
  //   rotate_half[2k] = -x[2k+1], rotate_half[2k+1] = x[2k]
  // This means element 2k uses angle f_{2k} and element 2k+1 uses f_{2k+1}
  // (different angles per element), matching cola's TextRotaryEmbedding.
  // Rotates the first rope_dim dimensions; the rest pass through unchanged.
  static std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& cos,
      const torch::Tensor& sin) {
    auto apply_rope = [&](const torch::Tensor& t) {
      int64_t d = cos.size(-1);  // rope_dim (full, doubled)
      auto rot = t.narrow(-1, 0, d);
      auto pass = t.narrow(-1, d, t.size(-1) - d);

      // CONSECUTIVE rotate_half matching rotary_embedding_torch:
      // pairs (2k, 2k+1): result[2k]=-x[2k+1], result[2k+1]=x[2k]
      auto rotate_half = [](const torch::Tensor& x) {
        int64_t L_ = x.size(0), H_ = x.size(1), sz = x.size(-1);
        auto paired = x.reshape({L_, H_, sz / 2, 2});
        auto even = paired.select(-1, 0);  // x[0::2]
        auto odd = paired.select(-1, 1);   // x[1::2]
        return torch::stack({-odd, even}, /*dim=*/-1).reshape({L_, H_, sz});
      };

      auto c = cos.unsqueeze(1);  // (L, 1, d)
      auto s = sin.unsqueeze(1);

      auto rot_new = rot * c + rotate_half(rot) * s;
      return torch::cat({rot_new, pass}, /*dim=*/-1);
    };
    return {apply_rope(q), apply_rope(k)};
  }

  // Apply rotary embedding to a single tensor of shape (L, heads, head_dim).
  // cos/sin have shape (L, rope_dim) — DOUBLED. CONSECUTIVE rotate_half.
  static torch::Tensor apply_rotary_emb_single(const torch::Tensor& x,
                                               const torch::Tensor& cos,
                                               const torch::Tensor& sin) {
    int64_t d = cos.size(-1);  // rope_dim (full, doubled)
    auto rot = x.narrow(-1, 0, d);
    auto pass = x.narrow(-1, d, x.size(-1) - d);

    // CONSECUTIVE rotate_half matching rotary_embedding_torch:
    // pairs (2k, 2k+1): result[2k]=-x[2k+1], result[2k+1]=x[2k]
    int64_t L_ = rot.size(0), H_ = rot.size(1), sz = rot.size(-1);
    auto paired = rot.reshape({L_, H_, sz / 2, 2});
    auto even = paired.select(-1, 0);  // x[0::2]
    auto odd = paired.select(-1, 1);   // x[1::2]
    auto rot_half =
        torch::stack({-odd, even}, /*dim=*/-1).reshape({L_, H_, sz});

    auto c = cos.unsqueeze(1);  // (L, 1, d)
    auto s = sin.unsqueeze(1);

    auto rot_new = rot * c + rot_half * s;
    return torch::cat({rot_new, pass}, /*dim=*/-1);
  }

 private:
  int64_t dim_;
  torch::Tensor inv_freq_;
};
TORCH_MODULE(ColaDiTRotaryEmbedding);

// ---------------------------------------------------------------------------
// AdaLN (Adaptive Layer Normalization)
// ---------------------------------------------------------------------------

// AdaLN applies shift/scale modulation ("in" mode) and gate modulation
// ("out" mode) conditioned on the timestep embedding.
class AdaLNImpl final : public torch::nn::Module {
 public:
  AdaLNImpl(int64_t dim,
            int64_t emb_dim,
            const std::vector<std::string>& layers,
            const std::vector<std::string>& modes = {"in", "out"})
      : layers_(layers), modes_(modes) {
    for (const auto& layer : layers) {
      if (std::find(modes.begin(), modes.end(), "in") != modes.end()) {
        // {layer}_in: SiLU -> Linear(dim, 2*dim) produces shift + scale
        auto seq = std::make_shared<torch::nn::SequentialImpl>();
        seq->push_back(torch::nn::SiLU());
        seq->push_back(torch::nn::Linear(dim, 2 * dim));
        register_module(layer + "_in", seq);
      }
      if (std::find(modes.begin(), modes.end(), "out") != modes.end()) {
        // {layer}_out: SiLU -> Linear(dim, dim) produces gate
        auto seq = std::make_shared<torch::nn::SequentialImpl>();
        seq->push_back(torch::nn::SiLU());
        seq->push_back(torch::nn::Linear(dim, dim));
        register_module(layer + "_out", seq);
      }
    }
  }

  // "in" mode: returns norm(hid) * (1 + scale) + shift
  // "out" mode: returns hid * gate + residual
  torch::Tensor forward(const torch::Tensor& hid,
                        const torch::Tensor& emb,
                        const std::string& layer,
                        const std::string& mode,
                        torch::nn::LayerNorm norm_layer = nullptr,
                        const torch::Tensor& residual = {}) {
    auto mod =
        named_modules()[layer + "_" + mode]->as<torch::nn::SequentialImpl>();
    torch::Tensor out = mod->forward(emb);

    // Repeat emb if it has fewer elements than hid (per-sample to per-token)
    if (out.size(0) != hid.size(0)) {
      // This shouldn't happen in the NA layout since emb is already per-token
      // but handle it for safety
      out = out.repeat_interleave(hid.size(0) / out.size(0), /*dim=*/0);
    }

    if (mode == "in") {
      auto chunks = out.chunk(2, /*dim=*/-1);
      auto shift = chunks[0];
      auto scale = chunks[1];
      return norm_layer(hid) * (1.0 + scale) + shift;
    }
    return hid * out + residual;
  }

 private:
  std::vector<std::string> layers_;
  std::vector<std::string> modes_;
};
TORCH_MODULE(AdaLN);

// ---------------------------------------------------------------------------
// MLP (GELU tanh approximation)
// ---------------------------------------------------------------------------

class ColaDiTMLPImpl final : public torch::nn::Module {
 public:
  ColaDiTMLPImpl(int64_t dim, int64_t expand_ratio) {
    proj_in_ =
        register_module("proj_in", torch::nn::Linear(dim, dim * expand_ratio));
    act_ = register_module(
        "act", torch::nn::GELU(torch::nn::GELUOptions().approximate("tanh")));
    proj_out_ =
        register_module("proj_out", torch::nn::Linear(dim * expand_ratio, dim));
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return proj_out_->forward(act_->forward(proj_in_->forward(x)));
  }

 private:
  torch::nn::Linear proj_in_{nullptr};
  torch::nn::GELU act_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
};
TORCH_MODULE(ColaDiTMLP);

// ---------------------------------------------------------------------------
// ColaDiT Attention (QKV proj, QK norm, RoPE, block-causal attention)
// Implements per-sample KV cache matching the official Python ColaDiTAttention.
// ---------------------------------------------------------------------------

class ColaDiTAttentionImpl final : public torch::nn::Module {
 public:
  ColaDiTAttentionImpl(int64_t txt_dim,
                       int64_t heads,
                       int64_t head_dim,
                       bool qk_bias,
                       int64_t rope_dim)
      : heads_(heads), head_dim_(head_dim) {
    int64_t inner_dim = heads * head_dim;
    proj_qkv_ = register_module(
        "proj_qkv",
        torch::nn::Linear(
            torch::nn::LinearOptions(txt_dim, inner_dim * 3).bias(qk_bias)));
    proj_out_ = register_module(
        "proj_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(inner_dim, txt_dim).bias(qk_bias)));
    norm_q_ = register_module(
        "norm_q",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim})));
    norm_k_ = register_module(
        "norm_k",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim})));
    rope_ = register_module("rope", ColaDiTRotaryEmbedding(rope_dim));
  }

  // Clear the per-sample KV cache.
  void set_kv_cache(bool /*flag*/) {
    k_cache_.clear();
    v_cache_.clear();
  }

  // Forward with optional KV cache support, matching the official Python
  // ColaDiTAttention.forward() semantics:
  //
  //   txt:          (L_q_total, txt_dim) — Q-side input (current block only)
  //   k_lens:       per-sample K-side lengths (cumulative: cache + current Q)
  //   q_lens:       per-sample Q-side lengths (= block_size during generation)
  //   update_kv:    append current K/V to cache, then read full cache as K
  //   use_kv_cache: prepend cached K/V to current K
  //
  // When both are False (unconditional pass): full_k = current K only.
  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        int64_t block_size,
                        const torch::Tensor& attn_mask,
                        bool update_kv = false,
                        bool use_kv_cache = false) {
    int64_t L_q = txt.size(0);  // Q-side total length
    int64_t B = static_cast<int64_t>(q_lens.size());

    // QKV projection
    auto qkv = proj_qkv_->forward(txt);
    qkv = qkv.reshape({L_q, 3, heads_, head_dim_});
    auto txt_q = qkv.select(1, 0);  // (L_q, heads, head_dim)
    auto txt_k = qkv.select(1, 1);
    auto txt_v = qkv.select(1, 2);

    // QK normalization.
    txt_q = norm_q_(txt_q);
    txt_k = norm_k_(txt_k);

    // --- KV cache bookkeeping (matches official Python) ----------------------
    // Split per-sample new K/V from the Q-side projection.
    std::vector<torch::Tensor> new_ks, new_vs;
    {
      int64_t offset = 0;
      for (int64_t i = 0; i < B; ++i) {
        int64_t ql = q_lens[i];
        new_ks.push_back(txt_k.narrow(0, offset, ql));
        new_vs.push_back(txt_v.narrow(0, offset, ql));
        offset += ql;
      }
    }

    torch::Tensor full_k, full_v;
    if (update_kv) {
      // Append to cache, then read full cache.
      if (k_cache_.empty()) {
        k_cache_.resize(B);
        v_cache_.resize(B);
        for (int64_t i = 0; i < B; ++i) {
          k_cache_[i] = new_ks[i].clone();
          v_cache_[i] = new_vs[i].clone();
        }
      } else {
        for (int64_t i = 0; i < B; ++i) {
          k_cache_[i] = torch::cat({k_cache_[i], new_ks[i]}, /*dim=*/0);
          v_cache_[i] = torch::cat({v_cache_[i], new_vs[i]}, /*dim=*/0);
        }
      }
      std::vector<torch::Tensor> full_ks, full_vs;
      for (int64_t i = 0; i < B; ++i) {
        full_ks.push_back(k_cache_[i]);
        full_vs.push_back(v_cache_[i]);
      }
      full_k = torch::cat(full_ks, /*dim=*/0);
      full_v = torch::cat(full_vs, /*dim=*/0);
    } else if (use_kv_cache && !k_cache_.empty()) {
      // Prepend cached K/V to current K/V (don't update cache).
      std::vector<torch::Tensor> full_ks, full_vs;
      for (int64_t i = 0; i < B; ++i) {
        full_ks.push_back(torch::cat({k_cache_[i], new_ks[i]}, /*dim=*/0));
        full_vs.push_back(torch::cat({v_cache_[i], new_vs[i]}, /*dim=*/0));
      }
      full_k = torch::cat(full_ks, /*dim=*/0);
      full_v = torch::cat(full_vs, /*dim=*/0);
    } else {
      // No cache (unconditional pass): K = current Q only.
      full_k = txt_k;
      full_v = txt_v;
    }

    int64_t L_k_total = full_k.size(0);

    // --- RoPE ----------------------------------------------------------------
    // K positions: [0, k_lens[i]) for each sample i.
    // Q positions: [k_lens[i] - q_lens[i], k_lens[i]) — tail of K.
    torch::Tensor cos_k, sin_k, cos_q, sin_q;
    {
      std::vector<torch::Tensor> ck_list, sk_list, cq_list, sq_list;
      for (int64_t i = 0; i < B; ++i) {
        auto [ck, sk] = rope_->get_cos_sin(k_lens[i], 0, txt.device());
        ck_list.push_back(ck);
        sk_list.push_back(sk);
        int64_t q_offset = k_lens[i] - q_lens[i];
        auto [cq, sq] = rope_->get_cos_sin(q_lens[i], q_offset, txt.device());
        cq_list.push_back(cq);
        sq_list.push_back(sq);
      }
      cos_k = torch::cat(ck_list, /*dim=*/0);
      sin_k = torch::cat(sk_list, /*dim=*/0);
      cos_q = torch::cat(cq_list, /*dim=*/0);
      sin_q = torch::cat(sq_list, /*dim=*/0);
    }

    // Apply RoPE in float32 to match official Python:
    //   apply_rotary_emb(freqs, txt_q.float()).to(txt_q.dtype)
    // Without this, bfloat16 cos/sin introduces per-dimension errors that
    // accumulate across layers and cause feature-direction drift.
    {
      auto q_dtype = txt_q.dtype();
      txt_q = ColaDiTRotaryEmbeddingImpl::apply_rotary_emb_single(
                  txt_q.to(torch::kFloat32),
                  cos_q.to(torch::kFloat32),
                  sin_q.to(torch::kFloat32))
                  .to(q_dtype);
    }
    {
      auto k_dtype = full_k.dtype();
      full_k = ColaDiTRotaryEmbeddingImpl::apply_rotary_emb_single(
                   full_k.to(torch::kFloat32),
                   cos_k.to(torch::kFloat32),
                   sin_k.to(torch::kFloat32))
                   .to(k_dtype);
    }

    // --- Attention -----------------------------------------------------------
    // Mirror official slow_attn: Q/K/V in bfloat16, softmax under bf16 autocast
    // so the backend promotes softmax to fp32 internally (matching training
    // numerics).
    auto q_na = txt_q.to(torch::kBFloat16).permute({1, 0, 2}).unsqueeze(0);
    auto k_na = full_k.to(torch::kBFloat16).permute({1, 0, 2}).unsqueeze(0);
    auto v_na = full_v.to(torch::kBFloat16).permute({1, 0, 2}).unsqueeze(0);

    torch::Tensor attn_out;
    {
      torch::NoGradGuard no_grad;
#if defined(USE_CUDA)
      torch::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
      torch::autocast::set_autocast_enabled(torch::kCUDA, true);
#endif
      float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
      auto attn = q_na.mul(scale).matmul(k_na.transpose(-2, -1));
      if (attn_mask.defined()) {
        attn = attn + attn_mask.to(torch::kBFloat16);
      }
      auto attn_weight = torch::softmax(attn, /*dim=*/-1);
      attn_out = attn_weight.matmul(v_na);
    }

    attn_out = attn_out.squeeze(0).permute({1, 0, 2}).reshape(
        {L_q, heads_ * head_dim_});
    // Match Python: rearrange(...).type_as(txt_q) before proj_out.
    attn_out = attn_out.to(txt_q.scalar_type());
    return proj_out_->forward(attn_out);  // (L_q, txt_dim)
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  torch::nn::Linear proj_qkv_{nullptr};
  torch::nn::Linear proj_out_{nullptr};
  torch::nn::LayerNorm norm_q_{nullptr};
  torch::nn::LayerNorm norm_k_{nullptr};
  ColaDiTRotaryEmbedding rope_{nullptr};
  // Per-sample KV cache: one (l_i_cum, heads, head_dim) tensor per sample.
  std::vector<torch::Tensor> k_cache_;
  std::vector<torch::Tensor> v_cache_;
};
TORCH_MODULE(ColaDiTAttention);

// ---------------------------------------------------------------------------
// ColaDiT Transformer Block
// ---------------------------------------------------------------------------

class ColaDiTBlockImpl final : public torch::nn::Module {
 public:
  ColaDiTBlockImpl(int64_t txt_dim,
                   int64_t emb_dim,
                   int64_t heads,
                   int64_t head_dim,
                   int64_t expand_ratio,
                   float norm_eps,
                   bool qk_bias,
                   int64_t rope_dim) {
    msa_norm_ = register_module(
        "msa_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({txt_dim})
                                 .eps(norm_eps)
                                 .elementwise_affine(false)));
    msa_ = register_module(
        "msa", ColaDiTAttention(txt_dim, heads, head_dim, qk_bias, rope_dim));
    mlp_norm_ = register_module(
        "mlp_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({txt_dim})
                                 .eps(norm_eps)
                                 .elementwise_affine(false)));
    mlp_ = register_module("mlp", ColaDiTMLP(txt_dim, expand_ratio));
    ada_ = register_module(
        "ada", AdaLN(txt_dim, emb_dim, std::vector<std::string>{"msa", "mlp"}));
  }

  // txt:          (L_q, dim) — Q-side hidden state (current block only)
  // k_lens:       per-sample cumulative K lengths (cache + current Q)
  // q_lens:       per-sample Q lengths (= block_size during generation)
  // emb:          (L_q, emb_dim) — AdaLN conditioning for Q positions only
  // update_kv / use_kv_cache: forwarded to ColaDiTAttention KV cache logic
  void set_kv_cache(bool flag) { msa_->set_kv_cache(flag); }

  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        const torch::Tensor& emb,
                        int64_t block_size,
                        const torch::Tensor& attn_mask,
                        bool update_kv = false,
                        bool use_kv_cache = false) {
    // Attention sublayer: AdaLN (in) → attention → AdaLN (out) + residual.
    // Caller enables bf16 autocast around the DiT forward pass.
    auto txt_msa = ada_->forward(txt, emb, "msa", "in", msa_norm_);
    txt_msa = msa_->forward(txt_msa,
                            k_lens,
                            q_lens,
                            block_size,
                            attn_mask,
                            update_kv,
                            use_kv_cache);
    auto txt_out = ada_->forward(txt_msa, emb, "msa", "out", nullptr, txt);

    // MLP sublayer: AdaLN (in) → MLP → AdaLN (out) + residual
    auto txt_mlp = ada_->forward(txt_out, emb, "mlp", "in", mlp_norm_);
    txt_mlp = mlp_->forward(txt_mlp);
    txt_out = ada_->forward(txt_mlp, emb, "mlp", "out", nullptr, txt_out);
    return txt_out;
  }

 private:
  torch::nn::LayerNorm msa_norm_{nullptr};
  ColaDiTAttention msa_{nullptr};
  torch::nn::LayerNorm mlp_norm_{nullptr};
  ColaDiTMLP mlp_{nullptr};
  AdaLN ada_{nullptr};
};
TORCH_MODULE(ColaDiTBlock);

// ---------------------------------------------------------------------------
// PatchIn1D / PatchOut1D — patchification wrappers
// Matches the Python PatchIn1D/PatchOut1D classes which contain a Linear "proj"
// ---------------------------------------------------------------------------

class PatchIn1DImpl final : public torch::nn::Module {
 public:
  PatchIn1DImpl(int64_t in_channels, int64_t patch_size, int64_t dim) {
    proj_ = register_module("proj",
                            torch::nn::Linear(in_channels * patch_size, dim));
  }

  torch::Tensor forward(const torch::Tensor& x) { return proj_->forward(x); }

 private:
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(PatchIn1D);

class PatchOut1DImpl final : public torch::nn::Module {
 public:
  PatchOut1DImpl(int64_t out_channels, int64_t patch_size, int64_t dim) {
    proj_ = register_module("proj",
                            torch::nn::Linear(dim, out_channels * patch_size));
  }

  torch::Tensor forward(const torch::Tensor& x) { return proj_->forward(x); }

 private:
  torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(PatchOut1D);

// ---------------------------------------------------------------------------
// ColaDiT Transformer (full model)
// ---------------------------------------------------------------------------

class ColaDiTTransformerImpl final : public torch::nn::Module {
 public:
  explicit ColaDiTTransformerImpl(const ModelContext& ctx) {
    const auto& args = ctx.get_model_args();
    int64_t txt_dim = args.txt_dim();
    int64_t emb_dim = args.emb_dim();
    int64_t heads = args.heads();
    int64_t head_dim = args.head_dim();
    int64_t expand_ratio = args.expand_ratio();
    int64_t num_layers = args.num_layers();
    float norm_eps = args.norm_eps();
    bool qk_bias = args.qk_bias();
    int64_t rope_dim = args.rope_dim();
    int64_t txt_in_channels = args.txt_in_channels();
    int64_t txt_out_channels = args.txt_out_channels();
    block_size_ = args.block_size();
    txt_dim_ = txt_dim;

    // Input projection: latent_dim -> txt_dim (PatchIn1D wraps Linear as
    // "proj")
    txt_in_ = register_module("txt_in", PatchIn1D(txt_in_channels, 1, txt_dim));

    // Timestep embedding: sinusoidal(256) -> MLP -> emb_dim
    emb_in_ =
        register_module("emb_in", ColaTimestepEmbedding(256, txt_dim, emb_dim));

    // Transformer blocks
    blocks_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
      auto block = ColaDiTBlock(txt_dim,
                                emb_dim,
                                heads,
                                head_dim,
                                expand_ratio,
                                norm_eps,
                                qk_bias,
                                rope_dim);
      blocks_.push_back(block);
      register_module("blocks_" + std::to_string(i), block);
    }

    // Output: norm + AdaLN + projection
    txt_out_norm_ = register_module(
        "txt_out_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({txt_dim}).eps(norm_eps)));
    txt_out_ada_ = register_module("txt_out_ada",
                                   AdaLN(txt_dim,
                                         emb_dim,
                                         std::vector<std::string>{"out"},
                                         std::vector<std::string>{"in"}));
    txt_out_ =
        register_module("txt_out", PatchOut1D(txt_out_channels, 1, txt_dim));
  }

  // Clear per-layer KV caches on all blocks.
  void set_kv_cache(bool flag) {
    for (auto& block : blocks_) {
      block->set_kv_cache(flag);
    }
  }

  // Forward pass — matches the official Python ColaDiTModel.forward().
  //
  // txt:          (L_q_total, txt_in_channels) — Q-side input (current block
  //               or prefix, depending on update_kv flag)
  // k_lens:       per-sample cumulative K lengths (cache + current Q)
  // q_lens:       per-sample Q lengths (= block_size during generation, or
  //               prefix length during prefetch with update_kv=True)
  // timestep:     per-token timestep (length = L_q_total)
  // update_kv:    commit current Q's K/V to per-layer cache
  // use_kv_cache: read per-layer cached K/V for the conditional pass
  torch::Tensor forward(const torch::Tensor& txt,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        const torch::Tensor& timestep,
                        bool update_kv = false,
                        bool use_kv_cache = false) {
    int64_t L_q = txt.size(0);

    auto hidden = txt_in_(txt);  // (L_q, txt_dim)

    // Timestep embedding — length = L_q.
    auto ts = timestep;
    if (ts.dim() == 0) {
      ts = ts.unsqueeze(0);
    }
    if (ts.size(0) == 1 && L_q > 1) {
      ts = ts.expand({L_q}, /*implicit=*/true);
    }
    // Pass timestep as-is; ColaTimestepEmbeddingImpl::forward will compute the
    // sinusoidal embedding in float32 and then cast to bfloat16 internally,
    // matching the official Python path exactly.
    auto emb = emb_in_(ts);  // (L_q, emb_dim)

    // Block-causal attention mask for Q × K.
    auto attn_mask = create_block_causal_mask(
        k_lens, q_lens, block_size_, hidden.scalar_type(), hidden.device());

    for (int64_t bi = 0; bi < static_cast<int64_t>(blocks_.size()); ++bi) {
      hidden = blocks_[bi]->forward(hidden,
                                    k_lens,
                                    q_lens,
                                    emb,
                                    block_size_,
                                    attn_mask,
                                    update_kv,
                                    use_kv_cache);
    }

    // Output: AdaLN + projection.
    hidden = txt_out_ada_->forward(hidden, emb, "out", "in", txt_out_norm_);
    hidden = txt_out_(hidden);

    return hidden;  // (L_q, txt_out_channels)
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    load_cola_module_from_state_dicts(*loader, this);
  }

  int64_t block_size() const { return block_size_; }

 private:
  int64_t block_size_;
  int64_t txt_dim_;
  PatchIn1D txt_in_{nullptr};
  ColaTimestepEmbedding emb_in_{nullptr};
  std::vector<ColaDiTBlock> blocks_;
  torch::nn::LayerNorm txt_out_norm_{nullptr};
  AdaLN txt_out_ada_{nullptr};
  PatchOut1D txt_out_{nullptr};
};
TORCH_MODULE(ColaDiTTransformer);

// ---------------------------------------------------------------------------
// REGISTER_MODEL_ARGS for ColaDiT config.json
// ---------------------------------------------------------------------------

REGISTER_MODEL_ARGS(cola_dit, [&] {
  LOAD_ARG_OR(model_type, "model_type", "cola_dit");
  LOAD_ARG(txt_dim, "txt_dim");
  LOAD_ARG(txt_in_channels, "txt_in_channels");
  LOAD_ARG(txt_out_channels, "txt_out_channels");
  LOAD_ARG(emb_dim, "emb_dim");
  LOAD_ARG(heads, "heads");
  LOAD_ARG(head_dim, "head_dim");
  LOAD_ARG(expand_ratio, "expand_ratio");
  LOAD_ARG(num_layers, "num_layers");
  LOAD_ARG(rope_dim, "rope_dim");
  LOAD_ARG(block_size, "block_size");
  LOAD_ARG(qk_bias, "qk_bias");
  LOAD_ARG_OR(norm_eps, "norm_eps", 1e-5);
  LOAD_ARG_OR(patch_size, "patch_size", 1);
});

}  // namespace xllm
