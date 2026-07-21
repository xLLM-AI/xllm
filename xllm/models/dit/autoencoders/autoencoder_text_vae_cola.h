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

#include "core/framework/model_context.h"
#include "models/dit/utils/cola_block_causal_mask.h"
#include "models/dit/utils/cola_weight_loader.h"
#include "models/model_registry.h"

namespace xllm {

// ---------------------------------------------------------------------------
// SwiGLU Activation
// ---------------------------------------------------------------------------
// Python: x, gate = x.chunk(2, dim=-1); return F.silu(gate) * x
// The input has shape (..., ffn_dim) where ffn_dim = 2 * hidden_dim.
// Output has shape (..., hidden_dim).

class SwiGLUImpl final : public torch::nn::Module {
 public:
  torch::Tensor forward(const torch::Tensor& x) {
    auto chunks = x.chunk(2, /*dim=*/-1);
    return torch::silu(chunks[1]) * chunks[0];
  }
};
TORCH_MODULE(SwiGLU);

// ---------------------------------------------------------------------------
// VAE Rotary Embedding (configurable theta)
// ---------------------------------------------------------------------------
// Same as ColaDiTRotaryEmbedding but with configurable theta.
// VAE uses theta=500000, DiT uses theta=10000.

class VAERotaryEmbeddingImpl final : public torch::nn::Module {
 public:
  explicit VAERotaryEmbeddingImpl(int64_t dim, int64_t theta = 500000)
      : dim_(dim), theta_(theta) {
    auto inv_freq =
        1.0 / torch::pow(static_cast<double>(theta),
                         torch::arange(0, dim, 2, torch::kFloat32) / dim);
    register_buffer("inv_freq", inv_freq);
  }

  std::pair<torch::Tensor, torch::Tensor> get_cos_sin(int64_t length,
                                                      int64_t offset,
                                                      torch::Device device) {
    if (!inv_freq_.defined()) {
      // Recompute if buffer was not loaded from checkpoint
      inv_freq_ =
          1.0 / torch::pow(static_cast<double>(theta_),
                           torch::arange(0, dim_, 2, torch::kFloat32) / dim_);
      inv_freq_ = inv_freq_.to(device);
    }
    auto positions = torch::arange(
        offset,
        offset + length,
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto freqs = positions.unsqueeze(1) * inv_freq_.unsqueeze(0);
    // Double the frequencies (same as Python's cat([freqs, freqs])) so that
    // the SPLIT rotate_half pairs (j, j+dim/2) both use the same angle θ_j,
    // matching cola's VAERotaryEmbedding.get_rotary_embedding().
    auto emb = torch::cat({freqs, freqs}, /*dim=*/-1);  // (L, dim)
    return {torch::cos(emb), torch::sin(emb)};
  }

  static std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& cos,
      const torch::Tensor& sin) {
    int64_t d = cos.size(-1);
    auto q_rot = q.narrow(-1, 0, d);
    auto q_pass = q.narrow(-1, d, q.size(-1) - d);
    auto k_rot = k.narrow(-1, 0, d);
    auto k_pass = k.narrow(-1, d, k.size(-1) - d);

    auto rotate_half = [](const torch::Tensor& x) {
      int64_t d = x.size(-1);
      return torch::cat({-x.narrow(-1, d / 2, d / 2), x.narrow(-1, 0, d / 2)},
                        /*dim=*/-1);
    };

    auto cos_expanded = cos.unsqueeze(1);
    auto sin_expanded = sin.unsqueeze(1);

    auto q_new = torch::cat(
        {q_rot * cos_expanded + rotate_half(q_rot) * sin_expanded, q_pass},
        /*dim=*/-1);
    auto k_new = torch::cat(
        {k_rot * cos_expanded + rotate_half(k_rot) * sin_expanded, k_pass},
        /*dim=*/-1);
    return {q_new, k_new};
  }

  // Apply rotary embedding to a single tensor of shape (L, heads, head_dim)
  static torch::Tensor apply_rotary_emb_single(const torch::Tensor& x,
                                               const torch::Tensor& cos,
                                               const torch::Tensor& sin) {
    int64_t d = cos.size(-1);
    auto x_rot = x.narrow(-1, 0, d);
    auto x_pass = x.narrow(-1, d, x.size(-1) - d);

    auto rotate_half = [](const torch::Tensor& t) {
      int64_t sz = t.size(-1);
      return torch::cat(
          {-t.narrow(-1, sz / 2, sz / 2), t.narrow(-1, 0, sz / 2)},
          /*dim=*/-1);
    };

    auto cos_expanded = cos.unsqueeze(1);
    auto sin_expanded = sin.unsqueeze(1);

    return torch::cat(
        {x_rot * cos_expanded + rotate_half(x_rot) * sin_expanded, x_pass},
        /*dim=*/-1);
  }

 private:
  int64_t dim_;
  int64_t theta_;
  torch::Tensor inv_freq_;
};
TORCH_MODULE(VAERotaryEmbedding);

// ---------------------------------------------------------------------------
// TextVAE Block (transformer block for encoder/decoder)
// ---------------------------------------------------------------------------
// Reference: modeling_cola_vae.py TextVAEBlock.
// Post-norm variant: x = residual + attn(norm(x)); x = residual +
// ffn(norm(x))

class TextVAEBlockImpl final : public torch::nn::Module {
 public:
  TextVAEBlockImpl(int64_t dim,
                   int64_t ffn_dim,
                   int64_t num_heads,
                   int64_t shared_heads_kv,
                   int64_t rope_theta,
                   bool qk_bias = false)
      : dim_(dim),
        num_heads_(num_heads),
        head_dim_(dim / num_heads),
        shared_heads_kv_(shared_heads_kv) {
    norm_attn_ = register_module(
        "norm_attn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    int64_t kv_dim = dim / shared_heads_kv;
    qkv_proj_ = register_module(
        "qkv_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, dim + kv_dim * 2).bias(qk_bias)));
    attn_out_proj_ =
        register_module("attn_out_proj", torch::nn::Linear(dim, dim));

    // QK norm is applied to the FULL dimension before reshaping to heads.
    // This matches the Python implementation where norm is on (L, dim).
    q_norm_ = register_module(
        "q_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    k_norm_ = register_module(
        "k_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({kv_dim})));

    rope_ = register_module("rope", VAERotaryEmbedding(head_dim_, rope_theta));

    norm_ffn_ = register_module(
        "norm_ffn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    // SwiGLU: proj outputs ffn_dim, SwiGLU splits into gate and up
    ffn_proj_ = register_module("ffn_proj", torch::nn::Linear(dim, ffn_dim));
    ffn_act_ = register_module("ffn_act", SwiGLU());
    // SwiGLU splits ffn_dim into two halves, so ffn_out input is ffn_dim/2
    ffn_out_ = register_module("ffn_out", torch::nn::Linear(ffn_dim / 2, dim));
  }

  // Clear the per-sample KV cache.
  void set_kv_cache(bool /*flag*/) {
    k_cache_.clear();
    v_cache_.clear();
  }

  // Forward matching the official Python TextVAEBlock.forward() with
  // per-sample KV cache support.
  //
  // x:           (L_q, dim) — Q-side input (current block during generation,
  //              or full sequence during prefix encode / self-attention)
  // k_lens:      per-sample cumulative K lengths (cache + current Q)
  // q_lens:      per-sample Q lengths (block_size during generation)
  // update_kv:   append current K/V to cache, read full cache as K/V
  //              (False by default — VAE encode blocks never cache)
  torch::Tensor forward(const torch::Tensor& x,
                        const std::vector<int64_t>& k_lens,
                        const std::vector<int64_t>& q_lens,
                        int64_t block_size,
                        const torch::Tensor& attn_mask,
                        bool update_kv = false) {
    int64_t L_q = x.size(0);
    int64_t B = static_cast<int64_t>(q_lens.size());
    int64_t kv_dim = dim_ / shared_heads_kv_;
    int64_t kv_heads = num_heads_ / shared_heads_kv_;

    // QKV projection on Q-side input.
    auto qkv = qkv_proj_->forward(x);
    auto qkv_chunks = qkv.split({dim_, kv_dim, kv_dim}, /*dim=*/-1);
    auto q = qkv_chunks[0];  // (L_q, dim)
    auto k = qkv_chunks[1];  // (L_q, kv_dim)
    auto v = qkv_chunks[2];  // (L_q, kv_dim)

    // --- KV cache bookkeeping (matches official Python) ----------------------
    // Cache raw K/V from qkv_proj; k_norm is applied to the concatenated
    // full_k AFTER cache assembly (LayerNorm is not linear:
    // norm(cat(a,b)) != cat(norm(a), norm(b))).
    std::vector<torch::Tensor> new_ks, new_vs;
    {
      int64_t offset = 0;
      for (int64_t i = 0; i < B; ++i) {
        int64_t ql = q_lens[i];
        new_ks.push_back(k.narrow(0, offset, ql));
        new_vs.push_back(v.narrow(0, offset, ql));
        offset += ql;
      }
    }

    torch::Tensor full_k, full_v;
    if (update_kv) {
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
      std::vector<torch::Tensor> fks, fvs;
      for (int64_t i = 0; i < B; ++i) {
        fks.push_back(k_cache_[i]);
        fvs.push_back(v_cache_[i]);
      }
      full_k = torch::cat(fks, /*dim=*/0);
      full_v = torch::cat(fvs, /*dim=*/0);
    } else if (!k_cache_.empty()) {
      // Read-only cache: prepend cached K/V.
      std::vector<torch::Tensor> fks, fvs;
      for (int64_t i = 0; i < B; ++i) {
        fks.push_back(torch::cat({k_cache_[i], new_ks[i]}, /*dim=*/0));
        fvs.push_back(torch::cat({v_cache_[i], new_vs[i]}, /*dim=*/0));
      }
      full_k = torch::cat(fks, /*dim=*/0);
      full_v = torch::cat(fvs, /*dim=*/0);
    } else {
      // No cache: K = current Q only (self-attention / unconditional).
      full_k = k;
      full_v = v;
    }

    // QK normalization — official TextVAEBlock applies q_norm to Q and k_norm
    // to the assembled full_k tensor (including cached prefix keys).
    auto dtype = q.dtype();
    q = q_norm_(q).to(dtype);
    full_k = k_norm_(full_k).to(dtype);

    int64_t L_k_total = full_k.size(0);

    // --- RoPE ----------------------------------------------------------------
    // K positions: [0, k_lens[i]) per sample.
    // Q positions: [k_lens[i] - q_lens[i], k_lens[i]) — Q at tail of K.
    torch::Tensor cos_k, sin_k, cos_q, sin_q;
    {
      std::vector<torch::Tensor> ck, sk, cq, sq;
      for (int64_t i = 0; i < B; ++i) {
        auto [ck_i, sk_i] = rope_->get_cos_sin(k_lens[i], 0, x.device());
        ck.push_back(ck_i);
        sk.push_back(sk_i);
        int64_t q_off = k_lens[i] - q_lens[i];
        auto [cq_i, sq_i] = rope_->get_cos_sin(q_lens[i], q_off, x.device());
        cq.push_back(cq_i);
        sq.push_back(sq_i);
      }
      cos_k = torch::cat(ck, /*dim=*/0);
      sin_k = torch::cat(sk, /*dim=*/0);
      cos_q = torch::cat(cq, /*dim=*/0);
      sin_q = torch::cat(sq, /*dim=*/0);
    }

    // Reshape q for RoPE: (L_q, num_heads, head_dim)
    q = q.reshape({L_q, num_heads_, head_dim_});
    // Reshape k for RoPE: (L_k, kv_heads, head_dim)
    full_k = full_k.reshape({L_k_total, kv_heads, head_dim_});

    // Apply RoPE in float32 matching official full_precision=True:
    //   q_ = q.float(); k_ = k.float(); apply rope; return .to(original_dtype)
    {
      auto q_dtype = q.dtype();
      q = VAERotaryEmbeddingImpl::apply_rotary_emb_single(
              q.to(torch::kFloat32),
              cos_q.to(torch::kFloat32),
              sin_q.to(torch::kFloat32))
              .to(q_dtype);
    }
    {
      auto k_dtype = full_k.dtype();
      full_k = VAERotaryEmbeddingImpl::apply_rotary_emb_single(
                   full_k.to(torch::kFloat32),
                   cos_k.to(torch::kFloat32),
                   sin_k.to(torch::kFloat32))
                   .to(k_dtype);
    }

    // Expand K/V heads if GQA (shared_heads_kv > 1).
    if (shared_heads_kv_ > 1) {
      full_k = full_k.repeat_interleave(shared_heads_kv_, /*dim=*/1);
    }
    full_v = full_v.reshape({L_k_total, kv_heads, head_dim_});
    if (shared_heads_kv_ > 1) {
      full_v = full_v.repeat_interleave(shared_heads_kv_, /*dim=*/1);
    }

    // --- Attention -----------------------------------------------------------
    // Cast Q/K/V to bfloat16 and run under bfloat16 autocast so that CUDA's
    // softmax kernel accumulates in float32 (matching VAE training numerics).
    auto q_na = q.to(torch::kBFloat16).permute({1, 0, 2}).unsqueeze(0);
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

    attn_out = attn_out.squeeze(0).permute({1, 0, 2}).reshape({L_q, dim_});
    // Match Python: rearrange(...).type_as(x) before attn_out_proj.
    auto attn_result = attn_out_proj_->forward(attn_out.to(x.scalar_type()));

    // --- Post-norm residual (post_norm=True) ---------------------------------
    // Attention sublayer: x_new = norm_attn(x) + attn
    auto q_part = norm_attn_(x) + attn_result;  // (L_q, dim)

    // FFN sublayer: x_new = q_part + norm_ffn(ffn(q_part))
    auto residual = q_part;
    auto ffn_out =
        ffn_out_->forward(ffn_act_->forward(ffn_proj_->forward(q_part)));
    q_part = residual + norm_ffn_(ffn_out);

    return q_part;  // (L_q, dim)
  }

 private:
  int64_t dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t shared_heads_kv_;
  torch::nn::LayerNorm norm_attn_{nullptr};
  torch::nn::Linear qkv_proj_{nullptr};
  torch::nn::Linear attn_out_proj_{nullptr};
  torch::nn::LayerNorm q_norm_{nullptr};
  torch::nn::LayerNorm k_norm_{nullptr};
  VAERotaryEmbedding rope_{nullptr};
  torch::nn::LayerNorm norm_ffn_{nullptr};
  torch::nn::Linear ffn_proj_{nullptr};
  SwiGLU ffn_act_{nullptr};
  torch::nn::Linear ffn_out_{nullptr};
  // Per-sample KV cache (raw K/V before RoPE, matching official Python).
  std::vector<torch::Tensor> k_cache_;
  std::vector<torch::Tensor> v_cache_;
};
TORCH_MODULE(TextVAEBlock);

// ---------------------------------------------------------------------------
// ColaTextVAE Encoder
// ---------------------------------------------------------------------------
// Reference: modeling_cola_vae.py ColaTextVAEModel.encode()
// Token embedding -> Conv1d patchification -> transformer blocks ->
// final_layer -> final_norm -> per-sample latents

class ColaTextVAEEncoderImpl final : public torch::nn::Module {
 public:
  ColaTextVAEEncoderImpl(const ModelContext& ctx) {
    const auto& args = ctx.get_model_args();
    int64_t vae_dim = args.vae_dim();
    dim_ = vae_dim;
    int64_t vae_num_heads = args.vae_num_heads();
    int64_t ffn_dim = args.ffn_dim();
    int64_t shared_heads_kv = args.shared_heads_kv();
    int64_t vae_rope_theta = args.vae_rope_theta();
    int64_t latent_dim = args.latent_dim();
    int64_t patch_size = args.vae_patch_size();
    int64_t encoder_num_blocks = args.encoder_num_blocks();
    bool use_variation = args.use_variation();
    bool encoder_last_ln = args.encoder_last_ln();
    bool qk_bias = args.qk_bias();
    int64_t vocab_size = args.vocab_size();

    wte_ =
        register_module("wte", torch::nn::Embedding(vocab_size + 1, vae_dim));
    patch_embedder_ = register_module(
        "patch_embedder",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(vae_dim, vae_dim, patch_size)
                              .stride(patch_size)));

    blocks_.reserve(encoder_num_blocks);
    for (int64_t i = 0; i < encoder_num_blocks; ++i) {
      auto block = TextVAEBlock(vae_dim,
                                ffn_dim,
                                vae_num_heads,
                                shared_heads_kv,
                                vae_rope_theta,
                                qk_bias);
      blocks_.push_back(block);
      register_module("blocks_" + std::to_string(i), block);
    }

    if (use_variation) {
      final_layer_ = register_module(
          "final_layer", torch::nn::Linear(vae_dim, latent_dim * 2));
      if (encoder_last_ln) {
        final_norm_ = register_module(
            "final_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({latent_dim})
                                     .elementwise_affine(false)));
      }
    } else {
      final_layer_ = register_module("final_layer",
                                     torch::nn::Linear(vae_dim, latent_dim));
      final_norm_ = register_module(
          "final_norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({latent_dim})
                                   .elementwise_affine(false)));
    }

    use_variation_ = use_variation;
    encoder_last_ln_ = encoder_last_ln;
    block_size_ = args.vae_block_size();
    patch_size_ = patch_size;
  }

  // Encode per-sample input_ids into per-sample latents.
  // Returns concatenated latents (L_total, latent_dim) and per-sample lengths.
  std::pair<torch::Tensor, std::vector<int64_t>> encode(
      const std::vector<torch::Tensor>& input_ids_list) {
    torch::NoGradGuard no_grad;

    // Embedding + Conv1d patchification per sample
    std::vector<torch::Tensor> per_sample;
    per_sample.reserve(input_ids_list.size());
    for (const auto& ids : input_ids_list) {
      if (ids.numel() == 0) {
        LOG(WARNING) << "ColaTextVAEEncoder: empty input_ids, skipping";
        continue;
      }
      auto x = wte_(ids.unsqueeze(0));      // (1, L_i, dim)
      x = x.permute({0, 2, 1});             // (1, dim, L_i)
      x = patch_embedder_(x);               // (1, dim, n_i)
      x = x.permute({0, 2, 1}).squeeze(0);  // (n_i, dim)
      per_sample.push_back(x);
    }

    if (per_sample.empty()) {
      LOG(WARNING) << "ColaTextVAEEncoder: all input samples are empty";
      return {torch::zeros({0, dim_}, wte_->weight.options()),
              std::vector<int64_t>{}};
    }

    // Build txt_shape for NA layout
    std::vector<int64_t> sample_lens;
    sample_lens.reserve(per_sample.size());
    for (const auto& t : per_sample) {
      sample_lens.push_back(t.size(0));
    }
    int64_t L_total = 0;
    for (auto l : sample_lens) L_total += l;

    // Concatenate all samples
    auto x = torch::cat(per_sample, /*dim=*/0);  // (L_total, dim)

    // Build block-causal attention mask
    torch::Tensor attn_mask;
    if (block_size_ > 0) {
      attn_mask = create_block_causal_mask(
          sample_lens, sample_lens, block_size_, x.scalar_type(), x.device());
    }

    // Run through transformer blocks
    for (auto& block : blocks_) {
      x = block->forward(x, sample_lens, sample_lens, block_size_, attn_mask);
    }

    // Final projection
    x = final_layer_(x);
    if (encoder_last_ln_ && use_variation_) {
      // Split mean/logvar, apply norm to mean only
      auto chunks = x.chunk(2, /*dim=*/-1);
      auto mean = final_norm_(chunks[0]);
      x = torch::cat({mean, chunks[1]}, /*dim=*/-1);
    } else if (final_norm_) {
      x = final_norm_(x);
    }

    return {x, sample_lens};
  }

 private:
  int64_t dim_ = 0;
  torch::nn::Embedding wte_{nullptr};
  torch::nn::Conv1d patch_embedder_{nullptr};
  std::vector<TextVAEBlock> blocks_;
  torch::nn::Linear final_layer_{nullptr};
  torch::nn::LayerNorm final_norm_{nullptr};
  bool use_variation_ = true;
  bool encoder_last_ln_ = true;
  int64_t block_size_ = 0;
  int64_t patch_size_ = 1;
};
TORCH_MODULE(ColaTextVAEEncoder);

// ---------------------------------------------------------------------------
// ColaTextVAE Decoder
//-----------
// Reference: modeling_cola_vae.py ColaTextVAEModel.decode()
// Latent projection -> transformer blocks -> unpatch -> final_norm ->
// vocab projection

class ColaTextVAEDecoderImpl final : public torch::nn::Module {
 public:
  ColaTextVAEDecoderImpl(const ModelContext& ctx) {
    const auto& args = ctx.get_model_args();
    int64_t vae_dim = args.vae_dim();
    int64_t vae_num_heads = args.vae_num_heads();
    int64_t ffn_dim = args.ffn_dim();
    int64_t shared_heads_kv = args.shared_heads_kv();
    int64_t vae_rope_theta = args.vae_rope_theta();
    int64_t latent_dim = args.latent_dim();
    int64_t patch_size = args.vae_patch_size();
    int64_t decoder_num_blocks = args.decoder_num_blocks();
    int64_t vocab_size = args.vocab_size();
    bool qk_bias = args.qk_bias();

    in_layer_ =
        register_module("in_layer", torch::nn::Linear(latent_dim, vae_dim));

    blocks_.reserve(decoder_num_blocks);
    for (int64_t i = 0; i < decoder_num_blocks; ++i) {
      auto block = TextVAEBlock(vae_dim,
                                ffn_dim,
                                vae_num_heads,
                                shared_heads_kv,
                                vae_rope_theta,
                                qk_bias);
      blocks_.push_back(block);
      register_module("blocks_" + std::to_string(i), block);
    }

    unpatch_layer_ = register_module(
        "unpatch_layer", torch::nn::Linear(vae_dim, patch_size * vae_dim));
    final_norm_ = register_module(
        "final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({vae_dim})));
    final_layer_ =
        register_module("final_layer", torch::nn::Linear(vae_dim, vocab_size));

    block_size_ = args.vae_block_size();
    patch_size_ = patch_size;
  }

  // Clear per-layer KV caches on all decoder blocks.
  void set_kv_cache(bool flag) {
    for (auto& block : blocks_) {
      block->set_kv_cache(flag);
    }
  }

  // Decode Q-side latents into vocabulary logits, using per-layer KV cache
  // to access the context (prefix + committed blocks).
  //
  // Matches the official Python ColaTextVAEModel.decode():
  //   z:          (L_q, latent_dim) — current block latents (Q-side only)
  //   k_lens:     per-sample cumulative K lengths (cache + current Q)
  //   q_lens:     per-sample Q lengths (= block_size during generation)
  //   update_kv:  commit current K/V to per-layer cache (True when
  //               prefetching prefix or committing a denoised block)
  //
  // Returns: (1, L_q * patch_size, vocab_size)
  torch::Tensor decode(const torch::Tensor& z,
                       const std::vector<int64_t>& k_lens,
                       const std::vector<int64_t>& q_lens,
                       bool update_kv = false) {
    torch::NoGradGuard no_grad;

    int64_t L_q = z.size(0);

    auto hidden = in_layer_->forward(z);  // (L_q, dim)

    // Block-causal attention mask for Q × K.
    torch::Tensor attn_mask;
    if (block_size_ > 0) {
      attn_mask = create_block_causal_mask(
          k_lens, q_lens, block_size_, hidden.scalar_type(), hidden.device());
    }

    // Pass Q-only input through each block; KV cache provides context K/V.
    for (auto& block : blocks_) {
      hidden = block->forward(
          hidden, k_lens, q_lens, block_size_, attn_mask, update_kv);
    }

    // Unpatch: (L_q, dim) → (L_q, dim*patch_size) → (L_q*patch_size, dim)
    hidden = unpatch_layer_->forward(hidden);
    hidden = hidden.reshape({L_q * patch_size_, -1});
    hidden = final_norm_(hidden);
    hidden = final_layer_->forward(hidden);  // (L_q*ps, vocab)
    return hidden.unsqueeze(0);              // (1, L_q*ps, vocab)
  }

 private:
  torch::nn::Linear in_layer_{nullptr};
  std::vector<TextVAEBlock> blocks_;
  torch::nn::Linear unpatch_layer_{nullptr};
  torch::nn::LayerNorm final_norm_{nullptr};
  torch::nn::Linear final_layer_{nullptr};
  int64_t block_size_ = 0;
  int64_t patch_size_ = 1;
};
TORCH_MODULE(ColaTextVAEDecoder);

// ---------------------------------------------------------------------------
// ColaTextVAEModel (main model combining encoder + decoder)
// ---------------------------------------------------------------------------
// Reference: modeling_cola_vae.py ColaTextVAEModel

class ColaTextVAEModelImpl final : public torch::nn::Module {
 public:
  explicit ColaTextVAEModelImpl(const ModelContext& ctx) {
    encoder_ = register_module("encoder", ColaTextVAEEncoder(ctx));
    decoder_ = register_module("decoder", ColaTextVAEDecoder(ctx));

    const auto& args = ctx.get_model_args();
    scaling_factor_ = args.scaling_factor();
    shifting_factor_ = args.shifting_factor();
    use_variation_ = args.use_variation();
    patch_size_ = args.vae_patch_size();
  }

  // Encode input_ids to latents with optional scaling/shifting.
  // Returns per-sample latents and their lengths.
  std::pair<torch::Tensor, std::vector<int64_t>> encode(
      const std::vector<torch::Tensor>& input_ids_list) {
    auto [latents, sample_lens] = encoder_->encode(input_ids_list);

    // Apply DiagonalGaussianDistribution if using variation
    if (use_variation_) {
      // Split into mean and logvar, sample from distribution
      auto chunks = latents.chunk(2, /*dim=*/-1);
      auto mean = chunks[0];
      auto logvar = torch::clamp(chunks[1], -30.0f, 20.0f);
      // Use mean (deterministic) for inference
      latents = mean;
    }

    // Apply scaling and shifting: z = (z - shifting) * scaling
    if (scaling_factor_ != 1.0f || shifting_factor_ != 0.0f) {
      latents = (latents - shifting_factor_) * scaling_factor_;
    }

    return {latents, sample_lens};
  }

  // Decode latents to vocabulary logits.
  torch::Tensor decode(const torch::Tensor& z,
                       const std::vector<int64_t>& k_lens,
                       const std::vector<int64_t>& q_lens,
                       bool update_kv = false) {
    return decoder_->decode(z, k_lens, q_lens, update_kv);
  }

  // Clear decoder KV caches (encoder has no cache; it runs self-attention
  // once).
  void set_kv_cache(bool flag) { decoder_->set_kv_cache(flag); }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    load_cola_module_from_state_dicts(
        *loader, encoder_.ptr().get(), "encoder.");
    load_cola_module_from_state_dicts(
        *loader, decoder_.ptr().get(), "decoder.");
  }

  float scaling_factor() const { return scaling_factor_; }
  float shifting_factor() const { return shifting_factor_; }
  int64_t patch_size() const { return patch_size_; }

 private:
  ColaTextVAEEncoder encoder_{nullptr};
  ColaTextVAEDecoder decoder_{nullptr};
  float scaling_factor_ = 1.0f;
  float shifting_factor_ = 0.0f;
  bool use_variation_ = true;
  int64_t patch_size_ = 1;
};
TORCH_MODULE(ColaTextVAEModel);

// ---------------------------------------------------------------------------
// REGISTER_MODEL_ARGS for ColaTextVAE config.json
// ---------------------------------------------------------------------------

REGISTER_MODEL_ARGS(cola_text_vae, [&] {
  LOAD_ARG_OR(model_type, "model_type", "cola_text_vae");
  LOAD_ARG(vocab_size, "vocab_size");
  LOAD_ARG(vae_dim, "dim");
  LOAD_ARG(vae_num_heads, "num_heads");
  LOAD_ARG(encoder_num_blocks, "encoder_num_blocks");
  LOAD_ARG(decoder_num_blocks, "decoder_num_blocks");
  LOAD_ARG(ffn_dim, "ffn_dim");
  LOAD_ARG(latent_dim, "latent_dim");
  LOAD_ARG(shared_heads_kv, "shared_heads_kv");
  LOAD_ARG_OR(vae_rope_theta, "rope_theta", 500000);
  LOAD_ARG_OR(vae_block_size, "block_size", 4);
  LOAD_ARG_OR(vae_patch_size, "patch_size", 1);
  LOAD_ARG_OR(encoder_last_ln, "encoder_last_ln", true);
  LOAD_ARG_OR(use_variation, "use_variation", true);
  LOAD_ARG_OR(qk_bias, "qk_bias", false);
  LOAD_ARG_OR(scaling_factor, "scaling_factor", 1.0f);
  LOAD_ARG_OR(shifting_factor, "shifting_factor", 0.0f);
});

}  // namespace xllm
