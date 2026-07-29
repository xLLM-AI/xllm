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

#pragma once

#include <torch/torch.h>

#include <cmath>
#include <tuple>
#include <vector>

#if defined(USE_NPU)
#include "core/kernels/npu/npu_ops_api.h"
#endif

namespace xllm::dit {

struct SparseAttnConfig {
  float sparsity = 0.5;
  int64_t pool_size = 128;
  int64_t sparse_start_step = 0;  // skip_timesteps
  bool enabled = false;
  std::string version = "rain_fusion";  // "rain_fusion" or "sparse_attention"
  int64_t inner_precise = 0;  // sparse_attention: 0=float32 softmax, 1=fp16
  int64_t mask_refresh_steps =
      1;  // sparse_attention: recompute mask every N steps
};

// Per-request dynamic state for sparse / RainFusion attention.
struct SparseAttnState {
  int64_t current_step = 0;
  std::vector<int64_t> latent_shape = {1, 1, 1};
  int64_t seq_len = -1;
  torch::Tensor
      cached_mask;  // sparse_attention: cached binary int8 block_sparse_mask
};

// =========================================================================
//  Common helpers (shared by both rain_fusion and sparse_attention)
// =========================================================================

inline torch::Tensor avgpool(const torch::Tensor& input,
                             int64_t pool_size,
                             const std::string& input_layout) {
  std::vector<torch::Tensor> pooled_parts;
  if (input_layout == "BSND") {
    int64_t batch = input.size(0);
    int64_t seqlen = input.size(1);
    int64_t headnum = input.size(2);
    int64_t dim = input.size(3);
    int64_t num_full_blocks = seqlen / pool_size;
    int64_t tail_size = seqlen % pool_size;

    if (num_full_blocks > 0) {
      auto full_blocks = input.slice(1, 0, num_full_blocks * pool_size);
      full_blocks = full_blocks.reshape(
          {batch, num_full_blocks, pool_size, headnum, dim});
      pooled_parts.emplace_back(full_blocks.mean(2));
    }
    if (tail_size > 0) {
      auto tail_block = input.slice(1, num_full_blocks * pool_size, seqlen);
      tail_block = tail_block.reshape({batch, 1, tail_size, headnum, dim});
      pooled_parts.emplace_back(tail_block.mean(2));
    }
  } else {
    // BNSD: [B, N, S, D]
    int64_t batch = input.size(0);
    int64_t headnum = input.size(1);
    int64_t seqlen = input.size(2);
    int64_t dim = input.size(3);
    int64_t num_full_blocks = seqlen / pool_size;
    int64_t tail_size = seqlen % pool_size;

    if (num_full_blocks > 0) {
      auto full_blocks = input.slice(2, 0, num_full_blocks * pool_size);
      full_blocks = full_blocks.reshape(
          {batch, headnum, num_full_blocks, pool_size, dim});
      pooled_parts.emplace_back(full_blocks.mean(3));
    }
    if (tail_size > 0) {
      auto tail_block = input.slice(2, num_full_blocks * pool_size, seqlen);
      tail_block = tail_block.reshape({batch, headnum, 1, tail_size, dim});
      pooled_parts.emplace_back(tail_block.mean(3));
    }
  }
  if (pooled_parts.size() > 1) {
    int64_t cat_dim = (input_layout == "BSND") ? 1 : 2;
    return torch::cat(pooled_parts, cat_dim);
  }
  return pooled_parts[0];
}

// =========================================================================
//  RainFusion attention (V2) — frame-pairing + aclnnRainFusionAttention
//  Internal helpers live in the rain_fusion namespace to scope
//  preprocessing from the sparse_attention variant.
// =========================================================================
#if defined(USE_NPU)
namespace rain_fusion {

// Layout helpers: merge B*N for NPU 8-dim compliance.
// The reference Wan2.2 code uses einops with (b n) pre-merge:
//   rearrange(tensor, 'b (f h w) n d -> (b n) f h w d', ...)
// This keeps all intermediate tensors ≤ 8 dims and eliminates
// per-layout code branches.

inline torch::Tensor to_flat(const torch::Tensor& t,
                             const std::string& layout) {
  if (layout == "BNSD") {
    // .contiguous() needed: SP padding strip via slice() makes tensor
    // non-contiguous; without it reshape does a hidden device copy.
    auto ct = t.contiguous();
    return ct.view({ct.size(0) * ct.size(1), ct.size(2), ct.size(3)});
  }
  // BSND: [B, S, N, D] → [B*N, S, D]
  auto ct = t.permute({0, 2, 1, 3}).contiguous();
  return ct.view({ct.size(0), ct.size(1), ct.size(2)});
}

inline torch::Tensor from_flat(const torch::Tensor& t, int64_t B, int64_t N) {
  // view requires contiguous; aclnn output may be non-contiguous
  auto ct = t.contiguous();
  return ct.view({B, N, ct.size(1), ct.size(2)});  // → [B, N, S, D]
}

// ---- Frame-pairing rearrange ----
// Splits first frame (placed at END), pairs remaining frames (fn=tq/2, fb=2).
// All internal ops work on [B*N, S, D] — max 8 dims after splitting.
// Output: [B, N, rearranged_rest+first_frame, D]
inline torch::Tensor rearrange_with_remaining(const torch::Tensor& tensor,
                                              int64_t tq,
                                              int64_t hq,
                                              int64_t wq,
                                              const std::string& input_layout) {
  int64_t B = (input_layout == "BNSD") ? tensor.size(0) : tensor.size(0);
  int64_t N = (input_layout == "BNSD") ? tensor.size(1) : tensor.size(2);
  int64_t D = (input_layout == "BNSD") ? tensor.size(3) : tensor.size(3);
  int64_t first_frame_len = hq * wq;
  int64_t rest_frames = tq - 1;
  int64_t fn = tq / 2;
  int64_t fb = 2;
  int64_t hn = hq / 8;
  int64_t wn = wq / 8;
  CHECK(tq % 2 == 1) << "V2 frame-pairing requires odd tq, got " << tq;

  // Flatten to [B*N, S, D], split first frame
  auto t = to_flat(tensor, input_layout);
  auto t_first = t.slice(1, 0, first_frame_len);
  auto t_rest = t.slice(1, first_frame_len, t.size(1));

  if (hq % 8 == 0 && wq % 8 == 0) {
    // Aligned: single-step frame-pairing (8 dims: B*N,fn,fb,hn,8,wn,8,D)
    t_rest = t_rest.reshape({B * N, fn, fb, hn, 8, wn, 8, D})
                 .permute({0, 1, 3, 5, 2, 4, 6, 7})
                 .contiguous()
                 .reshape({B * N, -1, D});
  } else {
    // Remainder: split h/w remainders, pair only aligned portion
    int64_t hq_block = (hq / 8) * 8;
    int64_t wq_block = (wq / 8) * 8;
    int64_t hq_rem = hq % 8;
    int64_t wq_rem = wq % 8;
    int64_t hn_a = hq_block / 8;
    int64_t wn_a = wq_block / 8;

    t_rest = t_rest.reshape({B * N, rest_frames, hq, wq, D});
    torch::Tensor t_h_r, t_w_r;
    if (hq_rem > 0) {
      auto s = t_rest.split({hq_block, hq_rem}, 2);
      t_rest = s[0];
      t_h_r = s[1].reshape({B * N, rest_frames, -1, wq, D});
    }
    if (wq_rem > 0) {
      auto s = t_rest.split({wq_block, wq_rem}, 3);
      t_rest = s[0];
      t_w_r = s[1].reshape({B * N, rest_frames, hq_block, -1, D});
    }
    // Frame-pairing aligned portion (8 dims)
    t_rest = t_rest.reshape({B * N, fn, fb, hn_a, 8, wn_a, 8, D})
                 .permute({0, 1, 3, 5, 2, 4, 6, 7})
                 .contiguous()
                 .reshape({B * N, -1, D});
    if (hq_rem > 0) {
      t_rest = torch::cat({t_rest, t_h_r.reshape({B * N, -1, D})}, 1);
    }
    if (wq_rem > 0) {
      t_rest = torch::cat({t_rest, t_w_r.reshape({B * N, -1, D})}, 1);
    }
  }
  t = torch::cat({t_rest, t_first}, 1);
  return from_flat(t, B, N);
}

// ---- Inverse rearrange ----
// Reverses frame-pairing: [B*N, paired_rest+first_frame, D] → [B,N,S,D]
// Works on flat layout; max 8 dims.
inline torch::Tensor bsa_inv_rearrange(const torch::Tensor& out,
                                       int64_t tq,
                                       int64_t hq,
                                       int64_t wq,
                                       const std::string& input_layout) {
  int64_t B = (input_layout == "BNSD") ? out.size(0) : out.size(0);
  int64_t N = (input_layout == "BNSD") ? out.size(1) : out.size(2);
  int64_t D = (input_layout == "BNSD") ? out.size(3) : out.size(3);
  int64_t first_frame_len = hq * wq;
  int64_t rest_frames = tq - 1;
  int64_t fn = tq / 2;
  int64_t fb = 2;
  int64_t hn = hq / 8;
  int64_t wn = wq / 8;

  auto t = to_flat(out, input_layout);
  auto t_first = t.slice(1, -first_frame_len, t.size(1));
  auto t_rest = t.slice(1, 0, -first_frame_len);

  if (hq % 8 == 0 && wq % 8 == 0) {
    // Aligned: un-pair (8 dims)
    t_rest = t_rest.reshape({B * N, fn, hn, wn, fb, 8, 8, D})
                 .permute({0, 1, 4, 2, 5, 3, 6, 7})
                 .contiguous()
                 .reshape({B * N, rest_frames * hq * wq, D});
  } else {
    // Remainder
    int64_t hq_block = (hq / 8) * 8;
    int64_t wq_block = (wq / 8) * 8;
    int64_t hq_rem = hq % 8;
    int64_t wq_rem = wq % 8;
    int64_t hn_a = hq_block / 8;
    int64_t wn_a = wq_block / 8;
    int64_t h_rem_size = hq_rem * wq;
    int64_t block_tokens = fn * hn_a * wn_a * 128;

    t_rest = t_rest.reshape({B * N, rest_frames * hq * wq, D});
    auto t_block = t_rest.slice(1, 0, block_tokens);
    torch::Tensor t_h_r, t_w_r;
    int64_t offset = block_tokens;
    if (hq_rem > 0) {
      t_h_r = t_rest.slice(1, offset, offset + h_rem_size * rest_frames);
      offset += h_rem_size * rest_frames;
    }
    if (wq_rem > 0) {
      t_w_r = t_rest.slice(1, offset);
    }

    t_block = t_block.reshape({B * N, fn, hn_a, wn_a, fb, 8, 8, D})
                  .permute({0, 1, 4, 2, 5, 3, 6, 7})
                  .contiguous()
                  .reshape({B * N, rest_frames, hq_block, wq_block, D});
    if (wq_rem > 0) {
      t_block = torch::cat(
          {t_block, t_w_r.reshape({B * N, rest_frames, hq_block, wq_rem, D})},
          3);
    }
    if (hq_rem > 0) {
      t_block = torch::cat(
          {t_block, t_h_r.reshape({B * N, rest_frames, hq_rem, wq, D})}, 2);
    }
    t_rest = t_block.reshape({B * N, rest_frames * hq * wq, D});
  }
  // Restore original order: [first_frame, un-rearranged_rest]
  // (matches reference do_tensor_inv_rearrange: concat(tensor_f, tensor_i))
  t = torch::cat({t_first, t_rest}, 1);
  return from_flat(t, B, N);
}

// ---- Mask helpers ----

inline std::pair<torch::Tensor, torch::Tensor> mask_to_select_idx(
    const torch::Tensor& mask_b0) {
  int64_t N = mask_b0.size(0);
  int64_t q_blocks = mask_b0.size(1);
  int64_t kv_blocks = mask_b0.size(2);

  auto row_indices =
      torch::arange(kv_blocks, mask_b0.options().dtype(torch::kInt64))
          .view({1, 1, kv_blocks})
          .expand({N, q_blocks, kv_blocks});

  auto sorted_vals =
      torch::where(mask_b0,
                   row_indices,
                   torch::full({}, 1000000000LL, row_indices.options()));
  auto sorted_sorted = std::get<0>(torch::sort(sorted_vals, -1));

  auto valid_count = mask_b0.sum(-1);
  auto keep_mask = row_indices < valid_count.unsqueeze(-1);

  auto select_idx = torch::where(keep_mask,
                                 sorted_sorted.to(torch::kInt64),
                                 torch::full({}, -1LL, row_indices.options()));
  auto select_num_idx = valid_count.to(torch::kInt64);
  return {select_idx, select_num_idx};
}

inline std::pair<torch::Tensor, torch::Tensor> get_blockwise_mask(
    const torch::Tensor& q_pool,
    const torch::Tensor& k_pool,
    int64_t txt_len,
    float sparsity,
    double scale,
    int64_t pool_size,
    int64_t tq,
    int64_t hq,
    int64_t wq,
    const std::string& input_layout) {
  torch::Tensor attn_scores;
  if (input_layout == "BSND") {
    attn_scores = torch::einsum("blnd,bsnd->bnls", {q_pool, k_pool}) * scale;
  } else {
    attn_scores = torch::einsum("bnld,bnsd->bnls", {q_pool, k_pool}) * scale;
  }

  auto score_matrix = torch::softmax(attn_scores, -1);
  int64_t cols = score_matrix.size(-1);
  int64_t keep_len =
      std::max(static_cast<int64_t>(std::ceil(cols * (1.0 - sparsity))),
               static_cast<int64_t>(1));

  auto topk_result = score_matrix.topk(keep_len, -1);
  auto thresholds = std::get<0>(topk_result).slice(-1, keep_len - 1, keep_len);
  auto mask = score_matrix >= thresholds;

  // RainFusion: first frame at END → protect LAST blocks
  int64_t first_frame_len = hq * wq;
  int64_t protect_len = (first_frame_len + txt_len + pool_size - 1) / pool_size;
  if (protect_len > 0) {
    mask.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(-protect_len, torch::indexing::None),
         torch::indexing::Slice()},
        true);
    mask.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(-protect_len, torch::indexing::None)},
        true);
  }
  // Reference transposes to [q_blocks, N, max_select] before passing to
  // aclnnRainFusionAttention (see Wan2.2 get_blockwise_mask).
  auto [si, sn] = mask_to_select_idx(mask[0]);      // [N, q_blocks, kv_blocks]
  return {si.transpose(0, 1), sn.transpose(0, 1)};  // → [q_blocks, N, ...]
}

// RainFusion block sparse attention — main entry point.
inline std::tuple<torch::Tensor, torch::Tensor> attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const SparseAttnConfig& config,
    SparseAttnState& state) {
  CHECK(query.dim() == 4) << "query must be 4D [B, N, S, D]";
  CHECK(query.dtype() == torch::kHalf || query.dtype() == torch::kBFloat16)
      << "query must be fp16 or bf16";
  CHECK(config.sparsity >= 0.0f && config.sparsity < 1.0f)
      << "sparsity must be in [0.0, 1.0)";
  CHECK(config.pool_size > 0 && config.pool_size % 128 == 0)
      << "pool_size must be positive multiple of 128";
  int64_t tq = state.latent_shape[0];
  int64_t hq = state.latent_shape[1];
  int64_t wq = state.latent_shape[2];
  int64_t num_heads = query.size(1);
  double scale = std::pow(static_cast<double>(query.size(-1)), -0.5);

  // Step 1: Cat QKV → rearrange once → chunk (matches reference:
  //   rainfusion_blockwise.py cat(q,k,v) → do_tensor_rearrange_pooling)
  auto qkv = torch::cat({query, key, value}, 0);
  auto qkv_r = rearrange_with_remaining(qkv, tq, hq, wq, "BNSD");
  auto qkv_chunks = qkv_r.chunk(3, 0);  // Q', K', V'
  auto q_r = qkv_chunks[0];
  auto k_r = qkv_chunks[1];
  auto v_r = qkv_chunks[2];

  // Step 2: Cat Q'+K' → avgpool together → compute mask
  auto qk = torch::cat({q_r, k_r}, 0);
  auto qk_pool = avgpool(qk, config.pool_size, "BNSD");
  auto pool_chunks = qk_pool.chunk(2, 0);
  auto [sel_idx, sel_num] = get_blockwise_mask(pool_chunks[0],
                                               pool_chunks[1],
                                               /*txt_len=*/0,
                                               config.sparsity,
                                               scale,
                                               config.pool_size,
                                               tq,
                                               hq,
                                               wq,
                                               "BNSD");

  // Step 3: RainFusion block sparse attention
  int64_t batch_size = query.size(0);
  std::vector<int64_t> seq_lens(batch_size, q_r.size(2));
  constexpr int64_t kInnerPrecise = 0;

  auto [attn_out, lse] = xllm::kernel::npu::npu_rain_fusion_attention(
      q_r,
      k_r,
      v_r,
      sel_idx,
      sel_num,
      {config.pool_size, config.pool_size},
      "BNSD",
      "BNSD",
      num_heads,
      scale,
      kInnerPrecise,
      std::optional<torch::IntArrayRef>(seq_lens),
      std::optional<torch::IntArrayRef>(seq_lens));

  // Step 4: Inverse rearrange
  auto out = bsa_inv_rearrange(attn_out, tq, hq, wq, "BNSD");

  return std::make_tuple(out, sel_idx);
}

}  // namespace rain_fusion

// =========================================================================
//  Sparse attention (V3) — block-decompose + aclnnBlockSparseAttention
//  Internal helpers live in the sparse_attention namespace to scope
//  preprocessing from the rain_fusion variant.
// =========================================================================
namespace sparse_attention {

// Rearrange: block-decompose (f hn hb wn wb) → (f hn wn hb wb).
// Remainder: first frame kept raw, rest block-decomposed per-frame.
inline torch::Tensor rearrange_v3(const torch::Tensor& tensor,
                                  int64_t tq,
                                  int64_t hq,
                                  int64_t wq,
                                  const std::string& input_layout) {
  int64_t first_frame_len = hq * wq;
  int64_t frame_num = tq;

  if (hq % 8 == 0 && wq % 8 == 0) {
    int64_t hn = hq / 8, wn = wq / 8;
    if (input_layout == "BSND") {
      return tensor
          .reshape({tensor.size(0),
                    frame_num,
                    hn,
                    8,
                    wn,
                    8,
                    tensor.size(2),
                    tensor.size(3)})
          .permute({0, 1, 2, 4, 3, 5, 6, 7})
          .contiguous()
          .reshape({tensor.size(0), -1, tensor.size(2), tensor.size(3)});
    }
    return tensor
        .reshape({tensor.size(0),
                  tensor.size(1),
                  frame_num,
                  hn,
                  8,
                  wn,
                  8,
                  tensor.size(3)})
        .permute({0, 1, 2, 3, 5, 4, 6, 7})
        .contiguous()
        .reshape({tensor.size(0), tensor.size(1), -1, tensor.size(3)});
  }

  int64_t hq_block = (hq / 8) * 8, wq_block = (wq / 8) * 8;
  int64_t hq_rem = hq % 8, wq_rem = wq % 8;
  int64_t hn = hq_block / 8, wn = wq_block / 8;

  if (input_layout == "BSND") {
    auto t_first = tensor.slice(1, 0, first_frame_len);
    auto t_rest = tensor.slice(1, first_frame_len);
    t_rest = t_rest.reshape({tensor.size(0),
                             frame_num - 1,
                             hq,
                             wq,
                             tensor.size(2),
                             tensor.size(3)});
    torch::Tensor t_h_r, t_w_r;
    if (hq_rem > 0) {
      auto s = t_rest.split({hq_block, hq_rem}, 2);
      t_rest = s[0];
      t_h_r = s[1].reshape(
          {tensor.size(0), frame_num - 1, -1, tensor.size(2), tensor.size(3)});
    }
    if (wq_rem > 0) {
      auto s = t_rest.split({wq_block, wq_rem}, 3);
      t_rest = s[0];
      t_w_r = s[1].reshape(
          {tensor.size(0), frame_num - 1, -1, tensor.size(2), tensor.size(3)});
    }
    t_rest = t_rest
                 .reshape({tensor.size(0),
                           frame_num - 1,
                           hn,
                           8,
                           wn,
                           8,
                           tensor.size(2),
                           tensor.size(3)})
                 .permute({0, 1, 2, 4, 3, 5, 6, 7})
                 .contiguous()
                 .reshape({tensor.size(0),
                           frame_num - 1,
                           hn * wn * 64,
                           tensor.size(2),
                           tensor.size(3)});
    if (hq_rem > 0) t_rest = torch::cat({t_rest, t_h_r}, 2);
    if (wq_rem > 0) t_rest = torch::cat({t_rest, t_w_r}, 2);
    t_rest =
        t_rest.reshape({tensor.size(0), -1, tensor.size(2), tensor.size(3)});
    return torch::cat({t_first, t_rest}, 1);
  }

  // BNSD remainder
  auto t_first = tensor.slice(2, 0, first_frame_len);
  auto t_rest = tensor.slice(2, first_frame_len);
  t_rest = t_rest.reshape(
      {tensor.size(0), tensor.size(1), frame_num - 1, hq, wq, tensor.size(3)});
  torch::Tensor t_h_r, t_w_r;
  if (hq_rem > 0) {
    auto s = t_rest.split({hq_block, hq_rem}, 3);
    t_rest = s[0];
    t_h_r = s[1].reshape(
        {tensor.size(0), tensor.size(1), frame_num - 1, -1, tensor.size(3)});
  }
  if (wq_rem > 0) {
    auto s = t_rest.split({wq_block, wq_rem}, 4);
    t_rest = s[0];
    t_w_r = s[1].reshape(
        {tensor.size(0), tensor.size(1), frame_num - 1, -1, tensor.size(3)});
  }
  t_rest = t_rest
               .reshape({tensor.size(0),
                         tensor.size(1),
                         frame_num - 1,
                         hn,
                         8,
                         wn,
                         8,
                         tensor.size(3)})
               .permute({0, 1, 2, 3, 5, 4, 6, 7})
               .contiguous()
               .reshape({tensor.size(0),
                         tensor.size(1),
                         frame_num - 1,
                         hn * wn * 64,
                         tensor.size(3)});
  if (hq_rem > 0) t_rest = torch::cat({t_rest, t_h_r}, 3);
  if (wq_rem > 0) t_rest = torch::cat({t_rest, t_w_r}, 3);
  t_rest = t_rest.reshape({tensor.size(0), tensor.size(1), -1, tensor.size(3)});
  return torch::cat({t_first, t_rest}, 2);
}

// Inverse rearrange — reverses rearrange_v3.
inline torch::Tensor bsa_inv_rearrange_v3(const torch::Tensor& out,
                                          int64_t tq,
                                          int64_t hq,
                                          int64_t wq,
                                          const std::string& input_layout) {
  int64_t hn = hq / 8, wn = wq / 8;

  if (hq % 8 == 0 && wq % 8 == 0) {
    if (input_layout == "BNSD") {
      int64_t n = out.size(1);
      return out.reshape({out.size(0), n, tq, hn, wn, 8, 8, out.size(3)})
          .permute({0, 1, 2, 3, 5, 4, 6, 7})
          .contiguous()
          .reshape({out.size(0), n, tq * hq * wq, out.size(3)});
    }
    int64_t n = out.size(2);
    return out.reshape({out.size(0), tq, hn, wn, 8, 8, n, out.size(3)})
        .permute({0, 1, 2, 4, 3, 5, 6, 7})
        .contiguous()
        .reshape({out.size(0), tq * hq * wq, n, out.size(3)});
  }

  int64_t first_frame_len = hq * wq;
  int64_t hq_block = (hq / 8) * 8, wq_block = (wq / 8) * 8;
  int64_t hq_rem = hq % 8, wq_rem = wq % 8;
  int64_t block_size = hn * wn * 64;
  int64_t h_rem_size = hq_rem * wq;

  if (input_layout == "BNSD") {
    int64_t n = out.size(1);
    auto o_first = out.slice(2, 0, first_frame_len);
    auto o_rest = out.slice(2, first_frame_len);
    o_rest = o_rest.reshape({out.size(0), n, tq - 1, hq * wq, out.size(3)});

    auto t_block = o_rest.slice(3, 0, block_size);
    torch::Tensor t_h_r, t_w_r;
    if (hq_rem > 0)
      t_h_r = o_rest.slice(3, block_size, block_size + h_rem_size);
    if (wq_rem > 0) t_w_r = o_rest.slice(3, block_size + h_rem_size);

    t_block =
        t_block.reshape({out.size(0), n, tq - 1, hn, wn, 8, 8, out.size(3)})
            .permute({0, 1, 2, 3, 5, 4, 6, 7})
            .contiguous()
            .reshape({out.size(0), n, tq - 1, hq_block, wq_block, out.size(3)});
    if (wq_rem > 0)
      t_block = torch::cat(
          {t_block,
           t_w_r.reshape(
               {out.size(0), n, tq - 1, hq_block, wq_rem, out.size(3)})},
          4);
    if (hq_rem > 0)
      t_block = torch::cat(
          {t_block,
           t_h_r.reshape({out.size(0), n, tq - 1, hq_rem, wq, out.size(3)})},
          3);
    auto o_rest_out =
        t_block.reshape({out.size(0), n, (tq - 1) * hq * wq, out.size(3)});
    return torch::cat({o_first, o_rest_out}, 2);
  }

  // BSND remainder
  int64_t n = out.size(2);
  auto o_first = out.slice(1, 0, first_frame_len);
  auto o_rest = out.slice(1, first_frame_len);
  o_rest = o_rest.reshape({out.size(0), tq - 1, hq * wq, n, out.size(3)});

  auto t_block = o_rest.slice(2, 0, block_size);
  torch::Tensor t_h_r, t_w_r;
  if (hq_rem > 0) t_h_r = o_rest.slice(2, block_size, block_size + h_rem_size);
  if (wq_rem > 0) t_w_r = o_rest.slice(2, block_size + h_rem_size);

  t_block =
      t_block.reshape({out.size(0), tq - 1, hn, wn, 8, 8, n, out.size(3)})
          .permute({0, 1, 2, 4, 3, 5, 6, 7})
          .contiguous()
          .reshape({out.size(0), tq - 1, hq_block, wq_block, n, out.size(3)});
  if (wq_rem > 0)
    t_block = torch::cat(
        {t_block,
         t_w_r.reshape(
             {out.size(0), tq - 1, hq_block, wq_rem, n, out.size(3)})},
        3);
  if (hq_rem > 0)
    t_block = torch::cat(
        {t_block,
         t_h_r.reshape({out.size(0), tq - 1, hq_rem, wq, n, out.size(3)})},
        2);
  auto o_rest_out =
      t_block.reshape({out.size(0), (tq - 1) * hq * wq, n, out.size(3)});
  return torch::cat({o_first, o_rest_out}, 1);
}

// Binary mask for aclnnBlockSparseAttention.  Returns int8
// [B,N,q_blocks,kv_blocks].
inline torch::Tensor get_blockwise_mask_v3(const torch::Tensor& q_pool,
                                           const torch::Tensor& k_pool,
                                           int64_t txt_len,
                                           float sparsity,
                                           double scale,
                                           int64_t pool_size,
                                           int64_t tq,
                                           int64_t hq,
                                           int64_t wq,
                                           const std::string& input_layout,
                                           bool protect_first_frame) {
  torch::Tensor attn_scores;
  if (input_layout == "BSND")
    attn_scores = torch::einsum("blnd,bsnd->bnls", {q_pool, k_pool}) * scale;
  else
    attn_scores = torch::einsum("bnld,bnsd->bnls", {q_pool, k_pool}) * scale;

  auto score_matrix = torch::softmax(attn_scores, -1);
  int64_t cols = score_matrix.size(-1);
  int64_t keep_len = std::max(
      static_cast<int64_t>(std::ceil(cols * (1.0 - sparsity))), int64_t(1));
  auto topk_result = score_matrix.topk(keep_len, -1);
  auto thresholds = std::get<0>(topk_result).slice(-1, keep_len - 1, keep_len);
  auto mask = score_matrix >= thresholds;

  int64_t text_block_num = (txt_len + pool_size - 1) / pool_size;
  if (text_block_num > 0) {
    mask.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(-text_block_num, torch::indexing::None),
         torch::indexing::Slice()},
        true);
    mask.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(),
         torch::indexing::Slice(-text_block_num, torch::indexing::None)},
        true);
  }
  if (protect_first_frame) {
    int64_t nblocks = (hq * wq + pool_size - 1) / pool_size;
    if (nblocks > 0) {
      mask.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(0, nblocks),
                       torch::indexing::Slice()},
                      true);
      mask.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(),
                       torch::indexing::Slice(0, nblocks)},
                      true);
    }
  }
  return mask.to(torch::kInt8);
}

// Sparse attention — main entry point. Ported from wan22_rainfusion.
inline std::tuple<torch::Tensor, torch::Tensor> attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const SparseAttnConfig& config,
    SparseAttnState& state) {
  CHECK(query.dim() == 4) << "query must be 4D [B, N, S, D]";
  int64_t tq = state.latent_shape[0], hq = state.latent_shape[1],
          wq = state.latent_shape[2];
  int64_t num_heads = query.size(1);
  double scale = std::pow(static_cast<double>(query.size(-1)), -0.5);
  int64_t batch_size = query.size(0);

  auto q_r = rearrange_v3(query, tq, hq, wq, "BNSD");
  auto k_r = rearrange_v3(key, tq, hq, wq, "BNSD");
  auto v_r = rearrange_v3(value, tq, hq, wq, "BNSD");

  torch::Tensor new_mask;
  bool use_cached = state.cached_mask.defined() &&
                    (config.mask_refresh_steps <= 0 ||
                     (state.current_step % config.mask_refresh_steps != 0));
  if (!use_cached) {
    auto qk = torch::cat({q_r, k_r}, 0);
    auto qk_pool = avgpool(qk, config.pool_size, "BNSD");
    auto chunks = qk_pool.chunk(2, 0);
    new_mask = get_blockwise_mask_v3(chunks[0],
                                     chunks[1],
                                     /*txt_len=*/0,
                                     config.sparsity,
                                     scale,
                                     config.pool_size,
                                     tq,
                                     hq,
                                     wq,
                                     "BNSD",
                                     /*protect_first_frame=*/true);
    state.cached_mask = new_mask;
  } else {
    new_mask = state.cached_mask;
  }

  std::vector<int64_t> seq_lens(batch_size, q_r.size(2));
  auto [attn_out, lse] = xllm::kernel::npu::npu_block_sparse_attention(
      q_r,
      k_r,
      v_r,
      new_mask,
      {config.pool_size, config.pool_size},
      "BNSD",
      "BNSD",
      num_heads,
      scale,
      config.inner_precise,
      /*softmax_lse_flag=*/0,
      std::optional<torch::IntArrayRef>(seq_lens),
      std::optional<torch::IntArrayRef>(seq_lens));

  auto out = bsa_inv_rearrange_v3(attn_out, tq, hq, wq, "BNSD");
  return std::make_tuple(out, new_mask);
}

}  // namespace sparse_attention
#endif  // defined(USE_NPU)

}  // namespace xllm::dit
