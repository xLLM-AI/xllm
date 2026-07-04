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

// RWKV-7 "Goose" decoder layer for xLLM.
//
// Implements the RWKV-7 time-mix (attention replacement) and channel-mix
// (FFN replacement) blocks.  Architecture reference:
//   https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7
//
// Weight naming convention follows the official BlinkDL checkpoint format:
//   blocks.{i}.att.*   → time-mix weights
//   blocks.{i}.ffn.*   → channel-mix weights
//   blocks.{i}.ln0/ln1/ln2 → layer norms

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

// RWKV-7 Time-Mix block (attention replacement).
//
// Key operations per token:
//   1. Token-shift mixing: xx = prev_token_embed - current_embed
//   2. Compute gating values via LoRA-style low-rank projections
//   3. In-context learning (a-gate) modulates the W-matrix update
//   4. W-matrix recurrence: state = state * decay + state @ ab + vk
//   5. Receptance-key residual connection for output correction
class RWKV7TimeMixImpl : public torch::nn::Module {
 public:
  RWKV7TimeMixImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  // Forward time-mix for all sequences in the current batch.
  //
  //   x           – layer-normed block input: [total_tokens, hidden_size]
  //   att_x_prev  – per-sequence shift state: [num_seqs, hidden_size]
  //   att_kv      – per-sequence W-matrix:   [num_seqs, n_heads, head_size,
  //   head_size]
  //                 Updated in-place during this call.
  //   is_layer0   – true for block 0 (initialises v_first)
  //   v_first     – cross-layer value state: [total_tokens, hidden_size]
  //   (in/out) seq_lens    – per-sequence query-token counts (host vector)
  //
  // Returns {time_mix_output [total_tokens, C],
  //          new_att_x_prev  [num_seqs, C]}
  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& att_x_prev,
      torch::Tensor& att_kv,
      bool is_layer0,
      torch::Tensor& v_first,
      const std::vector<int32_t>& seq_lens);

 private:
  // Compute per-token decay exp(-exp(w)) for one sequence.
  //   xw: [T, C] → output: [T, n_heads_, head_size_]
  torch::Tensor compute_decay(const torch::Tensor& xw) const;

  // Core RWKV-7 recurrence kernel (sequential over T) for one sequence.
  //   r, w, k, v, a, kk: [T, n_heads_, head_size_]
  //   state:              [n_heads_, head_size_, head_size_]
  // Returns {output [T, C], new_state [n_heads_, head_size_, head_size_]}.
  std::pair<torch::Tensor, torch::Tensor> rwkv7_recurrence(
      const torch::Tensor& r,
      const torch::Tensor& w,
      const torch::Tensor& k,
      const torch::Tensor& v,
      const torch::Tensor& a,
      const torch::Tensor& kk,
      const torch::Tensor& state) const;

  int64_t hidden_size_;
  int64_t n_heads_;
  int64_t head_size_;

  // Token-shift mixing scalars — registered parameters [1, 1, hidden_size].
  torch::Tensor x_r_, x_w_, x_k_, x_v_, x_a_, x_g_;

  // Decay LoRA: w = log_sigmoid(w0 + xw@w1@w2) - 0.5
  //   w0: [1, 1, C],  w1: [C, rank],  w2: [rank, C]
  //   Shapes of w1/w2 depend on the checkpoint rank; loaded dynamically.
  torch::Tensor w0_;  // registered parameter [1, 1, C]
  torch::Tensor w1_;  // plain tensor, shape from checkpoint
  torch::Tensor w2_;  // plain tensor, shape from checkpoint

  // A-gate LoRA: a = sigmoid(a0 + xa@a1@a2)
  torch::Tensor a0_;  // registered parameter [1, 1, C]
  torch::Tensor a1_;  // plain tensor
  torch::Tensor a2_;  // plain tensor

  // V-first blend LoRA: blend = sigmoid(v0 + xv@v1@v2)
  torch::Tensor v0_;  // registered parameter [1, 1, C]
  torch::Tensor v1_;  // plain tensor
  torch::Tensor v2_;  // plain tensor

  // Gate LoRA: g = sigmoid(xg@g1) @ g2
  torch::Tensor g1_;  // plain tensor
  torch::Tensor g2_;  // plain tensor

  // Key modifiers — registered parameters [1, 1, hidden_size]
  torch::Tensor k_k_;
  torch::Tensor k_a_;

  // Receptance-key interaction — registered parameter [n_heads_, head_size_]
  torch::Tensor r_k_;

  // Linear projections (no bias to match BlinkDL checkpoint format)
  torch::nn::Linear receptance_{nullptr};
  torch::nn::Linear key_{nullptr};
  torch::nn::Linear value_{nullptr};
  torch::nn::Linear output_{nullptr};

  // Post-attention group norm: GroupNorm(n_heads_, hidden_size_, eps=64e-5)
  torch::nn::GroupNorm ln_x_{nullptr};
};

TORCH_MODULE(RWKV7TimeMix);

// RWKV-7 Channel-Mix block (FFN replacement).
//
// Simple gated MLP with token shift:
//   k  = x + (prev_token - x) * x_k
//   output = value(relu(key(k)) ^ 2)
class RWKV7ChannelMixImpl : public torch::nn::Module {
 public:
  RWKV7ChannelMixImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  // Forward channel-mix for all sequences in the current batch.
  //
  //   x           – layer-normed input: [total_tokens, hidden_size]
  //   ffn_x_prev  – per-sequence shift state: [num_seqs, hidden_size]
  //   seq_lens    – per-sequence query-token counts
  //
  // Returns {channel_mix_output [total_tokens, C],
  //          new_ffn_x_prev     [num_seqs, C]}
  std::pair<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& ffn_x_prev,
      const std::vector<int32_t>& seq_lens);

 private:
  // Token-shift mixing scalar — registered parameter [1, 1, hidden_size]
  torch::Tensor x_k_;

  // Linear projections: key expands, value contracts (no bias)
  torch::nn::Linear key_{nullptr};    // hidden_size → intermediate_size
  torch::nn::Linear value_{nullptr};  // intermediate_size → hidden_size
};

TORCH_MODULE(RWKV7ChannelMix);

// Complete RWKV-7 block: (ln0 →) ln1 → time-mix → residual → ln2 → channel-mix
// → residual.
//
// ln0 is applied only in block 0 as a pre-normalisation of the token embedding.
class RWKV7DecoderLayerImpl : public torch::nn::Module {
 public:
  RWKV7DecoderLayerImpl(const ModelContext& context, int32_t layer_id);

  void load_state_dict(const StateDict& state_dict);

  // Forward one RWKV-7 block.
  //
  //   x          – block input: [total_tokens, hidden_size]
  //   kv_cache   – this layer's linear-attention state (conv + ssm tensors)
  //   input_params – batch metadata (seq_lens, linear_state_ids …)
  //   layer_id   – 0-based layer index (0 triggers ln0 + v_first
  //   initialisation) v_first    – cross-layer value state, updated in-place at
  //   layer 0
  //
  // Returns the block output: [total_tokens, hidden_size].
  torch::Tensor forward(const torch::Tensor& x,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        int32_t layer_id,
                        torch::Tensor& v_first);

 private:
  // Read per-sequence shift states and W-matrix from the KV cache.
  //
  // conv_cache layout (dim -1 = 3 * hidden_size):
  //   [0 : H]     → att_x_prev
  //   [H : 2H]    → ffn_x_prev
  //   [2H : 3H]   → unused (artefact of conv_cache shape formula)
  //
  // ssm_cache: [num_slots, n_heads_, head_size_, head_size_] → W-matrix state.
  //
  // Returns {att_x_prev [seqs, H], ffn_x_prev [seqs, H], att_kv [seqs, H, N,
  // N]}.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> read_state(
      KVCache& kv_cache,
      const torch::Tensor& state_indices,
      int32_t num_seqs) const;

  // Write updated states back to the KV cache.
  void write_state(KVCache& kv_cache,
                   const torch::Tensor& state_indices,
                   const torch::Tensor& att_x_prev,
                   const torch::Tensor& ffn_x_prev,
                   const torch::Tensor& att_kv) const;

  torch::nn::LayerNorm ln0_{nullptr};  // only registered for block 0
  torch::nn::LayerNorm ln1_{nullptr};
  torch::nn::LayerNorm ln2_{nullptr};

  RWKV7TimeMix att_{nullptr};
  RWKV7ChannelMix ffn_{nullptr};

  int64_t hidden_size_;
  int64_t n_heads_;
  int64_t head_size_;
  int32_t layer_id_;
  bool has_ln0_;
};

TORCH_MODULE(RWKV7DecoderLayer);

}  // namespace layer
}  // namespace xllm
