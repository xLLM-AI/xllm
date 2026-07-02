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

#include <cstdint>
#include <tuple>
#include <vector>

namespace xllm {
namespace layer {
namespace test {

struct Dsv4CompressorRefConfig {
  int64_t compress_ratio = 4;
  int64_t head_dim = 0;
  int64_t rope_head_dim = 0;
  bool rotate = false;
  double norm_eps = 1e-6;
};

struct Dsv4CompressorRefWeights {
  torch::Tensor wkv;
  torch::Tensor wgate;
  torch::Tensor norm;
  torch::Tensor ape;
};

struct Dsv4CompressorRefResult {
  torch::Tensor output;
  torch::Tensor kv_state;
  torch::Tensor score_state;
  torch::Tensor paged_state;
};

struct Dsv4AttentionRefConfig {
  int64_t hidden_dim = 0;
  int64_t q_lora_rank = 0;
  int64_t n_heads = 0;
  int64_t head_dim = 0;
  int64_t rope_head_dim = 0;
  int64_t o_groups = 0;
  int64_t o_lora_rank = 0;
  int64_t window_size = 0;
  int64_t compress_ratio = 1;
  int64_t index_n_heads = 0;
  int64_t index_head_dim = 0;
  int64_t index_topk = 0;
  int64_t max_seq_len = 0;
  double norm_eps = 1e-6;
  double rope_theta = 10000.0;
  double compress_rope_theta = 40000.0;
  double rope_factor = 40.0;
  int64_t original_seq_len = 0;
  int64_t beta_fast = 32;
  int64_t beta_slow = 1;
};

struct Dsv4IndexerRefWeights {
  torch::Tensor wq_b;
  torch::Tensor weights_proj;
  Dsv4CompressorRefWeights compressor;
};

struct Dsv4AttentionRefWeights {
  torch::Tensor wq_a;
  torch::Tensor q_norm;
  torch::Tensor wq_b;
  torch::Tensor wkv;
  torch::Tensor kv_norm;
  torch::Tensor wo_a;
  torch::Tensor wo_b;
  torch::Tensor attn_sink;
  Dsv4CompressorRefWeights compressor;
  Dsv4IndexerRefWeights indexer;
};

struct Dsv4AttentionRefState {
  torch::Tensor token_cache;
  torch::Tensor compressed_cache;
  torch::Tensor compress_kv_state;
  torch::Tensor compress_score_state;
  torch::Tensor index_cache;
  torch::Tensor index_kv_state;
  torch::Tensor index_score_state;
};

struct Dsv4AttentionRefResult {
  torch::Tensor output;
  torch::Tensor topk;
  torch::Tensor context_lens;
  Dsv4AttentionRefState state;
};

std::tuple<torch::Tensor, torch::Tensor> make_dsv4_rope_ref(
    int64_t rows,
    int64_t rope_dim,
    const torch::TensorOptions& options);

std::tuple<torch::Tensor, torch::Tensor> make_dsv4_freqs_ref(
    int64_t rows,
    int64_t rope_dim,
    int64_t original_seq_len,
    double base,
    double factor,
    int64_t beta_fast,
    int64_t beta_slow,
    const torch::TensorOptions& options);

void apply_dsv4_rotary_ref(torch::Tensor& output,
                           const torch::Tensor& sin,
                           const torch::Tensor& cos,
                           int64_t rope_dim);

void apply_dsv4_rotary_inv_ref(torch::Tensor& output,
                               const torch::Tensor& sin,
                               const torch::Tensor& cos,
                               int64_t rope_dim);

torch::Tensor rms_norm_ref(const torch::Tensor& input,
                           const torch::Tensor& weight,
                           double eps,
                           torch::ScalarType dtype);

Dsv4CompressorRefResult dsv4_compressor_ref(
    const torch::Tensor& hidden_states,
    const Dsv4CompressorRefWeights& weights,
    torch::Tensor kv_state,
    torch::Tensor score_state,
    const std::vector<int64_t>& start_pos,
    const std::vector<int64_t>& q_offsets,
    const torch::Tensor& sin,
    const torch::Tensor& cos,
    const torch::Tensor& hadamard,
    const Dsv4CompressorRefConfig& config);

Dsv4AttentionRefState make_dsv4_attention_state_ref(
    int64_t batch_size,
    const Dsv4AttentionRefConfig& config,
    const torch::TensorOptions& options);

Dsv4AttentionRefResult dsv4_attention_ref(
    const torch::Tensor& hidden_states,
    const Dsv4AttentionRefWeights& weights,
    Dsv4AttentionRefState state,
    const std::vector<int64_t>& start_pos,
    const std::vector<int64_t>& q_offsets,
    const Dsv4AttentionRefConfig& config);

}  // namespace test
}  // namespace layer
}  // namespace xllm
