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

#include "layers/mlu/deepseek_v4_ref_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

#include "util/linalg.h"

namespace {

bool has_compressed_kv(int64_t compress_ratio) {
  return compress_ratio == 4 || compress_ratio == 128;
}

int64_t coff_for_ratio(int64_t compress_ratio) {
  return compress_ratio == 4 ? 2 : 1;
}

int64_t q_len_at(const std::vector<int64_t>& offsets, int64_t seq_idx) {
  return offsets[seq_idx + 1] - offsets[seq_idx];
}

torch::Tensor linear_ref(const torch::Tensor& input,
                         const torch::Tensor& weight) {
  return torch::nn::functional::linear(
      input, weight.to(input.device()).to(input.scalar_type()));
}

torch::Tensor positions_tensor(const std::vector<int64_t>& positions,
                               const torch::Device& device) {
  return torch::tensor(
      positions, torch::TensorOptions().dtype(torch::kInt64).device(device));
}

std::vector<int64_t> token_positions(const std::vector<int64_t>& start_pos,
                                     const std::vector<int64_t>& q_offsets) {
  std::vector<int64_t> positions;
  positions.reserve(static_cast<size_t>(q_offsets.back()));
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    const int64_t q_len = q_len_at(q_offsets, seq_idx);
    for (int64_t token_idx = 0; token_idx < q_len; ++token_idx) {
      positions.emplace_back(start_pos[seq_idx] + token_idx);
    }
  }
  return positions;
}

std::vector<int64_t> compressed_positions(const std::vector<int64_t>& start_pos,
                                          const std::vector<int64_t>& q_offsets,
                                          int64_t compress_ratio) {
  std::vector<int64_t> positions;
  positions.reserve(static_cast<size_t>(q_offsets.back() / compress_ratio +
                                        start_pos.size()));
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    const int64_t q_len = q_len_at(q_offsets, seq_idx);
    for (int64_t token_idx = 0; token_idx < q_len; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      if ((pos + 1) % compress_ratio == 0) {
        positions.emplace_back(pos + 1 - compress_ratio);
      }
    }
  }
  return positions;
}

void apply_rotary(torch::Tensor& output,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  int64_t rope_dim,
                  bool inverse) {
  if (output.numel() == 0 || rope_dim == 0) {
    return;
  }
  torch::Tensor rope = output.slice(-1, output.size(-1) - rope_dim);
  torch::Tensor even = rope.slice(-1, 0, rope_dim, 2);
  torch::Tensor odd = rope.slice(-1, 1, rope_dim, 2);
  torch::Tensor sin_even = sin.slice(-1, 0, rope_dim, 2);
  torch::Tensor cos_even = cos.slice(-1, 0, rope_dim, 2);
  if (output.dim() == 3) {
    sin_even = sin_even.unsqueeze(1);
    cos_even = cos_even.unsqueeze(1);
  }
  torch::Tensor rotated_even;
  torch::Tensor rotated_odd;
  if (inverse) {
    rotated_even = even * cos_even + odd * sin_even;
    rotated_odd = odd * cos_even - even * sin_even;
  } else {
    rotated_even = even * cos_even - odd * sin_even;
    rotated_odd = odd * cos_even + even * sin_even;
  }
  rope.copy_(torch::stack({rotated_even, rotated_odd}, -1).flatten(-2));
}

torch::Tensor make_sincos_rows(const torch::Tensor& table,
                               const std::vector<int64_t>& positions,
                               int64_t rope_dim) {
  if (positions.empty()) {
    return torch::empty({0, rope_dim}, table.options());
  }
  return table.index_select(0, positions_tensor(positions, table.device()));
}

void assign_compressed_cache(torch::Tensor& cache,
                             const torch::Tensor& rows,
                             const std::vector<int64_t>& start_pos,
                             const std::vector<int64_t>& q_offsets,
                             int64_t compress_ratio) {
  int64_t row_idx = 0;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    const int64_t begin = q_offsets[seq_idx];
    const int64_t end = q_offsets[seq_idx + 1];
    for (int64_t token_idx = 0; token_idx < end - begin; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      if ((pos + 1) % compress_ratio != 0) {
        continue;
      }
      cache[seq_idx][pos / compress_ratio].copy_(rows[row_idx]);
      ++row_idx;
    }
  }
  CHECK_EQ(row_idx, rows.size(0));
}

torch::Tensor sparse_attn_ref(const torch::Tensor& q,
                              const torch::Tensor& kv,
                              const torch::Tensor& attn_sink,
                              double softmax_scale) {
  const int64_t n_heads = q.size(0);
  const int64_t head_dim = q.size(1);
  if (kv.size(0) == 0) {
    return torch::zeros({n_heads, head_dim}, q.options());
  }
  torch::Tensor q_f = q.to(torch::kFloat32);
  torch::Tensor kv_f = kv.to(torch::kFloat32);
  torch::Tensor score = torch::matmul(q_f, kv_f.t()) * softmax_scale;
  torch::Tensor max_score = std::get<0>(score.max(/*dim=*/1, true));
  torch::Tensor exp_score = torch::exp(score - max_score);
  torch::Tensor sink =
      attn_sink.to(q.device()).to(torch::kFloat32).view({n_heads, 1});
  // Equivalent to multiplying a normal attention output by
  // 1 / (exp(sink - lse) + 1): the sink only contributes to the denominator
  // and has a zero value vector.
  torch::Tensor denom =
      exp_score.sum(/*dim=*/1, true) + torch::exp(sink - max_score);
  return torch::matmul(exp_score, kv_f).div(denom).to(q.scalar_type());
}

int64_t max_topk_width(
    const xllm::layer::test::Dsv4AttentionRefConfig& config) {
  int64_t width = config.window_size;
  if (config.compress_ratio == 4) {
    width += config.index_topk;
  } else if (config.compress_ratio == 128) {
    width += config.max_seq_len / config.compress_ratio + 1;
  }
  return std::max<int64_t>(width, 1);
}

}  // namespace

namespace xllm {
namespace layer {
namespace test {

std::tuple<torch::Tensor, torch::Tensor> make_dsv4_rope_ref(
    int64_t rows,
    int64_t rope_dim,
    const torch::TensorOptions& options) {
  torch::Tensor base = torch::arange(0, rows * (rope_dim / 2), options)
                           .view({rows, rope_dim / 2})
                           .to(torch::kFloat32);
  torch::Tensor angles = base * 0.03125;
  torch::Tensor sin = torch::sin(angles)
                          .unsqueeze(-1)
                          .repeat({1, 1, 2})
                          .reshape({rows, rope_dim})
                          .to(options.dtype());
  torch::Tensor cos = torch::cos(angles)
                          .unsqueeze(-1)
                          .repeat({1, 1, 2})
                          .reshape({rows, rope_dim})
                          .to(options.dtype());
  return {sin, cos};
}

std::tuple<torch::Tensor, torch::Tensor> make_dsv4_freqs_ref(
    int64_t rows,
    int64_t rope_dim,
    int64_t original_seq_len,
    double base,
    double factor,
    int64_t beta_fast,
    int64_t beta_slow,
    const torch::TensorOptions& options) {
  CHECK_GT(rows, 0) << "rope rows must be positive";
  CHECK_GT(rope_dim, 0) << "rope_dim must be positive";
  CHECK_EQ(rope_dim % 2, 0) << "rope_dim must be even";
  const torch::TensorOptions float_options =
      options.dtype(torch::kFloat32).requires_grad(false);
  torch::Tensor exponent = torch::arange(0, rope_dim, 2, float_options) /
                           static_cast<double>(rope_dim);
  torch::Tensor freqs = torch::exp(-std::log(base) * exponent);
  if (original_seq_len > 0) {
    auto correction_dim =
        [rope_dim, base, original_seq_len](int64_t rotations) -> double {
      return static_cast<double>(rope_dim) *
             std::log(static_cast<double>(original_seq_len) /
                      (static_cast<double>(rotations) * 2.0 * M_PI)) /
             (2.0 * std::log(base));
    };
    const int64_t low = std::max<int64_t>(
        static_cast<int64_t>(std::floor(correction_dim(beta_fast))), 0);
    const int64_t high = std::min<int64_t>(
        static_cast<int64_t>(std::ceil(correction_dim(beta_slow))),
        rope_dim - 1);
    const double denom = low == high ? static_cast<double>(high - low) + 0.001
                                     : static_cast<double>(high - low);
    torch::Tensor ramp = ((torch::arange(0, rope_dim / 2, float_options) -
                           static_cast<double>(low)) /
                          denom)
                             .clamp(0, 1);
    torch::Tensor smooth = 1 - ramp;
    freqs = freqs / factor * (1 - smooth) + freqs * smooth;
  }
  torch::Tensor steps = torch::arange(0, rows, float_options);
  torch::Tensor angles = steps.unsqueeze(1) * freqs.unsqueeze(0);
  torch::Tensor sin = torch::sin(angles)
                          .unsqueeze(-1)
                          .repeat({1, 1, 2})
                          .reshape({rows, rope_dim})
                          .to(options.dtype());
  torch::Tensor cos = torch::cos(angles)
                          .unsqueeze(-1)
                          .repeat({1, 1, 2})
                          .reshape({rows, rope_dim})
                          .to(options.dtype());
  return {sin, cos};
}

void apply_dsv4_rotary_ref(torch::Tensor& output,
                           const torch::Tensor& sin,
                           const torch::Tensor& cos,
                           int64_t rope_dim) {
  apply_rotary(output, sin, cos, rope_dim, false);
}

void apply_dsv4_rotary_inv_ref(torch::Tensor& output,
                               const torch::Tensor& sin,
                               const torch::Tensor& cos,
                               int64_t rope_dim) {
  apply_rotary(output, sin, cos, rope_dim, true);
}

torch::Tensor rms_norm_ref(const torch::Tensor& input,
                           const torch::Tensor& weight,
                           double eps,
                           torch::ScalarType dtype) {
  torch::Tensor output = input.to(torch::kFloat32);
  torch::Tensor variance = output.square().mean(-1, true);
  output = output * torch::rsqrt(variance + eps);
  output = output * weight.to(output.device()).to(torch::kFloat32);
  return output.to(dtype);
}

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
    const Dsv4CompressorRefConfig& config) {
  const int64_t batch_size = static_cast<int64_t>(start_pos.size());
  const int64_t coff = config.compress_ratio == 4 ? 2 : 1;
  torch::Tensor kv_proj = torch::nn::functional::linear(
      hidden_states, weights.wkv.to(hidden_states.scalar_type()));
  torch::Tensor score_proj = torch::nn::functional::linear(
      hidden_states, weights.wgate.to(hidden_states.scalar_type()));
  torch::Tensor norm_weight = weights.norm.to(hidden_states.scalar_type());
  std::vector<torch::Tensor> rows;
  rows.reserve(hidden_states.size(0) / config.compress_ratio + batch_size);

  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t begin = q_offsets[seq_idx];
    const int64_t end = q_offsets[seq_idx + 1];
    for (int64_t token_idx = 0; token_idx < end - begin; ++token_idx) {
      const int64_t flat_idx = begin + token_idx;
      const int64_t pos = start_pos[seq_idx] + token_idx;
      const int64_t pos_mod = pos % config.compress_ratio;
      torch::Tensor kv_row = kv_proj[flat_idx];
      torch::Tensor score_row = score_proj[flat_idx] + weights.ape[pos_mod];

      if (coff == 2) {
        kv_state[seq_idx][config.compress_ratio + pos_mod].copy_(kv_row);
        score_state[seq_idx][config.compress_ratio + pos_mod].copy_(score_row);
      } else {
        kv_state[seq_idx][pos_mod].copy_(kv_row);
        score_state[seq_idx][pos_mod].copy_(score_row);
      }

      if ((pos + 1) % config.compress_ratio != 0) {
        continue;
      }

      if (coff == 2) {
        torch::Tensor curr_kv = kv_state[seq_idx].slice(
            0, config.compress_ratio, 2 * config.compress_ratio);
        torch::Tensor curr_score = score_state[seq_idx].slice(
            0, config.compress_ratio, 2 * config.compress_ratio);
        const bool has_prev_window = (pos + 1) > config.compress_ratio;
        torch::Tensor kv_pool;
        torch::Tensor score_pool;
        if (has_prev_window) {
          torch::Tensor prev_kv =
              kv_state[seq_idx].slice(0, 0, config.compress_ratio);
          torch::Tensor prev_score =
              score_state[seq_idx].slice(0, 0, config.compress_ratio);
          kv_pool = torch::cat({prev_kv.slice(1, 0, config.head_dim),
                                curr_kv.slice(1, config.head_dim)},
                               0);
          score_pool = torch::cat({prev_score.slice(1, 0, config.head_dim),
                                   curr_score.slice(1, config.head_dim)},
                                  0);
        } else {
          kv_pool = curr_kv.slice(1, config.head_dim);
          score_pool = curr_score.slice(1, config.head_dim);
        }
        rows.emplace_back((kv_pool * torch::softmax(score_pool, 0)).sum(0));
        kv_state[seq_idx].slice(0, 0, config.compress_ratio).copy_(curr_kv);
        score_state[seq_idx]
            .slice(0, 0, config.compress_ratio)
            .copy_(curr_score);
      } else {
        torch::Tensor kv_pool = kv_state[seq_idx];
        torch::Tensor score_pool = score_state[seq_idx];
        rows.emplace_back((kv_pool * torch::softmax(score_pool, 0)).sum(0));
      }
    }
  }

  torch::Tensor output;
  if (rows.empty()) {
    output = torch::empty({0, config.head_dim}, hidden_states.options());
  } else {
    output = torch::stack(rows, 0);
    output = rms_norm_ref(
        output, norm_weight, config.norm_eps, hidden_states.scalar_type());
    apply_dsv4_rotary_ref(output, sin, cos, config.rope_head_dim);
    if (config.rotate) {
      output = util::rotate_activation(output, hadamard);
    }
  }
  return {output, kv_state, score_state};
}

Dsv4AttentionRefState make_dsv4_attention_state_ref(
    int64_t batch_size,
    const Dsv4AttentionRefConfig& config,
    const torch::TensorOptions& options) {
  CHECK_GT(batch_size, 0) << "batch_size must be positive";
  CHECK_GT(config.max_seq_len, 0) << "max_seq_len must be positive";
  CHECK_GT(config.head_dim, 0) << "head_dim must be positive";
  Dsv4AttentionRefState state;
  state.token_cache =
      torch::zeros({batch_size, config.max_seq_len, config.head_dim}, options);
  if (!has_compressed_kv(config.compress_ratio)) {
    return state;
  }
  const int64_t compressed_len = config.max_seq_len / config.compress_ratio + 1;
  const int64_t coff = coff_for_ratio(config.compress_ratio);
  const torch::TensorOptions state_options = options.dtype(torch::kFloat32);
  const torch::TensorOptions score_options = options.dtype(torch::kFloat32);
  state.compressed_cache =
      torch::zeros({batch_size, compressed_len, config.head_dim}, options);
  state.compress_kv_state = torch::zeros(
      {batch_size, coff * config.compress_ratio, coff * config.head_dim},
      state_options);
  state.compress_score_state = torch::full(
      {batch_size, coff * config.compress_ratio, coff * config.head_dim},
      -std::numeric_limits<float>::infinity(),
      score_options);
  if (config.compress_ratio != 4) {
    return state;
  }
  state.index_cache = torch::zeros(
      {batch_size, compressed_len, config.index_head_dim}, options);
  state.index_kv_state = torch::zeros(
      {batch_size, 2 * config.compress_ratio, 2 * config.index_head_dim},
      state_options);
  state.index_score_state = torch::full(
      {batch_size, 2 * config.compress_ratio, 2 * config.index_head_dim},
      -std::numeric_limits<float>::infinity(),
      score_options);
  return state;
}

Dsv4AttentionRefResult dsv4_attention_ref(
    const torch::Tensor& hidden_states,
    const Dsv4AttentionRefWeights& weights,
    Dsv4AttentionRefState state,
    const std::vector<int64_t>& start_pos,
    const std::vector<int64_t>& q_offsets,
    const Dsv4AttentionRefConfig& config) {
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(hidden_states.size(1), config.hidden_dim);
  CHECK_EQ(q_offsets.size(), start_pos.size() + 1);
  CHECK_EQ(q_offsets.front(), 0);
  CHECK_EQ(q_offsets.back(), hidden_states.size(0));
  CHECK(config.compress_ratio == 1 || config.compress_ratio == 4 ||
        config.compress_ratio == 128)
      << "DeepSeek V4 ref only supports ratios 1, 4, and 128.";
  const int64_t batch_size = static_cast<int64_t>(start_pos.size());
  if (!state.token_cache.defined()) {
    state = make_dsv4_attention_state_ref(
        batch_size, config, hidden_states.options());
  }

  const torch::ScalarType dtype = hidden_states.scalar_type();
  const std::vector<int64_t> positions = token_positions(start_pos, q_offsets);
  const int64_t rope_rows = config.max_seq_len + 1;
  const bool compressed = has_compressed_kv(config.compress_ratio);
  const int64_t rope_original =
      compressed ? config.original_seq_len : static_cast<int64_t>(0);
  const double rope_base =
      compressed ? config.compress_rope_theta : config.rope_theta;
  auto [sin_table, cos_table] = make_dsv4_freqs_ref(rope_rows,
                                                    config.rope_head_dim,
                                                    rope_original,
                                                    rope_base,
                                                    config.rope_factor,
                                                    config.beta_fast,
                                                    config.beta_slow,
                                                    hidden_states.options());
  torch::Tensor sin =
      make_sincos_rows(sin_table, positions, config.rope_head_dim);
  torch::Tensor cos =
      make_sincos_rows(cos_table, positions, config.rope_head_dim);

  torch::Tensor qr = rms_norm_ref(linear_ref(hidden_states, weights.wq_a),
                                  weights.q_norm,
                                  config.norm_eps,
                                  dtype);
  torch::Tensor q =
      linear_ref(qr, weights.wq_b)
          .view({hidden_states.size(0), config.n_heads, config.head_dim});
  torch::Tensor q_f = q.to(torch::kFloat32);
  q = (q_f * torch::rsqrt(q_f.square().mean(-1, true) + config.norm_eps))
          .to(dtype);
  apply_dsv4_rotary_ref(q, sin, cos, config.rope_head_dim);

  torch::Tensor kv = rms_norm_ref(linear_ref(hidden_states, weights.wkv),
                                  weights.kv_norm,
                                  config.norm_eps,
                                  dtype);
  apply_dsv4_rotary_ref(kv, sin, cos, config.rope_head_dim);
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t begin = q_offsets[seq_idx];
    const int64_t end = q_offsets[seq_idx + 1];
    for (int64_t flat_idx = begin; flat_idx < end; ++flat_idx) {
      const int64_t pos = start_pos[seq_idx] + flat_idx - begin;
      state.token_cache[seq_idx][pos].copy_(kv[flat_idx]);
    }
  }

  torch::Tensor index_q;
  torch::Tensor index_weights;
  if (compressed) {
    auto [compressed_sin_table, compressed_cos_table] =
        make_dsv4_freqs_ref(rope_rows,
                            config.rope_head_dim,
                            config.original_seq_len,
                            config.compress_rope_theta,
                            config.rope_factor,
                            config.beta_fast,
                            config.beta_slow,
                            hidden_states.options());
    torch::Tensor input_compressed_sin =
        make_sincos_rows(compressed_sin_table, positions, config.rope_head_dim);
    torch::Tensor input_compressed_cos =
        make_sincos_rows(compressed_cos_table, positions, config.rope_head_dim);
    const std::vector<int64_t> c_positions =
        compressed_positions(start_pos, q_offsets, config.compress_ratio);
    torch::Tensor c_sin = make_sincos_rows(
        compressed_sin_table, c_positions, config.rope_head_dim);
    torch::Tensor c_cos = make_sincos_rows(
        compressed_cos_table, c_positions, config.rope_head_dim);
    Dsv4CompressorRefConfig compressor_config{config.compress_ratio,
                                              config.head_dim,
                                              config.rope_head_dim,
                                              false,
                                              config.norm_eps};
    Dsv4CompressorRefResult compressor_result =
        dsv4_compressor_ref(hidden_states,
                            weights.compressor,
                            state.compress_kv_state,
                            state.compress_score_state,
                            start_pos,
                            q_offsets,
                            c_sin,
                            c_cos,
                            torch::Tensor(),
                            compressor_config);
    state.compress_kv_state = compressor_result.kv_state;
    state.compress_score_state = compressor_result.score_state;
    assign_compressed_cache(state.compressed_cache,
                            compressor_result.output,
                            start_pos,
                            q_offsets,
                            config.compress_ratio);

    if (config.compress_ratio == 4) {
      torch::Tensor hadamard =
          util::create_hadamard_matrix(config.index_head_dim,
                                       torch::kFloat32,
                                       hidden_states.device(),
                                       /*normalize=*/true);
      index_q = linear_ref(qr, weights.indexer.wq_b)
                    .view({hidden_states.size(0),
                           config.index_n_heads,
                           config.index_head_dim});
      apply_dsv4_rotary_ref(index_q,
                            input_compressed_sin,
                            input_compressed_cos,
                            config.rope_head_dim);
      index_q = util::rotate_activation(index_q.to(dtype), hadamard.to(dtype));
      Dsv4CompressorRefConfig index_config{config.compress_ratio,
                                           config.index_head_dim,
                                           config.rope_head_dim,
                                           true,
                                           config.norm_eps};
      Dsv4CompressorRefResult index_result =
          dsv4_compressor_ref(hidden_states,
                              weights.indexer.compressor,
                              state.index_kv_state,
                              state.index_score_state,
                              start_pos,
                              q_offsets,
                              c_sin,
                              c_cos,
                              hadamard.to(dtype),
                              index_config);
      state.index_kv_state = index_result.kv_state;
      state.index_score_state = index_result.score_state;
      assign_compressed_cache(state.index_cache,
                              index_result.output,
                              start_pos,
                              q_offsets,
                              config.compress_ratio);
      const double index_scale =
          std::pow(static_cast<double>(config.index_head_dim), -0.5) *
          std::pow(static_cast<double>(config.index_n_heads), -0.5);
      index_weights =
          linear_ref(hidden_states, weights.indexer.weights_proj) * index_scale;
    }
  }

  const int64_t topk_width = max_topk_width(config);
  std::vector<int32_t> topk_values(
      static_cast<size_t>(hidden_states.size(0) * topk_width), -1);
  std::vector<int32_t> context_lens(static_cast<size_t>(hidden_states.size(0)),
                                    0);
  std::vector<torch::Tensor> attn_rows;
  attn_rows.reserve(static_cast<size_t>(hidden_states.size(0)));
  const double softmax_scale =
      std::pow(static_cast<double>(config.head_dim), -0.5);
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t begin = q_offsets[seq_idx];
    const int64_t end = q_offsets[seq_idx + 1];
    for (int64_t flat_idx = begin; flat_idx < end; ++flat_idx) {
      const int64_t pos = start_pos[seq_idx] + flat_idx - begin;
      const int64_t win_begin =
          std::max<int64_t>(0, pos - config.window_size + 1);
      std::vector<torch::Tensor> selected;
      selected.reserve(static_cast<size_t>(topk_width));
      int64_t topk_col = 0;
      for (int64_t kv_pos = win_begin; kv_pos <= pos; ++kv_pos) {
        selected.emplace_back(state.token_cache[seq_idx][kv_pos]);
        topk_values[static_cast<size_t>(flat_idx * topk_width + topk_col)] =
            static_cast<int32_t>(kv_pos);
        ++topk_col;
      }
      if (config.compress_ratio == 4) {
        const int64_t valid_count = (pos + 1) / config.compress_ratio;
        const int64_t selected_count =
            std::min<int64_t>(config.index_topk, valid_count);
        if (selected_count > 0) {
          torch::Tensor score =
              torch::matmul(index_q[flat_idx].to(torch::kFloat32),
                            state.index_cache[seq_idx]
                                .slice(0, 0, valid_count)
                                .to(torch::kFloat32)
                                .t());
          score = (score.relu() *
                   index_weights[flat_idx].to(torch::kFloat32).unsqueeze(-1))
                      .sum(0);
          torch::Tensor indices =
              std::get<1>(score.topk(selected_count, -1)).to(torch::kInt64);
          for (int64_t i = 0; i < selected_count; ++i) {
            const int64_t c_idx = indices[i].item<int64_t>();
            selected.emplace_back(state.compressed_cache[seq_idx][c_idx]);
            topk_values[static_cast<size_t>(flat_idx * topk_width + topk_col)] =
                static_cast<int32_t>(config.max_seq_len + c_idx);
            ++topk_col;
          }
        }
      } else if (config.compress_ratio == 128) {
        const int64_t valid_count = (pos + 1) / config.compress_ratio;
        for (int64_t c_idx = 0; c_idx < valid_count; ++c_idx) {
          selected.emplace_back(state.compressed_cache[seq_idx][c_idx]);
          topk_values[static_cast<size_t>(flat_idx * topk_width + topk_col)] =
              static_cast<int32_t>(config.max_seq_len + c_idx);
          ++topk_col;
        }
      }
      context_lens[static_cast<size_t>(flat_idx)] =
          static_cast<int32_t>(topk_col);
      torch::Tensor selected_kv =
          selected.empty()
              ? torch::empty({0, config.head_dim}, hidden_states.options())
              : torch::stack(selected, 0);
      attn_rows.emplace_back(sparse_attn_ref(
          q[flat_idx], selected_kv, weights.attn_sink, softmax_scale));
    }
  }

  torch::Tensor attn_output = torch::stack(attn_rows, 0);
  apply_dsv4_rotary_inv_ref(attn_output, sin, cos, config.rope_head_dim);
  torch::Tensor grouped =
      attn_output.reshape({hidden_states.size(0), config.o_groups, -1});
  torch::Tensor wo_a = weights.wo_a.to(hidden_states.device())
                           .to(dtype)
                           .view({config.o_groups, config.o_lora_rank, -1});
  torch::Tensor low_rank =
      torch::einsum("tgd,grd->tgr", {grouped, wo_a}).flatten(1);
  torch::Tensor output = linear_ref(low_rank, weights.wo_b);
  torch::Tensor topk =
      torch::from_blob(topk_values.data(),
                       {hidden_states.size(0), topk_width},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .to(hidden_states.device());
  torch::Tensor context_lens_tensor =
      torch::from_blob(context_lens.data(),
                       {hidden_states.size(0)},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .to(hidden_states.device());
  return {output, topk, context_lens_tensor, state};
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
