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

#include "layers/mlu/glm5_next/glm5_next_kda.h"

#include <framework/core/device.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "kernels/mlu/chunk_kda.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::layer {
namespace {

torch::Tensor make_chunk_indices(const torch::Tensor& cu_seqlens,
                                 int64_t chunk_size) {
  torch::Tensor lengths = cu_seqlens.diff();
  torch::Tensor num_chunks =
      ((lengths + chunk_size - 1) / chunk_size).to(torch::kLong);
  torch::Tensor chunk_offsets = torch::cumsum(num_chunks, /*dim=*/0);
  const int64_t total_chunks = chunk_offsets[-1].item<int64_t>();
  torch::Tensor flat_indices =
      torch::arange(total_chunks, cu_seqlens.options());
  torch::Tensor prefixes =
      torch::cat({torch::zeros({1}, chunk_offsets.options()),
                  chunk_offsets.slice(/*dim=*/0, /*start=*/0, /*end=*/-1)});
  torch::Tensor local_indices =
      flat_indices - torch::repeat_interleave(prefixes, num_chunks);
  torch::Tensor sequence_indices = (local_indices == 0).cumsum(/*dim=*/0) - 1;
  return torch::stack({sequence_indices, local_indices}, /*dim=*/1)
      .to(torch::kInt32);
}

std::tuple<torch::Tensor, torch::Tensor> packed_eager_kda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& log_gate,
    const torch::Tensor& raw_beta,
    const torch::Tensor& initial_state,
    const std::vector<int64_t>& sequence_lengths,
    bool use_qk_l2norm = true) {
  std::vector<torch::Tensor> outputs;
  std::vector<torch::Tensor> states;
  outputs.reserve(sequence_lengths.size());
  states.reserve(sequence_lengths.size());

  int64_t begin = 0;
  for (size_t sequence_id = 0; sequence_id < sequence_lengths.size();
       ++sequence_id) {
    const int64_t end = begin + sequence_lengths[sequence_id];
    torch::Tensor output;
    torch::Tensor state;
    std::tie(output, state) = glm5_next_kda_eager_recurrence(
        q.slice(/*dim=*/1, /*start=*/begin, /*end=*/end),
        k.slice(/*dim=*/1, /*start=*/begin, /*end=*/end),
        v.slice(/*dim=*/1, /*start=*/begin, /*end=*/end),
        log_gate.slice(/*dim=*/0, /*start=*/begin, /*end=*/end),
        raw_beta.slice(/*dim=*/0, /*start=*/begin, /*end=*/end),
        initial_state.slice(/*dim=*/0,
                            /*start=*/static_cast<int64_t>(sequence_id),
                            /*end=*/static_cast<int64_t>(sequence_id + 1)),
        /*l2norm_qk=*/use_qk_l2norm);
    outputs.emplace_back(output.squeeze(/*dim=*/0));
    states.emplace_back(state.squeeze(/*dim=*/0));
    begin = end;
  }
  return {torch::cat(outputs, /*dim=*/0), torch::stack(states, /*dim=*/0)};
}

double tensor_cosine_similarity(const torch::Tensor& actual,
                                const torch::Tensor& expected) {
  torch::Tensor actual_fp32 = actual.to(torch::kFloat32).flatten();
  torch::Tensor expected_fp32 = expected.to(torch::kFloat32).flatten();
  const double numerator =
      torch::sum(actual_fp32 * expected_fp32).item<double>();
  const double denominator =
      actual_fp32.norm().item<double>() * expected_fp32.norm().item<double>();
  return numerator / std::max(denominator, std::numeric_limits<double>::min());
}

TEST(Glm5NextKDATest, SafeGateUsesBoundedPerKeyFormula) {
  const torch::Tensor raw_gate =
      torch::tensor({{{0.25f, -0.5f}, {1.0f, -1.5f}}}, torch::kFloat32);
  const torch::Tensor a_log =
      torch::tensor({std::log(2.0f), std::log(0.5f)}, torch::kFloat32);
  const torch::Tensor dt_bias =
      torch::tensor({0.5f, -0.25f, -0.5f, 0.75f}, torch::kFloat32);
  constexpr float kLowerBound = -5.0f;

  const torch::Tensor actual =
      glm5_next_safe_gate(raw_gate, a_log, dt_bias, kLowerBound);
  const torch::Tensor expected =
      kLowerBound * torch::sigmoid(torch::exp(a_log).view({1, 2, 1}) *
                                   (raw_gate + dt_bias.view({1, 2, 2})));

  EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
  EXPECT_TRUE(torch::all(actual <= 0.0f).item<bool>());
  EXPECT_TRUE(torch::all(actual >= kLowerBound).item<bool>());
}

TEST(Glm5NextKDATest, OutputGateUsesSigmoidRatherThanSilu) {
  const torch::Tensor normalized =
      torch::tensor({{-2.0f, 3.0f}}, torch::kFloat32);
  const torch::Tensor output_gate =
      torch::tensor({{-1.0f, 2.0f}}, torch::kFloat32);

  const torch::Tensor actual =
      glm5_next_kda_apply_output_gate(normalized, output_gate);
  const torch::Tensor expected = normalized * torch::sigmoid(output_gate);
  const torch::Tensor silu_gated = normalized * torch::silu(output_gate);

  EXPECT_TRUE(torch::allclose(actual, expected, 1e-6, 1e-6));
  EXPECT_FALSE(torch::allclose(actual, silu_gated, 1e-6, 1e-6));
}

TEST(Glm5NextKDATest, EagerRecurrencePreservesStateAcrossSegments) {
  torch::manual_seed(20260827);
  constexpr int64_t kTokens = 5;
  constexpr int64_t kHeads = 2;
  constexpr int64_t kDim = 4;
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  const torch::Tensor q = torch::randn({1, kTokens, kHeads, kDim}, options);
  const torch::Tensor k = torch::randn({1, kTokens, kHeads, kDim}, options);
  const torch::Tensor v = torch::randn({1, kTokens, kHeads, kDim}, options);
  const torch::Tensor raw_gate = torch::randn({kTokens, kHeads, kDim}, options);
  const torch::Tensor a_log = torch::randn({kHeads}, options);
  const torch::Tensor dt_bias = torch::randn({kHeads * kDim}, options);
  const torch::Tensor log_gate =
      glm5_next_safe_gate(raw_gate, a_log, dt_bias, -5.0f);
  const torch::Tensor beta = torch::randn({kTokens, kHeads}, options);
  const torch::Tensor initial_state =
      torch::zeros({1, kHeads, kDim, kDim}, options);

  torch::Tensor full_output;
  torch::Tensor full_state;
  std::tie(full_output, full_state) = glm5_next_kda_eager_recurrence(
      q, k, v, log_gate, beta, initial_state, /*l2norm_qk=*/true);

  constexpr int64_t kSplit = 3;
  torch::Tensor first_output;
  torch::Tensor first_state;
  std::tie(first_output, first_state) = glm5_next_kda_eager_recurrence(
      q.slice(/*dim=*/1, /*start=*/0, /*end=*/kSplit),
      k.slice(/*dim=*/1, /*start=*/0, /*end=*/kSplit),
      v.slice(/*dim=*/1, /*start=*/0, /*end=*/kSplit),
      log_gate.slice(/*dim=*/0, /*start=*/0, /*end=*/kSplit),
      beta.slice(/*dim=*/0, /*start=*/0, /*end=*/kSplit),
      initial_state,
      /*l2norm_qk=*/true);

  torch::Tensor second_output;
  torch::Tensor second_state;
  std::tie(second_output, second_state) = glm5_next_kda_eager_recurrence(
      q.slice(/*dim=*/1, /*start=*/kSplit),
      k.slice(/*dim=*/1, /*start=*/kSplit),
      v.slice(/*dim=*/1, /*start=*/kSplit),
      log_gate.slice(/*dim=*/0, /*start=*/kSplit),
      beta.slice(/*dim=*/0, /*start=*/kSplit),
      first_state,
      /*l2norm_qk=*/true);

  const torch::Tensor segmented_output =
      torch::cat({first_output, second_output}, /*dim=*/1);
  EXPECT_TRUE(torch::allclose(segmented_output, full_output, 1e-5, 1e-5));
  EXPECT_TRUE(torch::allclose(second_state, full_state, 1e-5, 1e-5));
}

TEST(Glm5NextKDATest, ChunkPrefillMatchesPackedEagerAcrossBoundaries) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260831);

  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const int64_t chunk_size = kernel::mlu::kda_prefill_chunk_size();
  const std::vector<int64_t> sequence_lengths =
      chunk_size == 16 ? std::vector<int64_t>{1, 15, 16, 17, 31, 63, 64, 65}
                       : std::vector<int64_t>{1, 63, 64, 65, 127};
  const int64_t total_tokens =
      std::accumulate(sequence_lengths.begin(), sequence_lengths.end(), 0L);
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  torch::Tensor q =
      (torch::randn({1, total_tokens, kNumHeads, kHeadDim}, fp32_options) *
       0.1f)
          .to(torch::kBFloat16);
  torch::Tensor k =
      (torch::randn({1, total_tokens, kNumHeads, kHeadDim}, fp32_options) *
       0.1f)
          .to(torch::kBFloat16);
  torch::Tensor v =
      (torch::randn({1, total_tokens, kNumHeads, kHeadDim}, fp32_options) *
       0.1f)
          .to(torch::kBFloat16);
  torch::Tensor raw_gate =
      torch::randn({total_tokens, kNumHeads, kHeadDim}, fp32_options) * 0.1f;
  torch::Tensor a_log = torch::full({kNumHeads}, -2.0f, fp32_options);
  torch::Tensor dt_bias = torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  torch::Tensor log_gate = glm5_next_safe_gate(raw_gate, a_log, dt_bias, -5.0f);
  torch::Tensor raw_beta =
      torch::randn({total_tokens, kNumHeads}, fp32_options) * 0.1f;
  torch::Tensor activated_beta = torch::sigmoid(raw_beta);
  torch::Tensor initial_state =
      torch::randn({static_cast<int64_t>(sequence_lengths.size()),
                    kNumHeads,
                    kHeadDim,
                    kHeadDim},
                   fp32_options) *
      0.01f;
  std::vector<int32_t> cumulative_sequence_lengths = {0};
  cumulative_sequence_lengths.reserve(sequence_lengths.size() + 1);
  int32_t cumulative_length = 0;
  for (const int64_t sequence_length : sequence_lengths) {
    cumulative_length += static_cast<int32_t>(sequence_length);
    cumulative_sequence_lengths.push_back(cumulative_length);
  }
  torch::Tensor cu_seqlens =
      torch::tensor(cumulative_sequence_lengths, int_options);
  torch::Tensor chunk_indices = make_chunk_indices(cu_seqlens, chunk_size);

  torch::Tensor eager_warmup_output;
  torch::Tensor eager_warmup_state;
  std::tie(eager_warmup_output, eager_warmup_state) = packed_eager_kda(
      q, k, v, log_gate, raw_beta, initial_state, sequence_lengths);
  (void)eager_warmup_output;
  (void)eager_warmup_state;
  torch_mlu::synchronize();
  const auto eager_begin = std::chrono::steady_clock::now();
  torch::Tensor expected_output;
  torch::Tensor expected_state;
  std::tie(expected_output, expected_state) = packed_eager_kda(
      q, k, v, log_gate, raw_beta, initial_state, sequence_lengths);
  torch_mlu::synchronize();
  const auto eager_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - eager_begin);

  kernel::mlu::ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [warmup_output, warmup_state] =
      chunk_kda->forward(q,
                         k,
                         v,
                         log_gate,
                         activated_beta,
                         initial_state,
                         cu_seqlens,
                         chunk_indices,
                         /*output_final_state=*/true,
                         /*use_qk_l2norm=*/true);
  (void)warmup_output;
  (void)warmup_state;
  torch_mlu::synchronize();
  const auto chunk_begin = std::chrono::steady_clock::now();
  auto [actual_output, actual_state] =
      chunk_kda->forward(q,
                         k,
                         v,
                         log_gate,
                         activated_beta,
                         initial_state,
                         cu_seqlens,
                         chunk_indices,
                         /*output_final_state=*/true,
                         /*use_qk_l2norm=*/true);
  torch_mlu::synchronize();
  const auto chunk_elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - chunk_begin);

  torch::Tensor squeezed_output = actual_output.squeeze(/*dim=*/0);
  const double output_max_abs =
      torch::max(torch::abs(squeezed_output.to(torch::kFloat32) -
                            expected_output.to(torch::kFloat32)))
          .item<double>();
  const double state_max_abs =
      torch::max(torch::abs(actual_state - expected_state)).item<double>();
  const double output_cosine =
      tensor_cosine_similarity(squeezed_output, expected_output);
  const double state_cosine =
      tensor_cosine_similarity(actual_state, expected_state);

  EXPECT_TRUE(torch::allclose(squeezed_output,
                              expected_output,
                              /*rtol=*/3e-2,
                              /*atol=*/2e-2))
      << "output max_abs=" << output_max_abs << ", cosine=" << output_cosine;
  EXPECT_TRUE(torch::allclose(actual_state,
                              expected_state,
                              /*rtol=*/3e-2,
                              /*atol=*/2e-2))
      << "state max_abs=" << state_max_abs << ", cosine=" << state_cosine;
  LOG(INFO) << "GLM5-Next KDA eager/chunk latency for " << total_tokens
            << " packed tokens: eager=" << eager_elapsed.count()
            << " us, chunk=" << chunk_elapsed.count() << " us, speedup="
            << static_cast<double>(eager_elapsed.count()) /
                   static_cast<double>(chunk_elapsed.count())
            << ", output max_abs=" << output_max_abs
            << ", output cosine=" << output_cosine
            << ", state max_abs=" << state_max_abs
            << ", state cosine=" << state_cosine;
}

TEST(Glm5NextKDATest,
     ChunkPrefillSmallHeadsMatchPackedEagerWithOptionalNormalization) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260905);

  for (const int64_t num_heads : {1, 2, 4}) {
    SCOPED_TRACE(num_heads);
    for (const bool use_qk_l2norm : {false, true}) {
      SCOPED_TRACE(use_qk_l2norm);
      constexpr int64_t kHeadDim = 128;
      const int64_t chunk_size = kernel::mlu::kda_prefill_chunk_size();
      const std::vector<int64_t> sequence_lengths = {
          1, chunk_size - 1, chunk_size + 1};
      const int64_t total_tokens =
          std::accumulate(sequence_lengths.begin(), sequence_lengths.end(), 0L);
      const torch::TensorOptions fp32_options =
          torch::TensorOptions().dtype(torch::kFloat32).device(device);
      const torch::TensorOptions int_options =
          torch::TensorOptions().dtype(torch::kInt32).device(device);

      const std::vector<int64_t> input_shape = {
          1, total_tokens, num_heads, kHeadDim};
      torch::Tensor q =
          (torch::randn(input_shape, fp32_options) * 0.1f).to(torch::kBFloat16);
      torch::Tensor k =
          (torch::randn(input_shape, fp32_options) * 0.1f).to(torch::kBFloat16);
      torch::Tensor v =
          (torch::randn(input_shape, fp32_options) * 0.1f).to(torch::kBFloat16);
      torch::Tensor log_gate =
          -(torch::rand({total_tokens, num_heads, kHeadDim}, fp32_options) *
                0.02f +
            0.005f);
      torch::Tensor raw_beta =
          torch::randn({total_tokens, num_heads}, fp32_options) * 0.1f;
      torch::Tensor initial_state =
          torch::randn({static_cast<int64_t>(sequence_lengths.size()),
                        num_heads,
                        kHeadDim,
                        kHeadDim},
                       fp32_options) *
          0.01f;

      std::vector<int32_t> cumulative_sequence_lengths = {0};
      cumulative_sequence_lengths.reserve(sequence_lengths.size() + 1);
      int32_t cumulative_length = 0;
      for (const int64_t sequence_length : sequence_lengths) {
        cumulative_length += static_cast<int32_t>(sequence_length);
        cumulative_sequence_lengths.push_back(cumulative_length);
      }
      torch::Tensor cu_seqlens =
          torch::tensor(cumulative_sequence_lengths, int_options);
      torch::Tensor chunk_indices = make_chunk_indices(cu_seqlens, chunk_size);

      torch::Tensor expected_output;
      torch::Tensor expected_state;
      std::tie(expected_output, expected_state) =
          packed_eager_kda(q,
                           k,
                           v,
                           log_gate,
                           raw_beta,
                           initial_state,
                           sequence_lengths,
                           use_qk_l2norm);

      kernel::mlu::ChunkKDA chunk_kda(num_heads);
      chunk_kda->to(device);
      auto [actual_output, actual_state] =
          chunk_kda->forward(q,
                             k,
                             v,
                             log_gate,
                             torch::sigmoid(raw_beta),
                             initial_state,
                             cu_seqlens,
                             chunk_indices,
                             /*output_final_state=*/true,
                             use_qk_l2norm);
      torch_mlu::synchronize();

      EXPECT_TRUE(torch::allclose(actual_output.squeeze(/*dim=*/0),
                                  expected_output,
                                  /*rtol=*/3e-2,
                                  /*atol=*/2e-2));
      EXPECT_TRUE(torch::allclose(actual_state,
                                  expected_state,
                                  /*rtol=*/3e-2,
                                  /*atol=*/2e-2));
    }
  }
}

TEST(Glm5NextKDATest, ChunkPrefillStateFeedsExistingDecodeKernel) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260901);

  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kPrefillTokens = 65;
  constexpr int64_t kTotalTokens = kPrefillTokens + 1;
  torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  torch::Tensor q =
      (torch::randn({1, kTotalTokens, kNumHeads, kHeadDim}, fp32_options) *
       0.1f)
          .to(torch::kBFloat16);
  torch::Tensor k =
      (torch::randn({1, kTotalTokens, kNumHeads, kHeadDim}, fp32_options) *
       0.1f)
          .to(torch::kBFloat16);
  torch::Tensor v =
      (torch::randn({1, kTotalTokens, kNumHeads, kHeadDim}, fp32_options) *
       0.1f)
          .to(torch::kBFloat16);
  torch::Tensor raw_gate =
      torch::randn({kTotalTokens, kNumHeads, kHeadDim}, fp32_options) * 0.1f;
  torch::Tensor a_log = torch::full({kNumHeads}, -2.0f, fp32_options);
  torch::Tensor dt_bias = torch::zeros({kNumHeads * kHeadDim}, fp32_options);
  torch::Tensor log_gate = glm5_next_safe_gate(raw_gate, a_log, dt_bias, -5.0f);
  torch::Tensor raw_beta =
      torch::randn({kTotalTokens, kNumHeads}, fp32_options) * 0.1f;
  torch::Tensor initial_state =
      torch::randn({1, kNumHeads, kHeadDim, kHeadDim}, fp32_options) * 0.01f;

  torch::Tensor expected_output;
  torch::Tensor expected_state;
  std::tie(expected_output, expected_state) =
      glm5_next_kda_eager_recurrence(q,
                                     k,
                                     v,
                                     log_gate,
                                     raw_beta,
                                     initial_state,
                                     /*l2norm_qk=*/true);

  const std::vector<int32_t> cumulative_sequence_lengths = {
      0, static_cast<int32_t>(kPrefillTokens)};
  torch::Tensor prefill_cu_seqlens =
      torch::tensor(cumulative_sequence_lengths, int_options);
  torch::Tensor chunk_indices = make_chunk_indices(
      prefill_cu_seqlens, kernel::mlu::kda_prefill_chunk_size());
  kernel::mlu::ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [prefill_output, prefill_state] = chunk_kda->forward(
      q.slice(/*dim=*/1, /*start=*/0, /*end=*/kPrefillTokens),
      k.slice(/*dim=*/1, /*start=*/0, /*end=*/kPrefillTokens),
      v.slice(/*dim=*/1, /*start=*/0, /*end=*/kPrefillTokens),
      log_gate.slice(/*dim=*/0, /*start=*/0, /*end=*/kPrefillTokens),
      torch::sigmoid(
          raw_beta.slice(/*dim=*/0, /*start=*/0, /*end=*/kPrefillTokens)),
      initial_state,
      prefill_cu_seqlens,
      chunk_indices,
      /*output_final_state=*/true,
      /*use_qk_l2norm=*/true);

  torch::Tensor decode_raw_gate =
      raw_gate.slice(/*dim=*/0, /*start=*/kPrefillTokens)
          .reshape({1, kNumHeads * kHeadDim})
          .contiguous();
  torch::Tensor decode_raw_beta =
      raw_beta.slice(/*dim=*/0, /*start=*/kPrefillTokens).contiguous();
  torch::Tensor decode_q =
      q.slice(/*dim=*/1, /*start=*/kPrefillTokens).contiguous();
  torch::Tensor decode_k =
      k.slice(/*dim=*/1, /*start=*/kPrefillTokens).contiguous();
  torch::Tensor decode_v =
      v.slice(/*dim=*/1, /*start=*/kPrefillTokens).contiguous();
  torch::Tensor state_indices;
  torch::Tensor decode_cu_seqlens;
  auto [decode_output, decode_state] =
      kernel::mlu::fused_sigmoid_gating_delta_rule_update(
          a_log,
          decode_raw_gate,
          decode_raw_beta,
          dt_bias,
          decode_q,
          decode_k,
          decode_v,
          prefill_state,
          state_indices,
          decode_cu_seqlens,
          /*scale=*/1.0 / std::sqrt(static_cast<double>(kHeadDim)),
          /*use_qk_l2norm_in_kernel=*/true,
          /*softplus_beta=*/1.0f,
          /*softplus_threshold=*/20.0f,
          /*num_accepted_tokens_opt=*/std::nullopt,
          /*inplace_final_state=*/false,
          /*is_kda=*/true,
          /*kda_use_safe_gate=*/true,
          /*kda_gate_lower_bound=*/-5.0f);
  torch_mlu::synchronize();

  torch::Tensor actual_output = torch::cat(
      {prefill_output.squeeze(/*dim=*/0), decode_output.squeeze(/*dim=*/0)},
      /*dim=*/0);
  EXPECT_TRUE(torch::allclose(actual_output,
                              expected_output.squeeze(/*dim=*/0),
                              /*rtol=*/3e-2,
                              /*atol=*/2e-2));
  EXPECT_TRUE(torch::allclose(decode_state,
                              expected_state,
                              /*rtol=*/3e-2,
                              /*atol=*/2e-2));
}

TEST(Glm5NextKDATest, CorrelatedKeysRemainStableAcrossSafeGateRange) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(/*seed=*/20260910);

  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto bf16_options = fp32_options.dtype(torch::kBFloat16);
  const auto int_options = fp32_options.dtype(torch::kInt32);
  kernel::mlu::ChunkKDA chunk_kda(kNumHeads);

  // Correlated keys expose cancellation; strong decay exposes exponential
  // overflow. A full-chunk inverse must preserve this well-conditioned
  // recurrence.
  for (const int64_t tokens : {64, 128}) {
    SCOPED_TRACE(tokens);
    const torch::Tensor q =
        torch::randn({1, tokens, kNumHeads, kHeadDim}, bf16_options);
    const torch::Tensor k = torch::ones_like(q);
    const torch::Tensor v = torch::randn_like(q);
    const torch::Tensor initial_state =
        torch::randn({1, kNumHeads, kHeadDim, kHeadDim}, fp32_options) * 0.03f;
    const torch::Tensor cu_seqlens = torch::tensor(
        std::vector<int32_t>{0, static_cast<int32_t>(tokens)}, int_options);
    const torch::Tensor chunk_indices =
        make_chunk_indices(cu_seqlens, kernel::mlu::kda_prefill_chunk_size());
    for (const float decay : {0.0f, -1.0e-4f, -5.0f}) {
      SCOPED_TRACE(decay);
      const torch::Tensor log_gate =
          torch::full({tokens, kNumHeads, kHeadDim}, decay, fp32_options);
      for (const float beta_logit : {100.0f, std::log(99.0f)}) {
        SCOPED_TRACE(beta_logit);
        const torch::Tensor raw_beta =
            torch::full({tokens, kNumHeads}, beta_logit, fp32_options);
        const torch::Tensor beta = torch::sigmoid(raw_beta);
        auto [expected_output, expected_state] =
            glm5_next_kda_eager_recurrence(q,
                                           k,
                                           v,
                                           log_gate,
                                           raw_beta,
                                           initial_state,
                                           /*l2norm_qk=*/true);
        auto [actual_output, actual_state] =
            chunk_kda->forward(q,
                               k,
                               v,
                               log_gate,
                               beta,
                               initial_state,
                               cu_seqlens,
                               chunk_indices,
                               /*output_final_state=*/true,
                               /*use_qk_l2norm=*/true);
        EXPECT_TRUE(torch::isfinite(actual_output).all().item<bool>());
        EXPECT_TRUE(torch::isfinite(actual_state).all().item<bool>());
        EXPECT_TRUE(torch::allclose(
            actual_output, expected_output, /*rtol=*/3e-2, /*atol=*/2e-2));
        EXPECT_TRUE(torch::allclose(
            actual_state, expected_state, /*rtol=*/3e-2, /*atol=*/2e-2));
      }
    }
  }
}

TEST(Glm5NextKDATest, PrefillFeedsIndexedMtpCheckpointsAcrossRaggedRequests) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(/*seed=*/20260909);

  constexpr int64_t kSequences = 32;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kPoolSlots = 520;
  const auto fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const auto bf16_options = fp32_options.dtype(torch::kBFloat16);
  const torch::Tensor a_log =
      torch::full({kNumHeads}, /*fill_value=*/-2.0f, fp32_options);
  const torch::Tensor bias =
      torch::randn({kNumHeads * kHeadDim}, fp32_options) * 4.0f;

  // Each request contributes a partial chunk and has its own warm state.
  const torch::Tensor prefill_q =
      torch::randn({1, kSequences, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor prefill_k = torch::randn_like(prefill_q);
  const torch::Tensor prefill_v = torch::randn_like(prefill_q);
  const torch::Tensor prefill_raw_gate =
      torch::randn({kSequences, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor prefill_gate =
      glm5_next_safe_gate(prefill_raw_gate, a_log, bias, /*lower_bound=*/-5.0f);
  const torch::Tensor prefill_beta =
      torch::sigmoid(torch::randn({kSequences, kNumHeads}, fp32_options));
  const torch::Tensor initial_state =
      torch::randn({kSequences, kNumHeads, kHeadDim, kHeadDim}, fp32_options) *
      0.03f;
  const torch::Tensor prefill_cu = torch::arange(kSequences + 1, int_options);
  const torch::Tensor chunk_indices =
      make_chunk_indices(prefill_cu, kernel::mlu::kda_prefill_chunk_size());
  kernel::mlu::ChunkKDA chunk_kda(kNumHeads);
  auto [prefill_output, prefill_state] =
      chunk_kda->forward(prefill_q,
                         prefill_k,
                         prefill_v,
                         prefill_gate,
                         prefill_beta,
                         initial_state,
                         prefill_cu,
                         chunk_indices,
                         /*output_final_state=*/true,
                         /*use_qk_l2norm=*/true);
  ASSERT_TRUE(torch::isfinite(prefill_output).all().item<bool>());
  ASSERT_TRUE(torch::isfinite(prefill_state).all().item<bool>());

  const std::array<std::pair<int64_t, bool>, 6> decode_cases = {
      {{3, false}, {3, true}, {4, false}, {4, true}, {8, false}, {8, true}}};
  for (const auto& [checkpoint_width, ragged] : decode_cases) {
    SCOPED_TRACE(checkpoint_width);
    SCOPED_TRACE(ragged ? "ragged" : "dense");
    std::vector<int32_t> boundaries{0};
    std::vector<int32_t> accepted;
    std::vector<int32_t> slots;
    boundaries.reserve(kSequences + 1);
    accepted.reserve(kSequences);
    slots.reserve(kSequences * checkpoint_width);
    for (int64_t sequence = 0; sequence < kSequences; ++sequence) {
      const int32_t length =
          ragged ? static_cast<int32_t>(sequence % (checkpoint_width + 1))
                 : static_cast<int32_t>(checkpoint_width);
      boundaries.emplace_back(boundaries.back() + length);
      accepted.emplace_back(
          static_cast<int32_t>(sequence % checkpoint_width + 1));
      for (int64_t checkpoint = 0; checkpoint < checkpoint_width;
           ++checkpoint) {
        // Reverse the physical request order and leave gaps in the pool.
        slots.emplace_back(
            static_cast<int32_t>(16 * (kSequences - sequence) + checkpoint));
      }
    }
    // A padded row with tokens must produce zero output and preserve the pool.
    const int64_t padding_sequence = kSequences - 1;
    std::fill(slots.begin() + padding_sequence * checkpoint_width,
              slots.end(),
              /*value=*/0);
    // Sequence 2 starts from checkpoint 2; only its second write is skipped.
    slots[2 * checkpoint_width + 1] = 0;
    if (checkpoint_width > 4) {
      // A missing window-boundary checkpoint must not reset the recurrence.
      // Sequence 7 starts from checkpoint 7 and has tokens in both windows.
      slots[7 * checkpoint_width + 3] = 0;
    }
    torch::Tensor state_pool =
        torch::randn({kPoolSlots, kNumHeads, kHeadDim, kHeadDim},
                     fp32_options) *
        0.03f;
    for (int64_t sequence = 0; sequence < padding_sequence; ++sequence) {
      const int32_t initial_slot =
          slots[sequence * checkpoint_width + accepted[sequence] - 1];
      state_pool[initial_slot].copy_(prefill_state[sequence]);
    }
    torch::Tensor expected_pool = state_pool.clone();
    const int64_t total_tokens = boundaries.back();
    torch::Tensor q =
        torch::randn({1, total_tokens, kNumHeads, kHeadDim}, bf16_options);
    torch::Tensor k = torch::randn_like(q);
    torch::Tensor v = torch::randn_like(q);
    torch::Tensor raw_gate =
        torch::randn({total_tokens, kNumHeads * kHeadDim}, bf16_options);
    torch::Tensor raw_beta =
        torch::randn({total_tokens, kNumHeads}, bf16_options);
    const torch::Tensor log_gate =
        glm5_next_safe_gate(raw_gate.view({total_tokens, kNumHeads, kHeadDim}),
                            a_log,
                            bias,
                            /*lower_bound=*/-5.0f);
    torch::Tensor expected_output = torch::zeros_like(v);
    std::array<bool, kPoolSlots> written_slots{};
    for (int64_t sequence = 0; sequence < padding_sequence; ++sequence) {
      const int32_t initial_slot =
          slots[sequence * checkpoint_width + accepted[sequence] - 1];
      torch::Tensor state = state_pool
                                .slice(
                                    /*dim=*/0, initial_slot, initial_slot + 1)
                                .clone();
      for (int64_t token = boundaries[sequence];
           token < boundaries[sequence + 1];
           ++token) {
        torch::Tensor output;
        std::tie(output, state) = glm5_next_kda_eager_recurrence(
            q.slice(/*dim=*/1, token, token + 1),
            k.slice(/*dim=*/1, token, token + 1),
            v.slice(/*dim=*/1, token, token + 1),
            log_gate.slice(/*dim=*/0, token, token + 1),
            raw_beta.slice(/*dim=*/0, token, token + 1),
            state,
            /*l2norm_qk=*/true);
        expected_output.slice(/*dim=*/1, token, token + 1).copy_(output);
        const int64_t checkpoint = token - boundaries[sequence];
        const int32_t final_slot =
            slots[sequence * checkpoint_width + checkpoint];
        if (final_slot > 0) {
          expected_pool[final_slot].copy_(state.squeeze(/*dim=*/0));
          written_slots[final_slot] = true;
        }
      }
    }
    torch::Tensor state_indices =
        torch::tensor(slots, int_options).view({kSequences, checkpoint_width});
    torch::Tensor cu_seqlens = torch::tensor(boundaries, int_options);
    const torch::Tensor accepted_tokens = torch::tensor(accepted, int_options);
    auto [actual_output, actual_pool] =
        kernel::mlu::fused_sigmoid_gating_delta_rule_update(
            a_log,
            raw_gate,
            raw_beta,
            bias,
            q,
            k,
            v,
            state_pool,
            state_indices,
            cu_seqlens,
            /*scale=*/1.0 / std::sqrt(static_cast<double>(kHeadDim)),
            /*use_qk_l2norm_in_kernel=*/true,
            /*softplus_beta=*/1.0f,
            /*softplus_threshold=*/20.0f,
            accepted_tokens,
            /*inplace_final_state=*/true,
            /*is_kda=*/true,
            /*kda_use_safe_gate=*/true,
            /*kda_gate_lower_bound=*/-5.0f);
    EXPECT_EQ(actual_pool.data_ptr(), state_pool.data_ptr());
    EXPECT_TRUE(torch::allclose(
        actual_output, expected_output, /*rtol=*/3e-2, /*atol=*/2e-3));
    // Compare every checkpoint, including untouched and empty-request slots.
    EXPECT_TRUE(torch::allclose(
        actual_pool, expected_pool, /*rtol=*/5e-4, /*atol=*/5e-5));
    std::vector<int64_t> untouched_slots;
    untouched_slots.reserve(kPoolSlots);
    for (int64_t slot = 0; slot < kPoolSlots; ++slot) {
      if (!written_slots[slot]) {
        untouched_slots.emplace_back(slot);
      }
    }
    const torch::Tensor untouched_indices =
        torch::tensor(untouched_slots, int_options.dtype(torch::kInt64));
    EXPECT_TRUE(
        torch::equal(actual_pool.index_select(/*dim=*/0, untouched_indices),
                     expected_pool.index_select(/*dim=*/0, untouched_indices)));
    const torch::Tensor padding_output = actual_output.slice(
        /*dim=*/1, boundaries[padding_sequence], boundaries.back());
    EXPECT_TRUE(
        torch::equal(padding_output, torch::zeros_like(padding_output)));
    EXPECT_TRUE(torch::isfinite(actual_output).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(actual_pool).all().item<bool>());
  }
}

TEST(Glm5NextKDATest, NonInplaceDecodeReturnsEveryStateAcrossTokenWindows) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(/*seed=*/20260911);

  constexpr int64_t kSequences = 3;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kPoolSlots = 5;
  constexpr int64_t kTotalTokens = 15;
  const std::vector<int32_t> boundaries = {0, 5, 11, 15};
  const std::vector<int32_t> initial_slots = {3, 1, 4};
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions bf16_options =
      fp32_options.dtype(torch::kBFloat16);
  const torch::TensorOptions int_options = fp32_options.dtype(torch::kInt32);
  const torch::Tensor a_log = torch::zeros({kNumHeads}, fp32_options);
  const torch::Tensor bias =
      torch::randn({kNumHeads * kHeadDim}, fp32_options) * 0.1f;
  torch::Tensor q =
      torch::randn({1, kTotalTokens, kNumHeads, kHeadDim}, bf16_options);
  torch::Tensor k = torch::randn_like(q);
  torch::Tensor v = torch::randn_like(q);
  torch::Tensor raw_gate =
      (torch::randn({kTotalTokens, kNumHeads * kHeadDim}, fp32_options) *
           0.25f -
       6.0f)
          .to(torch::kBFloat16);
  torch::Tensor raw_beta =
      torch::randn({kTotalTokens, kNumHeads}, bf16_options);
  torch::Tensor initial_pool =
      torch::randn({kPoolSlots, kNumHeads, kHeadDim, kHeadDim}, fp32_options) *
      0.03f;
  const torch::Tensor original_pool = initial_pool.clone();
  const torch::Tensor log_gate =
      glm5_next_safe_gate(raw_gate.view({kTotalTokens, kNumHeads, kHeadDim}),
                          a_log,
                          bias,
                          /*lower_bound=*/-5.0f);
  torch::Tensor expected_output = torch::empty_like(v);
  torch::Tensor expected_states =
      torch::empty({kTotalTokens, kNumHeads, kHeadDim, kHeadDim}, fp32_options);
  for (int64_t sequence = 0; sequence < kSequences; ++sequence) {
    const int64_t initial_slot = initial_slots[sequence];
    torch::Tensor state =
        original_pool.slice(/*dim=*/0, initial_slot, initial_slot + 1).clone();
    for (int64_t token = boundaries[sequence]; token < boundaries[sequence + 1];
         ++token) {
      torch::Tensor output;
      std::tie(output, state) = glm5_next_kda_eager_recurrence(
          q.slice(/*dim=*/1, token, token + 1),
          k.slice(/*dim=*/1, token, token + 1),
          v.slice(/*dim=*/1, token, token + 1),
          log_gate.slice(/*dim=*/0, token, token + 1),
          raw_beta.slice(/*dim=*/0, token, token + 1),
          state,
          /*l2norm_qk=*/true);
      expected_output.slice(/*dim=*/1, token, token + 1).copy_(output);
      expected_states[token].copy_(state.squeeze(/*dim=*/0));
    }
  }

  // Capacity one identifies the initial slot only: non-inplace output must
  // continue beyond the four-token window and return a state for every token.
  torch::Tensor state_indices =
      torch::tensor(initial_slots, int_options).view({kSequences, 1});
  torch::Tensor cu_seqlens = torch::tensor(boundaries, int_options);
  const torch::Tensor accepted_tokens = torch::ones({kSequences}, int_options);
  auto [actual_output, actual_states] =
      kernel::mlu::fused_sigmoid_gating_delta_rule_update(
          a_log,
          raw_gate,
          raw_beta,
          bias,
          q,
          k,
          v,
          initial_pool,
          state_indices,
          cu_seqlens,
          /*scale=*/1.0 / std::sqrt(static_cast<double>(kHeadDim)),
          /*use_qk_l2norm_in_kernel=*/true,
          /*softplus_beta=*/1.0f,
          /*softplus_threshold=*/20.0f,
          accepted_tokens,
          /*inplace_final_state=*/false,
          /*is_kda=*/true,
          /*kda_use_safe_gate=*/true,
          /*kda_gate_lower_bound=*/-5.0f);
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::equal(initial_pool, original_pool));
  EXPECT_NE(actual_states.data_ptr(), initial_pool.data_ptr());
  ASSERT_EQ(actual_states.sizes(), expected_states.sizes());
  EXPECT_EQ(actual_states.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::isfinite(actual_output).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(actual_states).all().item<bool>());
  EXPECT_TRUE(torch::allclose(
      actual_output, expected_output, /*rtol=*/3e-2, /*atol=*/2e-2));
  EXPECT_TRUE(torch::allclose(
      actual_states, expected_states, /*rtol=*/3e-2, /*atol=*/2e-2));
}

}  // namespace
}  // namespace xllm::layer
