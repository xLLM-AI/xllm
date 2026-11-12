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

#include "kernels/mlu/chunk_kda.h"

#include <framework/core/device.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace xllm::kernel::mlu {
namespace {

constexpr int64_t kNumHeads = 8;
constexpr int64_t kHeadDim = 128;

torch::Tensor make_cu_seqlens(const std::vector<int64_t>& sequence_lengths,
                              const torch::Device& device) {
  std::vector<int32_t> cumulative_lengths = {0};
  cumulative_lengths.reserve(sequence_lengths.size() + 1);
  int64_t cumulative_length = 0;
  for (const int64_t sequence_length : sequence_lengths) {
    cumulative_length += sequence_length;
    cumulative_lengths.push_back(static_cast<int32_t>(cumulative_length));
  }
  return torch::tensor(
      cumulative_lengths,
      torch::TensorOptions().dtype(torch::kInt32).device(device));
}

torch::Tensor make_chunk_indices(const torch::Tensor& cu_seqlens,
                                 int64_t chunk_size) {
  torch::Tensor lengths = cu_seqlens.diff();
  torch::Tensor chunk_counts =
      ((lengths + chunk_size - 1) / chunk_size).to(torch::kLong);
  torch::Tensor chunk_offsets = torch::cumsum(chunk_counts, /*dim=*/0);
  const int64_t total_chunks = chunk_offsets[-1].item<int64_t>();
  torch::Tensor flat_indices =
      torch::arange(total_chunks, cu_seqlens.options());
  torch::Tensor prefixes =
      torch::cat({torch::zeros({1}, chunk_offsets.options()),
                  chunk_offsets.slice(/*dim=*/0, /*start=*/0, /*end=*/-1)});
  torch::Tensor local_indices =
      flat_indices - torch::repeat_interleave(prefixes, chunk_counts);
  torch::Tensor sequence_indices = (local_indices == 0).cumsum(/*dim=*/0) - 1;
  return torch::stack({sequence_indices, local_indices}, /*dim=*/1)
      .to(torch::kInt32);
}

torch::Tensor make_scaled_identity_states(const torch::Tensor& state_scales,
                                          int64_t num_sequences,
                                          const torch::TensorOptions& options) {
  return torch::eye(kHeadDim, options).view({1, 1, kHeadDim, kHeadDim}) *
         state_scales.view({num_sequences, kNumHeads, 1, 1});
}

TEST(ChunkKDATest,
     PackedHighConcurrencyPreservesStateIsolationAcrossChunkBoundaries) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kNumSequences = 128;
  const int64_t chunk_size = kda_prefill_chunk_size();
  const std::vector<int64_t> boundary_lengths = {1,
                                                 chunk_size - 1,
                                                 chunk_size,
                                                 chunk_size + 1,
                                                 2 * chunk_size - 1,
                                                 2 * chunk_size,
                                                 2 * chunk_size + 1};
  std::vector<int64_t> sequence_lengths;
  sequence_lengths.reserve(kNumSequences);
  for (int64_t sequence_id = 0; sequence_id < kNumSequences; ++sequence_id) {
    sequence_lengths.push_back(
        boundary_lengths[static_cast<size_t>(sequence_id) %
                         boundary_lengths.size()]);
  }
  const int64_t total_tokens = std::accumulate(
      sequence_lengths.begin(), sequence_lengths.end(), int64_t{0});
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  const torch::Tensor qkv =
      torch::ones({1, total_tokens, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor log_gate =
      torch::zeros({total_tokens, kNumHeads, kHeadDim}, fp32_options);
  const torch::Tensor beta =
      torch::zeros({total_tokens, kNumHeads}, fp32_options);
  const torch::Tensor sequence_scales =
      torch::arange(kNumSequences, fp32_options).view({kNumSequences, 1}) /
      static_cast<double>(2 * kNumSequences);
  const torch::Tensor head_scales =
      torch::arange(kNumHeads, fp32_options).view({1, kNumHeads}) /
      static_cast<double>(4 * kNumHeads);
  const torch::Tensor state_scales = 0.25f + sequence_scales + head_scales;
  const torch::Tensor initial_state =
      make_scaled_identity_states(state_scales, kNumSequences, fp32_options);
  const torch::Tensor cu_seqlens = make_cu_seqlens(sequence_lengths, device);
  const torch::Tensor chunk_indices =
      make_chunk_indices(cu_seqlens, chunk_size);

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [output, final_state] = chunk_kda->forward(qkv,
                                                  qkv,
                                                  qkv,
                                                  log_gate,
                                                  beta,
                                                  initial_state,
                                                  cu_seqlens,
                                                  chunk_indices,
                                                  /*output_final_state=*/true,
                                                  /*use_qk_l2norm=*/true);
  torch_mlu::synchronize();

  const torch::Tensor length_tensor =
      torch::tensor(sequence_lengths,
                    torch::TensorOptions().dtype(torch::kLong).device(device));
  const torch::Tensor token_scales =
      torch::repeat_interleave(state_scales, length_tensor, /*dim=*/0);
  const torch::Tensor expected_output =
      (token_scales / static_cast<double>(kHeadDim))
          .unsqueeze(/*dim=*/-1)
          .expand({total_tokens, kNumHeads, kHeadDim});

  ASSERT_EQ(output.sizes(),
            torch::IntArrayRef({1, total_tokens, kNumHeads, kHeadDim}));
  ASSERT_EQ(final_state.sizes(), initial_state.sizes());
  EXPECT_TRUE(torch::allclose(output.squeeze(/*dim=*/0).to(torch::kFloat32),
                              expected_output,
                              /*rtol=*/0.0,
                              /*atol=*/1e-4));
  EXPECT_TRUE(torch::allclose(final_state,
                              initial_state,
                              /*rtol=*/0.0,
                              /*atol=*/1e-6));
}

TEST(ChunkKDATest, LongContextProcessesAllWorkspaceGroupsAt128K) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  constexpr int64_t kLongContextTokens = 128 * 1024;
  constexpr int64_t kWorkspaceGroupChunks = 128;
  const int64_t chunk_size = kda_prefill_chunk_size();
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  const torch::Tensor qkv =
      torch::ones({1, kLongContextTokens, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor log_gate =
      torch::zeros({1, 1, 1}, fp32_options)
          .expand({kLongContextTokens, kNumHeads, kHeadDim});
  const torch::Tensor beta =
      torch::zeros({kLongContextTokens, kNumHeads}, fp32_options);
  const torch::Tensor initial_state =
      torch::eye(kHeadDim, fp32_options)
          .view({1, 1, kHeadDim, kHeadDim})
          .expand({1, kNumHeads, kHeadDim, kHeadDim})
          .contiguous();
  const std::vector<int64_t> sequence_lengths = {kLongContextTokens};
  const torch::Tensor cu_seqlens = make_cu_seqlens(sequence_lengths, device);
  const torch::Tensor chunk_indices =
      make_chunk_indices(cu_seqlens, chunk_size);
  ASSERT_GT(chunk_indices.size(0), kWorkspaceGroupChunks);

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [output, final_state] = chunk_kda->forward(qkv,
                                                  qkv,
                                                  qkv,
                                                  log_gate,
                                                  beta,
                                                  initial_state,
                                                  cu_seqlens,
                                                  chunk_indices,
                                                  /*output_final_state=*/true,
                                                  /*use_qk_l2norm=*/true);
  torch_mlu::synchronize();

  ASSERT_EQ(output.sizes(),
            torch::IntArrayRef({1, kLongContextTokens, kNumHeads, kHeadDim}));
  ASSERT_EQ(final_state.sizes(), initial_state.sizes());
  const float output_min = output.amin().item<float>();
  const float output_max = output.amax().item<float>();
  constexpr float kExpectedOutput = 1.0f / static_cast<float>(kHeadDim);
  EXPECT_TRUE(std::isfinite(output_min));
  EXPECT_TRUE(std::isfinite(output_max));
  EXPECT_NEAR(output_min, kExpectedOutput, 1e-4f);
  EXPECT_NEAR(output_max, kExpectedOutput, 1e-4f);
  EXPECT_TRUE(torch::allclose(final_state,
                              initial_state,
                              /*rtol=*/0.0,
                              /*atol=*/1e-6));
}

TEST(ChunkKDATest, Random128KMatchesEightKSegmentedStateChaining) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260904);

  constexpr int64_t kLongContextTokens = 128 * 1024;
  constexpr int64_t kSegmentTokens = 8 * 1024;
  const int64_t chunk_size = kda_prefill_chunk_size();
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const std::vector<int64_t> input_shape = {
      1, kLongContextTokens, kNumHeads, kHeadDim};

  // Share one immutable random tensor for q/k/v so the 128K regression stays
  // below the per-process MLU memory quota.
  const torch::Tensor qkv = torch::randn(input_shape, bf16_options) * 0.1f;
  const torch::Tensor log_gate =
      -(torch::rand({kLongContextTokens, kNumHeads, kHeadDim}, fp32_options) *
            0.02f +
        0.005f);
  const torch::Tensor beta =
      torch::rand({kLongContextTokens, kNumHeads}, fp32_options) * 0.2f + 0.05f;
  const torch::Tensor initial_state =
      torch::randn({1, kNumHeads, kHeadDim, kHeadDim}, fp32_options) * 0.01f;

  const std::vector<int64_t> full_sequence_lengths = {kLongContextTokens};
  const torch::Tensor full_cu_seqlens =
      make_cu_seqlens(full_sequence_lengths, device);
  const torch::Tensor full_chunk_indices =
      make_chunk_indices(full_cu_seqlens, chunk_size);
  const std::vector<int64_t> segment_sequence_lengths = {kSegmentTokens};
  const torch::Tensor segment_cu_seqlens =
      make_cu_seqlens(segment_sequence_lengths, device);
  const torch::Tensor segment_chunk_indices =
      make_chunk_indices(segment_cu_seqlens, chunk_size);
  ASSERT_EQ(kLongContextTokens % kSegmentTokens, 0);
  ASSERT_GT(full_chunk_indices.size(0), segment_chunk_indices.size(0));

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [full_output, full_final_state] =
      chunk_kda->forward(qkv,
                         qkv,
                         qkv,
                         log_gate,
                         beta,
                         initial_state,
                         full_cu_seqlens,
                         full_chunk_indices,
                         /*output_final_state=*/true,
                         /*use_qk_l2norm=*/true);

  torch::Tensor segmented_state = initial_state;
  for (int64_t segment_begin = 0; segment_begin < kLongContextTokens;
       segment_begin += kSegmentTokens) {
    const int64_t segment_end = segment_begin + kSegmentTokens;
    auto [segment_output, segment_final_state] =
        chunk_kda->forward(qkv.slice(/*dim=*/1,
                                     /*start=*/segment_begin,
                                     /*end=*/segment_end),
                           qkv.slice(/*dim=*/1,
                                     /*start=*/segment_begin,
                                     /*end=*/segment_end),
                           qkv.slice(/*dim=*/1,
                                     /*start=*/segment_begin,
                                     /*end=*/segment_end),
                           log_gate.slice(/*dim=*/0,
                                          /*start=*/segment_begin,
                                          /*end=*/segment_end),
                           beta.slice(/*dim=*/0,
                                      /*start=*/segment_begin,
                                      /*end=*/segment_end),
                           segmented_state,
                           segment_cu_seqlens,
                           segment_chunk_indices,
                           /*output_final_state=*/true,
                           /*use_qk_l2norm=*/true);
    EXPECT_TRUE(torch::allclose(full_output.slice(/*dim=*/1,
                                                  /*start=*/segment_begin,
                                                  /*end=*/segment_end),
                                segment_output,
                                /*rtol=*/1e-3,
                                /*atol=*/1e-3))
        << "output mismatch in segment starting at token " << segment_begin;
    segmented_state = segment_final_state;
  }
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::allclose(full_final_state,
                              segmented_state,
                              /*rtol=*/1e-4,
                              /*atol=*/1e-4));
}

TEST(ChunkKDATest, OmittingFinalStateKeepsOutputContract) {
  torch::Device device(torch::kPrivateUse1, /*index=*/0);
  torch::DeviceGuard guard(device);

  const int64_t chunk_size = kda_prefill_chunk_size();
  const int64_t token_count = chunk_size + 1;
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::Tensor qkv =
      torch::zeros({1, token_count, kNumHeads, kHeadDim}, bf16_options);
  const torch::Tensor log_gate =
      torch::zeros({token_count, kNumHeads, kHeadDim}, fp32_options);
  const torch::Tensor beta =
      torch::zeros({token_count, kNumHeads}, fp32_options);
  const torch::Tensor initial_state =
      torch::zeros({1, kNumHeads, kHeadDim, kHeadDim}, fp32_options);
  const std::vector<int64_t> sequence_lengths = {token_count};
  const torch::Tensor cu_seqlens = make_cu_seqlens(sequence_lengths, device);
  const torch::Tensor chunk_indices =
      make_chunk_indices(cu_seqlens, chunk_size);

  ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
  auto [output, final_state] = chunk_kda->forward(qkv,
                                                  qkv,
                                                  qkv,
                                                  log_gate,
                                                  beta,
                                                  initial_state,
                                                  cu_seqlens,
                                                  chunk_indices,
                                                  /*output_final_state=*/false,
                                                  /*use_qk_l2norm=*/true);
  torch_mlu::synchronize();

  EXPECT_EQ(output.sizes(),
            torch::IntArrayRef({1, token_count, kNumHeads, kHeadDim}));
  EXPECT_TRUE(torch::equal(output, torch::zeros_like(output)));
  EXPECT_FALSE(final_state.defined());
}

void check_raw_gate_against_recurrence(int64_t num_heads, bool use_qk_l2norm) {
  torch::Device device(torch::kPrivateUse1, 0);
  torch::DeviceGuard guard(device);
  torch::manual_seed(20260915);
  const int64_t chunk_size = kda_prefill_chunk_size();
  const std::vector<int64_t> sequence_lengths = {
      1, chunk_size - 1, 2 * chunk_size + 1};
  const int64_t total_tokens = std::accumulate(
      sequence_lengths.begin(), sequence_lengths.end(), int64_t{0});
  const int64_t num_sequences = static_cast<int64_t>(sequence_lengths.size());
  const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
  const std::vector<int64_t> shape = {total_tokens, num_heads, kHeadDim};
  const torch::Tensor q =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor k =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor v =
      (torch::randn(shape, cpu_options) * 0.1f).to(torch::kBFloat16);
  const torch::Tensor raw_gate =
      (torch::randn(shape, cpu_options) * 2.0f - 2.0f).to(torch::kBFloat16);
  const torch::Tensor raw_beta =
      torch::randn({total_tokens, num_heads}, cpu_options).to(torch::kBFloat16);
  const torch::Tensor a_log = torch::randn({num_heads}, cpu_options) * 0.2f;
  const torch::Tensor dt_bias =
      torch::randn({num_heads * kHeadDim}, cpu_options) * 0.1f;
  constexpr float kGateLowerBound = -5.0f;
  const torch::Tensor gate =
      kGateLowerBound *
      torch::sigmoid(torch::exp(a_log).view({1, num_heads, 1}) *
                     (raw_gate.to(torch::kFloat32) +
                      dt_bias.view({1, num_heads, kHeadDim})));
  const torch::Tensor beta = torch::sigmoid(raw_beta.to(torch::kFloat32));
  const torch::Tensor initial_state =
      torch::randn({num_sequences, num_heads, kHeadDim, kHeadDim},
                   cpu_options) *
      0.01f;
  torch::Tensor q_reference = q.to(torch::kFloat32);
  torch::Tensor k_reference = k.to(torch::kFloat32);
  if (use_qk_l2norm) {
    q_reference *=
        torch::rsqrt(torch::sum(q_reference.square(), -1, true) + 1e-6f);
    k_reference *=
        torch::rsqrt(torch::sum(k_reference.square(), -1, true) + 1e-6f);
  }
  q_reference /= std::sqrt(static_cast<float>(kHeadDim));
  const torch::Tensor v_reference = v.to(torch::kFloat32);
  torch::Tensor expected_output = torch::empty(shape, cpu_options);
  torch::Tensor expected_state = initial_state.clone();
  int64_t token = 0;
  for (int64_t sequence = 0; sequence < num_sequences; ++sequence) {
    torch::Tensor state = expected_state[sequence];
    for (int64_t local_token = 0; local_token < sequence_lengths[sequence];
         ++local_token, ++token) {
      state *= torch::exp(gate[token]).unsqueeze(-2);
      const torch::Tensor key = k_reference[token].unsqueeze(-2);
      const torch::Tensor delta =
          (v_reference[token] - torch::sum(state * key, -1)) *
          beta[token].unsqueeze(-1);
      state += delta.unsqueeze(-1) * key;
      expected_output[token].copy_(
          torch::sum(state * q_reference[token].unsqueeze(-2), -1));
    }
  }
  const torch::Tensor cu_seqlens = make_cu_seqlens(sequence_lengths, device);
  const torch::Tensor chunk_indices =
      make_chunk_indices(cu_seqlens, chunk_size);
  ChunkKDA chunk_kda(num_heads);
  auto [output, final_state] =
      chunk_kda->forward_raw_gate(q.unsqueeze(0).to(device),
                                  k.unsqueeze(0).to(device),
                                  v.unsqueeze(0).to(device),
                                  raw_gate.to(device),
                                  a_log.to(device),
                                  dt_bias.to(device),
                                  kGateLowerBound,
                                  raw_beta.to(device),
                                  initial_state.to(device),
                                  cu_seqlens,
                                  chunk_indices,
                                  /*output_final_state=*/true,
                                  use_qk_l2norm);
  torch_mlu::synchronize();
  EXPECT_TRUE(
      torch::allclose(output.squeeze(0).to(torch::kCPU).to(torch::kFloat32),
                      expected_output,
                      /*rtol=*/5e-3,
                      /*atol=*/2e-5));
  EXPECT_TRUE(torch::allclose(final_state.to(torch::kCPU),
                              expected_state,
                              /*rtol=*/1e-3,
                              /*atol=*/2e-5));
}

TEST(ChunkKDATest, RawGatePackedBoundariesMatchTokenRecurrence) {
  check_raw_gate_against_recurrence(kNumHeads, /*use_qk_l2norm=*/true);
}

TEST(ChunkKDATest, RawGateSmallHeadCountRetainsGenericPath) {
  check_raw_gate_against_recurrence(/*num_heads=*/2, /*use_qk_l2norm=*/true);
}

TEST(ChunkKDATest, RawGateWithoutNormalizationMatchesTokenRecurrence) {
  check_raw_gate_against_recurrence(kNumHeads, /*use_qk_l2norm=*/false);
}

}  // namespace
}  // namespace xllm::kernel::mlu
