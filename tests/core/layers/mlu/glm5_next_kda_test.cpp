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

#include <c10/core/DeviceGuard.h>
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/model/llm_model_params.h"
#include "framework/model/model_args.h"
#include "framework/model/model_params_view.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/mlu/chunk_kda.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"
#include "kernels/param.h"
#include "kv_cache/storage/kv_cache.h"
#include "kv_cache/storage/kv_cache_utils.h"
#include "layers/common/attention_metadata.h"
#include "tests/core/layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {

torch::Tensor l2_normalize(const torch::Tensor& input) {
  return input / torch::sqrt(torch::sum(input * input, -1, true) + 1e-6f);
}

torch::Tensor glm5_next_safe_gate(const torch::Tensor& raw_gate,
                                  const torch::Tensor& A_log,
                                  const torch::Tensor& dt_bias,
                                  float lower_bound) {
  const int64_t num_heads = raw_gate.size(-2);
  const int64_t head_dim = raw_gate.size(-1);

  torch::Tensor gate_fp32 = raw_gate.to(torch::kFloat32);
  torch::Tensor a_scale =
      torch::exp(A_log.to(torch::kFloat32)).view({1, num_heads, 1});
  torch::Tensor bias =
      dt_bias.to(torch::kFloat32).view({1, num_heads, head_dim});
  return lower_bound * torch::sigmoid(a_scale * (gate_fp32 + bias));
}

std::tuple<torch::Tensor, torch::Tensor> glm5_next_kda_eager_recurrence(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& log_gate,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    bool l2norm_qk) {
  torch::Tensor gate = log_gate.unsqueeze(0);
  torch::Tensor beta_values = beta.unsqueeze(0);

  torch::Tensor q_fp32 = q.to(torch::kFloat32);
  torch::Tensor k_fp32 = k.to(torch::kFloat32);
  if (l2norm_qk) {
    q_fp32 = l2_normalize(q_fp32);
    k_fp32 = l2_normalize(k_fp32);
  }
  const double scale = 1.0 / std::sqrt(static_cast<double>(q.size(3)));
  q_fp32 = q_fp32 * scale;

  torch::Tensor state = initial_state.to(torch::kFloat32).clone();
  torch::Tensor v_fp32 = v.to(torch::kFloat32);
  torch::Tensor gate_fp32 = gate.to(torch::kFloat32);
  torch::Tensor beta_fp32 = torch::sigmoid(beta_values.to(torch::kFloat32));
  std::vector<torch::Tensor> outputs;
  outputs.reserve(static_cast<size_t>(q.size(1)));

  for (int64_t token = 0; token < q.size(1); ++token) {
    torch::Tensor q_t = q_fp32.select(1, token);
    torch::Tensor k_t = k_fp32.select(1, token);
    torch::Tensor v_t = v_fp32.select(1, token);
    torch::Tensor gate_t = gate_fp32.select(1, token);
    torch::Tensor beta_t = beta_fp32.select(1, token);

    state = state * torch::exp(gate_t).unsqueeze(-2);
    torch::Tensor prediction = torch::sum(state * k_t.unsqueeze(-2), -1);
    torch::Tensor delta = (v_t - prediction) * beta_t.unsqueeze(-1);
    state = state + delta.unsqueeze(-1) * k_t.unsqueeze(-2);
    outputs.push_back(torch::sum(state * q_t.unsqueeze(-2), -1));
  }

  return {torch::stack(outputs, 1).to(v.scalar_type()), state};
}

constexpr std::array<const char*, 6> kProjectionNames = {"q_proj.weight",
                                                         "k_proj.weight",
                                                         "v_proj.weight",
                                                         "b_proj.weight",
                                                         "f_a_proj.weight",
                                                         "g_a_proj.weight"};

TEST(Glm5NextKDAInputProjectionTest, LoadsShardsInAnyOrderAndReplicatesGates) {
  torch::InferenceMode guard;
  torch::manual_seed(20260908);
  constexpr int64_t kHiddenSize = 64;
  constexpr int64_t kNumHeads = 8;
  constexpr int64_t kHeadDim = 16;
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const std::array<int64_t, 6> widths = {128, 128, 128, 8, 16, 16};
  std::array<torch::Tensor, 6> weights;
  for (size_t index = 0; index < weights.size(); ++index) {
    weights[index] = torch::randn({widths[index], kHiddenSize}, options);
  }
  for (const int32_t tp_size : {1, 2, 8}) {
    for (int32_t rank = 0; rank < tp_size; ++rank) {
      Glm5NextKDAInputProjection projection(
          kHiddenSize, kNumHeads, kHeadDim, rank, tp_size, options);
      for (const size_t index : {5, 2, 0, 4, 1, 3}) {
        projection->load_state_dict(
            StateDict({{kProjectionNames[index], weights[index]}}));
      }
      projection->verify_loaded_weights("");
      projection->load_state_dict(
          StateDict({{"q_proj.weight", torch::zeros_like(weights[0])}}));
      std::vector<torch::Tensor> expected;
      expected.reserve(weights.size());
      for (size_t index = 0; index < weights.size(); ++index) {
        const int64_t width =
            index < 4 ? widths[index] / tp_size : widths[index];
        const int64_t offset = index < 4 ? rank * width : 0;
        expected.emplace_back(weights[index].narrow(0, offset, width));
      }
      EXPECT_TRUE(torch::equal(projection->named_parameters()["weight"],
                               torch::cat(expected, 0)));
    }
  }
}

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

TEST(Glm5NextKDATest, PaddedVerifyLayerPreservesStateAndReplays) {
  const torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.hidden_size(128)
      .linear_num_key_heads(8)
      .linear_num_value_heads(8)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .rms_norm_eps(1e-6);
  Glm5NextKDA layer(args, QuantArgs(), parallel, options);
  torch::manual_seed(20260918);
  for (auto& parameter : layer->named_parameters()) {
    parameter.value().normal_(0.0, 0.03);
  }
  for (int32_t width : {2, 3, 4, 5}) {
    SCOPED_TRACE(width);
    auto conv = torch::randn({3, width + 2, 3 * 8 * 128}, options);
    auto ssm =
        torch::randn({3 * width, 8, 128, 128}, options.dtype(torch::kFloat32)) *
        0.03;
    KVCache eager_cache(
        LinearAttentionKVCacheTensors{conv.clone(), ssm.clone()});
    KVCache graph_cache(
        LinearAttentionKVCacheTensors{conv.clone(), ssm.clone()});
    auto hidden = torch::randn({3 * width, 128}, options);
    LlmModelParams padded;
    padded.is_spec_verify = true;
    padded.embedding.linear_state_ids = {1, 0, 0};
    padded.embedding.linear_state_indices = torch::tensor({1, 0, 0}, ints);
    padded.num_accepted_tokens = torch::tensor({width, 1, width}, ints);
    LlmModelParams real = padded.clone();
    real.embedding.linear_state_ids = {1};
    real.embedding.linear_state_indices = torch::tensor({1}, ints);
    real.num_accepted_tokens = padded.num_accepted_tokens.narrow(0, 0, 1);
    ModelInputSnapshot padded_snapshot(std::move(padded));
    ModelInputSnapshot real_snapshot(std::move(real));
    ModelParamsView padded_view = padded_snapshot.view();
    ModelParamsView real_view = real_snapshot.view();
    AttentionMetadata padded_meta{};
    padded_meta.is_chunked_prefill = true;
    padded_meta.max_query_len = width;
    padded_meta.q_cu_seq_lens =
        torch::tensor({0, width, 2 * width, 3 * width}, ints);
    AttentionMetadata real_meta = padded_meta;
    real_meta.q_cu_seq_lens = torch::tensor({0, width}, ints);
    auto expected = layer->forward(
        hidden.narrow(0, 0, width), real_meta, eager_cache, real_view);
    auto warmup = layer->forward(hidden, padded_meta, graph_cache, padded_view);
    EXPECT_TRUE(
        torch::allclose(warmup.narrow(0, 0, width), expected, 0.02, 0.002));
    EXPECT_TRUE(
        torch::equal(warmup.narrow(0, width, 2 * width),
                     torch::zeros_like(warmup.narrow(0, width, 2 * width))));
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    torch::Tensor output;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      output = layer->forward(hidden, padded_meta, graph_cache, padded_view);
      graph.capture_end();
    }
    for (int32_t round = 0; round < 3; ++round) {
      EXPECT_TRUE(torch::equal(graph_cache.get_conv_cache(),
                               eager_cache.get_conv_cache()));
      EXPECT_TRUE(torch::allclose(graph_cache.get_ssm_cache(),
                                  eager_cache.get_ssm_cache(),
                                  0.001,
                                  0.0001));
      EXPECT_TRUE(torch::equal(graph_cache.get_ssm_cache().narrow(0, 0, width),
                               ssm.narrow(0, 0, width)));
      hidden.normal_();
      padded_view.num_accepted_tokens[0].fill_(round % width + 1);
      expected = layer->forward(
          hidden.narrow(0, 0, width), real_meta, eager_cache, real_view);
      graph.replay();
      EXPECT_TRUE(
          torch::allclose(output.narrow(0, 0, width), expected, 0.02, 0.002));
      EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
    }
    EXPECT_TRUE(torch::equal(graph_cache.get_conv_cache(),
                             eager_cache.get_conv_cache()));
    EXPECT_TRUE(torch::allclose(graph_cache.get_ssm_cache(),
                                eager_cache.get_ssm_cache(),
                                0.001,
                                0.0001));
  }
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

  torch::Tensor expected_output;
  torch::Tensor expected_state;
  std::tie(expected_output, expected_state) = packed_eager_kda(
      q, k, v, log_gate, raw_beta, initial_state, sequence_lengths);
  kernel::mlu::ChunkKDA chunk_kda(kNumHeads);
  chunk_kda->to(device);
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

  const std::array<std::pair<int64_t, bool>, 10> decode_cases = {{{2, false},
                                                                  {2, true},
                                                                  {3, false},
                                                                  {3, true},
                                                                  {4, false},
                                                                  {4, true},
                                                                  {5, false},
                                                                  {5, true},
                                                                  {8, false},
                                                                  {8, true}}};
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
    const int64_t padding_sequence = kSequences - 3;
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

void check_four_token_verify(bool ragged, float gate_value, bool correlated) {
  constexpr int64_t kSequences = 16;
  constexpr int64_t kWidth = 4;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 128;
  constexpr int64_t kTokens = kSequences * kWidth;
  constexpr int64_t kSlots = kTokens + 1;
  const torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  torch::manual_seed(20260922);
  const auto fp32 =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  const auto bf16 = fp32.dtype(torch::kBFloat16);
  const auto ints = fp32.dtype(torch::kInt32);
  torch::Tensor q = torch::randn({1, kTokens, kHeads, kDim}, bf16);
  torch::Tensor k = torch::randn_like(q);
  if (correlated) {
    k = q.slice(1, 0, 1).expand_as(q).contiguous();
    q = k + torch::randn_like(k) * 0.01f;
  }
  torch::Tensor v = torch::randn_like(q);
  torch::Tensor raw_gate =
      torch::full({kTokens, kHeads * kDim}, gate_value, bf16);
  torch::Tensor raw_beta = torch::full({kTokens, kHeads}, 2.0f, bf16);
  const torch::Tensor a_log = correlated ? torch::zeros({kHeads}, fp32)
                                         : torch::randn({kHeads}, fp32) * 0.1f;
  const torch::Tensor bias = correlated
                                 ? torch::zeros({kHeads * kDim}, fp32)
                                 : torch::randn({kHeads * kDim}, fp32) * 0.1f;
  if (!correlated) {
    raw_gate.add_(torch::randn_like(raw_gate) * 0.5f);
    raw_beta.copy_(torch::randn_like(raw_beta));
  }
  const torch::Tensor log_gate =
      glm5_next_safe_gate(raw_gate.view({kTokens, kHeads, kDim}),
                          a_log,
                          bias,
                          /*lower_bound=*/-5.0f);
  torch::Tensor state =
      torch::randn({kSlots, kHeads, kDim, kDim}, fp32) * 0.03f;
  const torch::Tensor original_state = state.clone();
  std::vector<int32_t> offsets{0};
  std::vector<int32_t> slots;
  offsets.reserve(kSequences + 1);
  slots.reserve(kTokens);
  for (int64_t seq = 0; seq < kSequences; ++seq) {
    const int32_t length =
        ragged ? static_cast<int32_t>(seq % (kWidth + 1)) : kWidth;
    offsets.emplace_back(offsets.back() + length);
    for (int64_t token = 0; token < kWidth; ++token) {
      slots.emplace_back(
          static_cast<int32_t>((kSequences - seq - 1) * kWidth + token + 1));
    }
  }
  std::fill(slots.end() - kWidth, slots.end(), 0);
  slots[2 * kWidth + 1] = 0;
  torch::Tensor indices = torch::tensor(slots, ints).view({kSequences, kWidth});
  torch::Tensor cu = torch::tensor(offsets, ints);
  torch::Tensor accepted = torch::ones({kSequences}, ints);
  const auto run = [&]() {
    return kernel::mlu::fused_sigmoid_gating_delta_rule_update(
        a_log,
        raw_gate,
        raw_beta,
        bias,
        q,
        k,
        v,
        state,
        indices,
        cu,
        /*scale=*/1.0 / std::sqrt(static_cast<double>(kDim)),
        /*use_qk_l2norm_in_kernel=*/true,
        /*softplus_beta=*/1.0f,
        /*softplus_threshold=*/20.0f,
        accepted,
        /*inplace_final_state=*/true,
        /*is_kda=*/true,
        /*kda_use_safe_gate=*/true,
        /*kda_gate_lower_bound=*/-5.0f);
  };
  run();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor graph_output;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, 0));
    graph.capture_begin();
    graph_output = run().first;
    graph.capture_end();
  }
  for (int32_t round : {0, 1, 3}) {
    SCOPED_TRACE(round);
    state.copy_(original_state);
    std::vector<int32_t> counts;
    counts.reserve(kSequences);
    for (int64_t seq = 0; seq < kSequences; ++seq) {
      counts.emplace_back(static_cast<int32_t>((seq + round) % kWidth + 1));
    }
    counts[2] = 3;
    if (ragged) {
      counts[0] = 0;
    }
    accepted.copy_(torch::tensor(counts, ints));
    torch::Tensor expected_output = torch::zeros_like(v);
    torch::Tensor expected_state = original_state.clone();
    std::array<bool, kSlots> written{};
    for (int64_t seq = 0; seq < kSequences; ++seq) {
      if (offsets[seq] == offsets[seq + 1]) {
        continue;
      }
      const int32_t initial_slot = slots[seq * kWidth + counts[seq] - 1];
      if (initial_slot <= 0) {
        continue;
      }
      torch::Tensor recurrence =
          original_state[initial_slot].unsqueeze(0).clone();
      for (int64_t token = offsets[seq]; token < offsets[seq + 1]; ++token) {
        torch::Tensor output;
        std::tie(output, recurrence) =
            glm5_next_kda_eager_recurrence(q.slice(1, token, token + 1),
                                           k.slice(1, token, token + 1),
                                           v.slice(1, token, token + 1),
                                           log_gate.slice(0, token, token + 1),
                                           raw_beta.slice(0, token, token + 1),
                                           recurrence,
                                           /*l2norm_qk=*/true);
        expected_output.slice(1, token, token + 1).copy_(output);
        const int32_t slot = slots[seq * kWidth + token - offsets[seq]];
        if (slot <= 0) {
          continue;
        }
        expected_state[slot].copy_(recurrence.squeeze(0));
        written[slot] = true;
      }
    }
    auto [actual_output, actual_state] = run();
    EXPECT_EQ(actual_state.data_ptr(), state.data_ptr());
    EXPECT_TRUE(torch::allclose(
        actual_output, expected_output, /*rtol=*/3e-2, /*atol=*/2e-3));
    EXPECT_TRUE(torch::allclose(
        actual_state, expected_state, /*rtol=*/5e-4, /*atol=*/5e-5));
    state.copy_(original_state);
    graph.replay();
    EXPECT_TRUE(torch::allclose(
        graph_output, expected_output, /*rtol=*/3e-2, /*atol=*/2e-3));
    EXPECT_TRUE(
        torch::allclose(state, expected_state, /*rtol=*/5e-4, /*atol=*/5e-5));
    EXPECT_TRUE(torch::isfinite(graph_output).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(state).all().item<bool>());
    std::vector<int64_t> untouched;
    untouched.reserve(kSlots);
    for (int64_t slot = 0; slot < kSlots; ++slot) {
      if (!written[slot]) {
        untouched.emplace_back(slot);
      }
    }
    const torch::Tensor untouched_indices =
        torch::tensor(untouched, ints.dtype(torch::kInt64));
    EXPECT_TRUE(
        torch::equal(state.index_select(0, untouched_indices),
                     original_state.index_select(0, untouched_indices)));
    EXPECT_TRUE(
        torch::equal(graph_output.slice(1, offsets.back()),
                     torch::zeros_like(graph_output.slice(1, offsets.back()))));
  }
}

TEST(Glm5NextKDATest, FourTokenVerifyPreservesAllCheckpointsAndReplays) {
  check_four_token_verify(/*ragged=*/false,
                          /*gate_value=*/-2.0f,
                          /*correlated=*/false);
}

TEST(Glm5NextKDATest, FourTokenVerifyHandlesRaggedAndNullSlots) {
  check_four_token_verify(/*ragged=*/true,
                          /*gate_value=*/-2.0f,
                          /*correlated=*/false);
}

TEST(Glm5NextKDATest, FourTokenVerifyCorrelatedKeysAcrossSafeGateRange) {
  for (float gate_value : {-30.0f, 30.0f}) {
    SCOPED_TRACE(gate_value);
    check_four_token_verify(/*ragged=*/false,
                            gate_value,
                            /*correlated=*/true);
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
