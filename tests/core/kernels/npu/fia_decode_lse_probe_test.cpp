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

// DCP-2 probe: can npu_fused_infer_attention emit a CORRECT softmax_lse in the
// decode setting (single query token + paged KV + block_table + GQA)?
// DCP decode-merge needs per-rank LSE, but xLLM decode currently runs
// batch_decode (ATB paged attention) which emits no LSE. The plan is to switch
// decode to FIA with softmax_lse_flag=true. This probe verifies, on real NPU:
//   (1) FIA decode output matches batch_decode output (attention numerics OK);
//   (2) FIA softmax_lse is finite and order-of-magnitude sane.
// It also verifies the DCP-specific batch metadata shape and reports whether
// FIA accepts a zero local-KV shard before DCP-2 production merge is designed.

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {
namespace test {
namespace {

class FiaDecodeLseProbe : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }
  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  torch::Device device_ = torch::Device("npu:0");
};

torch::Tensor make_slot_mapping(const std::vector<int32_t>& slots_host,
                                const torch::Device& device) {
  return torch::tensor(slots_host, torch::TensorOptions().dtype(torch::kInt32))
      .to(device);
}

void write_paged_kv_cache(torch::Tensor& key,
                          torch::Tensor& value,
                          torch::Tensor& k_cache,
                          torch::Tensor& v_cache,
                          const std::vector<int32_t>& slots_host,
                          const torch::Device& device) {
  const torch::Tensor slot_mapping = make_slot_mapping(slots_host, device);
  std::optional<torch::Tensor> value_opt = value;
  std::optional<torch::Tensor> v_cache_opt = v_cache;
  reshape_paged_cache(key, value_opt, k_cache, v_cache_opt, slot_mapping);
}

float max_abs_diff(const torch::Tensor& expected, const torch::Tensor& actual) {
  const torch::Tensor expected_cpu =
      expected.cpu().to(torch::kFloat32).reshape({-1});
  const torch::Tensor actual_cpu =
      actual.cpu().to(torch::kFloat32).reshape({-1});
  CHECK_EQ(expected_cpu.numel(), actual_cpu.numel());
  return (expected_cpu - actual_cpu).abs().max().item<float>();
}

std::pair<torch::Tensor, torch::Tensor> reference_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    double scale,
    bool causal) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(value.sizes(), key.sizes());
  CHECK_EQ(query.size(2), key.size(2));
  CHECK_EQ(query.size(1) % key.size(1), 0);

  const torch::Tensor query_cpu = query.cpu().to(torch::kFloat32);
  const torch::Tensor key_cpu = key.cpu().to(torch::kFloat32);
  const torch::Tensor value_cpu = value.cpu().to(torch::kFloat32);
  const int64_t expansion_factor = query_cpu.size(1) / key_cpu.size(1);
  const torch::Tensor expanded_key =
      key_cpu.unsqueeze(2)
          .expand({key_cpu.size(0),
                   key_cpu.size(1),
                   expansion_factor,
                   key_cpu.size(2)})
          .reshape({key_cpu.size(0), query_cpu.size(1), key_cpu.size(2)});
  const torch::Tensor expanded_value =
      value_cpu.unsqueeze(2)
          .expand({value_cpu.size(0),
                   value_cpu.size(1),
                   expansion_factor,
                   value_cpu.size(2)})
          .reshape({value_cpu.size(0), query_cpu.size(1), value_cpu.size(2)});

  torch::Tensor scores =
      torch::einsum("qhd,khd->qhk", {query_cpu, expanded_key}) * scale;
  if (causal) {
    CHECK_EQ(query_cpu.size(0), key_cpu.size(0));
    const torch::Tensor causal_mask =
        torch::triu(torch::ones({query_cpu.size(0), key_cpu.size(0)},
                                torch::TensorOptions().dtype(torch::kBool)),
                    1);
    scores = scores.masked_fill(causal_mask.unsqueeze(1),
                                -std::numeric_limits<float>::infinity());
  }
  const torch::Tensor lse = torch::logsumexp(scores, -1, true);
  const torch::Tensor output = torch::einsum(
      "qhk,khd->qhd", {torch::softmax(scores, -1), expanded_value});
  return {output, lse};
}

torch::Tensor make_fia_causal_mask(const torch::Device& device) {
  const torch::TensorOptions options =
      torch::TensorOptions().device(device).dtype(torch::kFloat32);
  return torch::triu(torch::ones({2048, 2048}, options), 1)
      .to(torch::kInt8)
      .contiguous();
}

// One decode step: batch=1, q_len=1, ctx_len tokens already in paged KV cache.
// GQA: num_heads=8, num_kv_heads=2 (num_heads > num_kv_heads).
TEST_F(FiaDecodeLseProbe, DecodeFiaOutputMatchesBatchDecodeAndLseIsFinite) {
  // Dims mirror real Qwen3.5-2B attention (k cache shape [nblk,128,2,256]):
  // block_size=128, num_kv_heads=2, head_dim=256, num_heads=8 (GQA).
  const int64_t ctx_len = 200;  // history tokens in KV cache (>1 block)
  const int64_t block_size = 128;
  const int64_t num_blocks = 8;
  const int64_t num_heads = 8;
  const int64_t num_kv_heads = 2;
  const int64_t head_dim = 256;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  auto opts = torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  // --- Fill paged KV cache with ctx_len tokens via the real write path. ---
  torch::Tensor key =
      torch::randn({ctx_len, num_kv_heads, head_dim}, opts) * 0.1;
  torch::Tensor value =
      torch::randn({ctx_len, num_kv_heads, head_dim}, opts) * 0.1;
  torch::Tensor k_cache =
      torch::zeros({num_blocks, block_size, num_kv_heads, head_dim}, opts);
  torch::Tensor v_cache = torch::zeros_like(k_cache);

  std::vector<int32_t> slots_host;
  slots_host.reserve(ctx_len);
  for (int64_t t = 0; t < ctx_len; ++t) {
    slots_host.push_back(static_cast<int32_t>(t));  // contiguous slots 0..ctx-1
  }
  write_paged_kv_cache(key, value, k_cache, v_cache, slots_host, device_);

  // block_table: sequence occupies blocks 0..ceil(ctx/block_size)-1.
  const int64_t n_used_blocks = (ctx_len + block_size - 1) / block_size;
  std::vector<int32_t> bt_host;
  for (int64_t b = 0; b < n_used_blocks; ++b) {
    bt_host.push_back(static_cast<int32_t>(b));
  }
  torch::Tensor block_table =
      torch::tensor(bt_host, torch::TensorOptions().dtype(torch::kInt32))
          .to(device_)
          .view({1, n_used_blocks});

  // --- Decode query: 1 token. ---
  torch::Tensor query = torch::randn({1, num_heads, head_dim}, opts) * 0.1;
  // context_lens must be a CPU host int32 tensor: ATB PagedAttention marks it
  // as hostData (Input(context_lens, /*isHost=*/true)); a device tensor makes
  // PagedAttentionOperation setup fail.
  torch::Tensor seq_lens =
      torch::tensor({static_cast<int32_t>(ctx_len)},
                    torch::TensorOptions().dtype(torch::kInt32));

  // --- (A) golden: existing batch_decode (no LSE). ---
  torch::Tensor out_golden = torch::zeros({1, num_heads, head_dim}, opts);
  batch_decode(query,
               k_cache,
               v_cache,
               static_cast<float>(scale),
               block_table,
               seq_lens,
               out_golden);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  // KV cache viewed to 3D [num_blocks, block_size, num_kv_heads*head_dim]
  // (avoids FIA reading head_dim=256 and rejecting it in TND). Decode is
  // non-causal (no mask), so sparse_mode MUST be 0 (FIA: "when attnMask is not
  // provided, sparseMode must be 0"). This differs from chunked_prefill which
  // passes a causal mask + sparse_mode=3.
  torch::Tensor k_view = k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  torch::Tensor v_view = v_cache.view({v_cache.size(0), v_cache.size(1), -1});
  std::vector<int64_t> actual_seq_lengths = {1};  // q tokens per seq
  std::vector<int64_t> actual_seq_lengths_kv = {ctx_len};
  std::optional<torch::Tensor> no_mask = std::nullopt;
  std::optional<torch::Tensor> bt_opt = block_table;
  auto [out_fia, lse_fia] =
      npu_fused_infer_attention(query,
                                k_view,
                                v_view,
                                no_mask,
                                bt_opt,
                                actual_seq_lengths,
                                actual_seq_lengths_kv,
                                num_heads,
                                num_kv_heads,
                                scale,
                                block_size,
                                /*sparse_mode=*/0,
                                "TND",
                                /*softmax_lse_flag=*/true);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  // (1) output numerics match batch_decode.
  const float max_diff = max_abs_diff(out_golden, out_fia);
  const float golden_absmax =
      out_golden.cpu().to(torch::kFloat32).abs().max().item<float>();
  LOG(INFO) << "[DCP2-probe] output max|golden-fia|=" << max_diff
            << " golden_absmax=" << golden_absmax;
  EXPECT_LT(max_diff, 2e-2f)
      << "FIA decode output diverges from batch_decode (max_diff=" << max_diff
      << ")";

  // (2) LSE finite + sane.
  ASSERT_TRUE(lse_fia.defined() && lse_fia.numel() > 0)
      << "FIA returned empty softmax_lse under softmax_lse_flag=true";
  const torch::Tensor lse = lse_fia.cpu().to(torch::kFloat32);
  const bool all_finite = torch::isfinite(lse).all().item<bool>();
  const float lse_min = lse.min().item<float>();
  const float lse_max = lse.max().item<float>();
  LOG(INFO) << "[DCP2-probe] lse shape=" << lse.sizes() << " min=" << lse_min
            << " max=" << lse_max << " finite=" << all_finite;
  EXPECT_TRUE(all_finite) << "FIA softmax_lse has nan/inf";
  // LSE = log(sum exp(scores)) over ctx_len keys; must be finite real number.
  EXPECT_GT(lse_max, -1e30f) << "LSE unreasonably small";
  EXPECT_LT(lse_max, 1e30f) << "LSE unreasonably large";
}

TEST_F(FiaDecodeLseProbe, MultiTokenRawKvCausalOutputAndLseMatchReference) {
  const int64_t num_heads = 8;
  const int64_t num_kv_heads = 2;
  const int64_t head_dim = 128;
  const int64_t token_count = 7;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const torch::TensorOptions options =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  const torch::Tensor query =
      torch::randn({token_count, num_heads, head_dim}, options) * 0.1;
  const torch::Tensor key =
      torch::randn({token_count, num_kv_heads, head_dim}, options) * 0.1;
  const torch::Tensor value =
      torch::randn({token_count, num_kv_heads, head_dim}, options) * 0.1;
  const torch::Tensor causal_mask = make_fia_causal_mask(device_);
  const std::vector<int64_t> cumulative_query_lengths = {3, 7};

  const auto [output, lse] =
      npu_fused_infer_attention(query,
                                key,
                                value,
                                std::make_optional(causal_mask),
                                /*block_table=*/std::nullopt,
                                cumulative_query_lengths,
                                cumulative_query_lengths,
                                num_heads,
                                num_kv_heads,
                                scale,
                                /*block_size=*/0,
                                /*sparse_mode=*/3,
                                "TND",
                                /*softmax_lse_flag=*/true);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  ASSERT_EQ(output.sizes(),
            torch::IntArrayRef({token_count, num_heads, head_dim}));
  ASSERT_EQ(lse.sizes(), torch::IntArrayRef({token_count, num_heads, 1}));
  std::vector<torch::Tensor> reference_outputs;
  std::vector<torch::Tensor> reference_lses;
  int64_t sequence_begin = 0;
  for (const int64_t sequence_end : cumulative_query_lengths) {
    const int64_t sequence_length = sequence_end - sequence_begin;
    const auto [sequence_output, sequence_lse] =
        reference_attention(query.narrow(0, sequence_begin, sequence_length),
                            key.narrow(0, sequence_begin, sequence_length),
                            value.narrow(0, sequence_begin, sequence_length),
                            scale,
                            /*causal=*/true);
    reference_outputs.emplace_back(sequence_output);
    reference_lses.emplace_back(sequence_lse);
    sequence_begin = sequence_end;
  }
  const torch::Tensor reference_output = torch::cat(reference_outputs, 0);
  const torch::Tensor reference_lse = torch::cat(reference_lses, 0);
  const float output_max_diff = max_abs_diff(reference_output, output);
  const float lse_max_diff = max_abs_diff(reference_lse, lse);
  LOG(INFO) << "[DCP4-probe][raw-multi-token] output_max_diff="
            << output_max_diff << " lse_max_diff=" << lse_max_diff
            << " output_shape=" << output.sizes()
            << " lse_shape=" << lse.sizes();
  EXPECT_LT(output_max_diff, 3e-2f);
  EXPECT_LT(lse_max_diff, 3e-2f);
}

TEST_F(FiaDecodeLseProbe, MultiTokenPagedContextTruncatesSharedPartialBlock) {
  const int64_t num_heads = 8;
  const int64_t num_kv_heads = 1;
  const int64_t head_dim = 128;
  const int64_t block_size = 128;
  const int64_t num_blocks = 4;
  const int64_t context_len = 130;
  const int64_t token_count = 5;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const torch::TensorOptions options =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);

  const torch::Tensor query =
      torch::randn({token_count, num_heads, head_dim}, options) * 0.1;
  torch::Tensor context_key =
      torch::randn({context_len, num_kv_heads, head_dim}, options) * 0.1;
  torch::Tensor context_value =
      torch::randn({context_len, num_kv_heads, head_dim}, options) * 0.1;
  torch::Tensor k_cache =
      torch::zeros({num_blocks, block_size, num_kv_heads, head_dim}, options);
  torch::Tensor v_cache = torch::zeros_like(k_cache);

  std::vector<int32_t> context_slots;
  context_slots.reserve(context_len);
  for (int64_t token = 0; token < context_len; ++token) {
    const int64_t physical_block = token < block_size ? 1 : 3;
    const int64_t block_offset = token % block_size;
    context_slots.emplace_back(
        static_cast<int32_t>(physical_block * block_size + block_offset));
  }
  write_paged_kv_cache(
      context_key, context_value, k_cache, v_cache, context_slots, device_);

  const torch::Tensor block_table =
      torch::tensor(std::vector<int32_t>{0, 2, 1, 3},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(device_)
          .view({2, 2});
  const std::vector<int64_t> cumulative_query_lengths = {2, 5};
  const std::vector<int64_t> local_context_lengths = {0, context_len};
  const torch::Tensor k_view =
      k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  const torch::Tensor v_view =
      v_cache.view({v_cache.size(0), v_cache.size(1), -1});
  const auto [baseline_output, baseline_lse] =
      npu_fused_infer_attention(query,
                                k_view,
                                v_view,
                                /*atten_mask=*/std::nullopt,
                                std::make_optional(block_table),
                                cumulative_query_lengths,
                                local_context_lengths,
                                num_heads,
                                num_kv_heads,
                                scale,
                                block_size,
                                /*sparse_mode=*/0,
                                "TND",
                                /*softmax_lse_flag=*/true);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  torch::Tensor current_key =
      torch::full({3, num_kv_heads, head_dim}, 50.0, options);
  torch::Tensor current_value =
      torch::full({3, num_kv_heads, head_dim}, 100.0, options);
  const std::vector<int32_t> current_slots = {
      static_cast<int32_t>(3 * block_size + 2),
      static_cast<int32_t>(3 * block_size + 3),
      static_cast<int32_t>(3 * block_size + 4)};
  write_paged_kv_cache(
      current_key, current_value, k_cache, v_cache, current_slots, device_);
  const auto [polluted_output, polluted_lse] =
      npu_fused_infer_attention(query,
                                k_view,
                                v_view,
                                /*atten_mask=*/std::nullopt,
                                std::make_optional(block_table),
                                cumulative_query_lengths,
                                local_context_lengths,
                                num_heads,
                                num_kv_heads,
                                scale,
                                block_size,
                                /*sparse_mode=*/0,
                                "TND",
                                /*softmax_lse_flag=*/true);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  ASSERT_EQ(polluted_output.sizes(),
            torch::IntArrayRef({token_count, num_heads, head_dim}));
  ASSERT_EQ(polluted_lse.sizes(),
            torch::IntArrayRef({token_count, num_heads, 1}));
  const torch::Tensor positive_query = query.narrow(0, 2, 3);
  const auto [reference_output, reference_lse] =
      reference_attention(positive_query,
                          context_key,
                          context_value,
                          scale,
                          /*causal=*/false);
  const torch::Tensor baseline_positive_output =
      baseline_output.narrow(0, 2, 3);
  const torch::Tensor baseline_positive_lse = baseline_lse.narrow(0, 2, 3);
  const torch::Tensor polluted_positive_output =
      polluted_output.narrow(0, 2, 3);
  const torch::Tensor polluted_positive_lse = polluted_lse.narrow(0, 2, 3);
  const float output_pollution_diff =
      max_abs_diff(baseline_positive_output, polluted_positive_output);
  const float lse_pollution_diff =
      max_abs_diff(baseline_positive_lse, polluted_positive_lse);
  const float output_reference_diff =
      max_abs_diff(reference_output, polluted_positive_output);
  const float lse_reference_diff =
      max_abs_diff(reference_lse, polluted_positive_lse);
  LOG(INFO) << "[DCP4-probe][paged-partial-block] output_pollution_diff="
            << output_pollution_diff
            << " lse_pollution_diff=" << lse_pollution_diff
            << " output_reference_diff=" << output_reference_diff
            << " lse_reference_diff=" << lse_reference_diff;
  EXPECT_LT(output_pollution_diff, 1e-3f)
      << "FIA over-read nonzero current KV beyond local_context_len";
  EXPECT_LT(lse_pollution_diff, 1e-3f)
      << "FIA LSE included current KV beyond local_context_len";
  EXPECT_LT(output_reference_diff, 3e-2f);
  EXPECT_LT(lse_reference_diff, 3e-2f);
}

// DCP gathers Q heads across two ranks before each rank runs FIA against its
// local KV. This models rank 1 of a dcp_size=2 group: two requests have 72 and
// 128 local KV tokens, while FIA sees R * Hq_local = 2 * 4 Q heads and one
// local KV head. For TND decode, actual_seq_lengths must be cumulative Q ends.
TEST_F(FiaDecodeLseProbe,
       BatchDecodeUsesCumulativeQueryLengthsAndGatheredGqaHeads) {
  const int64_t dcp_size = 2;
  const int64_t local_num_q_heads = 4;
  const int64_t num_heads = dcp_size * local_num_q_heads;
  const int64_t num_kv_heads = 1;
  const int64_t head_dim = 256;
  const int64_t block_size = 128;
  const int64_t num_blocks = 4;
  const int64_t first_local_kv_len = 72;
  const int64_t second_local_kv_len = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  const auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  const torch::Tensor first_key =
      torch::randn({first_local_kv_len, num_kv_heads, head_dim}, opts) * 0.1;
  const torch::Tensor first_value =
      torch::randn({first_local_kv_len, num_kv_heads, head_dim}, opts) * 0.1;
  const torch::Tensor second_key =
      torch::randn({second_local_kv_len, num_kv_heads, head_dim}, opts) * 0.1;
  const torch::Tensor second_value =
      torch::randn({second_local_kv_len, num_kv_heads, head_dim}, opts) * 0.1;
  torch::Tensor key = torch::cat({first_key, second_key}, /*dim=*/0);
  torch::Tensor value = torch::cat({first_value, second_value}, /*dim=*/0);
  torch::Tensor k_cache =
      torch::zeros({num_blocks, block_size, num_kv_heads, head_dim}, opts);
  torch::Tensor v_cache = torch::zeros_like(k_cache);

  std::vector<int32_t> slots_host;
  slots_host.reserve(first_local_kv_len + second_local_kv_len);
  for (int64_t token = 0; token < first_local_kv_len; ++token) {
    slots_host.push_back(static_cast<int32_t>(block_size + token));
  }
  for (int64_t token = 0; token < second_local_kv_len; ++token) {
    slots_host.push_back(static_cast<int32_t>(3 * block_size + token));
  }
  write_paged_kv_cache(key, value, k_cache, v_cache, slots_host, device_);

  // Local table keeps original physical block ids selected by DCP-1c.
  const torch::Tensor block_table =
      torch::tensor(std::vector<int32_t>{1, 3},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(device_)
          .view({2, 1});
  const torch::Tensor query =
      torch::randn({2, num_heads, head_dim}, opts) * 0.1;
  const torch::Tensor local_kv_lens = torch::tensor(
      std::vector<int32_t>{static_cast<int32_t>(first_local_kv_len),
                           static_cast<int32_t>(second_local_kv_len)},
      torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor out_golden = torch::zeros({2, num_heads, head_dim}, opts);
  batch_decode(query,
               k_cache,
               v_cache,
               static_cast<float>(scale),
               block_table,
               local_kv_lens,
               out_golden);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  const torch::Tensor k_view =
      k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  const torch::Tensor v_view =
      v_cache.view({v_cache.size(0), v_cache.size(1), -1});
  const std::vector<int64_t> actual_seq_lengths = {1, 2};
  const std::vector<int64_t> actual_seq_lengths_kv = {first_local_kv_len,
                                                      second_local_kv_len};
  const auto [out_fia, lse_fia] =
      npu_fused_infer_attention(query,
                                k_view,
                                v_view,
                                std::nullopt,
                                block_table,
                                actual_seq_lengths,
                                actual_seq_lengths_kv,
                                num_heads,
                                num_kv_heads,
                                scale,
                                block_size,
                                /*sparse_mode=*/0,
                                "TND",
                                /*softmax_lse_flag=*/true);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  const float output_max_diff = max_abs_diff(out_golden, out_fia);
  LOG(INFO) << "[DCP2-probe][batch] output max|golden-fia|=" << output_max_diff
            << " q_lengths={1,2} kv_lengths={" << first_local_kv_len << ","
            << second_local_kv_len << "}";
  EXPECT_LT(output_max_diff, 2e-2f);

  ASSERT_EQ(lse_fia.dim(), 3);
  EXPECT_EQ(lse_fia.size(0), 2);
  EXPECT_EQ(lse_fia.size(1), num_heads);
  EXPECT_EQ(lse_fia.size(2), 1);
  EXPECT_TRUE(torch::isfinite(lse_fia.cpu()).all().item<bool>());
}

// A short request can have no block owned by this rank while a later request
// in the same decode batch has local KV. Probe a leading zero explicitly: FIA
// must at least accept the metadata and preserve the positive row's result.
TEST_F(FiaDecodeLseProbe, LeadingZeroLocalKvDoesNotCorruptPositiveRow) {
  const int64_t num_heads = 8;
  const int64_t num_kv_heads = 1;
  const int64_t head_dim = 256;
  const int64_t block_size = 128;
  const int64_t num_blocks = 4;
  const int64_t positive_local_kv_len = 72;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  const auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  torch::Tensor key =
      torch::randn({positive_local_kv_len, num_kv_heads, head_dim}, opts) * 0.1;
  torch::Tensor value =
      torch::randn({positive_local_kv_len, num_kv_heads, head_dim}, opts) * 0.1;
  torch::Tensor k_cache =
      torch::zeros({num_blocks, block_size, num_kv_heads, head_dim}, opts);
  torch::Tensor v_cache = torch::zeros_like(k_cache);

  std::vector<int32_t> slots_host;
  slots_host.reserve(positive_local_kv_len);
  for (int64_t token = 0; token < positive_local_kv_len; ++token) {
    slots_host.push_back(static_cast<int32_t>(3 * block_size + token));
  }
  write_paged_kv_cache(key, value, k_cache, v_cache, slots_host, device_);

  // The first row has no local KV. Its table entry is intentionally ignored by
  // actual_seq_lengths_kv; the second row owns physical block 3.
  const torch::Tensor block_table =
      torch::tensor(std::vector<int32_t>{1, 3},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(device_)
          .view({2, 1});
  const torch::Tensor query =
      torch::randn({2, num_heads, head_dim}, opts) * 0.1;
  const torch::Tensor positive_query =
      query.slice(/*dim=*/0, /*start=*/1, /*end=*/2).contiguous();
  const torch::Tensor positive_block_table =
      block_table.slice(/*dim=*/0, /*start=*/1, /*end=*/2).contiguous();
  const torch::Tensor positive_kv_len = torch::tensor(
      std::vector<int32_t>{static_cast<int32_t>(positive_local_kv_len)},
      torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor positive_golden = torch::zeros({1, num_heads, head_dim}, opts);
  batch_decode(positive_query,
               k_cache,
               v_cache,
               static_cast<float>(scale),
               positive_block_table,
               positive_kv_len,
               positive_golden);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  const torch::Tensor k_view =
      k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  const torch::Tensor v_view =
      v_cache.view({v_cache.size(0), v_cache.size(1), -1});
  const std::vector<int64_t> actual_seq_lengths = {1, 2};
  const std::vector<int64_t> actual_seq_lengths_kv = {0, positive_local_kv_len};
  const auto [out_fia, lse_fia] =
      npu_fused_infer_attention(query,
                                k_view,
                                v_view,
                                std::nullopt,
                                block_table,
                                actual_seq_lengths,
                                actual_seq_lengths_kv,
                                num_heads,
                                num_kv_heads,
                                scale,
                                block_size,
                                /*sparse_mode=*/0,
                                "TND",
                                /*softmax_lse_flag=*/true);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  const torch::Tensor positive_fia =
      out_fia.slice(/*dim=*/0, /*start=*/1, /*end=*/2).contiguous();
  const float positive_max_diff = max_abs_diff(positive_golden, positive_fia);
  EXPECT_LT(positive_max_diff, 2e-2f)
      << "leading zero local KV corrupted the following positive row";

  const torch::Tensor zero_out =
      out_fia.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
          .cpu()
          .to(torch::kFloat32);
  const torch::Tensor zero_lse =
      lse_fia.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
          .cpu()
          .to(torch::kFloat32);
  LOG(INFO) << "[DCP2-probe][leading-zero] positive max|golden-fia|="
            << positive_max_diff
            << " zero_out_absmax=" << zero_out.abs().max().item<float>()
            << " zero_lse_min=" << zero_lse.min().item<float>()
            << " zero_lse_max=" << zero_lse.max().item<float>()
            << " zero_lse_finite="
            << torch::isfinite(zero_lse).all().item<bool>();
}

// DCP-1c can legitimately select no table column when every request is shorter
// than this rank's first owned block. Keep this separate so a possible FIA
// fatal for [batch, 0] does not hide the positive-batch probe result.
TEST_F(FiaDecodeLseProbe, AllZeroLocalKvWithEmptyBlockTableReportsFiaBehavior) {
  const int64_t num_heads = 8;
  const int64_t num_kv_heads = 1;
  const int64_t head_dim = 256;
  const int64_t block_size = 128;
  const int64_t num_blocks = 1;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  const auto opts =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  const torch::Tensor query =
      torch::randn({2, num_heads, head_dim}, opts) * 0.1;
  const torch::Tensor k_cache =
      torch::zeros({num_blocks, block_size, num_kv_heads, head_dim}, opts);
  const torch::Tensor v_cache = torch::zeros_like(k_cache);
  const torch::Tensor block_table = torch::empty(
      {2, 0}, torch::TensorOptions().device(device_).dtype(torch::kInt32));
  const torch::Tensor k_view =
      k_cache.view({k_cache.size(0), k_cache.size(1), -1});
  const torch::Tensor v_view =
      v_cache.view({v_cache.size(0), v_cache.size(1), -1});
  const std::vector<int64_t> actual_seq_lengths = {1, 2};
  const std::vector<int64_t> actual_seq_lengths_kv = {0, 0};

  const auto [out_fia, lse_fia] =
      npu_fused_infer_attention(query,
                                k_view,
                                v_view,
                                std::nullopt,
                                block_table,
                                actual_seq_lengths,
                                actual_seq_lengths_kv,
                                num_heads,
                                num_kv_heads,
                                scale,
                                block_size,
                                /*sparse_mode=*/0,
                                "TND",
                                /*softmax_lse_flag=*/true);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  ASSERT_EQ(out_fia.sizes(), torch::IntArrayRef({2, num_heads, head_dim}));
  ASSERT_EQ(lse_fia.sizes(), torch::IntArrayRef({2, num_heads, 1}));
  const torch::Tensor out_cpu = out_fia.cpu().to(torch::kFloat32);
  const torch::Tensor lse_cpu = lse_fia.cpu().to(torch::kFloat32);
  LOG(INFO) << "[DCP2-probe][all-zero-empty-table] out_absmax="
            << out_cpu.abs().max().item<float>()
            << " lse_min=" << lse_cpu.min().item<float>()
            << " lse_max=" << lse_cpu.max().item<float>()
            << " lse_finite=" << torch::isfinite(lse_cpu).all().item<bool>();
}

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::npu
