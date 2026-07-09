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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#include "kernels/mlu/chunk_gated_delta_rule.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm {
namespace layer {
namespace {

using xllm::kernel::mlu::causal_conv1d_fn;
using xllm::kernel::mlu::causal_conv1d_update_decode;
using xllm::kernel::mlu::ChunkGatedDeltaRule;
using xllm::kernel::mlu::fused_gdn_gating;
using xllm::kernel::mlu::fused_recurrent_gated_delta_rule_packed_decode;

// ---------------------------------------------------------------------------
// Shared fixture: initializes a single MLU device and exposes seeded tensor
// helpers that match the conventions used by neighbouring MLU layer tests.
// ---------------------------------------------------------------------------
class Qwen3_5GatedDeltaNetOpsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    torch::Device device(Platform::type_torch(), 0);
    Device xllm_device(device);
    xllm_device.set_seed(42);
    device_ = device;
  }

  // Seeded noise in [-stddev, +stddev) on the MLU device, bf16 by default.
  torch::Tensor MakeNoise(const std::string& key,
                          torch::IntArrayRef shape,
                          float stddev,
                          torch::ScalarType dtype = torch::kBFloat16) {
    auto raw = test::seeded_tensor(key, shape, dtype, device_);
    return (raw - 0.5f) * (std::sqrt(12.0f) * stddev);
  }

  torch::Tensor MakeOnes(torch::IntArrayRef shape, torch::ScalarType dtype) {
    return torch::ones(shape,
                       torch::TensorOptions().dtype(dtype).device(device_));
  }

  torch::Tensor Zeros(torch::IntArrayRef shape, torch::ScalarType dtype) {
    return torch::zeros(shape,
                        torch::TensorOptions().dtype(dtype).device(device_));
  }

  void Sync() {
    Device xllm_device(device_);
    xllm_device.synchronize_default_stream();
  }

  static torch::Device device_;
};

torch::Device Qwen3_5GatedDeltaNetOpsTest::device_ = torch::kCPU;

// Build the (batch, token_block_offset, tot) triple used by the MLU
// causal_conv1d_fn kernel. Mirrors the layout prepared by the Qwen3.5 model
// (block_size = 8, padded with pad_slot_id = -1).
struct ConvBatchMeta {
  torch::Tensor batch;
  torch::Tensor token_block_offset;
  int32_t tot = 0;
};

ConvBatchMeta MakeConvBatchMeta(const torch::Tensor& q_cu_seq_lens,
                                const torch::Device& device) {
  constexpr int32_t block_size = 8;
  constexpr int32_t pad_slot_id = -1;
  constexpr int64_t default_max_num_programs = 1024;

  auto seqlens = q_cu_seq_lens.diff();
  auto nums = (seqlens + block_size - 1) / block_size;
  nums = nums.to(torch::kLong);
  int32_t tot = nums.sum().item<int32_t>();
  torch::Tensor range_batch = torch::arange(nums.size(0), nums.options());
  torch::Tensor mlist_tensor = torch::repeat_interleave(range_batch, nums);
  int64_t mlist_len = mlist_tensor.size(0);
  int64_t max_num_programs = std::max(default_max_num_programs, mlist_len) * 2;

  auto opts = torch::dtype(torch::kInt32).device(device);
  torch::Tensor batch_ptr = torch::full({max_num_programs}, pad_slot_id, opts);
  torch::Tensor token_block_offset_ptr =
      torch::full({max_num_programs}, pad_slot_id, opts);

  std::vector<torch::Tensor> vec;
  vec.reserve(nums.size(0));
  for (int64_t i = 0; i < nums.size(0); ++i) {
    vec.emplace_back(torch::arange(nums[i].item<int64_t>(), nums.options()));
  }
  torch::Tensor offsetlist_tensor = torch::cat(vec, -1).to(torch::kInt32);
  batch_ptr.narrow(0, 0, mlist_len).copy_(mlist_tensor);
  token_block_offset_ptr.narrow(0, 0, mlist_len).copy_(offsetlist_tensor);

  return {batch_ptr, token_block_offset_ptr, tot};
}

// Build chunk_indices [num_chunks, 2] (int32) from cu_seqlens with
// chunk_size=64, matching ChunkGatedDeltaRuleImpl::prepare_chunk_indices.
torch::Tensor MakeChunkIndices(const torch::Tensor& cu_seqlens,
                               int64_t chunk_size) {
  auto lengths = cu_seqlens.narrow(0, 1, cu_seqlens.size(0) - 1) -
                 cu_seqlens.narrow(0, 0, cu_seqlens.size(0) - 1);
  torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
  num_chunks = num_chunks.to(torch::kLong);
  torch::Tensor cumsum = torch::cumsum(num_chunks, 0);
  int64_t total = cumsum[-1].item<int64_t>();
  torch::Tensor arange_total = torch::arange(total, cu_seqlens.options());
  torch::Tensor zeros = torch::zeros({1}, cumsum.options());
  torch::Tensor prefix =
      torch::cat({zeros, cumsum.slice(/*dim=*/0, /*start=*/0, /*end=*/-1)});
  torch::Tensor repeats_prefix = torch::repeat_interleave(prefix, num_chunks);
  torch::Tensor indices = arange_total - repeats_prefix;
  torch::Tensor mask = indices == 0;
  torch::Tensor col0 = mask.cumsum(0) - 1;
  return torch::stack({col0, indices}, /*dim=*/1)
      .to(cu_seqlens)
      .to(torch::kInt32);
}

// ===========================================================================
// fused_gdn_gating: computes (g, beta) from A_log, a, b, dt_bias.
// ===========================================================================
TEST_F(Qwen3_5GatedDeltaNetOpsTest, FusedGdnGatingShapeAndDeterminism) {
  // num_heads must be one of {4,8,12,16,24,32,48,64} (algo table).
  const int64_t num_heads = 16;
  const int64_t num_tokens = 32;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);

  auto a = MakeNoise("gdn_gating.a", {num_tokens, num_heads}, 0.02f);
  auto b = MakeNoise("gdn_gating.b", {num_tokens, num_heads}, 0.02f);
  auto a_log = MakeNoise("gdn_gating.A_log", {num_heads}, 0.02f);
  auto dt_bias = MakeOnes({num_heads}, torch::kBFloat16);

  auto [g, beta] = fused_gdn_gating(
      a_log, a, b, dt_bias, /*beta=*/1.0f, /*threshold=*/20.0f);
  Sync();

  // g: [1, num_tokens, num_heads] fp32, beta: [1, num_tokens, num_heads] bf16.
  EXPECT_EQ(g.sizes(), torch::IntArrayRef({1, num_tokens, num_heads}));
  EXPECT_EQ(g.scalar_type(), torch::kFloat32);
  EXPECT_EQ(beta.sizes(), torch::IntArrayRef({1, num_tokens, num_heads}));
  EXPECT_EQ(beta.scalar_type(), torch::kBFloat16);

  auto g_cpu = g.flatten().to(torch::kFloat32).cpu();
  EXPECT_TRUE(torch::isfinite(g_cpu).all().item<bool>()) << "g must be finite";

  // Determinism: same inputs -> same outputs.
  auto [g2, beta2] = fused_gdn_gating(
      a_log, a, b, dt_bias, /*beta=*/1.0f, /*threshold=*/20.0f);
  Sync();
  EXPECT_TRUE(torch::allclose(g, g2, /*rtol=*/1e-5, /*atol=*/1e-6));
  EXPECT_TRUE(torch::allclose(beta, beta2, /*rtol=*/1e-3, /*atol=*/1e-4));
}

// ===========================================================================
// causal_conv1d_fn: prefill path. x is [channels, num_tokens].
// ===========================================================================
TEST_F(Qwen3_5GatedDeltaNetOpsTest, CausalConv1dFnPrefill) {
  // channels must be in the pre-compiled dim_to_algo_id set.
  const int64_t channels = 1024;
  const int64_t conv_kernel = 4;
  const int64_t state_len = conv_kernel - 1;
  const int64_t batch_size = 2;
  const int64_t seq_len = 8;
  const int64_t num_tokens = batch_size * seq_len;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  auto x = MakeNoise("conv1d_fn.x", {channels, num_tokens}, 0.02f);
  auto weight = MakeNoise("conv1d_fn.weight", {channels, conv_kernel}, 0.02f);
  // conv_states: [num_cache_lines, channels, state_len]
  auto conv_states = Zeros({batch_size, channels, state_len}, torch::kBFloat16);
  auto q_cu_seq_lens =
      torch::arange(0, (batch_size + 1) * seq_len, seq_len, opts_int);
  auto batch_meta = MakeConvBatchMeta(q_cu_seq_lens, device_);

  // cache_indices: per-batch conv-state slot; has_initial_states: false.
  auto cache_indices = torch::arange(0, batch_size, opts_int);
  auto has_initial_states = torch::zeros(
      {batch_size}, torch::TensorOptions().dtype(torch::kBool).device(device_));

  auto out = causal_conv1d_fn(x,
                              weight,
                              conv_states,
                              q_cu_seq_lens,
                              batch_meta.batch,
                              batch_meta.token_block_offset,
                              /*nt=*/static_cast<int32_t>(num_tokens),
                              /*bias=*/std::nullopt,
                              cache_indices,
                              has_initial_states,
                              /*initial_state_idx=*/std::nullopt,
                              /*num_accepted_tokens=*/std::nullopt,
                              /*inplace_final_state=*/true);
  Sync();

  EXPECT_EQ(out.sizes(), x.sizes());
  EXPECT_EQ(out.scalar_type(), x.scalar_type());
  auto out_cpu = out.flatten().to(torch::kFloat32).cpu();
  EXPECT_TRUE(torch::isfinite(out_cpu).all().item<bool>())
      << "conv1d_fn output must be finite";

  // Determinism with fresh conv_state (inplace_final_state mutates it).
  auto conv_states2 =
      Zeros({batch_size, channels, state_len}, torch::kBFloat16);
  auto out2 = causal_conv1d_fn(x,
                               weight,
                               conv_states2,
                               q_cu_seq_lens,
                               batch_meta.batch,
                               batch_meta.token_block_offset,
                               static_cast<int32_t>(num_tokens),
                               std::nullopt,
                               cache_indices,
                               has_initial_states,
                               std::nullopt,
                               std::nullopt,
                               true);
  Sync();
  EXPECT_TRUE(torch::allclose(out, out2, /*rtol=*/1e-3, /*atol=*/1e-4))
      << "conv1d_fn should be deterministic for fixed inputs";
}

// ===========================================================================
// causal_conv1d_update_decode: decode path. x is [batch, dim, seqlen].
// ===========================================================================
TEST_F(Qwen3_5GatedDeltaNetOpsTest, CausalConv1dUpdateDecode) {
  // dim must be in the pre-compiled dim_to_algo_id set.
  const int64_t dim = 1024;
  const int64_t width = 4;
  const int64_t state_len = width - 1;
  const int64_t batch_size = 4;
  const int64_t seqlen = 1;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  auto x = MakeNoise("conv1d_update.x", {batch_size, dim, seqlen}, 0.02f);
  auto conv_state =
      MakeNoise("conv1d_update.state", {batch_size, dim, state_len}, 0.02f);
  auto weight = MakeNoise("conv1d_update.weight", {dim, width}, 0.02f);
  auto conv_state_indices = torch::arange(0, batch_size, opts_int);

  // The kernel shifts new tokens into conv_state in place, so capture the
  // pristine state before any run and feed a fresh clone to each run.
  auto conv_state_orig = conv_state.clone();
  auto conv_state_run1 = conv_state_orig.clone();
  auto out = causal_conv1d_update_decode(x,
                                         conv_state_run1,
                                         weight,
                                         /*bias=*/std::nullopt,
                                         conv_state_indices,
                                         /*pad_slot_id=*/-1);
  Sync();

  EXPECT_EQ(out.sizes(), x.sizes());
  EXPECT_EQ(out.scalar_type(), x.scalar_type());
  auto out_cpu = out.flatten().to(torch::kFloat32).cpu();
  EXPECT_TRUE(torch::isfinite(out_cpu).all().item<bool>())
      << "conv1d_update_decode output must be finite";

  // Determinism: rerun with a fresh clone of the pristine state.
  auto conv_state_run2 = conv_state_orig.clone();
  auto out2 = causal_conv1d_update_decode(
      x, conv_state_run2, weight, std::nullopt, conv_state_indices, -1);
  Sync();
  EXPECT_TRUE(torch::allclose(out, out2, /*rtol=*/1e-3, /*atol=*/1e-4))
      << "conv1d_update_decode should be deterministic for fixed inputs";
}

// ===========================================================================
// fused_recurrent_gated_delta_rule_packed_decode: decode path.
// mixed_qkv: [B, qkv_dim], ssm_cache: [num_slots, HV, V, K] (fp32).
// ===========================================================================
TEST_F(Qwen3_5GatedDeltaNetOpsTest, FusedRecurrentGatedDeltaRulePackedDecode) {
  // K == V == 128 (qwen3.5) keeps the state K/V layout ambiguity harmless.
  const int64_t H = 8;
  const int64_t HV = 8;
  const int64_t K = 128;
  const int64_t V = 128;
  const int64_t batch_size = 4;
  const int64_t qk_dim = 2 * H * K;
  const int64_t qkv_dim = qk_dim + HV * V;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  auto mixed_qkv =
      MakeNoise("recurrent_decode.mixed_qkv", {batch_size, qkv_dim}, 0.02f);
  auto a = MakeNoise("recurrent_decode.a", {batch_size, HV}, 0.02f);
  auto b = MakeNoise("recurrent_decode.b", {batch_size, HV}, 0.02f);
  auto a_log = MakeNoise("recurrent_decode.A_log", {HV}, 0.02f);
  auto dt_bias = MakeOnes({HV}, torch::kBFloat16);
  auto ssm_cache = MakeNoise("recurrent_decode.ssm_cache",
                             {batch_size, HV, V, K},
                             0.01f,
                             torch::kFloat32);
  auto ssm_state_indices = torch::arange(0, batch_size, opts_int);

  // The kernel updates ssm_cache in place; capture the pristine state and feed
  // a fresh clone to each run.
  auto ssm_cache_orig = ssm_cache.clone();
  auto ssm_cache_run1 = ssm_cache_orig.clone();
  double scale = 1.0 / std::sqrt(static_cast<double>(K));
  auto [out, final_state] = fused_recurrent_gated_delta_rule_packed_decode(
      mixed_qkv,
      a,
      b,
      a_log,
      dt_bias,
      scale,
      ssm_cache_run1,
      ssm_state_indices,
      /*use_qk_l2norm_in_kernel=*/true);
  Sync();

  EXPECT_EQ(out.sizes(), torch::IntArrayRef({batch_size, 1, HV, V}));
  EXPECT_EQ(out.scalar_type(), mixed_qkv.scalar_type());
  auto out_cpu = out.flatten().to(torch::kFloat32).cpu();
  EXPECT_TRUE(torch::isfinite(out_cpu).all().item<bool>())
      << "recurrent decode output must be finite";
  EXPECT_TRUE(torch::isfinite(final_state.flatten().to(torch::kFloat32).cpu())
                  .all()
                  .item<bool>())
      << "recurrent decode final state must be finite";

  // Determinism: rerun with a fresh clone of the pristine ssm_cache.
  auto ssm_cache_run2 = ssm_cache_orig.clone();
  auto [out2, _] =
      fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv,
                                                     a,
                                                     b,
                                                     a_log,
                                                     dt_bias,
                                                     scale,
                                                     ssm_cache_run2,
                                                     ssm_state_indices,
                                                     true);
  Sync();
  EXPECT_TRUE(torch::allclose(out, out2, /*rtol=*/1e-3, /*atol=*/1e-4))
      << "recurrent decode should be deterministic for fixed inputs";
}

// ===========================================================================
// ChunkGatedDeltaRule: prefill chunked kernel.
// num_k_heads in {1,2,4,8,16,32}, num_v_heads in {1,2,4,6,8,12,16,24,32,48,64}
// and num_v_heads % num_k_heads == 0.
// ===========================================================================
TEST_F(Qwen3_5GatedDeltaNetOpsTest, ChunkGatedDeltaRuleForward) {
  const int64_t num_k_heads = 4;  // Hg
  const int64_t num_v_heads = 8;  // H
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 128;
  const int64_t seq_len = 128;  // multiple of chunk_size (64)
  const int64_t batch_size = 1;
  auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  // q, k: [1, T, Hg, K]; v: [1, T, H, V]
  auto q = MakeNoise(
      "chunk.q", {batch_size, seq_len, num_k_heads, head_k_dim}, 0.01f);
  auto k = MakeNoise(
      "chunk.k", {batch_size, seq_len, num_k_heads, head_k_dim}, 0.01f);
  auto v = MakeNoise(
      "chunk.v", {batch_size, seq_len, num_v_heads, head_v_dim}, 0.01f);
  // g: [1, T, H] fp32; beta: [1, T, H] bf16
  auto g = MakeNoise(
      "chunk.g", {batch_size, seq_len, num_v_heads}, 0.002f, torch::kFloat32);
  auto beta =
      MakeNoise("chunk.beta", {batch_size, seq_len, num_v_heads}, 0.02f);
  // initial_state: mirror the layer path -- ssm_cache [B, H, K, V] transposed
  // to [B, H, V, K], fp32.
  auto initial_state =
      MakeNoise("chunk.initial_state",
                {batch_size, num_v_heads, head_v_dim, head_k_dim},
                0.01f,
                torch::kFloat32);
  auto cu_seqlens =
      torch::arange(0, (batch_size + 1) * seq_len, seq_len, opts_int);
  constexpr int64_t chunk_size = 64;
  auto chunk_indices = MakeChunkIndices(cu_seqlens, chunk_size);

  auto chunk_gdr = ChunkGatedDeltaRule(num_k_heads, num_v_heads);
  chunk_gdr->to(device_);

  auto [o, final_state] = chunk_gdr->forward(q,
                                             k,
                                             v,
                                             g,
                                             beta,
                                             initial_state,
                                             cu_seqlens,
                                             chunk_indices,
                                             /*output_final_state=*/true,
                                             /*use_qk_l2norm_in_kernel=*/true);
  Sync();

  EXPECT_EQ(o.sizes(),
            torch::IntArrayRef({batch_size, seq_len, num_v_heads, head_v_dim}));
  EXPECT_EQ(o.scalar_type(), opts.dtype());
  auto o_cpu = o.flatten().to(torch::kFloat32).cpu();
  EXPECT_TRUE(torch::isfinite(o_cpu).all().item<bool>())
      << "chunk GDN output must be finite";
  EXPECT_TRUE(final_state.defined());
  EXPECT_TRUE(torch::isfinite(final_state.flatten().to(torch::kFloat32).cpu())
                  .all()
                  .item<bool>())
      << "chunk GDN final state must be finite";

  // Determinism: rebuild inputs with the same seeds for a second run.
  auto q2 = MakeNoise(
      "chunk.q", {batch_size, seq_len, num_k_heads, head_k_dim}, 0.01f);
  auto k2 = MakeNoise(
      "chunk.k", {batch_size, seq_len, num_k_heads, head_k_dim}, 0.01f);
  auto v2 = MakeNoise(
      "chunk.v", {batch_size, seq_len, num_v_heads, head_v_dim}, 0.01f);
  auto g2 = MakeNoise(
      "chunk.g", {batch_size, seq_len, num_v_heads}, 0.002f, torch::kFloat32);
  auto beta2 =
      MakeNoise("chunk.beta", {batch_size, seq_len, num_v_heads}, 0.02f);
  auto init2 = MakeNoise("chunk.initial_state",
                         {batch_size, num_v_heads, head_v_dim, head_k_dim},
                         0.01f,
                         torch::kFloat32);
  auto [o2, _] = chunk_gdr->forward(
      q2, k2, v2, g2, beta2, init2, cu_seqlens, chunk_indices, true, true);
  Sync();
  EXPECT_TRUE(torch::allclose(o, o2, /*rtol=*/1e-3, /*atol=*/1e-4))
      << "chunk GDN should be deterministic for fixed inputs";
}

}  // namespace
}  // namespace layer
}  // namespace xllm
