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

#include <framework/core/device.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <optional>
#include <vector>

#include "kernels/mlu/chunk_gated_delta_rule.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"
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
using xllm::kernel::mlu::fused_post_conv_prep;
using xllm::kernel::mlu::fused_recurrent_gated_delta_rule;
using xllm::kernel::mlu::fused_recurrent_gated_delta_rule_packed_decode;
using xllm::kernel::mlu::fused_sigmoid_gating_delta_rule_update;

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

class Qwen3_5GatedDeltaNetSpecVerifyOpsTest
    : public Qwen3_5GatedDeltaNetOpsTest,
      public ::testing::WithParamInterface<int64_t> {};

std::string spec_verify_width_name(
    const ::testing::TestParamInfo<int64_t>& info) {
  return "Width" + std::to_string(info.param);
}

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
  constexpr int32_t block_size = 64;
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

torch::Tensor L2NormalizeLastDim(const torch::Tensor& x) {
  torch::Tensor x_float = x.to(torch::kFloat32);
  torch::Tensor square_sum =
      (x_float * x_float).sum(/*dim=*/-1, /*keepdim=*/true);
  torch::Tensor inv_norm = torch::rsqrt(square_sum + 1e-6);
  return (x_float * inv_norm).to(x.scalar_type());
}

torch::Tensor CausalConv1dUpdateRef(const torch::Tensor& x,
                                    torch::Tensor& conv_state,
                                    const torch::Tensor& weight,
                                    const std::optional<torch::Tensor>& bias,
                                    bool silu_activation) {
  torch::Tensor x_work = x;
  bool unsqueeze = x_work.dim() == 2;
  if (unsqueeze) {
    x_work = x_work.unsqueeze(/*dim=*/-1);
  }

  const int64_t seqlen = x_work.size(2);
  const int64_t width = weight.size(1);
  const int64_t state_len = conv_state.size(2);
  torch::Tensor x_new =
      torch::cat({conv_state, x_work}, /*dim=*/-1).to(weight.scalar_type());
  conv_state.copy_(x_new.slice(/*dim=*/2, -state_len));

  std::vector<torch::Tensor> outputs;
  outputs.reserve(seqlen);
  for (int64_t t = 0; t < seqlen; ++t) {
    torch::Tensor window = x_new.slice(/*dim=*/2, t, t + width);
    torch::Tensor out_t = (window * weight.unsqueeze(/*dim=*/0)).sum(/*dim=*/2);
    if (bias.has_value()) {
      out_t = out_t + bias.value().unsqueeze(/*dim=*/0);
    }
    outputs.emplace_back(out_t);
  }
  torch::Tensor out = torch::stack(outputs, /*dim=*/2);
  if (silu_activation) {
    out = torch::silu(out);
  }
  out = out.to(x.scalar_type());
  if (unsqueeze) {
    out = out.squeeze(/*dim=*/-1);
  }
  return out;
}

torch::Tensor causal_conv1d_update_spec_verify_ref(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& conv_state_indices,
    const torch::Tensor& num_accepted_tokens) {
  const int64_t batch_size = conv_state_indices.size(0);
  const int64_t width = weight.size(1);
  const int64_t state_len = conv_state.size(2);
  torch::Tensor out = torch::empty_like(x);

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t query_start = query_start_loc.index({i}).item<int32_t>();
    const int64_t query_end = query_start_loc.index({i + 1}).item<int32_t>();
    const int64_t state_index = conv_state_indices.index({i}).item<int32_t>();
    const int64_t accepted_offset =
        num_accepted_tokens.index({i}).item<int32_t>() - 1;
    torch::Tensor old_state =
        conv_state.select(/*dim=*/0, /*index=*/state_index)
            .slice(/*dim=*/1,
                   /*start=*/accepted_offset,
                   /*end=*/accepted_offset + width - 1)
            .clone();
    torch::Tensor x_seq =
        x.select(/*dim=*/0, /*index=*/0)
            .slice(/*dim=*/1, /*start=*/query_start, /*end=*/query_end);
    torch::Tensor combined =
        torch::cat({old_state, x_seq}, /*dim=*/1).to(weight.scalar_type());

    for (int64_t t = 0; t < query_end - query_start; ++t) {
      torch::Tensor window =
          combined.slice(/*dim=*/1, /*start=*/t, /*end=*/t + width);
      torch::Tensor out_token =
          torch::silu((window * weight).sum(/*dim=*/1)).to(x.scalar_type());
      out.select(/*dim=*/0, /*index=*/0)
          .slice(/*dim=*/1,
                 /*start=*/query_start + t,
                 /*end=*/query_start + t + 1)
          .copy_(out_token.unsqueeze(/*dim=*/1));
    }

    torch::Tensor new_state =
        combined.slice(/*dim=*/1, /*start=*/1, /*end=*/1 + state_len);
    conv_state.select(/*dim=*/0, /*index=*/state_index)
        .copy_(new_state.to(conv_state.scalar_type()));
  }
  return out;
}

std::pair<torch::Tensor, torch::Tensor> FusedRecurrentPackedDecodeRef(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    double scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& ssm_state_indices,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_k_dim,
    int64_t head_v_dim) {
  torch::Tensor q = mixed_qkv.slice(/*dim=*/1, 0, num_k_heads * head_k_dim)
                        .view({mixed_qkv.size(0), num_k_heads, head_k_dim});
  torch::Tensor k =
      mixed_qkv
          .slice(
              /*dim=*/1, num_k_heads * head_k_dim, 2 * num_k_heads * head_k_dim)
          .view({mixed_qkv.size(0), num_k_heads, head_k_dim});
  torch::Tensor v = mixed_qkv.slice(/*dim=*/1, 2 * num_k_heads * head_k_dim)
                        .view({mixed_qkv.size(0), num_v_heads, head_v_dim});

  q = L2NormalizeLastDim(q).to(torch::kFloat32) * scale;
  k = L2NormalizeLastDim(k).to(torch::kFloat32);
  v = v.to(torch::kFloat32);
  torch::Tensor g =
      (-torch::exp(a_log.to(torch::kFloat32)) *
       torch::nn::functional::softplus(
           a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32),
           torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(
               20.0)));
  torch::Tensor beta = torch::sigmoid(b.to(torch::kFloat32));

  torch::Tensor final_state = initial_state.clone();
  torch::Tensor out = torch::zeros(
      {mixed_qkv.size(0), 1, num_v_heads, head_v_dim}, mixed_qkv.options());
  const int64_t kv_group = num_v_heads / num_k_heads;
  for (int64_t n = 0; n < mixed_qkv.size(0); ++n) {
    const int64_t state_idx = ssm_state_indices[n].item<int32_t>();
    if (state_idx <= 0) {
      continue;
    }
    for (int64_t hv = 0; hv < num_v_heads; ++hv) {
      const int64_t h = hv / kv_group;
      torch::Tensor state_h =
          final_state.select(/*dim=*/0, state_idx).select(/*dim=*/0, hv);
      torch::Tensor k_t = k.select(/*dim=*/0, n).select(/*dim=*/0, h);
      torch::Tensor q_t = q.select(/*dim=*/0, n).select(/*dim=*/0, h);
      torch::Tensor v_t = v.select(/*dim=*/0, n).select(/*dim=*/0, hv);
      state_h.mul_(torch::exp(g.select(/*dim=*/0, n).select(/*dim=*/0, hv)));
      torch::Tensor v_new =
          (v_t - (state_h * k_t.unsqueeze(/*dim=*/0)).sum(/*dim=*/-1)) *
          beta.select(/*dim=*/0, n).select(/*dim=*/0, hv);
      state_h.add_(v_new.unsqueeze(/*dim=*/1) * k_t.unsqueeze(/*dim=*/0));
      out.select(/*dim=*/0, n)
          .select(/*dim=*/0, 0)
          .select(/*dim=*/0, hv)
          .copy_((state_h * q_t.unsqueeze(/*dim=*/0))
                     .sum(/*dim=*/-1)
                     .to(out.scalar_type()));
    }
  }
  return {out, final_state};
}

std::pair<torch::Tensor, torch::Tensor> FusedSigmoidSpecVerifyRef(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    double scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& ssm_state_indices,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& num_accepted_tokens) {
  const int64_t num_k_heads = q.size(2);
  const int64_t num_v_heads = v.size(2);
  const int64_t kv_group = num_v_heads / num_k_heads;

  torch::Tensor q_ref =
      L2NormalizeLastDim(q.squeeze(/*dim=*/0)).to(torch::kFloat32) * scale;
  torch::Tensor k_ref =
      L2NormalizeLastDim(k.squeeze(/*dim=*/0)).to(torch::kFloat32);
  torch::Tensor v_ref = v.squeeze(/*dim=*/0).to(torch::kFloat32);
  torch::Tensor g =
      (-torch::exp(a_log.to(torch::kFloat32)) *
       torch::nn::functional::softplus(
           a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32),
           torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(
               20.0)));
  torch::Tensor beta = torch::sigmoid(b.to(torch::kFloat32));

  torch::Tensor out_ref = torch::empty_like(v);
  torch::Tensor final_state_ref = initial_state.clone();
  const int64_t num_sequences = cu_seqlens.size(0) - 1;
  for (int64_t i = 0; i < num_sequences; ++i) {
    const int64_t bos = cu_seqlens.index({i}).item<int32_t>();
    const int64_t eos = cu_seqlens.index({i + 1}).item<int32_t>();
    const int64_t accepted_offset =
        num_accepted_tokens.index({i}).item<int32_t>() - 1;
    const int64_t initial_state_idx =
        ssm_state_indices.index({i, accepted_offset}).item<int32_t>();
    torch::Tensor running_state =
        final_state_ref.select(/*dim=*/0, initial_state_idx).clone();
    for (int64_t n = bos; n < eos; ++n) {
      torch::Tensor out_token = torch::empty(
          {num_v_heads, v.size(3)}, v.options().dtype(torch::kFloat32));
      for (int64_t hv = 0; hv < num_v_heads; ++hv) {
        const int64_t h = hv / kv_group;
        torch::Tensor state_h = running_state.select(/*dim=*/0, hv);
        torch::Tensor k_t = k_ref.select(/*dim=*/0, n).select(/*dim=*/0, h);
        torch::Tensor q_t = q_ref.select(/*dim=*/0, n).select(/*dim=*/0, h);
        torch::Tensor v_t = v_ref.select(/*dim=*/0, n).select(/*dim=*/0, hv);
        state_h.mul_(torch::exp(g.select(/*dim=*/0, n).select(/*dim=*/0, hv)));
        torch::Tensor v_new =
            (v_t - (state_h * k_t.unsqueeze(/*dim=*/0)).sum(/*dim=*/-1)) *
            beta.select(/*dim=*/0, n).select(/*dim=*/0, hv);
        state_h.add_(v_new.unsqueeze(/*dim=*/1) * k_t.unsqueeze(/*dim=*/0));
        out_token.select(/*dim=*/0, hv)
            .copy_((state_h * q_t.unsqueeze(/*dim=*/0)).sum(/*dim=*/-1));
      }
      const int64_t state_offset = n - bos;
      const int64_t final_state_idx =
          ssm_state_indices.index({i, state_offset}).item<int32_t>();
      final_state_ref.select(/*dim=*/0, final_state_idx).copy_(running_state);
      out_ref.select(/*dim=*/0, /*index=*/0)
          .select(/*dim=*/0, n)
          .copy_(out_token.to(out_ref.scalar_type()));
    }
  }
  return {out_ref, final_state_ref};
}

torch::Tensor select_rebased_checkpoint_rows(
    const torch::Tensor& state,
    const torch::Tensor& ssm_state_indices) {
  return state.index({ssm_state_indices.reshape({-1})});
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
// fused_post_conv_prep: split + q/k l2norm + gating after causal conv.
// ===========================================================================
TEST_F(Qwen3_5GatedDeltaNetOpsTest, FusedPostConvPrepMatchesReference) {
  const int64_t num_k_heads = 16;
  const int64_t num_v_heads = 32;
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 128;
  const int64_t num_tokens = 32;
  const int64_t qk_dim = num_k_heads * head_k_dim;
  const int64_t qkv_dim = 2 * qk_dim + num_v_heads * head_v_dim;

  auto conv_output =
      MakeNoise("post_conv.conv_output", {num_tokens, qkv_dim}, 0.02f);
  auto a = MakeNoise("post_conv.a", {num_tokens, num_v_heads}, 0.02f);
  auto b = MakeNoise("post_conv.b", {num_tokens, num_v_heads}, 0.02f);
  auto a_log = MakeNoise("post_conv.A_log", {num_v_heads}, 0.02f);
  auto dt_bias = MakeNoise("post_conv.dt_bias", {num_v_heads}, 0.02f);

  auto q_ref = conv_output.slice(/*dim=*/-1, /*start=*/0, qk_dim)
                   .view({num_tokens, num_k_heads, head_k_dim})
                   .contiguous();
  auto k_ref = conv_output.slice(/*dim=*/-1, qk_dim, 2 * qk_dim)
                   .view({num_tokens, num_k_heads, head_k_dim})
                   .contiguous();
  auto v_ref = conv_output.slice(/*dim=*/-1, 2 * qk_dim)
                   .view({num_tokens, num_v_heads, head_v_dim})
                   .contiguous();
  q_ref = L2NormalizeLastDim(q_ref);
  k_ref = L2NormalizeLastDim(k_ref);
  torch::Tensor softplus = torch::nn::functional::softplus(
      a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32),
      torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
  torch::Tensor g_ref = -torch::exp(a_log.to(torch::kFloat32)) * softplus;
  torch::Tensor beta_ref = torch::sigmoid(b.to(torch::kFloat32));

  auto [q, k, v, g, beta] = fused_post_conv_prep(conv_output,
                                                 a,
                                                 b,
                                                 a_log,
                                                 dt_bias,
                                                 num_k_heads,
                                                 head_k_dim,
                                                 head_v_dim,
                                                 /*apply_l2norm=*/true,
                                                 /*output_g_exp=*/false);
  Sync();

  EXPECT_EQ(q.sizes(),
            torch::IntArrayRef({num_tokens, num_k_heads, head_k_dim}));
  EXPECT_EQ(k.sizes(),
            torch::IntArrayRef({num_tokens, num_k_heads, head_k_dim}));
  EXPECT_EQ(v.sizes(),
            torch::IntArrayRef({num_tokens, num_v_heads, head_v_dim}));
  EXPECT_EQ(g.sizes(), torch::IntArrayRef({num_tokens, num_v_heads}));
  EXPECT_EQ(beta.sizes(), torch::IntArrayRef({num_tokens, num_v_heads}));
  EXPECT_EQ(q.scalar_type(), conv_output.scalar_type());
  EXPECT_EQ(k.scalar_type(), conv_output.scalar_type());
  EXPECT_EQ(v.scalar_type(), conv_output.scalar_type());
  EXPECT_EQ(g.scalar_type(), torch::kFloat32);
  EXPECT_EQ(beta.scalar_type(), torch::kFloat32);

  EXPECT_TRUE(torch::allclose(q, q_ref, /*rtol=*/1e-2, /*atol=*/1e-2));
  EXPECT_TRUE(torch::allclose(k, k_ref, /*rtol=*/1e-2, /*atol=*/1e-2));
  EXPECT_TRUE(torch::allclose(v, v_ref, /*rtol=*/1e-3, /*atol=*/1e-3));
  EXPECT_TRUE(torch::allclose(g, g_ref, /*rtol=*/1e-4, /*atol=*/1e-4));
  EXPECT_TRUE(torch::allclose(beta, beta_ref, /*rtol=*/1e-4, /*atol=*/1e-4));
}

TEST_F(Qwen3_5GatedDeltaNetOpsTest, FusedPostConvPrepSupportsNoL2Norm) {
  const int64_t num_k_heads = 2;
  const int64_t num_v_heads = 4;
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 128;
  const int64_t num_tokens = 8;
  const int64_t qk_dim = num_k_heads * head_k_dim;
  const int64_t qkv_dim = 2 * qk_dim + num_v_heads * head_v_dim;

  auto conv_output =
      MakeNoise("post_conv.no_l2.conv_output", {num_tokens, qkv_dim}, 0.02f);
  auto a = MakeNoise("post_conv.no_l2.a", {num_tokens, num_v_heads}, 0.02f);
  auto b = MakeNoise("post_conv.no_l2.b", {num_tokens, num_v_heads}, 0.02f);
  auto a_log = MakeNoise("post_conv.no_l2.A_log", {num_v_heads}, 0.02f);
  auto dt_bias = MakeNoise("post_conv.no_l2.dt_bias", {num_v_heads}, 0.02f);

  auto q_ref = conv_output.slice(/*dim=*/-1, /*start=*/0, qk_dim)
                   .view({num_tokens, num_k_heads, head_k_dim})
                   .contiguous();
  auto k_ref = conv_output.slice(/*dim=*/-1, qk_dim, 2 * qk_dim)
                   .view({num_tokens, num_k_heads, head_k_dim})
                   .contiguous();

  auto [q, k, v, g, beta] = fused_post_conv_prep(conv_output,
                                                 a,
                                                 b,
                                                 a_log,
                                                 dt_bias,
                                                 num_k_heads,
                                                 head_k_dim,
                                                 head_v_dim,
                                                 /*apply_l2norm=*/false,
                                                 /*output_g_exp=*/false);
  Sync();

  EXPECT_TRUE(torch::allclose(q, q_ref, /*rtol=*/1e-3, /*atol=*/1e-3));
  EXPECT_TRUE(torch::allclose(k, k_ref, /*rtol=*/1e-3, /*atol=*/1e-3));
  EXPECT_TRUE(torch::isfinite(v.to(torch::kFloat32)).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(g).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(beta).all().item<bool>());
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
      MakeNoise("conv1d_update.state", {batch_size + 1, dim, state_len}, 0.02f);
  auto weight = MakeNoise("conv1d_update.weight", {dim, width}, 0.02f);
  auto conv_state_indices = torch::arange(1, batch_size + 1, opts_int);

  // The kernel shifts new tokens into conv_state in place, so capture the
  // pristine state before any run and feed a fresh clone to each run.
  auto conv_state_orig = conv_state.clone();
  auto conv_state_ref = conv_state_orig.index({conv_state_indices}).clone();
  auto out_ref = CausalConv1dUpdateRef(x,
                                       conv_state_ref,
                                       weight,
                                       /*bias=*/std::nullopt,
                                       /*silu_activation=*/true);
  auto conv_state_run1 = conv_state_orig.clone();
  auto out = causal_conv1d_update_decode(x,
                                         conv_state_run1,
                                         weight,
                                         /*bias=*/std::nullopt,
                                         conv_state_indices,
                                         /*activation=*/true,
                                         /*pad_slot_id=*/-1);
  Sync();

  EXPECT_EQ(out.sizes(), x.sizes());
  EXPECT_EQ(out.scalar_type(), x.scalar_type());
  auto out_cpu = out.flatten().to(torch::kFloat32).cpu();
  EXPECT_TRUE(torch::isfinite(out_cpu).all().item<bool>())
      << "conv1d_update_decode output must be finite";
  auto out_abs_diff =
      (out.to(torch::kFloat32) - out_ref.to(torch::kFloat32)).abs();
  auto state_abs_diff =
      (conv_state_run1.index({conv_state_indices}).to(torch::kFloat32) -
       conv_state_ref.to(torch::kFloat32))
          .abs();
  EXPECT_TRUE(torch::allclose(out, out_ref, /*rtol=*/1e-2, /*atol=*/5e-2))
      << "max out diff: " << out_abs_diff.max().item<float>()
      << ", mean out diff: " << out_abs_diff.mean().item<float>();
  EXPECT_TRUE(torch::allclose(conv_state_run1.index({conv_state_indices}),
                              conv_state_ref,
                              /*rtol=*/1e-2,
                              /*atol=*/5e-2))
      << "max state diff: " << state_abs_diff.max().item<float>()
      << ", mean state diff: " << state_abs_diff.mean().item<float>();

  // Determinism: rerun with a fresh clone of the pristine state.
  auto conv_state_run2 = conv_state_orig.clone();
  auto out2 = causal_conv1d_update_decode(x,
                                          conv_state_run2,
                                          weight,
                                          std::nullopt,
                                          conv_state_indices,
                                          /*activation=*/true,
                                          /*pad_slot_id=*/-1);
  Sync();
  EXPECT_TRUE(torch::allclose(out, out2, /*rtol=*/1e-3, /*atol=*/1e-4))
      << "conv1d_update_decode should be deterministic for fixed inputs";
}

TEST_F(Qwen3_5GatedDeltaNetOpsTest,
       CausalConv1dUpdateDecodeSupportsDefaultsAndNoActivation) {
  const int64_t dim = 1024;
  const int64_t width = 4;
  const int64_t batch_size = 2;

  auto x = MakeNoise("conv1d_update.defaults.x", {batch_size, dim}, 0.02f);
  auto conv_state = MakeNoise(
      "conv1d_update.defaults.state", {batch_size, dim, width - 1}, 0.02f);
  auto weight = MakeNoise("conv1d_update.defaults.weight", {dim, width}, 0.02f);

  auto conv_state_ref = conv_state.clone();
  auto out_ref = CausalConv1dUpdateRef(x,
                                       conv_state_ref,
                                       weight,
                                       /*bias=*/std::nullopt,
                                       /*silu_activation=*/false);
  auto out =
      causal_conv1d_update_decode(x,
                                  conv_state,
                                  weight,
                                  /*bias=*/std::nullopt,
                                  /*conv_state_indices_opt=*/std::nullopt,
                                  /*activation=*/false,
                                  /*pad_slot_id=*/-1);
  Sync();

  EXPECT_EQ(out.sizes(), x.sizes());
  EXPECT_TRUE(torch::allclose(out, out_ref, /*rtol=*/1e-2, /*atol=*/5e-2));
  EXPECT_TRUE(torch::allclose(conv_state,
                              conv_state_ref,
                              /*rtol=*/1e-2,
                              /*atol=*/5e-2));
}

TEST_F(Qwen3_5GatedDeltaNetOpsTest, CausalConv1dUpdateDecodeSupportsVarlen) {
  const int64_t dim = 1024;
  const int64_t width = 4;
  const int64_t batch_size = 2;
  const int64_t num_tokens = 2;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  auto x = MakeNoise("conv1d_update.varlen.x", {num_tokens, dim}, 0.02f);
  auto conv_state = MakeNoise(
      "conv1d_update.varlen.state", {batch_size, dim, width - 1}, 0.02f);
  auto weight = MakeNoise("conv1d_update.varlen.weight", {dim, width}, 0.02f);
  auto conv_state_indices = torch::arange(0, batch_size, opts_int);
  auto query_start_loc = torch::tensor(std::vector<int32_t>{0, 1, 2}, opts_int);

  auto x_ref = x.view({batch_size, 1, dim})
                   .transpose(/*dim0=*/1, /*dim1=*/2)
                   .contiguous();
  auto conv_state_ref = conv_state.clone();
  auto out_ref = CausalConv1dUpdateRef(x_ref,
                                       conv_state_ref,
                                       weight,
                                       /*bias=*/std::nullopt,
                                       /*silu_activation=*/true)
                     .squeeze(/*dim=*/-1);
  auto out = causal_conv1d_update_decode(x,
                                         conv_state,
                                         weight,
                                         /*bias=*/std::nullopt,
                                         conv_state_indices,
                                         /*activation=*/true,
                                         /*pad_slot_id=*/-1,
                                         query_start_loc,
                                         /*max_query_len=*/1);
  Sync();

  EXPECT_EQ(out.sizes(), x.sizes());
  EXPECT_TRUE(torch::allclose(out, out_ref, /*rtol=*/1e-2, /*atol=*/5e-2));
  EXPECT_TRUE(torch::allclose(conv_state,
                              conv_state_ref,
                              /*rtol=*/1e-2,
                              /*atol=*/5e-2));
}

void run_causal_conv1d_update_spec_verify_test(
    const torch::Device& device,
    const std::string& key_prefix,
    int32_t q_max_seq_len,
    const std::vector<int32_t>& accepted_prefixes) {
  const int64_t dim = 384;
  const int64_t width = 4;
  const int64_t batch_size = static_cast<int64_t>(accepted_prefixes.size());
  const int64_t state_len = q_max_seq_len + 2;
  const int64_t total_tokens = batch_size * q_max_seq_len;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device);

  auto x =
      (test::seeded_tensor(
           key_prefix + ".x", {total_tokens, dim}, torch::kBFloat16, device) -
       0.5f) *
      (std::sqrt(12.0f) * 0.02f);
  auto conv_state = (test::seeded_tensor(key_prefix + ".state",
                                         {batch_size + 1, dim, state_len},
                                         torch::kBFloat16,
                                         device) -
                     0.5f) *
                    (std::sqrt(12.0f) * 0.02f);
  auto weight =
      (test::seeded_tensor(
           key_prefix + ".weight", {dim, width}, torch::kBFloat16, device) -
       0.5f) *
      (std::sqrt(12.0f) * 0.02f);
  auto query_start_loc =
      torch::arange(0, total_tokens + 1, q_max_seq_len, opts_int);
  auto conv_state_indices = torch::arange(1, batch_size + 1, opts_int);
  auto num_accepted_tokens = torch::tensor(accepted_prefixes, opts_int);

  auto conv_state_orig = conv_state.clone();
  auto conv_state_ref = conv_state_orig.clone();
  torch::Tensor x_ref =
      x.transpose(/*dim0=*/0, /*dim1=*/1).unsqueeze(/*dim=*/0).contiguous();
  auto out_ref = causal_conv1d_update_spec_verify_ref(x_ref,
                                                      conv_state_ref,
                                                      weight,
                                                      query_start_loc,
                                                      conv_state_indices,
                                                      num_accepted_tokens)
                     .squeeze(/*dim=*/0)
                     .transpose(/*dim0=*/0, /*dim1=*/1)
                     .contiguous();

  auto conv_state_run = conv_state_orig.clone();
  auto out = causal_conv1d_update_decode(x,
                                         conv_state_run,
                                         weight,
                                         /*bias_opt=*/std::nullopt,
                                         conv_state_indices,
                                         /*activation=*/true,
                                         /*pad_slot_id=*/-1,
                                         query_start_loc,
                                         q_max_seq_len,
                                         num_accepted_tokens);
  Device xllm_device(device);
  xllm_device.synchronize_default_stream();

  EXPECT_EQ(out.sizes(), x.sizes());
  EXPECT_EQ(out.scalar_type(), x.scalar_type());
  auto out_abs_diff =
      (out.to(torch::kFloat32) - out_ref.to(torch::kFloat32)).abs();
  auto state_abs_diff =
      (conv_state_run.index({conv_state_indices}).to(torch::kFloat32) -
       conv_state_ref.index({conv_state_indices}).to(torch::kFloat32))
          .abs();
  EXPECT_TRUE(torch::allclose(out, out_ref, /*rtol=*/1e-2, /*atol=*/5e-2))
      << "max out diff: " << out_abs_diff.max().item<float>()
      << ", mean out diff: " << out_abs_diff.mean().item<float>();
  EXPECT_TRUE(torch::allclose(conv_state_run.index({conv_state_indices}),
                              conv_state_ref.index({conv_state_indices}),
                              /*rtol=*/1e-2,
                              /*atol=*/5e-2))
      << "max state diff: " << state_abs_diff.max().item<float>()
      << ", mean state diff: " << state_abs_diff.mean().item<float>();
}

TEST_P(Qwen3_5GatedDeltaNetSpecVerifyOpsTest,
       CausalConv1dUpdateUsesAcceptedPrefix) {
  const int32_t q_max_seq_len = static_cast<int32_t>(GetParam());
  const std::string key_prefix =
      "conv1d_spec_width" + std::to_string(q_max_seq_len);
  run_causal_conv1d_update_spec_verify_test(
      device_, key_prefix, q_max_seq_len, {1, q_max_seq_len});
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
                             {batch_size + 1, HV, V, K},
                             0.01f,
                             torch::kFloat32);
  auto ssm_state_indices = torch::arange(1, batch_size + 1, opts_int);

  // The kernel updates ssm_cache in place; capture the pristine state and feed
  // a fresh clone to each run.
  auto ssm_cache_orig = ssm_cache.clone();
  double scale = 1.0 / std::sqrt(static_cast<double>(K));
  auto [out_ref, final_state_ref] =
      FusedRecurrentPackedDecodeRef(mixed_qkv,
                                    a,
                                    b,
                                    a_log,
                                    dt_bias,
                                    scale,
                                    ssm_cache_orig,
                                    ssm_state_indices,
                                    H,
                                    HV,
                                    K,
                                    V);
  auto ssm_cache_run1 = ssm_cache_orig.clone();
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
  auto out_abs_diff =
      (out.to(torch::kFloat32) - out_ref.to(torch::kFloat32)).abs();
  auto state_abs_diff =
      (final_state.to(torch::kFloat32) - final_state_ref.to(torch::kFloat32))
          .abs();
  EXPECT_TRUE(torch::allclose(out, out_ref, /*rtol=*/1e-2, /*atol=*/2e-2))
      << "max out diff: " << out_abs_diff.max().item<float>()
      << ", mean out diff: " << out_abs_diff.mean().item<float>();
  EXPECT_TRUE(torch::allclose(final_state,
                              final_state_ref,
                              /*rtol=*/1e-2,
                              /*atol=*/2e-2))
      << "max state diff: " << state_abs_diff.max().item<float>()
      << ", mean state diff: " << state_abs_diff.mean().item<float>();

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
// fused_sigmoid_gating_delta_rule_update: fused gating + recurrent update.
// ===========================================================================
TEST_F(Qwen3_5GatedDeltaNetOpsTest,
       FusedSigmoidGatingDeltaRuleUpdateMatchesReference) {
  const int64_t num_k_heads = 16;
  const int64_t num_v_heads = 32;
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 128;
  const int64_t batch_size = 1;
  const int64_t num_tokens = 4;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  auto q = MakeNoise("sigmoid_update.q",
                     {batch_size, num_tokens, num_k_heads, head_k_dim},
                     0.02f);
  auto k = MakeNoise("sigmoid_update.k",
                     {batch_size, num_tokens, num_k_heads, head_k_dim},
                     0.02f);
  auto v = MakeNoise("sigmoid_update.v",
                     {batch_size, num_tokens, num_v_heads, head_v_dim},
                     0.02f);
  auto a = MakeNoise("sigmoid_update.a", {num_tokens, num_v_heads}, 0.02f);
  auto b = MakeNoise("sigmoid_update.b", {num_tokens, num_v_heads}, 0.02f);
  auto a_log = MakeNoise("sigmoid_update.A_log", {num_v_heads}, 0.02f);
  auto dt_bias = MakeNoise("sigmoid_update.dt_bias", {num_v_heads}, 0.02f);
  auto initial_state =
      MakeNoise("sigmoid_update.initial_state",
                {num_tokens + 1, num_v_heads, head_v_dim, head_k_dim},
                0.01f,
                torch::kFloat32);
  auto ssm_state_indices =
      torch::arange(1, num_tokens + 1, opts_int).unsqueeze(0);
  auto cu_seqlens = torch::tensor(
      std::vector<int32_t>{0, static_cast<int32_t>(num_tokens)}, opts_int);
  auto num_accepted_tokens = torch::ones({1}, opts_int);

  torch::Tensor g =
      (-torch::exp(a_log.to(torch::kFloat32)) *
       torch::nn::functional::softplus(
           a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32),
           torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(
               20.0)));
  torch::Tensor beta = torch::sigmoid(b.to(torch::kFloat32));
  double scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));
  torch::Tensor q_ref =
      L2NormalizeLastDim(q.squeeze(/*dim=*/0)).to(torch::kFloat32) * scale;
  torch::Tensor k_ref =
      L2NormalizeLastDim(k.squeeze(/*dim=*/0)).to(torch::kFloat32);
  torch::Tensor v_ref = v.squeeze(/*dim=*/0).to(torch::kFloat32);
  torch::Tensor out_ref = torch::empty_like(v);
  torch::Tensor final_state_ref = initial_state.clone();
  torch::Tensor running_state =
      final_state_ref.select(/*dim=*/0, /*index=*/1).clone();
  const int64_t kv_group = num_v_heads / num_k_heads;
  for (int64_t t = 0; t < num_tokens; ++t) {
    torch::Tensor out_token = torch::empty({num_v_heads, head_v_dim},
                                           v.options().dtype(torch::kFloat32));
    for (int64_t hv = 0; hv < num_v_heads; ++hv) {
      const int64_t h = hv / kv_group;
      torch::Tensor state_h = running_state.select(/*dim=*/0, hv);
      torch::Tensor k_t = k_ref.select(/*dim=*/0, t).select(/*dim=*/0, h);
      torch::Tensor q_t = q_ref.select(/*dim=*/0, t).select(/*dim=*/0, h);
      torch::Tensor v_t = v_ref.select(/*dim=*/0, t).select(/*dim=*/0, hv);
      state_h.mul_(torch::exp(g.select(/*dim=*/0, t).select(/*dim=*/0, hv)));
      torch::Tensor v_new =
          (v_t - (state_h * k_t.unsqueeze(/*dim=*/0)).sum(/*dim=*/-1)) *
          beta.select(/*dim=*/0, t).select(/*dim=*/0, hv);
      state_h.add_(v_new.unsqueeze(/*dim=*/1) * k_t.unsqueeze(/*dim=*/0));
      out_token.select(/*dim=*/0, hv)
          .copy_((state_h * q_t.unsqueeze(/*dim=*/0)).sum(/*dim=*/-1));
    }
    final_state_ref.select(/*dim=*/0, /*index=*/t + 1).copy_(running_state);
    out_ref.select(/*dim=*/0, /*index=*/0)
        .select(/*dim=*/0, t)
        .copy_(out_token.to(out_ref.scalar_type()));
  }

  auto update_state = initial_state.clone();
  auto [out, final_state] =
      fused_sigmoid_gating_delta_rule_update(a_log,
                                             a,
                                             b,
                                             dt_bias,
                                             q,
                                             k,
                                             v,
                                             update_state,
                                             ssm_state_indices,
                                             cu_seqlens,
                                             scale,
                                             /*use_qk_l2norm_in_kernel=*/true,
                                             /*softplus_beta=*/1.0f,
                                             /*softplus_threshold=*/20.0f,
                                             num_accepted_tokens);
  Sync();

  EXPECT_EQ(out.sizes(), out_ref.sizes());
  EXPECT_EQ(final_state.sizes(), final_state_ref.sizes());
  auto out_abs_diff =
      (out.to(torch::kFloat32) - out_ref.to(torch::kFloat32)).abs();
  auto state_abs_diff =
      (final_state.to(torch::kFloat32) - final_state_ref.to(torch::kFloat32))
          .abs();
  EXPECT_TRUE(torch::allclose(out, out_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max out diff: " << out_abs_diff.max().item<float>()
      << ", mean out diff: " << out_abs_diff.mean().item<float>();
  EXPECT_TRUE(torch::allclose(final_state,
                              final_state_ref,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "max state diff: " << state_abs_diff.max().item<float>()
      << ", mean state diff: " << state_abs_diff.mean().item<float>();
}

TEST_F(Qwen3_5GatedDeltaNetOpsTest,
       FusedSigmoidGatingDeltaRuleUpdateSupportsDenseKdaAndOutOfPlaceState) {
  const int64_t batch_size = 2;
  const int64_t num_tokens = 2;
  const int64_t num_k_heads = 1;
  const int64_t num_v_heads = 1;
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 256;

  auto q = MakeNoise("sigmoid_update.kda.q",
                     {batch_size, num_tokens, num_k_heads, head_k_dim},
                     0.02f);
  auto k = MakeNoise("sigmoid_update.kda.k",
                     {batch_size, num_tokens, num_k_heads, head_k_dim},
                     0.02f);
  auto v = MakeNoise("sigmoid_update.kda.v",
                     {batch_size, num_tokens, num_v_heads, head_v_dim},
                     0.02f);
  auto a = MakeNoise("sigmoid_update.kda.a",
                     {batch_size * num_tokens, num_v_heads * head_k_dim},
                     0.02f);
  auto b = MakeNoise(
      "sigmoid_update.kda.b", {batch_size * num_tokens, num_v_heads}, 0.02f);
  auto a_log = MakeNoise("sigmoid_update.kda.A_log", {num_v_heads}, 0.02f);
  auto dt_bias = MakeNoise(
      "sigmoid_update.kda.dt_bias", {num_v_heads * head_k_dim}, 0.02f);
  auto initial_state =
      MakeNoise("sigmoid_update.kda.initial_state",
                {batch_size * num_tokens, num_v_heads, head_v_dim, head_k_dim},
                0.01f,
                torch::kFloat32);

  double scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));
  auto update_state = initial_state.clone();
  torch::Tensor q_ref = L2NormalizeLastDim(q).to(torch::kFloat32) * scale;
  torch::Tensor k_ref = L2NormalizeLastDim(k).to(torch::kFloat32);
  torch::Tensor v_ref = v.to(torch::kFloat32);
  torch::Tensor a_ref =
      a.view({batch_size, num_tokens, num_v_heads, head_k_dim});
  torch::Tensor b_ref = b.view({batch_size, num_tokens, num_v_heads});
  torch::Tensor dt_bias_ref =
      dt_bias.view({num_v_heads, head_k_dim}).to(torch::kFloat32);
  torch::Tensor out_ref = torch::empty_like(v);
  torch::Tensor final_state_ref = torch::empty_like(initial_state);
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    torch::Tensor running_state =
        initial_state.select(/*dim=*/0, batch * num_tokens).clone();
    for (int64_t token = 0; token < num_tokens; ++token) {
      for (int64_t hv = 0; hv < num_v_heads; ++hv) {
        torch::Tensor state_h = running_state.select(/*dim=*/0, hv);
        torch::Tensor decay = torch::exp(
            -torch::exp(a_log.select(/*dim=*/0, hv).to(torch::kFloat32)) *
            torch::nn::functional::softplus(
                a_ref.select(/*dim=*/0, batch)
                        .select(/*dim=*/0, token)
                        .select(/*dim=*/0, hv)
                        .to(torch::kFloat32) +
                    dt_bias_ref.select(/*dim=*/0, hv),
                torch::nn::functional::SoftplusFuncOptions()
                    .beta(1.0)
                    .threshold(20.0)));
        torch::Tensor q_token = q_ref.select(/*dim=*/0, batch)
                                    .select(/*dim=*/0, token)
                                    .select(/*dim=*/0, hv);
        torch::Tensor k_token = k_ref.select(/*dim=*/0, batch)
                                    .select(/*dim=*/0, token)
                                    .select(/*dim=*/0, hv);
        torch::Tensor v_token = v_ref.select(/*dim=*/0, batch)
                                    .select(/*dim=*/0, token)
                                    .select(/*dim=*/0, hv);
        state_h.mul_(decay.unsqueeze(/*dim=*/0));
        torch::Tensor beta = torch::sigmoid(b_ref.select(/*dim=*/0, batch)
                                                .select(/*dim=*/0, token)
                                                .select(/*dim=*/0, hv)
                                                .to(torch::kFloat32));
        torch::Tensor value_delta =
            (v_token -
             (state_h * k_token.unsqueeze(/*dim=*/0)).sum(/*dim=*/-1)) *
            beta;
        state_h.add_(value_delta.unsqueeze(/*dim=*/1) *
                     k_token.unsqueeze(/*dim=*/0));
        out_ref.select(/*dim=*/0, batch)
            .select(/*dim=*/0, token)
            .select(/*dim=*/0, hv)
            .copy_((state_h * q_token.unsqueeze(/*dim=*/0))
                       .sum(/*dim=*/-1)
                       .to(out_ref.scalar_type()));
      }
      final_state_ref.select(/*dim=*/0, batch * num_tokens + token)
          .copy_(running_state);
    }
  }

  torch::Tensor state_indices;
  torch::Tensor cu_seqlens;
  auto [out, final_state] = fused_sigmoid_gating_delta_rule_update(
      a_log,
      a,
      b,
      dt_bias,
      q,
      k,
      v,
      update_state,
      state_indices,
      cu_seqlens,
      scale,
      /*use_qk_l2norm_in_kernel=*/true,
      /*softplus_beta=*/1.0f,
      /*softplus_threshold=*/20.0f,
      /*num_accepted_tokens_opt=*/std::nullopt,
      /*inplace_final_state=*/false,
      /*is_kda=*/true);
  Sync();

  EXPECT_EQ(
      out.sizes(),
      torch::IntArrayRef({batch_size, num_tokens, num_v_heads, head_v_dim}));
  EXPECT_EQ(
      final_state.sizes(),
      torch::IntArrayRef(
          {batch_size * num_tokens, num_v_heads, head_v_dim, head_k_dim}));
  EXPECT_TRUE(torch::allclose(out, out_ref, /*rtol=*/1e-2, /*atol=*/1e-2));
  EXPECT_TRUE(torch::allclose(
      final_state, final_state_ref, /*rtol=*/1e-2, /*atol=*/1e-2));
}

TEST_F(Qwen3_5GatedDeltaNetOpsTest,
       FusedSigmoidGatingDeltaRuleUpdateCommonApiPassesAcceptedPrefix) {
  const int64_t num_k_heads = 16;
  const int64_t num_v_heads = 32;
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 128;
  const int64_t batch_size = 1;
  const int64_t num_tokens = 4;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  auto q = MakeNoise("sigmoid_update_api.q",
                     {batch_size, num_tokens, num_k_heads, head_k_dim},
                     0.02f);
  auto k = MakeNoise("sigmoid_update_api.k",
                     {batch_size, num_tokens, num_k_heads, head_k_dim},
                     0.02f);
  auto v = MakeNoise("sigmoid_update_api.v",
                     {batch_size, num_tokens, num_v_heads, head_v_dim},
                     0.02f);
  auto a = MakeNoise("sigmoid_update_api.a", {num_tokens, num_v_heads}, 0.02f);
  auto b = MakeNoise("sigmoid_update_api.b", {num_tokens, num_v_heads}, 0.02f);
  auto a_log = MakeNoise("sigmoid_update_api.A_log", {num_v_heads}, 0.02f);
  auto dt_bias = MakeNoise("sigmoid_update_api.dt_bias", {num_v_heads}, 0.02f);
  auto initial_state =
      MakeNoise("sigmoid_update_api.initial_state",
                {num_tokens + 1, num_v_heads, head_v_dim, head_k_dim},
                0.01f,
                torch::kFloat32);
  auto ssm_state_indices =
      torch::arange(1, num_tokens + 1, opts_int).unsqueeze(0);
  auto cu_seqlens = torch::tensor(
      std::vector<int32_t>{0, static_cast<int32_t>(num_tokens)}, opts_int);
  auto num_accepted_tokens = torch::full({1}, /*value=*/3, opts_int);
  double scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));

  auto expected_state = initial_state.clone();
  auto [expected_out, expected_final_state] =
      fused_sigmoid_gating_delta_rule_update(a_log,
                                             a,
                                             b,
                                             dt_bias,
                                             q,
                                             k,
                                             v,
                                             expected_state,
                                             ssm_state_indices,
                                             cu_seqlens,
                                             scale,
                                             /*use_qk_l2norm_in_kernel=*/true,
                                             /*softplus_beta=*/1.0f,
                                             /*softplus_threshold=*/20.0f,
                                             num_accepted_tokens);
  Sync();

  xllm::kernel::FusedSigmoidGatingDeltaRuleUpdateParams params;
  params.A_log = a_log;
  params.a = a.clone();
  params.dt_bias = dt_bias;
  params.q = q.clone();
  params.k = k.clone();
  params.v = v.clone();
  params.b = b.clone();
  params.initial_state_source = initial_state.clone();
  params.initial_state_indices = ssm_state_indices;
  params.cu_seqlens = cu_seqlens;
  params.scale = static_cast<float>(scale);
  params.use_qk_l2norm_in_kernel = true;
  params.softplus_beta = 1.0f;
  params.softplus_threshold = 20.0f;
  params.num_accepted_tokens = num_accepted_tokens;
  torch::Tensor out =
      xllm::kernel::fused_sigmoid_gating_delta_rule_update(params);
  Sync();

  EXPECT_EQ(out.sizes(), expected_out.sizes());
  EXPECT_TRUE(torch::allclose(out,
                              expected_out,
                              /*rtol=*/1e-3,
                              /*atol=*/1e-4));
  EXPECT_TRUE(torch::allclose(params.initial_state_source,
                              expected_final_state,
                              /*rtol=*/1e-3,
                              /*atol=*/1e-4));
}

void run_rebased_ssm_checkpoint_group_test(const torch::Device& device,
                                           const std::string& key_prefix,
                                           int64_t q_max_seq_len) {
  const int64_t num_k_heads = 1;
  const int64_t num_v_heads = 2;
  const int64_t head_k_dim = 128;
  const int64_t head_v_dim = 128;
  const int64_t batch_size = 2;
  const int64_t num_tokens = batch_size * q_max_seq_len;
  const int64_t num_state_slots = num_tokens + 1;
  auto opts_int = torch::TensorOptions().dtype(torch::kInt32).device(device);

  auto make_noise = [&](const std::string& name,
                        torch::IntArrayRef shape,
                        float stddev,
                        torch::ScalarType dtype = torch::kBFloat16) {
    auto raw =
        test::seeded_tensor(key_prefix + "." + name, shape, dtype, device);
    return (raw - 0.5f) * (std::sqrt(12.0f) * stddev);
  };

  auto q1 =
      make_noise("iter1.q", {1, num_tokens, num_k_heads, head_k_dim}, 0.02f);
  auto k1 =
      make_noise("iter1.k", {1, num_tokens, num_k_heads, head_k_dim}, 0.02f);
  auto v1 =
      make_noise("iter1.v", {1, num_tokens, num_v_heads, head_v_dim}, 0.02f);
  auto a1 = make_noise("iter1.a", {num_tokens, num_v_heads}, 0.02f);
  auto b1 = make_noise("iter1.b", {num_tokens, num_v_heads}, 0.02f);
  auto q2 =
      make_noise("iter2.q", {1, num_tokens, num_k_heads, head_k_dim}, 0.02f);
  auto k2 =
      make_noise("iter2.k", {1, num_tokens, num_k_heads, head_k_dim}, 0.02f);
  auto v2 =
      make_noise("iter2.v", {1, num_tokens, num_v_heads, head_v_dim}, 0.02f);
  auto a2 = make_noise("iter2.a", {num_tokens, num_v_heads}, 0.02f);
  auto b2 = make_noise("iter2.b", {num_tokens, num_v_heads}, 0.02f);
  auto a_log = make_noise("A_log", {num_v_heads}, 0.02f);
  auto dt_bias = make_noise("dt_bias", {num_v_heads}, 0.02f);
  auto initial_state =
      make_noise("initial_state",
                 {num_state_slots, num_v_heads, head_v_dim, head_k_dim},
                 0.01f,
                 torch::kFloat32);
  auto ssm_state_indices = torch::arange(1, num_state_slots, opts_int)
                               .view({batch_size, q_max_seq_len});
  auto cu_seqlens = torch::arange(0, num_tokens + 1, q_max_seq_len, opts_int);
  auto first_accepted =
      torch::full({batch_size}, static_cast<int32_t>(q_max_seq_len), opts_int);
  const int32_t middle_accepted =
      static_cast<int32_t>(std::max<int64_t>(2, (q_max_seq_len + 1) / 2));
  torch::Tensor second_accepted =
      torch::tensor(std::vector<int32_t>{1, middle_accepted}, opts_int);
  double scale = 1.0 / std::sqrt(static_cast<double>(head_k_dim));

  auto [out_ref1, state_ref1] = FusedSigmoidSpecVerifyRef(q1,
                                                          k1,
                                                          v1,
                                                          a1,
                                                          b1,
                                                          a_log,
                                                          dt_bias,
                                                          scale,
                                                          initial_state,
                                                          ssm_state_indices,
                                                          cu_seqlens,
                                                          first_accepted);
  auto state_run1 = initial_state.clone();
  auto [out1, final_state1] =
      fused_sigmoid_gating_delta_rule_update(a_log,
                                             a1,
                                             b1,
                                             dt_bias,
                                             q1,
                                             k1,
                                             v1,
                                             state_run1,
                                             ssm_state_indices,
                                             cu_seqlens,
                                             scale,
                                             /*use_qk_l2norm_in_kernel=*/true,
                                             /*softplus_beta=*/1.0f,
                                             /*softplus_threshold=*/20.0f,
                                             first_accepted);
  Device xllm_device(device);
  xllm_device.synchronize_default_stream();

  EXPECT_TRUE(torch::allclose(out1, out_ref1, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "Rebased Checkpoint Group first iteration output mismatch";
  EXPECT_TRUE(torch::allclose(
      select_rebased_checkpoint_rows(final_state1, ssm_state_indices),
      select_rebased_checkpoint_rows(state_ref1, ssm_state_indices),
      /*rtol=*/1e-2,
      /*atol=*/1e-2))
      << "Rebased Checkpoint Group first iteration checkpoint mismatch";
  EXPECT_FALSE(torch::allclose(final_state1.index({ssm_state_indices.select(
                                   /*dim=*/1, /*index=*/0)}),
                               initial_state.index({ssm_state_indices.select(
                                   /*dim=*/1, /*index=*/0)}),
                               /*rtol=*/1e-5,
                               /*atol=*/1e-6))
      << "Rebased Checkpoint Group offset 0 must be rewritten";

  auto [out_ref2, state_ref2] = FusedSigmoidSpecVerifyRef(q2,
                                                          k2,
                                                          v2,
                                                          a2,
                                                          b2,
                                                          a_log,
                                                          dt_bias,
                                                          scale,
                                                          state_ref1,
                                                          ssm_state_indices,
                                                          cu_seqlens,
                                                          second_accepted);
  auto state_run2 = final_state1.clone();
  auto [out2, final_state2] =
      fused_sigmoid_gating_delta_rule_update(a_log,
                                             a2,
                                             b2,
                                             dt_bias,
                                             q2,
                                             k2,
                                             v2,
                                             state_run2,
                                             ssm_state_indices,
                                             cu_seqlens,
                                             scale,
                                             /*use_qk_l2norm_in_kernel=*/true,
                                             /*softplus_beta=*/1.0f,
                                             /*softplus_threshold=*/20.0f,
                                             second_accepted);
  xllm_device.synchronize_default_stream();

  EXPECT_TRUE(torch::allclose(out2, out_ref2, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "Rebased Checkpoint Group accepted prefix 1 must resume from "
         "rewritten offset 0";
  EXPECT_TRUE(torch::allclose(
      select_rebased_checkpoint_rows(final_state2, ssm_state_indices),
      select_rebased_checkpoint_rows(state_ref2, ssm_state_indices),
      /*rtol=*/1e-2,
      /*atol=*/1e-2))
      << "Rebased Checkpoint Group second iteration checkpoint mismatch";
}

TEST_P(Qwen3_5GatedDeltaNetSpecVerifyOpsTest,
       FusedSigmoidGatingDeltaRuleUpdateRebasedCheckpoint) {
  const int64_t q_max_seq_len = GetParam();
  const std::string key_prefix =
      "sigmoid_rebased_width" + std::to_string(q_max_seq_len);
  run_rebased_ssm_checkpoint_group_test(device_, key_prefix, q_max_seq_len);
}

INSTANTIATE_TEST_SUITE_P(SpecVerifyWidths,
                         Qwen3_5GatedDeltaNetSpecVerifyOpsTest,
                         ::testing::Values(int64_t{2}, int64_t{3}, int64_t{4}),
                         spec_verify_width_name);

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
