/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kWidth = 4;
constexpr int32_t kStateLen = kWidth - 1;

class TileLangCausalConv1dDecodeWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

torch::Tensor causal_conv1d_decode_cpu_ref(const torch::Tensor& x,
                                           const torch::Tensor& conv_state,
                                           const torch::Tensor& weight,
                                           const torch::Tensor& bias,
                                           const torch::Tensor& init_indices,
                                           const torch::Tensor& current_indices,
                                           bool has_silu) {
  const int64_t batch = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t width = weight.size(1);
  const int64_t hist_len = width - 1;

  auto x_f = x.to(torch::kFloat32);
  auto weight_f = weight.to(torch::kFloat32);
  auto conv_state_f = conv_state.to(torch::kFloat32).clone();
  auto bias_f = bias.to(torch::kFloat32);
  auto init_l = init_indices.to(torch::kInt64);
  auto current_l = current_indices.to(torch::kInt64);

  auto out = torch::zeros({batch, dim}, torch::kFloat32);

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t read_line = init_l[b].item<int64_t>();
    const int64_t write_line = current_l[b].item<int64_t>();

    std::vector<torch::Tensor> history;
    history.reserve(hist_len);
    for (int64_t h = 0; h < hist_len; ++h) {
      history.emplace_back(
          conv_state_f[read_line].index({torch::indexing::Slice(), h}).clone());
    }

    auto acc = torch::zeros({dim}, torch::kFloat32);
    for (int64_t w = 0; w < hist_len; ++w) {
      acc += weight_f.index({torch::indexing::Slice(), w}) * history[w];
    }
    acc += weight_f.index({torch::indexing::Slice(), width - 1}) * x_f[b];
    acc += bias_f;
    if (has_silu) {
      acc = acc / (1.0F + torch::exp(-acc));
    }
    out[b] = acc;

    for (int64_t h = 0; h < hist_len - 1; ++h) {
      history[h] = history[h + 1].clone();
    }
    history[hist_len - 1] = x_f[b].clone();

    for (int64_t h = 0; h < hist_len; ++h) {
      conv_state_f[write_line].index_put_({torch::indexing::Slice(), h},
                                          history[h]);
    }
  }

  conv_state.copy_(conv_state_f.to(conv_state.dtype()));
  return out.to(x.dtype());
}

torch::Tensor prepare_weight_t(const torch::Tensor& weight) {
  return weight.transpose(0, 1).contiguous();
}

torch::Tensor prepare_conv_state_t(const torch::Tensor& conv_state) {
  return conv_state.permute({0, 2, 1}).contiguous();
}

torch::Tensor conv_state_t_to_pytorch(const torch::Tensor& conv_state_t) {
  return conv_state_t.permute({0, 2, 1}).contiguous();
}

struct CausalConv1dDecodeTestCase {
  std::string name;
  int64_t dim;
  int64_t batch_size;
  bool has_silu;
  int64_t num_cache_lines;
  int64_t seed;
};

void run_causal_conv1d_decode_case(
    const CausalConv1dDecodeTestCase& test_case) {
  ASSERT_GT(test_case.batch_size, 0);
  ASSERT_GT(test_case.dim, 0);
  ASSERT_GT(test_case.num_cache_lines, 0);

  const auto npu_device = torch::Device("npu:0");
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);
  const auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(npu_device);

  torch::manual_seed(test_case.seed);

  auto x_raw = torch::randn({test_case.batch_size, test_case.dim}, bf16_opts);
  auto conv_state_raw = torch::randn(
      {test_case.num_cache_lines, test_case.dim, kStateLen}, bf16_opts);
  auto weight_raw = torch::randn({test_case.dim, kWidth}, bf16_opts);
  auto bias_raw = torch::randn({test_case.dim}, bf16_opts);

  auto init_indices = torch::arange(0, test_case.batch_size, i32_opts);
  auto current_indices = torch::arange(0, test_case.batch_size, i32_opts);
  auto initial_state_mode = torch::ones(test_case.batch_size, i32_opts);

  auto x_cpu = x_raw.cpu();
  auto cs_cpu = conv_state_raw.cpu().clone();
  auto w_cpu = weight_raw.cpu();
  auto b_cpu = bias_raw.cpu();
  auto init_cpu = init_indices.cpu();
  auto curr_cpu = current_indices.cpu();

  auto golden = causal_conv1d_decode_cpu_ref(
      x_cpu, cs_cpu, w_cpu, b_cpu, init_cpu, curr_cpu, test_case.has_silu);

  auto weight_t = prepare_weight_t(weight_raw);
  auto conv_state_t = prepare_conv_state_t(conv_state_raw);

  auto y = causal_conv1d_decode(/*conv_state=*/conv_state_t,
                                /*x=*/x_raw,
                                /*weight=*/weight_t,
                                /*bias=*/bias_raw,
                                /*init_indices=*/init_indices,
                                /*current_indices=*/current_indices,
                                /*initial_state_mode=*/initial_state_mode,
                                /*has_silu=*/test_case.has_silu);

  float max_diff = (y.cpu().to(torch::kFloat32) - golden.to(torch::kFloat32))
                       .abs()
                       .max()
                       .item<float>();
  std::cout << "[causal_conv1d_decode_wrapper_test] case=" << test_case.name
            << ", max_diff=" << max_diff << std::endl;

  EXPECT_TRUE(torch::allclose(y.cpu().to(torch::kFloat32),
                              golden.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "causal_conv1d_decode output differs from CPU reference, max_diff="
      << max_diff;

  auto cs_result = conv_state_t_to_pytorch(conv_state_t);
  float cs_max_diff =
      (cs_result.cpu().to(torch::kFloat32) - cs_cpu.to(torch::kFloat32))
          .abs()
          .max()
          .item<float>();
  EXPECT_TRUE(torch::allclose(cs_result.cpu().to(torch::kFloat32),
                              cs_cpu.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "conv_state mismatch, max_diff=" << cs_max_diff;
}

TEST_F(TileLangCausalConv1dDecodeWrapperTest, DecodeBatch1Silu) {
  const std::vector<CausalConv1dDecodeTestCase> cases = {
      {.name = "decode_bs1_sl1",
       .dim = 2048,
       .batch_size = 1,
       .has_silu = true,
       .num_cache_lines = 4,
       .seed = 42},
      {.name = "qwen35_2b_tp2_decode_bs1_sl1",
       .dim = 3072,
       .batch_size = 1,
       .has_silu = true,
       .num_cache_lines = 4,
       .seed = 43},
  };
  for (const auto& tc : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << tc.name);
    run_causal_conv1d_decode_case(tc);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
