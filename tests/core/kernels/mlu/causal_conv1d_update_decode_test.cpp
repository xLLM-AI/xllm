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

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace {

using xllm::kernel::mlu::causal_conv1d_update_decode;

torch::Tensor bf16_randn(torch::IntArrayRef shape, const torch::Device& dev) {
  return torch::randn(
      shape, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
}

bool tensors_allclose(const torch::Tensor& a,
                      const torch::Tensor& b,
                      double rtol,
                      double atol) {
  torch::Tensor ac = a.to(torch::kCPU).to(torch::kFloat32).contiguous();
  torch::Tensor bc = b.to(torch::kCPU).to(torch::kFloat32).contiguous();
  if (ac.sizes() != bc.sizes()) {
    return false;
  }
  torch::Tensor diff = (ac - bc).abs();
  torch::Tensor tol = atol + rtol * bc.abs();
  return (diff <= tol).all().item<bool>();
}

class CausalConv1dUpdateDecodeJitTest : public ::testing::Test {
 protected:
  torch::Device device() { return torch::Device(torch::kPrivateUse1, 0); }
};

TEST_F(CausalConv1dUpdateDecodeJitTest, SecondCallHitsCache) {
  torch::DeviceGuard guard(device());
  int32_t dim = 1024;
  int32_t width = 4;
  int32_t batch = 4;
  torch::Tensor x = bf16_randn({batch, dim, 1}, device());
  torch::Tensor conv_state_orig = bf16_randn({batch, dim, width - 1}, device());
  torch::Tensor weight = bf16_randn({dim, width}, device());
  torch::Tensor conv_state_indices = torch::arange(
      0, batch, torch::TensorOptions().dtype(torch::kInt32).device(device()));

  torch::Tensor cs1 = conv_state_orig.clone();
  torch::Tensor cs2 = conv_state_orig.clone();

  auto t0 = std::chrono::steady_clock::now();
  torch::Tensor out1 = causal_conv1d_update_decode(x,
                                                   cs1,
                                                   weight,
                                                   std::nullopt,
                                                   conv_state_indices,
                                                   /*activation=*/true,
                                                   /*pad_slot_id=*/-1);
  torch_mlu::synchronize();
  auto t1 = std::chrono::steady_clock::now();
  torch::Tensor out2 = causal_conv1d_update_decode(x,
                                                   cs2,
                                                   weight,
                                                   std::nullopt,
                                                   conv_state_indices,
                                                   /*activation=*/true,
                                                   /*pad_slot_id=*/-1);
  torch_mlu::synchronize();
  auto t2 = std::chrono::steady_clock::now();

  int64_t first_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  int64_t second_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  LOG(INFO) << "first call (compile+launch): " << first_ms
            << "ms; second call (cached): " << second_ms << "ms";
  EXPECT_TRUE(tensors_allclose(out1, out2, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "output mismatch";
  EXPECT_TRUE(tensors_allclose(cs1, cs2, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "conv_state mismatch";
}

TEST_F(CausalConv1dUpdateDecodeJitTest, AcceptsUncompiledDim) {
  torch::DeviceGuard guard(device());
  int32_t dim = 1000;
  int32_t width = 4;
  int32_t batch = 2;
  torch::Tensor x = bf16_randn({batch, dim, 1}, device());
  torch::Tensor conv_state = bf16_randn({batch, dim, width - 1}, device());
  torch::Tensor weight = bf16_randn({dim, width}, device());
  torch::Tensor conv_state_indices = torch::arange(
      0, batch, torch::TensorOptions().dtype(torch::kInt32).device(device()));

  torch::Tensor out = causal_conv1d_update_decode(x,
                                                  conv_state,
                                                  weight,
                                                  std::nullopt,
                                                  conv_state_indices,
                                                  /*activation=*/true,
                                                  /*pad_slot_id=*/-1);
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::isfinite(out.to(torch::kFloat32)).all().item<bool>());
}

TEST_F(CausalConv1dUpdateDecodeJitTest,
       PreservesAdjacentSlotAcrossConsecutiveDecode) {
  torch::DeviceGuard guard(device());
  const int32_t dim = 2048;
  const int32_t width = 4;
  const int32_t num_slots = 3;
  const float canary = 7.0f;
  torch::TensorOptions bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device());
  torch::TensorOptions int_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(device());
  torch::Tensor x = torch::zeros({1, dim, 1}, bf16_opts);
  torch::Tensor weight = torch::ones({dim, width}, bf16_opts);
  torch::Tensor conv_state =
      torch::zeros({num_slots, dim, width - 1}, bf16_opts);
  conv_state[1].fill_(canary);
  torch::Tensor adjacent_slot = conv_state[1].clone();
  torch::Tensor first_slot = torch::zeros({1}, int_opts);

  causal_conv1d_update_decode(x,
                              conv_state,
                              weight,
                              std::nullopt,
                              first_slot,
                              /*activation=*/false,
                              /*pad_slot_id=*/-1);
  torch_mlu::synchronize();

  EXPECT_TRUE(torch::equal(conv_state[1], adjacent_slot))
      << "decoding slot 0 must not overwrite adjacent slot 1";

  torch::Tensor second_slot = torch::ones({1}, int_opts);
  torch::Tensor out = causal_conv1d_update_decode(x,
                                                  conv_state,
                                                  weight,
                                                  std::nullopt,
                                                  second_slot,
                                                  /*activation=*/false,
                                                  /*pad_slot_id=*/-1);
  torch_mlu::synchronize();
  torch::Tensor expected = torch::full_like(out, canary * (width - 1));

  EXPECT_TRUE(torch::equal(out, expected))
      << "the next decode must observe the original state of slot 1";
}

TEST_F(CausalConv1dUpdateDecodeJitTest,
       SpecVarlenKeepsAcceptedStateWindowLayout) {
  torch::DeviceGuard guard(device());
  constexpr int32_t kDim = 4;
  constexpr int32_t kWidth = 4;
  constexpr int32_t kBatch = 2;
  constexpr int32_t kMaxQueryLen = 3;
  constexpr int32_t kStateLen = kWidth - 1 + kMaxQueryLen - 1;
  torch::TensorOptions bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device());
  torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device());
  torch::Tensor x =
      torch::arange(1, 1 + 4 * kDim, bf16_options).view({4, kDim});
  torch::Tensor conv_state =
      torch::arange(101, 101 + kBatch * kDim * kStateLen, bf16_options)
          .view({kBatch, kDim, kStateLen});
  torch::Tensor original_state = conv_state.clone();
  torch::Tensor weight = torch::zeros({kDim, kWidth}, bf16_options);
  torch::Tensor conv_state_indices = torch::arange(0, kBatch, int_options);
  torch::Tensor query_start_loc =
      torch::tensor(std::vector<int32_t>{0, 1, 4}, int_options);
  torch::Tensor num_accepted_tokens =
      torch::tensor(std::vector<int32_t>{1, 2}, int_options);

  torch::Tensor expected_state = original_state.clone();
  torch::Tensor expected_first_prefix =
      torch::cat({original_state[0].slice(/*dim=*/1, /*start=*/1, /*end=*/3),
                  x[0].unsqueeze(/*dim=*/1)},
                 /*dim=*/1);
  expected_state[0]
      .slice(/*dim=*/1, /*start=*/0, /*end=*/3)
      .copy_(expected_first_prefix);
  torch::Tensor expected_second = torch::cat(
      {original_state[1].slice(/*dim=*/1, /*start=*/2, /*end=*/4),
       x.slice(/*dim=*/0, /*start=*/1).transpose(/*dim0=*/0, /*dim1=*/1)},
      /*dim=*/1);
  expected_state[1].copy_(expected_second);

  causal_conv1d_update_decode(x,
                              conv_state,
                              weight,
                              /*bias_opt=*/std::nullopt,
                              conv_state_indices,
                              /*activation=*/false,
                              /*pad_slot_id=*/-1,
                              query_start_loc,
                              /*max_query_len=*/kMaxQueryLen,
                              num_accepted_tokens);
  torch_mlu::synchronize();

  EXPECT_TRUE(tensors_allclose(conv_state,
                               expected_state,
                               /*rtol=*/0.0,
                               /*atol=*/0.0))
      << "spec varlen must preserve accepted state window layout while "
         "writing only the actual query tokens";
}

}  // namespace
}  // namespace xllm
