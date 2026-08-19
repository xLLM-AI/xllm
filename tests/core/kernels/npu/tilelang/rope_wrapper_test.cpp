/* Copyright 2025-2026 The xLLM Authors.

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangRopeWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
  }
};

struct RopeTestCase {
  std::string name;
  int64_t num_tokens;
  int64_t num_heads;
  int64_t full_head_dim;
  int64_t rope_dim;
  int64_t start_dim;
  int64_t seed;
};

torch::Tensor torch_rope_ref(const torch::Tensor& x,
                             const torch::Tensor& sin,
                             const torch::Tensor& cos) {
  auto cos_ref = cos;
  auto sin_ref = sin;
  if (cos_ref.dim() == 2) {
    cos_ref = cos_ref.unsqueeze(1);
    sin_ref = sin_ref.unsqueeze(1);
  }

  auto x_fp32 = x.to(torch::kFloat32);
  auto cos_fp32 = cos_ref.to(torch::kFloat32);
  auto sin_fp32 = sin_ref.to(torch::kFloat32);

  auto x_reshaped =
      x_fp32.view({x_fp32.size(0), x_fp32.size(1), x_fp32.size(2) / 2, 2});
  auto x0 = x_reshaped.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              0});
  auto x1 = x_reshaped.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              1});
  auto x_rotated = torch::stack({-x1, x0}, /*dim=*/-1).flatten(-2);

  auto out = x_fp32 * cos_fp32 + x_rotated * sin_fp32;
  return out.to(torch::kBFloat16);
}

double measure_npu_event_ms(const std::function<void()>& fn,
                            int32_t device_id,
                            int warmup_iters = 5,
                            int measure_iters = 100) {
  CHECK_GT(measure_iters, 0) << "measure_iters must be > 0";
  CHECK_GE(warmup_iters, 0) << "warmup_iters must be >= 0";

  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  for (int i = 0; i < warmup_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "warmup stream synchronize failed";

  aclrtEvent start_event = nullptr;
  aclrtEvent end_event = nullptr;
  CHECK_EQ(aclrtCreateEvent(&start_event), ACL_SUCCESS)
      << "aclrtCreateEvent(start) failed";
  CHECK_EQ(aclrtCreateEvent(&end_event), ACL_SUCCESS)
      << "aclrtCreateEvent(end) failed";

  CHECK_EQ(aclrtRecordEvent(start_event, stream), ACL_SUCCESS)
      << "aclrtRecordEvent(start) failed";
  for (int i = 0; i < measure_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtRecordEvent(end_event, stream), ACL_SUCCESS)
      << "aclrtRecordEvent(end) failed";
  CHECK_EQ(aclrtSynchronizeEvent(end_event), ACL_SUCCESS)
      << "aclrtSynchronizeEvent(end) failed";

  float elapsed_ms = 0.0F;
  CHECK_EQ(aclrtEventElapsedTime(&elapsed_ms, start_event, end_event),
           ACL_SUCCESS)
      << "aclrtEventElapsedTime failed";
  CHECK_EQ(aclrtDestroyEvent(start_event), ACL_SUCCESS)
      << "aclrtDestroyEvent(start) failed";
  CHECK_EQ(aclrtDestroyEvent(end_event), ACL_SUCCESS)
      << "aclrtDestroyEvent(end) failed";

  return static_cast<double>(elapsed_ms) / static_cast<double>(measure_iters);
}

torch::Tensor maybe_narrow(const torch::Tensor& tensor,
                           int64_t start_dim,
                           int64_t rope_dim) {
  if (start_dim == 0 && rope_dim == tensor.size(2)) {
    return tensor;
  }
  return tensor.narrow(/*dim=*/2, /*start=*/start_dim, /*length=*/rope_dim);
}

void run_apply_rotary_case(const RopeTestCase& test_case) {
  ASSERT_GT(test_case.num_tokens, 0);
  ASSERT_GT(test_case.num_heads, 0);
  ASSERT_GT(test_case.full_head_dim, 0);
  ASSERT_GT(test_case.rope_dim, 0);
  ASSERT_GE(test_case.start_dim, 0);
  ASSERT_LE(test_case.start_dim + test_case.rope_dim, test_case.full_head_dim);

  const auto npu_device = torch::Device("npu:0");
  const int32_t device_id = npu_device.index();
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);

  torch::manual_seed(test_case.seed);
  auto q_full = torch::randn(
      {test_case.num_tokens, test_case.num_heads, test_case.full_head_dim},
      bf16_opts);
  auto k_full = torch::randn(
      {test_case.num_tokens, test_case.num_heads, test_case.full_head_dim},
      bf16_opts);
  auto sin_cache =
      torch::randn({test_case.num_tokens, test_case.rope_dim}, bf16_opts);
  auto cos_cache =
      torch::randn({test_case.num_tokens, test_case.rope_dim}, bf16_opts);

  auto q_input = maybe_narrow(q_full, test_case.start_dim, test_case.rope_dim);
  auto k_input = maybe_narrow(k_full, test_case.start_dim, test_case.rope_dim);

  if (test_case.start_dim > 0) {
    EXPECT_EQ(q_input.storage_offset(), test_case.start_dim);
    EXPECT_EQ(k_input.storage_offset(), test_case.start_dim);
    if (test_case.num_tokens * test_case.num_heads > 1) {
      EXPECT_FALSE(q_input.is_contiguous());
      EXPECT_FALSE(k_input.is_contiguous());
    }
  }

  auto q_ref = torch_rope_ref(q_input, sin_cache, cos_cache);
  auto k_ref = torch_rope_ref(k_input, sin_cache, cos_cache);
  auto q_runtime_full = q_full.clone();
  auto k_runtime_full = k_full.clone();
  auto q =
      maybe_narrow(q_runtime_full, test_case.start_dim, test_case.rope_dim);
  auto k =
      maybe_narrow(k_runtime_full, test_case.start_dim, test_case.rope_dim);
  rope_in_place(q, sin_cache, cos_cache);
  rope_in_place(k, sin_cache, cos_cache);

  auto q_bench_full = q_full.clone();
  auto k_bench_full = k_full.clone();
  auto q_bench =
      maybe_narrow(q_bench_full, test_case.start_dim, test_case.rope_dim);
  auto k_bench =
      maybe_narrow(k_bench_full, test_case.start_dim, test_case.rope_dim);
  const double ref_elapsed_ms = measure_npu_event_ms(
      [&]() {
        [[maybe_unused]] auto q_ref_bench =
            torch_rope_ref(q_input, sin_cache, cos_cache);
        [[maybe_unused]] auto k_ref_bench =
            torch_rope_ref(k_input, sin_cache, cos_cache);
      },
      device_id);
  const double tl_elapsed_ms = measure_npu_event_ms(
      [&]() {
        rope_in_place(q_bench, sin_cache, cos_cache);
        rope_in_place(k_bench, sin_cache, cos_cache);
      },
      device_id);

  const double speedup =
      tl_elapsed_ms > 0.0 ? ref_elapsed_ms / tl_elapsed_ms : 0.0;
  std::cout << "[rope_wrapper_test] case=" << test_case.name
            << ", ref_ms=" << ref_elapsed_ms
            << ", tilelang_ms=" << tl_elapsed_ms << ", speedup=" << speedup
            << "x" << std::endl;

  auto q_max_diff = (q.to(torch::kFloat32) - q_ref.to(torch::kFloat32))
                        .abs()
                        .max()
                        .item<float>();
  auto k_max_diff = (k.to(torch::kFloat32) - k_ref.to(torch::kFloat32))
                        .abs()
                        .max()
                        .item<float>();

  EXPECT_TRUE(torch::allclose(q, q_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "q mismatch: tilelang output differs from interleaved rope reference"
      << ", max_diff=" << q_max_diff;
  EXPECT_TRUE(torch::allclose(k, k_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "k mismatch: tilelang output differs from interleaved rope reference"
      << ", max_diff=" << k_max_diff;
}

TEST_F(TileLangRopeWrapperTest, ApplyRotaryMatchesNpuReferenceVariant128x128) {
  const std::vector<RopeTestCase> cases = {
      {.name = "baseline_16x4_hd128_rd128",
       .num_tokens = 16,
       .num_heads = 4,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 20260213},
      {.name = "large_tokens_2051x2_hd128_rd128",
       .num_tokens = 2051,
       .num_heads = 2,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 20260214},
      {.name = "tiny_1x1_hd128_rd128",
       .num_tokens = 1,
       .num_heads = 1,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 101},
      {.name = "odd_tokens_7x3_hd128_rd128",
       .num_tokens = 7,
       .num_heads = 3,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 102},
      {.name = "token_dim_64x4_hd128_rd128",
       .num_tokens = 64,
       .num_heads = 4,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 107},
      {.name = "chunk_boundary_8x5_hd128_rd128",
       .num_tokens = 8,
       .num_heads = 5,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 103},
      {.name = "cross_chunk_9x5_hd128_rd128",
       .num_tokens = 9,
       .num_heads = 5,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 104},
      {.name = "head_dim_4x64_hd128_rd128",
       .num_tokens = 4,
       .num_heads = 64,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 108},
      {.name = "medium_127x8_hd128_rd128",
       .num_tokens = 127,
       .num_heads = 8,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 105},
      {.name = "large_heads_33x16_hd128_rd128",
       .num_tokens = 33,
       .num_heads = 16,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 106},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name
                                      << ", num_tokens=" << test_case.num_tokens
                                      << ", num_heads=" << test_case.num_heads);
    run_apply_rotary_case(test_case);
  }
}

TEST_F(TileLangRopeWrapperTest, ApplyRotaryMatchesNpuReferenceVariant576x64) {
  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kFullHeadDim = 576;
  constexpr int64_t kStartDim = 512;
  constexpr int64_t kRopeDim = 64;

  const std::vector<RopeTestCase> cases = {
      {.name = "1x576_start512_rope64",
       .num_tokens = 1,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260226},
      {.name = "8x576_start512_rope64",
       .num_tokens = 8,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260227},
      {.name = "47x576_start512_rope64",
       .num_tokens = 47,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260301},
      {.name = "48x576_start512_rope64",
       .num_tokens = 48,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260302},
      {.name = "49x576_start512_rope64",
       .num_tokens = 49,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260303},
      {.name = "95x576_start512_rope64",
       .num_tokens = 95,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260304},
      {.name = "96x576_start512_rope64",
       .num_tokens = 96,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260305},
      {.name = "97x576_start512_rope64",
       .num_tokens = 97,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260306},
      {.name = "128x576_start512_rope64",
       .num_tokens = 128,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260228},
      {.name = "512x576_start512_rope64",
       .num_tokens = 512,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260307},
      {.name = "1024x576_start512_rope64",
       .num_tokens = 1024,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260308},
      {.name = "2048x576_start512_rope64",
       .num_tokens = 2048,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260225},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name
                                      << ", num_tokens=" << test_case.num_tokens
                                      << ", num_heads=" << test_case.num_heads);
    run_apply_rotary_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
