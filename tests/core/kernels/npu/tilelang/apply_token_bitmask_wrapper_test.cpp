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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr float kDisallowedTokenMask = -1.0e9F;

class ApplyTokenBitmaskWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

torch::Tensor apply_reference_on_device(torch::Tensor logits,
                                        const torch::Tensor& bitmask) {
  const int64_t vocab_size = logits.size(1);
  const auto int64_options =
      torch::TensorOptions().dtype(torch::kInt64).device(logits.device());
  const torch::Tensor token_ids = torch::arange(vocab_size, int64_options);
  const torch::Tensor word_indices = torch::floor_divide(token_ids, 32);
  const torch::Tensor bit_indices = torch::remainder(token_ids, 32);
  const torch::Tensor words = bitmask.to(torch::kInt64)
                                  .bitwise_and(0xffffffffLL)
                                  .index_select(1, word_indices);
  const torch::Tensor allowed =
      torch::bitwise_and(torch::bitwise_right_shift(words, bit_indices), 1);
  const auto mask_options =
      torch::TensorOptions().dtype(logits.dtype()).device(logits.device());
  const torch::Tensor additive =
      torch::where(allowed.to(torch::kBool),
                   torch::zeros({1}, mask_options),
                   torch::full({1}, kDisallowedTokenMask, mask_options));
  logits.add_(additive);
  return logits;
}

torch::Tensor apply_cpu_reference(const torch::Tensor& logits,
                                  const torch::Tensor& bitmask) {
  const torch::Device output_device = logits.device();
  return apply_reference_on_device(logits.to(torch::kCPU),
                                   bitmask.to(torch::kCPU))
      .to(output_device);
}

torch::Tensor make_mask(int64_t rows,
                        int64_t vocab_size,
                        const std::string& pattern) {
  const int64_t num_words = (vocab_size + 31) / 32;
  auto mask = torch::zeros({rows, num_words},
                           torch::TensorOptions().dtype(torch::kInt32));
  auto accessor = mask.accessor<int32_t, 2>();
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t word = 0; word < num_words; ++word) {
      if (pattern == "all_allowed" || (pattern == "mixed" && row == 0)) {
        accessor[row][word] = -1;
      } else if (pattern == "alternating" || pattern == "mixed") {
        accessor[row][word] = static_cast<int32_t>(0xAAAAAAAAU);
      } else if (pattern == "boundaries") {
        accessor[row][word] = static_cast<int32_t>(0x80000001U);
      }
    }
  }
  return mask;
}

double measure_npu_event_ms(const std::function<void()>& fn,
                            int32_t device_id,
                            int32_t warmup_iters,
                            int32_t measure_iters) {
  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  for (int32_t iter = 0; iter < warmup_iters; ++iter) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

  aclrtEvent start_event = nullptr;
  aclrtEvent end_event = nullptr;
  CHECK_EQ(aclrtCreateEvent(&start_event), ACL_SUCCESS);
  CHECK_EQ(aclrtCreateEvent(&end_event), ACL_SUCCESS);
  CHECK_EQ(aclrtRecordEvent(start_event, stream), ACL_SUCCESS);
  for (int32_t iter = 0; iter < measure_iters; ++iter) {
    fn();
  }
  CHECK_EQ(aclrtRecordEvent(end_event, stream), ACL_SUCCESS);
  CHECK_EQ(aclrtSynchronizeEvent(end_event), ACL_SUCCESS);

  float elapsed_ms = 0.0F;
  CHECK_EQ(aclrtEventElapsedTime(&elapsed_ms, start_event, end_event),
           ACL_SUCCESS);
  CHECK_EQ(aclrtDestroyEvent(start_event), ACL_SUCCESS);
  CHECK_EQ(aclrtDestroyEvent(end_event), ACL_SUCCESS);
  return static_cast<double>(elapsed_ms) / measure_iters;
}

double measure_completed_wall_ms(const std::function<void()>& fn,
                                 int32_t device_id,
                                 int32_t warmup_iters,
                                 int32_t measure_iters) {
  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  for (int32_t iter = 0; iter < warmup_iters; ++iter) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

  const auto start = std::chrono::steady_clock::now();
  for (int32_t iter = 0; iter < measure_iters; ++iter) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  return elapsed / measure_iters;
}

void run_parity_case(int64_t rows,
                     int64_t vocab_size,
                     torch::ScalarType dtype,
                     const std::string& pattern) {
  const auto device = torch::Device("npu:0");
  torch::manual_seed(20260804 + rows + vocab_size);
  const auto options = torch::TensorOptions().dtype(dtype).device(device);
  const torch::Tensor input = torch::randn({rows, vocab_size}, options);
  const torch::Tensor bitmask = make_mask(rows, vocab_size, pattern).to(device);
  torch::Tensor expected = apply_cpu_reference(input, bitmask);
  torch::Tensor actual = input.clone();

  ASSERT_TRUE(can_apply_token_bitmask_inplace(actual, bitmask));
  apply_token_bitmask_inplace(actual, bitmask);
  ASSERT_EQ(aclrtSynchronizeStream(
                c10_npu::getCurrentNPUStream(device.index()).stream()),
            ACL_SUCCESS);

  if (!torch::equal(actual, expected)) {
    const torch::Tensor actual_head =
        actual.index({0, torch::indexing::Slice(0, 32)}).to(torch::kCPU);
    const torch::Tensor expected_head =
        expected.index({0, torch::indexing::Slice(0, 32)}).to(torch::kCPU);
    const int64_t mismatch_count =
        actual.ne(expected).sum().to(torch::kCPU).item<int64_t>();
    ADD_FAILURE() << "dtype=" << dtype << ", rows=" << rows
                  << ", vocab_size=" << vocab_size << ", pattern=" << pattern
                  << ", mismatch_count=" << mismatch_count
                  << ", actual_head=" << actual_head
                  << ", expected_head=" << expected_head;
  }
  const torch::Tensor allowed = bitmask.eq(-1).all(1);
  if (allowed.any().item<bool>()) {
    const torch::Tensor allowed_rows = torch::nonzero(allowed).view(-1);
    EXPECT_TRUE(torch::equal(actual.index_select(0, allowed_rows),
                             input.index_select(0, allowed_rows)));
  }
}

TEST_F(ApplyTokenBitmaskWrapperTest, MatchesReferenceAcrossDtypesAndPatterns) {
  const std::vector<torch::ScalarType> dtypes = {
      torch::kFloat16, torch::kBFloat16, torch::kFloat32};
  for (const torch::ScalarType dtype : dtypes) {
    run_parity_case(1, 32, dtype, "all_allowed");
    run_parity_case(1, 32, dtype, "all_disallowed");
    run_parity_case(1, 32, dtype, "alternating");
    run_parity_case(1, 32, dtype, "boundaries");
    run_parity_case(4, 150464, dtype, "mixed");
  }
}

TEST_F(ApplyTokenBitmaskWrapperTest, RejectsInputsThatRequireFallback) {
  const auto device = torch::Device("npu:0");
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  for (const int64_t vocab_size : {31, 33}) {
    torch::Tensor logits = torch::randn({1, vocab_size}, options);
    torch::Tensor bitmask = make_mask(1, vocab_size, "boundaries").to(device);
    EXPECT_FALSE(can_apply_token_bitmask_inplace(logits, bitmask));

    const torch::Tensor expected = apply_cpu_reference(logits, bitmask);
    torch::Tensor actual = logits.clone();
    ::xllm::apply_token_bitmask_inplace(actual, bitmask);
    ASSERT_EQ(aclrtSynchronizeStream(
                  c10_npu::getCurrentNPUStream(device.index()).stream()),
              ACL_SUCCESS);
    EXPECT_TRUE(torch::equal(actual, expected));
  }

  torch::Tensor storage = torch::zeros({1, 64}, options);
  torch::Tensor non_contiguous = storage.slice(1, 0, 64, 2);
  torch::Tensor bitmask = make_mask(1, 32, "all_allowed").to(device);
  ASSERT_FALSE(non_contiguous.is_contiguous());
  EXPECT_FALSE(can_apply_token_bitmask_inplace(non_contiguous, bitmask));
}

TEST_F(ApplyTokenBitmaskWrapperTest, ReportsCompletedDeviceTiming) {
  constexpr int64_t kRows = 4;
  constexpr int64_t kVocabSize = 150464;
  const auto device = torch::Device("npu:0");
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::Tensor input = torch::randn({kRows, kVocabSize}, options);
  torch::Tensor bitmask =
      make_mask(kRows, kVocabSize, "alternating").to(device);
  torch::Tensor reference_logits = input.clone();
  torch::Tensor fused_logits = input.clone();

  const auto reference_fn = [&]() {
    reference_logits.copy_(input);
    reference_logits = apply_reference_on_device(reference_logits, bitmask);
  };
  const auto fused_fn = [&]() {
    fused_logits.copy_(input);
    apply_token_bitmask_inplace(fused_logits, bitmask);
  };

  const double reference_completed_ms =
      measure_completed_wall_ms(reference_fn, device.index(), 2, 10);
  const double fused_completed_ms =
      measure_completed_wall_ms(fused_fn, device.index(), 2, 10);
  const double fused_device_ms =
      measure_npu_event_ms([&]() { fused_fn(); }, device.index(), 2, 10);

  std::cout << "[apply_token_bitmask_wrapper_test] rows=" << kRows
            << ", vocab=" << kVocabSize
            << ", reference_completed_ms=" << reference_completed_ms
            << ", fused_completed_ms=" << fused_completed_ms
            << ", fused_device_ms=" << fused_device_ms << ", completed_speedup="
            << reference_completed_ms / fused_completed_ms << "x" << std::endl;
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
