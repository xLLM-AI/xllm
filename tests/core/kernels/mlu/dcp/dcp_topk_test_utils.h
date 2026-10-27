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

#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>

namespace xllm::test {

inline constexpr int32_t kSentinel = 123456789;
using TopkResult = std::pair<torch::Tensor, torch::Tensor>;

// The oracle intentionally uses scalar integer arithmetic, not the device scan.
inline TopkResult local_reference(const torch::Tensor& slots,
                                  const torch::Tensor& lengths,
                                  int32_t rank,
                                  int32_t size,
                                  int32_t physical_block_size) {
  const int64_t q = slots.size(0);
  const int64_t k = slots.size(1);
  auto output = torch::zeros({q, k}, torch::kInt32);
  auto counts = torch::zeros({q}, torch::kInt32);
  const auto src = slots.accessor<int32_t, 2>();
  const auto lens = lengths.accessor<int32_t, 1>();
  auto dst = output.accessor<int32_t, 2>();
  auto cnt = counts.accessor<int32_t, 1>();
  for (int64_t row = 0; row < q; ++row) {
    for (int64_t col = 0; col < lens[row]; ++col) {
      const int64_t slot = src[row][col];
      if (slot < 0 || (slot / physical_block_size) % size != rank) {
        continue;
      }
      dst[row][cnt[row]++] = static_cast<int32_t>(
          (slot / physical_block_size / size) * physical_block_size +
          slot % physical_block_size);
    }
  }
  return {output, counts};
}

// Sentinel guards surround even dense outputs, preserving their dispatch path.
class GuardedTopk final {
 public:
  explicit GuardedTopk(int64_t q,
                       int64_t k,
                       const torch::Device& device,
                       bool strided = false,
                       bool strided_lens = false)
      : strided_(strided), strided_lens_(strided_lens) {
    const auto opts =
        torch::TensorOptions().dtype(torch::kInt32).device(device);
    storage_ = torch::full({q * k * (strided ? 2 : 1) + 2}, kSentinel, opts);
    output = storage_.slice(0, 1, -1, strided ? 2 : 1).view({q, k});
    lens_storage_ =
        torch::full({q * (strided_lens ? 2 : 1) + 2}, kSentinel, opts);
    lengths = lens_storage_.slice(0, 1, -1, strided_lens ? 2 : 1);
  }

  void check(const TopkResult& expected) const {
    EXPECT_TRUE(torch::equal(output.cpu(), expected.first));
    EXPECT_TRUE(torch::equal(lengths.cpu(), expected.second));
    const auto storage = storage_.cpu();
    const auto lens = lens_storage_.cpu();
    EXPECT_EQ(storage[0].item<int32_t>(), kSentinel);
    EXPECT_EQ(storage[-1].item<int32_t>(), kSentinel);
    EXPECT_EQ(lens[0].item<int32_t>(), kSentinel);
    EXPECT_EQ(lens[-1].item<int32_t>(), kSentinel);
    if (strided_) {
      EXPECT_TRUE((storage.slice(0, 2, -1, 2) == kSentinel).all().item<bool>());
    }
    if (strided_lens_) {
      EXPECT_TRUE((lens.slice(0, 2, -1, 2) == kSentinel).all().item<bool>());
    }
  }

  torch::Tensor output;
  torch::Tensor lengths;

 private:
  bool strided_;
  bool strided_lens_;
  torch::Tensor storage_;
  torch::Tensor lens_storage_;
};

}  // namespace xllm::test
