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
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "framework/sampling/beam_searcher.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace test {
namespace {

constexpr int32_t kDeviceId = 0;

struct BeamCase {
  std::string name;
  int64_t num_group;
  int64_t beam_width;
  int64_t topk;
};

std::string PrintBeamCase(const testing::TestParamInfo<BeamCase>& info) {
  return info.param.name;
}

// Pure-CPU oracle that mirrors the device selection rule:
// per group, combine logprobs + top_logprobs over (beam, topk) candidates and
// keep the `beam_width` largest by descending value (no ties in the inputs).
struct Reference {
  std::vector<int32_t> src_seq_idxes;
  std::vector<int32_t> out_tokens;
  std::vector<float> out_logprobs;
};

Reference compute_reference(const torch::Tensor& logprobs,
                            const torch::Tensor& top_tokens,
                            const torch::Tensor& top_logprobs,
                            int64_t num_group,
                            int64_t beam_width,
                            int64_t topk) {
  auto lp = logprobs.accessor<float, 1>();
  auto tt = top_tokens.accessor<int64_t, 2>();
  auto tlp = top_logprobs.accessor<float, 2>();

  Reference ref;
  ref.src_seq_idxes.reserve(num_group * beam_width);
  ref.out_tokens.reserve(num_group * beam_width);
  ref.out_logprobs.reserve(num_group * beam_width);

  for (int64_t g = 0; g < num_group; ++g) {
    struct Candidate {
      float value;
      int64_t src_local;
      int64_t col;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(beam_width * topk);
    for (int64_t b = 0; b < beam_width; ++b) {
      int64_t seq = g * beam_width + b;
      for (int64_t c = 0; c < topk; ++c) {
        float value = lp[seq] + tlp[seq][c];
        candidates.push_back({value, b, c});
      }
    }
    std::stable_sort(candidates.begin(),
                     candidates.end(),
                     [](const Candidate& a, const Candidate& b) {
                       return a.value > b.value;
                     });
    for (int64_t k = 0; k < beam_width; ++k) {
      const Candidate& cand = candidates[k];
      int64_t src_seq = g * beam_width + cand.src_local;
      ref.src_seq_idxes.push_back(static_cast<int32_t>(src_seq));
      ref.out_tokens.push_back(static_cast<int32_t>(tt[src_seq][cand.col]));
      ref.out_logprobs.push_back(cand.value);
    }
  }
  return ref;
}

class BeamSearchMluTest : public ::testing::TestWithParam<BeamCase> {
 protected:
  void SetUp() override {
    if (torch_mlu::device_count() == 0) {
      GTEST_SKIP() << "No MLU device available";
    }
    device_ = torch::Device(torch::kPrivateUse1, kDeviceId);
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

TEST_P(BeamSearchMluTest, MatchesCpuReference) {
  const auto& c = GetParam();
  const int64_t num_seq = c.num_group * c.beam_width;

  // Build no-tie inputs: every combined candidate value is distinct and
  // exactly representable in float32.
  auto logprobs = torch::empty({num_seq}, torch::kFloat32);
  auto top_logprobs = torch::empty({num_seq, c.topk}, torch::kFloat32);
  auto top_tokens = torch::empty({num_seq, c.topk}, torch::kInt64);
  auto lp = logprobs.accessor<float, 1>();
  auto tlp = top_logprobs.accessor<float, 2>();
  auto tt = top_tokens.accessor<int64_t, 2>();
  for (int64_t s = 0; s < num_seq; ++s) {
    lp[s] = static_cast<float>(s + 1) * 2.0f;
    for (int64_t col = 0; col < c.topk; ++col) {
      tlp[s][col] = -static_cast<float>(s * c.topk + col + 1) * 0.125f;
      tt[s][col] = s * 100 + col + 1;
    }
  }

  Reference ref = compute_reference(
      logprobs, top_tokens, top_logprobs, c.num_group, c.beam_width, c.topk);

  auto out = beam_search(logprobs.to(device_),
                         top_tokens.to(device_),
                         top_logprobs.to(device_),
                         c.beam_width);

  auto src_cpu = out.src_seq_idxes.cpu();
  auto tok_cpu = out.out_tokens.cpu();
  auto prob_cpu = out.out_logprobs.cpu();

  ASSERT_EQ(src_cpu.dtype(), torch::kInt32);
  ASSERT_EQ(tok_cpu.dtype(), torch::kInt32);
  ASSERT_EQ(prob_cpu.dtype(), torch::kFloat32);
  ASSERT_EQ(src_cpu.numel(), num_seq);
  ASSERT_EQ(tok_cpu.numel(), num_seq);
  ASSERT_EQ(prob_cpu.numel(), num_seq);

  auto src_acc = src_cpu.accessor<int32_t, 1>();
  auto tok_acc = tok_cpu.accessor<int32_t, 1>();
  auto prob_acc = prob_cpu.accessor<float, 1>();
  for (int64_t i = 0; i < num_seq; ++i) {
    EXPECT_EQ(src_acc[i], ref.src_seq_idxes[i]) << "src_seq_idx at " << i;
    EXPECT_EQ(tok_acc[i], ref.out_tokens[i]) << "out_token at " << i;
    EXPECT_FLOAT_EQ(prob_acc[i], ref.out_logprobs[i]) << "out_logprob at " << i;
  }
}

std::vector<BeamCase> GenerateCases() {
  return {
      {"SingleGroup_Beam2_Topk2", 1, 2, 2},
      {"SingleGroup_Beam4_Topk4", 1, 4, 4},
      {"MultiGroup_Beam2_Topk2", 3, 2, 2},
      {"MultiGroup_Beam4_Topk4", 2, 4, 4},
  };
}

INSTANTIATE_TEST_SUITE_P(BeamSearchMlu,
                         BeamSearchMluTest,
                         testing::ValuesIn(GenerateCases()),
                         PrintBeamCase);

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::mlu
