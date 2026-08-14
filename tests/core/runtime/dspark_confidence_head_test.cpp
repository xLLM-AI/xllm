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

// Numerical parity between DSparkConfidenceHead::forward_batched (O11) and the
// original per-step forward it replaced. The batched path hoists the whole
// gamma-step ConfidenceHead out of the sample loop; it must produce bit-close
// results to gamma independent per-step calls, otherwise adaptive pruning would
// silently change behavior.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "models/llm/npu/qwen3_dspark.h"

namespace xllm::npu::model {
namespace {

constexpr int64_t kHidden = 64;
constexpr int64_t kMarkovRank = 16;
constexpr int64_t kVocab = 128;

torch::TensorOptions f32_cpu() {
  return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
}

// Build a ConfidenceHead with random weights via its public load_state_dict.
DSparkConfidenceHead make_confidence_head(bool with_markov, int64_t seed) {
  torch::manual_seed(seed);
  DSparkConfidenceHead head;
  head.initialize(f32_cpu(), kHidden, kMarkovRank, with_markov);
  const int64_t in_dim = with_markov ? kHidden + kMarkovRank : kHidden;
  std::unordered_map<std::string, torch::Tensor> w;
  w["confidence_head.proj.weight"] = torch::randn({1, in_dim}, f32_cpu());
  w["confidence_head.proj.bias"] = torch::randn({1}, f32_cpu());
  StateDict sd(std::move(w));
  head.load_state_dict(sd);
  return head;
}

DSparkMarkovHead make_markov_head(int64_t seed) {
  torch::manual_seed(seed + 999);
  DSparkMarkovHead head;
  head.initialize(f32_cpu(), kMarkovRank);
  std::unordered_map<std::string, torch::Tensor> w;
  // markov_w1: [vocab, rank]; markov_w2: [draft_vocab, rank]. Only markov_w1 is
  // exercised by confidence (markov_embed); w2 just needs to satisfy load.
  w["markov_head.markov_w1.weight"] =
      torch::randn({kVocab, kMarkovRank}, f32_cpu());
  w["markov_head.markov_w2.weight"] =
      torch::randn({kVocab, kMarkovRank}, f32_cpu());
  StateDict sd(std::move(w));
  head.load_state_dict(sd);
  return head;
}

// Reference: gamma independent per-step forward() calls, stacked to [B, gamma].
torch::Tensor per_step_reference(const DSparkConfidenceHead& head,
                                 const DSparkMarkovHead& markov,
                                 const torch::Tensor& hidden_all,   // [B, g, H]
                                 const torch::Tensor& prev_matrix,  // [B, g]
                                 bool with_markov) {
  const int64_t gamma = hidden_all.size(1);
  std::vector<torch::Tensor> cols;
  cols.reserve(gamma);
  for (int64_t k = 0; k < gamma; ++k) {
    torch::Tensor step_hidden = hidden_all.select(/*dim=*/1, k);  // [B, H]
    torch::Tensor prev_k = prev_matrix.select(/*dim=*/1, k);      // [B]
    torch::Tensor embed =
        with_markov ? markov.markov_embed(prev_k) : torch::Tensor();
    cols.push_back(head.forward(step_hidden, embed));  // [B]
  }
  return torch::stack(cols, /*dim=*/1);  // [B, gamma]
}

void check_parity(bool with_markov, int64_t batch, int64_t gamma) {
  DSparkConfidenceHead head = make_confidence_head(with_markov, /*seed=*/7);
  DSparkMarkovHead markov = make_markov_head(/*seed=*/7);

  torch::manual_seed(1234);
  torch::Tensor hidden_all = torch::randn({batch, gamma, kHidden}, f32_cpu());
  torch::Tensor prev_matrix = torch::randint(
      0, kVocab, {batch, gamma}, torch::TensorOptions().dtype(torch::kLong));

  torch::Tensor ref =
      per_step_reference(head, markov, hidden_all, prev_matrix, with_markov);

  torch::Tensor markov_embed_all =
      with_markov ? markov.markov_embed(prev_matrix) : torch::Tensor();
  torch::Tensor batched = head.forward_batched(hidden_all, markov_embed_all);

  ASSERT_EQ(batched.sizes(), ref.sizes());
  double max_abs = (batched - ref).abs().max().item<double>();
  EXPECT_LT(max_abs, 1e-5) << "with_markov=" << with_markov
                           << " batch=" << batch << " gamma=" << gamma
                           << " max_abs=" << max_abs;
}

}  // namespace

TEST(DSparkConfidenceHeadParity, WithMarkov_Gamma7) {
  check_parity(/*with_markov=*/true, /*batch=*/5, /*gamma=*/7);
}

TEST(DSparkConfidenceHeadParity, WithMarkov_Gamma16) {
  check_parity(/*with_markov=*/true, /*batch=*/3, /*gamma=*/16);
}

TEST(DSparkConfidenceHeadParity, WithMarkov_Gamma1) {
  check_parity(/*with_markov=*/true, /*batch=*/4, /*gamma=*/1);
}

TEST(DSparkConfidenceHeadParity, NoMarkov_Gamma16) {
  check_parity(/*with_markov=*/false, /*batch=*/3, /*gamma=*/16);
}

// forward and forward_batched apply the same fixed confidence_temperature(), so
// both paths scale identically and parity holds. An extra batch/gamma shape for
// coverage beyond the cases above.
TEST(DSparkConfidenceHeadParity, WithMarkov_Batch6Gamma8) {
  check_parity(/*with_markov=*/true, /*batch=*/6, /*gamma=*/8);
}

}  // namespace xllm::npu::model
