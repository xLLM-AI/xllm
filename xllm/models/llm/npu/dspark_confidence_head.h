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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>

#include "framework/state_dict/state_dict.h"

namespace xllm::npu::model {

// DSpark ConfidenceHead: given a draft-step hidden state and the previous
// token id, produces an acceptance-prob logit. When
// confidence_head_with_markov=true (the standard config), the head is applied
// on `concat(hidden, markov_embedding[prev])` — a low-rank Markov feature
// that reuses markov_w1 (the same embedding table the MarkovHead uses).
//
// Weights (bias-less linear that xllm sees as `confidence_head.proj`):
//   proj.weight: [1, hidden_size + markov_rank]  when with_markov=true
//   proj.weight: [1, hidden_size]                 otherwise
//   proj.bias:   [1]
//
// forward returns [num_reqs] float acceptance probabilities in [0, 1] (sigmoid
// applied). The controller consumes them as per-step draft accept
// probabilities.
class DSparkConfidenceHead final {
 public:
  DSparkConfidenceHead() = default;

  void initialize(const torch::TensorOptions& options,
                  int64_t hidden_size,
                  int64_t markov_rank,
                  bool with_markov) {
    tensor_options_ = options;
    hidden_size_ = hidden_size;
    markov_rank_ = markov_rank;
    with_markov_ = with_markov;
    CHECK_GT(hidden_size_, 0) << "DSpark ConfidenceHead requires hidden_size>0";
    if (with_markov_) {
      CHECK_GT(markov_rank_, 0)
          << "DSpark ConfidenceHead with_markov requires markov_rank>0";
    }
  }

  // vLLM keys: confidence_head.proj.{weight,bias}
  void load_state_dict(const StateDict& state_dict) {
    if (hidden_size_ <= 0) {
      // Not initialized (enable_confidence_head=false); skip.
      return;
    }
    torch::Tensor w = state_dict.get_tensor("confidence_head.proj.weight");
    torch::Tensor b = state_dict.get_tensor("confidence_head.proj.bias");
    if (w.defined()) {
      proj_weight_ = w.to(tensor_options_);
    }
    if (b.defined()) {
      proj_bias_ = b.to(tensor_options_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_.defined())
        << "Failed to find " << prefix << "confidence_head.proj.weight";
    CHECK(proj_bias_.defined())
        << "Failed to find " << prefix << "confidence_head.proj.bias";
    CHECK_EQ(proj_weight_.dim(), 2)
        << "confidence_head.proj.weight must be [1, in_dim]";
    CHECK_EQ(proj_bias_.dim(), 1) << "confidence_head.proj.bias must be [1]";
    CHECK_EQ(proj_weight_.size(0), 1)
        << "confidence_head.proj.weight must output a single logit";
    const int64_t expected_in =
        with_markov_ ? hidden_size_ + markov_rank_ : hidden_size_;
    CHECK_EQ(proj_weight_.size(1), expected_in)
        << "confidence_head.proj.weight in_dim mismatch, expected "
        << expected_in << " got " << proj_weight_.size(1);
    CHECK_EQ(proj_bias_.size(0), 1)
        << "confidence_head.proj.bias size mismatch";
  }

  // hidden: [num_reqs, hidden_size]
  // markov_embed: [num_reqs, markov_rank] — from `markov_w1[prev_token_ids]`.
  //   Only consumed when `with_markov_` is true.
  // returns: [num_reqs] float32 acceptance probabilities in [0, 1].
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& markov_embed) const {
    CHECK(defined()) << "DSpark ConfidenceHead weights are not initialized";
    CHECK_EQ(hidden.dim(), 2) << "ConfidenceHead hidden must be [B, H]";
    CHECK_EQ(hidden.size(-1), hidden_size_)
        << "ConfidenceHead hidden size mismatch";
    torch::Tensor input;
    if (with_markov_) {
      CHECK(markov_embed.defined())
          << "ConfidenceHead with_markov requires markov_embed";
      CHECK_EQ(markov_embed.size(0), hidden.size(0))
          << "ConfidenceHead markov_embed batch mismatch";
      CHECK_EQ(markov_embed.size(-1), markov_rank_)
          << "ConfidenceHead markov_embed rank mismatch";
      input = torch::cat({hidden.to(proj_weight_.dtype()),
                          markov_embed.to(proj_weight_.dtype())},
                         /*dim=*/-1);
    } else {
      input = hidden.to(proj_weight_.dtype());
    }
    namespace F = torch::nn::functional;
    torch::Tensor logit = F::linear(input, proj_weight_, proj_bias_);
    const double temperature = confidence_temperature();
    if (temperature != 1.0) {
      logit = logit / temperature;
    }
    return torch::sigmoid(logit).squeeze(-1).to(torch::kFloat32);
  }

  // Batched over the whole draft block: applies the same head as `forward` to
  // all gamma steps at once, so the sample loop pays one linear+sigmoid instead
  // of gamma. Numerically identical to gamma per-step `forward` calls.
  //   hidden_all:       [B, gamma, hidden_size]
  //   markov_embed_all: [B, gamma, markov_rank]  (consumed only when
  //   with_markov_)
  // returns:            [B, gamma] float32 acceptance probabilities in [0, 1].
  torch::Tensor forward_batched(const torch::Tensor& hidden_all,
                                const torch::Tensor& markov_embed_all) const {
    CHECK(defined()) << "DSpark ConfidenceHead weights are not initialized";
    CHECK_EQ(hidden_all.dim(), 3)
        << "ConfidenceHead hidden_all must be [B, gamma, H]";
    CHECK_EQ(hidden_all.size(-1), hidden_size_)
        << "ConfidenceHead hidden size mismatch";
    const int64_t batch = hidden_all.size(0);
    const int64_t gamma = hidden_all.size(1);
    torch::Tensor input;
    if (with_markov_) {
      CHECK(markov_embed_all.defined())
          << "ConfidenceHead with_markov requires markov_embed_all";
      CHECK_EQ(markov_embed_all.dim(), 3)
          << "ConfidenceHead markov_embed_all must be [B, gamma, rank]";
      CHECK_EQ(markov_embed_all.size(0), batch)
          << "ConfidenceHead markov_embed_all batch mismatch";
      CHECK_EQ(markov_embed_all.size(1), gamma)
          << "ConfidenceHead markov_embed_all gamma mismatch";
      CHECK_EQ(markov_embed_all.size(-1), markov_rank_)
          << "ConfidenceHead markov_embed_all rank mismatch";
      input = torch::cat({hidden_all.to(proj_weight_.dtype()),
                          markov_embed_all.to(proj_weight_.dtype())},
                         /*dim=*/-1);
    } else {
      input = hidden_all.to(proj_weight_.dtype());
    }
    namespace F = torch::nn::functional;
    torch::Tensor logit =
        F::linear(input.reshape({batch * gamma, -1}), proj_weight_, proj_bias_)
            .view({batch, gamma});
    const double temperature = confidence_temperature();
    if (temperature != 1.0) {
      logit = logit / temperature;
    }
    return torch::sigmoid(logit).to(torch::kFloat32);
  }

  bool defined() const {
    return proj_weight_.defined() && proj_bias_.defined();
  }

 private:
  // Optional temperature scaling on the confidence logit before sigmoid, shared
  // by both `forward` and `forward_batched` so the two paths stay numerically
  // identical. Raw neural confidence estimates are typically overconfident (Guo
  // et al. 2017); the DSpark paper (Section 3.2.1 "Post-hoc Calibration") calls
  // for Sequential Temperature Scaling to keep the cumulative product ∏ c_i
  // aligned with the empirical acceptance rate, which higher T approximates by
  // flattening confidence toward 0.5. The scaling path is kept for that future
  // calibration, but T is fixed at 1.0 (no scaling) here; when it becomes
  // tunable it should arrive as a ModelArgs field, not a process-global knob.
  static constexpr double confidence_temperature() { return 1.0; }

  torch::Tensor proj_weight_;  // [1, hidden_size (+ markov_rank)]
  torch::Tensor proj_bias_;    // [1]
  torch::TensorOptions tensor_options_;
  int64_t hidden_size_ = 0;
  int64_t markov_rank_ = 0;
  bool with_markov_ = false;
};

}  // namespace xllm::npu::model
