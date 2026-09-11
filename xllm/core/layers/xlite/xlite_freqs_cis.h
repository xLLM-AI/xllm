/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions or
limitations under the License.
==============================================================================*/

// Precompute rotary freqs_cis for xlite. MHA: real [cos||sin]; MLA/DSA:
// complex64.

#pragma once

#include <torch/torch.h>
#include <xlite/xlite.h>

#include <cstdint>

namespace xllm::xlite {

// DeepSeek YaRN params (from ModelArgs rope_scaling_*, MLA only).
struct YarnConfig {
  float rope_factor = 0.0f;
  float beta_fast = 0.0f;
  float beta_slow = 0.0f;
  int64_t original_seq_len = 0;
};

class XliteFreqsCis {
 public:
  static torch::Tensor Precompute(const XModelConfig& cfg,
                                  const torch::TensorOptions& opts,
                                  const YarnConfig& yarn = {}) {
    uint32_t dim = cfg.ropeHeadDim;  // MHA: headDim; MLA: qk_rope_head_dim
    uint64_t end = cfg.maxSeqLen;
    float theta = cfg.ropeTheta;
    auto inv_freq =
        1.0 / torch::pow(theta,
                         torch::arange(
                             0, static_cast<int64_t>(dim), 2, torch::kFloat32) /
                             static_cast<float>(dim));

    if (cfg.attnType == XMODEL_ATTN_MLA || cfg.attnType == XMODEL_ATTN_DSA) {
      // MLA/DSA: complex64 interleaved [c0,s0,c1,s1,...] via torch.polar +
      // YaRN.
      if (yarn.rope_factor > 0.0f && yarn.original_seq_len > 0 &&
          static_cast<int64_t>(end) > yarn.original_seq_len) {
        inv_freq = apply_yarn(inv_freq, dim, theta, yarn);
      }
      auto t = torch::arange(static_cast<int64_t>(end), torch::kFloat32);
      auto freqs = torch::outer(t, inv_freq);
      return torch::polar(torch::ones_like(freqs), freqs).to(opts.device());
    }

    // MHA: real [cos||sin].
    auto t = torch::arange(static_cast<int64_t>(end), torch::kFloat32);
    auto table = torch::outer(t, inv_freq);
    return torch::cat({table.cos(), table.sin()}, /*dim=*/-1)
        .to(opts.dtype())
        .to(opts.device());
  }

 private:
  // YaRN: freqs/factor*(1-smooth) + freqs*smooth; smooth from beta_fast/slow
  // correction.
  static torch::Tensor apply_yarn(torch::Tensor inv_freq,
                                  uint32_t dim,
                                  float theta,
                                  const YarnConfig& yarn) {
    int64_t half = dim / 2;
    auto corr_dim = [&](float num_rotations) {
      return static_cast<float>(dim) *
             std::log(static_cast<double>(yarn.original_seq_len) /
                      (num_rotations * 2.0 * M_PI)) /
             (2.0 * std::log(static_cast<double>(theta)));
    };
    int64_t low = static_cast<int64_t>(std::floor(corr_dim(yarn.beta_fast)));
    int64_t high = static_cast<int64_t>(std::ceil(corr_dim(yarn.beta_slow)));
    low = std::max(low, static_cast<int64_t>(0));
    high = std::min(high, half - 1);
    if (low == high) {
      high += 1;
    }
    auto ramp = torch::clamp(
        (torch::arange(half, torch::kFloat32) - static_cast<float>(low)) /
            static_cast<float>(high - low),
        0.0,
        1.0);
    auto smooth = 1.0 - ramp;
    return inv_freq / yarn.rope_factor * (1.0 - smooth) + inv_freq * smooth;
  }
};

}  // namespace xllm::xlite