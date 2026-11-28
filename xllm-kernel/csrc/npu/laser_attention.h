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

#include <c10/util/Logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include <cmath>
#include <limits>
#include <optional>

namespace xllm_kernel {

// Both hosts use the same input contract and padding. The caller supplies the
// ACLNN launcher for its own packaged OPP.
template <typename Launch>
torch::Tensor run_laser_attention(const torch::Tensor& q_bnsd,
                                  const torch::Tensor& k_bnsd,
                                  const torch::Tensor& v_bnsd,
                                  double scale_value,
                                  int64_t head_num,
                                  Launch launch) {
  CHECK(q_bnsd.numel() > 0);
  CHECK(k_bnsd.numel() > 0);
  CHECK(q_bnsd.dim() == 4) << "Laser attention expects BNSD tensors";
  CHECK(k_bnsd.dim() == 4);
  CHECK(v_bnsd.dim() == 4);
  CHECK(q_bnsd.device().is_privateuseone());
  CHECK(q_bnsd.device() == k_bnsd.device());
  CHECK(q_bnsd.device() == v_bnsd.device());
  const torch::ScalarType input_dtype = q_bnsd.scalar_type();
  CHECK(input_dtype == torch::kHalf || input_dtype == torch::kBFloat16);
  CHECK(input_dtype == k_bnsd.scalar_type());
  CHECK(input_dtype == v_bnsd.scalar_type());
  CHECK(q_bnsd.size(0) == k_bnsd.size(0));
  CHECK(q_bnsd.size(1) == head_num);
  CHECK(q_bnsd.size(1) == k_bnsd.size(1));
  CHECK(q_bnsd.size(3) == k_bnsd.size(3));
  CHECK(k_bnsd.sizes() == v_bnsd.sizes());
  CHECK(q_bnsd.size(3) <= 128);
  CHECK(std::isfinite(scale_value));
  CHECK(scale_value > 0.0);

  if (q_bnsd.size(2) < 2048 || q_bnsd.size(2) != k_bnsd.size(2)) {
    const auto results = at_npu::native::custom_ops::npu_fusion_attention(
        q_bnsd.contiguous(),
        k_bnsd.contiguous(),
        v_bnsd.contiguous(),
        head_num,
        "BNSD",
        /*pse=*/std::nullopt,
        /*padding_mask=*/std::nullopt,
        /*atten_mask=*/std::nullopt,
        scale_value,
        /*keep_prob=*/1.0,
        /*pre_tokens=*/std::numeric_limits<int64_t>::max(),
        /*next_tokens=*/std::numeric_limits<int64_t>::max());
    return std::get<0>(results);
  }

  const int64_t rows = q_bnsd.size(2);
  const int64_t head_dim = q_bnsd.size(3);
  const int64_t pad_rows = (256 - rows % 256) % 256;
  const int64_t pad_dim = 128 - head_dim;
  const auto align = [&](const torch::Tensor& t) {
    if (pad_rows == 0 && pad_dim == 0) {
      return t.contiguous();
    }
    return torch::constant_pad_nd(t, {0, pad_dim, 0, pad_rows}, 0).contiguous();
  };
  const auto q = align(q_bnsd);
  const auto k = align(k_bnsd);
  const auto v = align(v_bnsd);
  const auto options = q.options().dtype(torch::kFloat);
  auto output = torch::empty(q.sizes(), options);
  auto lse = torch::empty({q.size(0), head_num, q.size(2)}, options);

  // Laser interprets pre_tokens as the number of trailing keys to mask.
  const int64_t pre_tokens =
      pad_rows ? pad_rows : std::numeric_limits<int32_t>::max();
  const int64_t next_tokens = 1;
  static constexpr char layout[] = "BNSD";
  const double keep_prob = 1.0;
  const bool high_precision = true;
  const std::optional<torch::Tensor> no_mask;
  launch(q,
         k,
         v,
         no_mask,
         no_mask,
         no_mask,
         scale_value,
         head_num,
         layout,
         keep_prob,
         pre_tokens,
         next_tokens,
         high_precision,
         lse,
         output);
  return output.slice(2, 0, rows)
      .slice(3, 0, head_dim)
      .to(q_bnsd.scalar_type())
      .contiguous();
}

}  // namespace xllm_kernel
