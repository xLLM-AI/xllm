/* Copyright 2026 The xLLM Authors.

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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor fused_adalayer_norm(const torch::Tensor& input,
                                  const torch::Tensor& scale,
                                  const torch::Tensor& shift,
                                  std::optional<torch::Tensor> weight,
                                  std::optional<torch::Tensor> bias,
                                  double eps) {
  CHECK(input.dim() == 3) << "Input tensor must be 3D [B, S, H].";
  CHECK(input.numel() > 0) << "Input tensor should not be empty.";
  CHECK(scale.dim() == 2 || scale.dim() == 3)
      << "Scale tensor dim must be 2 or 3.";
  CHECK(shift.dim() == 2 || shift.dim() == 3)
      << "Shift tensor dim must be 2 or 3.";
  CHECK(scale.size(-1) == input.size(-1))
      << "Scale last dim must equal input last dim.";
  CHECK(shift.size(-1) == input.size(-1))
      << "Shift last dim must equal input last dim.";
  if (weight.has_value()) {
    CHECK(weight.value().dim() == 1 && weight.value().size(0) == input.size(-1))
        << "Weight dim must equal input last dim.";
  }
  if (bias.has_value()) {
    CHECK(bias.value().dim() == 1 && bias.value().size(0) == input.size(-1))
        << "Bias dim must equal input last dim.";
  }

  // aclnnAdaLayerNorm only supports broadcast modulation where scale/shift are
  // [B, H] or [B, 1, H] (a single modulation vector broadcast over the
  // sequence dimension). A genuine token-wise modulation [B, S, H] with S > 1
  // is not directly supported. In that case, mirror the mindiesd reference:
  // fold the sequence dimension into the batch so each token becomes its own
  // "row" -- input -> [B*S, 1, H], scale/shift -> [B*S, H] -- run the op, then
  // restore the original [B, S, H] shape.
  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.size(1);
  const int64_t hidden_size = input.size(2);

  auto is_tokenwise = [&](const torch::Tensor& t) {
    return t.dim() == 3 && t.size(1) == seq_len && seq_len != 1;
  };
  const bool tokenwise = is_tokenwise(scale) || is_tokenwise(shift);

  torch::Tensor x_arg = input;
  // aclnnAdaLayerNorm's 2D path is faster than its 3D broadcast path.
  // Squeeze [B, 1, H] -> [B, H] to use the faster 2D code path.
  auto normalize_modulation = [](const torch::Tensor& t) {
    if (t.dim() == 3 && t.size(1) == 1) {
      return t.squeeze(1);  // [B, 1, H] -> [B, H]
    }
    return t;
  };
  torch::Tensor scale_arg = normalize_modulation(scale);
  torch::Tensor shift_arg = normalize_modulation(shift);

  if (tokenwise) {
    auto expand_modulation = [&](const torch::Tensor& t) {
      torch::Tensor expanded;
      if (t.dim() == 2) {
        expanded = t.unsqueeze(1).expand({batch_size, seq_len, hidden_size});
      } else if (t.size(1) == 1) {
        expanded = t.expand({batch_size, seq_len, hidden_size});
      } else {
        expanded = t;
      }
      return expanded.reshape({batch_size * seq_len, hidden_size});
    };
    x_arg = input.reshape({batch_size * seq_len, 1, hidden_size});
    scale_arg = expand_modulation(scale);
    shift_arg = expand_modulation(shift);
  }

  auto output = torch::empty(x_arg.sizes().vec(), x_arg.options());

  EXEC_NPU_CMD(aclnnAdaLayerNorm,
               x_arg,
               scale_arg,
               shift_arg,
               weight,
               bias,
               eps,
               output);

  if (tokenwise) {
    output = output.reshape({batch_size, seq_len, hidden_size});
  }

  return output;
}

}  // namespace xllm::kernel::npu
