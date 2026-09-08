/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {

torch::Tensor causal_conv1d_update(
    torch::Tensor x,
    torch::Tensor conv_state,
    torch::Tensor weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& query_start_loc,
    int32_t max_query_len,
    bool activation,
    const std::optional<torch::Tensor>& initial_state_idx,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_mode_opt) {
  const bool has_silu = activation;
  const int32_t dim = static_cast<int32_t>(x.size(-1));

  auto bias_work = bias.has_value() && bias.value().defined()
                       ? bias.value()
                       : torch::zeros({dim}, x.options());

  auto cu_seqlens =
      query_start_loc.has_value()
          ? query_start_loc.value().to(torch::kInt32)
          : torch::arange(
                0,
                x.size(0) + 1,
                std::max(max_query_len, int32_t{1}),
                torch::TensorOptions().dtype(torch::kInt32).device(x.device()));

  int64_t batch = cu_seqlens.size(0) - 1;
  if (batch <= 0) {
    return x;
  }

  auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(x.device());

  torch::Tensor init_indices;
  torch::Tensor current_indices;
  if (conv_state_indices.has_value()) {
    auto ci = conv_state_indices.value().to(torch::kInt32);
    if (ci.dim() == 1) {
      init_indices = ci;
      current_indices = ci;
    } else {
      auto ci_0 = ci.select(1, 0);
      auto ci_1 = ci.select(1, 1);
      if (initial_state_idx.has_value()) {
        auto isi = initial_state_idx.value().to(torch::kInt32);
        init_indices = torch::where(isi == 0, ci_0, ci_1);
      } else {
        init_indices = ci_0;
      }
      if (block_idx_last_scheduled_token.has_value()) {
        auto bilt = block_idx_last_scheduled_token.value().to(torch::kInt32);
        current_indices = torch::where(bilt == 0, ci_0, ci_1);
      } else {
        current_indices = ci_0;
      }
    }
  } else {
    init_indices = torch::arange(batch, i32_opts);
    current_indices = init_indices;
  }

  torch::Tensor initial_state_mode;
  if (initial_state_mode_opt.has_value()) {
    initial_state_mode = initial_state_mode_opt.value().to(torch::kInt32);
  } else {
    initial_state_mode = torch::ones({batch}, i32_opts);
  }

  const bool is_3d = (x.dim() == 3);
  auto x_flat = is_3d ? x.reshape({-1, dim}) : x;

  if (has_causal_conv1d_decode_specialization(batch, dim, has_silu)) {
    auto conv_state_nonconst = conv_state;
    auto y = causal_conv1d_decode(
        /*conv_state=*/conv_state_nonconst,
        /*x=*/x_flat,
        /*weight=*/weight,
        /*bias=*/bias_work,
        /*init_indices=*/init_indices,
        /*current_indices=*/current_indices,
        /*initial_state_mode=*/initial_state_mode,
        /*has_silu=*/has_silu);

    if (is_3d) {
      y = y.view(x.sizes());
    }
    return y;
  }

  // Fallback: per-batch loop using causal_conv1d (batch=1 kernel, fp16).
  auto original_dtype = x.scalar_type();
  bool need_cast = (original_dtype != torch::kFloat16);

  auto x_fp16 = need_cast ? x_flat.to(torch::kFloat16) : x_flat;
  auto weight_fp16 = need_cast ? weight.to(torch::kFloat16) : weight;
  auto conv_state_fp16 =
      need_cast ? conv_state.to(torch::kFloat16).clone() : conv_state.clone();
  auto bias_fp16 = need_cast ? bias_work.to(torch::kFloat16) : bias_work;

  auto y_fp16 = torch::empty({x_flat.size(0), dim}, x_fp16.options());
  auto cu_seqlens_cpu = cu_seqlens.to(torch::kCPU);
  const int32_t* cu_ptr = cu_seqlens_cpu.data_ptr<int32_t>();

  for (int64_t b = 0; b < batch; ++b) {
    int32_t seq_start_b = cu_ptr[b];
    int32_t seq_end_b = cu_ptr[b + 1];
    int32_t sb_len = seq_end_b - seq_start_b;
    if (sb_len <= 0) {
      continue;
    }

    auto x_b = x_fp16.slice(0, seq_start_b, seq_end_b);
    auto init_b = init_indices.slice(0, b, b + 1);
    auto curr_b = current_indices.slice(0, b, b + 1);
    auto ism_b = initial_state_mode.slice(0, b, b + 1);

    auto cu_b = torch::tensor(
        {0, sb_len},
        torch::TensorOptions().dtype(torch::kInt32).device(x.device()));

    auto y_b = causal_conv1d(conv_state_fp16,
                             x_b,
                             weight_fp16,
                             bias_fp16,
                             cu_b,
                             init_b,
                             curr_b,
                             ism_b,
                             has_silu);

    y_fp16.slice(0, seq_start_b, seq_end_b).copy_(y_b);
  }

  if (need_cast) {
    conv_state.copy_(conv_state_fp16.to(original_dtype));
  } else {
    conv_state.copy_(conv_state_fp16);
  }
  auto y = need_cast ? y_fp16.to(original_dtype) : y_fp16;

  if (is_3d) {
    y = y.view(x.sizes());
  }
  return y;
}

}  // namespace xllm::kernel::npu::tilelang
