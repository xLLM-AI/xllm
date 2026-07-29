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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>
#include <string>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_CAUSAL_CONV1D_REGISTRY_INC
#error "XLLM_TL_CAUSAL_CONV1D_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_CAUSAL_CONV1D_REGISTRY_INC

CausalConv1dSpecialization build_runtime_specialization(
    const torch::Tensor& x,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& weight_t,
    int64_t original_dim) {
  CHECK_EQ(x.dim(), 2) << "TileLang causal_conv1d: x must be 2D [T, D]";
  CHECK_EQ(cu_seqlens.dim(), 1)
      << "TileLang causal_conv1d: cu_seqlens must be 1D";
  CHECK_GE(weight_t.dim(), 2)
      << "TileLang causal_conv1d: weight_t must be >=2D [width, dim]";
  CHECK_GE(cu_seqlens.size(0), 2)
      << "TileLang causal_conv1d: cu_seqlens must have at least 2 elements";

  const int32_t batch_size = static_cast<int32_t>(cu_seqlens.size(0) - 1);
  const int32_t dim = static_cast<int32_t>(original_dim);
  const int32_t width = static_cast<int32_t>(weight_t.size(0));

  CHECK_GT(batch_size, 0) << "TileLang causal_conv1d: batch_size must be > 0";
  CHECK_GT(dim, 0) << "TileLang causal_conv1d: dim must be > 0";
  CHECK_GT(width, 0) << "TileLang causal_conv1d: width must be > 0";

  CHECK_LE(batch_size,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang causal_conv1d: batch_size exceeds int range";
  CHECK_LE(dim, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang causal_conv1d: dim exceeds int range";

  return make_causal_conv1d_specialization(
      CausalConv1dBatchSize{batch_size},
      CausalConv1dDim{dim},
      CausalConv1dWidth{width},
      CausalConv1dHasSilu{0},
      CausalConv1dDType{to_tilelang_dtype(x.scalar_type())});
}

void check_supported(const torch::Tensor& x,
                     torch::Tensor& conv_state_t,
                     const torch::Tensor& weight_t,
                     const torch::Tensor& bias,
                     const torch::Tensor& cu_seqlens,
                     const torch::Tensor& init_indices,
                     const torch::Tensor& current_indices,
                     const torch::Tensor& initial_state_mode) {
  CHECK(x.defined()) << "TileLang causal_conv1d: x must be defined";
  CHECK(conv_state_t.defined())
      << "TileLang causal_conv1d: conv_state must be defined";
  CHECK(weight_t.defined()) << "TileLang causal_conv1d: weight must be defined";
  CHECK(bias.defined()) << "TileLang causal_conv1d: bias must be defined";
  CHECK(cu_seqlens.defined())
      << "TileLang causal_conv1d: cu_seqlens must be defined";
  CHECK(init_indices.defined())
      << "TileLang causal_conv1d: init_indices must be defined";
  CHECK(current_indices.defined())
      << "TileLang causal_conv1d: current_indices must be defined";
  CHECK(initial_state_mode.defined())
      << "TileLang causal_conv1d: initial_state_mode must be defined";

  CHECK(x.scalar_type() == c10::ScalarType::Half)
      << "TileLang causal_conv1d: only float16 is supported, got "
      << x.scalar_type();

  CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
        conv_state_t.device().type() == c10::DeviceType::PrivateUse1 &&
        weight_t.device().type() == c10::DeviceType::PrivateUse1 &&
        bias.device().type() == c10::DeviceType::PrivateUse1 &&
        cu_seqlens.device().type() == c10::DeviceType::PrivateUse1 &&
        init_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        current_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        initial_state_mode.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang causal_conv1d: all tensors must be on NPU";

  CHECK_EQ(x.dim(), 2) << "TileLang causal_conv1d: x must be 2D [T, D]";
  CHECK_EQ(conv_state_t.dim(), 3)
      << "TileLang causal_conv1d: conv_state must be 3D [C, state_len, D]";
  CHECK_GE(weight_t.dim(), 2) << "TileLang causal_conv1d: weight must be >=2D";
  CHECK_EQ(bias.dim(), 1) << "TileLang causal_conv1d: bias must be 1D [D]";
  CHECK_EQ(cu_seqlens.dim(), 1)
      << "TileLang causal_conv1d: cu_seqlens must be 1D [B+1]";
  CHECK_EQ(init_indices.dim(), 1)
      << "TileLang causal_conv1d: init_indices must be 1D [B]";
  CHECK_EQ(current_indices.dim(), 1)
      << "TileLang causal_conv1d: current_indices must be 1D [B]";
  CHECK_EQ(initial_state_mode.dim(), 1)
      << "TileLang causal_conv1d: initial_state_mode must be 1D [B]";

  const int64_t dim = x.size(1);
  const int64_t batch = cu_seqlens.size(0) - 1;

  CHECK_EQ(conv_state_t.size(2), dim)
      << "TileLang causal_conv1d: conv_state dim mismatch";
  CHECK_EQ(bias.size(0), dim) << "TileLang causal_conv1d: bias dim mismatch";
  CHECK_EQ(init_indices.size(0), batch)
      << "TileLang causal_conv1d: init_indices batch mismatch";
  CHECK_EQ(current_indices.size(0), batch)
      << "TileLang causal_conv1d: current_indices batch mismatch";
  CHECK_EQ(initial_state_mode.size(0), batch)
      << "TileLang causal_conv1d: initial_state_mode batch mismatch";

  CHECK_EQ(x.dtype(), conv_state_t.dtype())
      << "TileLang causal_conv1d: x/conv_state dtype mismatch";
  CHECK_EQ(x.dtype(), weight_t.dtype())
      << "TileLang causal_conv1d: x/weight dtype mismatch";
  CHECK_EQ(x.dtype(), bias.dtype())
      << "TileLang causal_conv1d: x/bias dtype mismatch";

  CHECK_EQ(cu_seqlens.dtype(), torch::kInt32)
      << "TileLang causal_conv1d: cu_seqlens must be int32";
  CHECK_EQ(init_indices.dtype(), torch::kInt32)
      << "TileLang causal_conv1d: init_indices must be int32";
  CHECK_EQ(current_indices.dtype(), torch::kInt32)
      << "TileLang causal_conv1d: current_indices must be int32";
  CHECK_EQ(initial_state_mode.dtype(), torch::kInt32)
      << "TileLang causal_conv1d: initial_state_mode must be int32";

  CHECK(x.is_contiguous()) << "TileLang causal_conv1d: x must be contiguous";
  CHECK(conv_state_t.is_contiguous())
      << "TileLang causal_conv1d: conv_state must be contiguous";
  CHECK(weight_t.is_contiguous())
      << "TileLang causal_conv1d: weight must be contiguous";
  CHECK(bias.is_contiguous())
      << "TileLang causal_conv1d: bias must be contiguous";
  CHECK(cu_seqlens.is_contiguous())
      << "TileLang causal_conv1d: cu_seqlens must be contiguous";
  CHECK(init_indices.is_contiguous())
      << "TileLang causal_conv1d: init_indices must be contiguous";
  CHECK(current_indices.is_contiguous())
      << "TileLang causal_conv1d: current_indices must be contiguous";
  CHECK(initial_state_mode.is_contiguous())
      << "TileLang causal_conv1d: initial_state_mode must be contiguous";

  CHECK_GE(conv_state_t.size(1), weight_t.size(0) - 1)
      << "TileLang causal_conv1d: state_len must be >= width-1";
}

torch::Tensor _pad_last_dim(const torch::Tensor& t, int64_t padded_dim) {
  int64_t dim = t.size(-1);
  if (padded_dim <= dim) {
    return t;
  }
  int64_t pad_size = padded_dim - dim;
  auto opts = t.options();
  auto shape = t.sizes();
  std::vector<int64_t> new_shape(shape.begin(), shape.end());
  new_shape.back() = padded_dim;
  auto padded = torch::zeros(new_shape, opts);
  padded.slice(-1, 0, dim).copy_(t);
  return padded.contiguous();
}

int64_t _compute_padded_dim(int64_t dim, int64_t vec_core_num) {
  int64_t block_dim = (dim + vec_core_num - 1) / vec_core_num;
  return block_dim * vec_core_num;
}

void run_tilelang_causal_conv1d_once(const torch::Tensor& x,
                                     torch::Tensor& conv_state_t,
                                     const torch::Tensor& weight_t,
                                     const torch::Tensor& bias,
                                     const torch::Tensor& cu_seqlens,
                                     const torch::Tensor& init_indices,
                                     const torch::Tensor& current_indices,
                                     const torch::Tensor& initial_state_mode,
                                     torch::Tensor& y,
                                     int64_t padded_dim,
                                     int64_t original_dim) {
  const auto specialization =
      build_runtime_specialization(x, cu_seqlens, weight_t, original_dim);
  const auto* entry = find_causal_conv1d_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang causal_conv1d: no compiled variant. Available variants: "
      << available_causal_conv1d_variant_keys();

  const int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(x.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(weight_t.data_ptr())),
      reinterpret_cast<uint8_t*>(conv_state_t.data_ptr()),
      reinterpret_cast<uint8_t*>(const_cast<void*>(init_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(current_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(cu_seqlens.data_ptr())),
      reinterpret_cast<uint8_t*>(
          const_cast<void*>(initial_state_mode.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(bias.data_ptr())),
      reinterpret_cast<uint8_t*>(y.data_ptr()),
      static_cast<int64_t>(x.size(0)),
      static_cast<int64_t>(conv_state_t.size(0)),
      static_cast<int64_t>(conv_state_t.size(1)),
      stream);
}

}  // namespace

torch::Tensor causal_conv1d(torch::Tensor& conv_state,
                            const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& bias,
                            const torch::Tensor& cu_seqlens,
                            const torch::Tensor& init_indices,
                            const torch::Tensor& current_indices,
                            const torch::Tensor& initial_state_mode,
                            bool has_silu) {
  check_supported(x,
                  conv_state,
                  weight,
                  bias,
                  cu_seqlens,
                  init_indices,
                  current_indices,
                  initial_state_mode);

  const int64_t total_tokens = x.size(0);
  const int64_t dim = x.size(1);
  const int64_t vec_core_num = 48;
  const int64_t padded_dim = _compute_padded_dim(dim, vec_core_num);

  torch::Tensor x_padded = _pad_last_dim(x, padded_dim);
  torch::Tensor weight_padded = _pad_last_dim(weight, padded_dim);
  torch::Tensor bias_padded = _pad_last_dim(bias, padded_dim);
  torch::Tensor conv_state_padded = _pad_last_dim(conv_state, padded_dim);
  torch::Tensor y = torch::zeros({total_tokens, padded_dim}, x.options());

  run_tilelang_causal_conv1d_once(x_padded,
                                  conv_state_padded,
                                  weight_padded,
                                  bias_padded,
                                  cu_seqlens,
                                  init_indices,
                                  current_indices,
                                  initial_state_mode,
                                  y,
                                  padded_dim,
                                  dim);

  if (padded_dim > dim) {
    conv_state.copy_(conv_state_padded.slice(-1, 0, dim).contiguous());
  }

  torch::Tensor output =
      (padded_dim > dim) ? y.slice(1, 0, dim).contiguous() : y;

  if (has_silu) {
    output = torch::silu(output);
  }

  return output;
}

}  // namespace xllm::kernel::npu::tilelang
