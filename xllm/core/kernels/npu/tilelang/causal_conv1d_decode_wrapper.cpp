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

#ifndef XLLM_TL_CAUSAL_CONV1D_DECODE_REGISTRY_INC
#error "XLLM_TL_CAUSAL_CONV1D_DECODE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_CAUSAL_CONV1D_DECODE_REGISTRY_INC

CausalConv1dDecodeSpecialization build_decode_runtime_specialization(
    const torch::Tensor& x,
    const torch::Tensor& weight_t,
    bool has_silu) {
  CHECK_EQ(x.dim(), 2) << "TileLang causal_conv1d_decode: x must be 2D [B, D]";
  CHECK_GE(weight_t.dim(), 2)
      << "TileLang causal_conv1d_decode: weight_t must be >=2D [width, dim]";

  const int32_t batch_size = static_cast<int32_t>(x.size(0));
  const int32_t dim = static_cast<int32_t>(x.size(1));
  const int32_t width = static_cast<int32_t>(weight_t.size(0));

  CHECK_GT(batch_size, 0)
      << "TileLang causal_conv1d_decode: batch_size must be > 0";
  CHECK_GT(dim, 0) << "TileLang causal_conv1d_decode: dim must be > 0";
  CHECK_GT(width, 0) << "TileLang causal_conv1d_decode: width must be > 0";

  CHECK_LE(batch_size,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang causal_conv1d_decode: batch_size exceeds int range";
  CHECK_LE(dim, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang causal_conv1d_decode: dim exceeds int range";

  return make_causal_conv1d_decode_specialization(
      CausalConv1dDecodeDim{dim},
      CausalConv1dDecodeWidth{width},
      CausalConv1dDecodeHasSilu{has_silu ? 1 : 0},
      CausalConv1dDecodeDType{to_tilelang_dtype(x.scalar_type())});
}

void check_decode_supported(const torch::Tensor& x,
                            torch::Tensor& conv_state_t,
                            const torch::Tensor& weight_t,
                            const torch::Tensor& bias,
                            const torch::Tensor& init_indices,
                            const torch::Tensor& current_indices,
                            const torch::Tensor& initial_state_mode,
                            bool has_silu) {
  CHECK(x.defined()) << "TileLang causal_conv1d_decode: x must be defined";
  CHECK(conv_state_t.defined())
      << "TileLang causal_conv1d_decode: conv_state must be defined";
  CHECK(weight_t.defined())
      << "TileLang causal_conv1d_decode: weight must be defined";
  CHECK(bias.defined())
      << "TileLang causal_conv1d_decode: bias must be defined";
  CHECK(init_indices.defined())
      << "TileLang causal_conv1d_decode: init_indices must be defined";
  CHECK(current_indices.defined())
      << "TileLang causal_conv1d_decode: current_indices must be defined";
  CHECK(initial_state_mode.defined())
      << "TileLang causal_conv1d_decode: initial_state_mode must be defined";

  CHECK(x.scalar_type() == c10::ScalarType::BFloat16)
      << "TileLang causal_conv1d_decode: only bfloat16 is supported, got "
      << x.scalar_type();

  CHECK(x.device().type() == c10::DeviceType::PrivateUse1 &&
        conv_state_t.device().type() == c10::DeviceType::PrivateUse1 &&
        weight_t.device().type() == c10::DeviceType::PrivateUse1 &&
        bias.device().type() == c10::DeviceType::PrivateUse1 &&
        init_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        current_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        initial_state_mode.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang causal_conv1d_decode: all tensors must be on NPU";

  CHECK_EQ(x.dim(), 2) << "TileLang causal_conv1d_decode: x must be 2D [B, D]";
  CHECK_EQ(conv_state_t.dim(), 3) << "TileLang causal_conv1d_decode: "
                                     "conv_state must be 3D [C, state_len, D]";
  CHECK_GE(weight_t.dim(), 2)
      << "TileLang causal_conv1d_decode: weight must be >=2D";
  CHECK_EQ(bias.dim(), 1)
      << "TileLang causal_conv1d_decode: bias must be 1D [D]";
  CHECK_EQ(init_indices.dim(), 1)
      << "TileLang causal_conv1d_decode: init_indices must be 1D [B]";
  CHECK_EQ(current_indices.dim(), 1)
      << "TileLang causal_conv1d_decode: current_indices must be 1D [B]";
  CHECK_EQ(initial_state_mode.dim(), 1)
      << "TileLang causal_conv1d_decode: initial_state_mode must be 1D [B]";

  const int64_t batch = x.size(0);
  const int64_t dim = x.size(1);

  CHECK_EQ(conv_state_t.size(2), dim)
      << "TileLang causal_conv1d_decode: conv_state dim mismatch";
  CHECK_EQ(bias.size(0), dim)
      << "TileLang causal_conv1d_decode: bias dim mismatch";
  CHECK_EQ(init_indices.size(0), batch)
      << "TileLang causal_conv1d_decode: init_indices batch mismatch";
  CHECK_EQ(current_indices.size(0), batch)
      << "TileLang causal_conv1d_decode: current_indices batch mismatch";
  CHECK_EQ(initial_state_mode.size(0), batch)
      << "TileLang causal_conv1d_decode: initial_state_mode batch mismatch";

  CHECK_EQ(x.dtype(), conv_state_t.dtype())
      << "TileLang causal_conv1d_decode: x/conv_state dtype mismatch";
  CHECK_EQ(x.dtype(), weight_t.dtype())
      << "TileLang causal_conv1d_decode: x/weight dtype mismatch";
  CHECK_EQ(x.dtype(), bias.dtype())
      << "TileLang causal_conv1d_decode: x/bias dtype mismatch";

  CHECK_EQ(init_indices.dtype(), torch::kInt32)
      << "TileLang causal_conv1d_decode: init_indices must be int32";
  CHECK_EQ(current_indices.dtype(), torch::kInt32)
      << "TileLang causal_conv1d_decode: current_indices must be int32";
  CHECK_EQ(initial_state_mode.dtype(), torch::kInt32)
      << "TileLang causal_conv1d_decode: initial_state_mode must be int32";

  CHECK(x.is_contiguous())
      << "TileLang causal_conv1d_decode: x must be contiguous";
  CHECK(conv_state_t.is_contiguous())
      << "TileLang causal_conv1d_decode: conv_state must be contiguous";
  CHECK(weight_t.is_contiguous())
      << "TileLang causal_conv1d_decode: weight must be contiguous";
  CHECK(bias.is_contiguous())
      << "TileLang causal_conv1d_decode: bias must be contiguous";
  CHECK(init_indices.is_contiguous())
      << "TileLang causal_conv1d_decode: init_indices must be contiguous";
  CHECK(current_indices.is_contiguous())
      << "TileLang causal_conv1d_decode: current_indices must be contiguous";
  CHECK(initial_state_mode.is_contiguous())
      << "TileLang causal_conv1d_decode: initial_state_mode must be contiguous";

  CHECK_GE(conv_state_t.size(1), weight_t.size(0) - 1)
      << "TileLang causal_conv1d_decode: state_len must be >= width-1";
}

void run_tilelang_causal_conv1d_decode_once(
    const torch::Tensor& x,
    torch::Tensor& conv_state_t,
    const torch::Tensor& weight_t,
    const torch::Tensor& bias,
    const torch::Tensor& init_indices,
    const torch::Tensor& current_indices,
    const torch::Tensor& initial_state_mode,
    torch::Tensor& y,
    bool has_silu) {
  const auto specialization =
      build_decode_runtime_specialization(x, weight_t, has_silu);
  const auto* entry = find_causal_conv1d_decode_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang causal_conv1d_decode: no compiled variant for batch_size="
      << x.size(0) << ", dim=" << x.size(1) << ". Available variants: "
      << available_causal_conv1d_decode_variant_keys();

  const int32_t device_id = x.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();

  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(x.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(weight_t.data_ptr())),
      reinterpret_cast<uint8_t*>(conv_state_t.data_ptr()),
      reinterpret_cast<uint8_t*>(const_cast<void*>(init_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(current_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(
          const_cast<void*>(initial_state_mode.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(bias.data_ptr())),
      reinterpret_cast<uint8_t*>(y.data_ptr()),
      static_cast<int64_t>(x.size(0)),
      static_cast<int64_t>(x.size(1)),
      static_cast<int64_t>(conv_state_t.size(0)),
      static_cast<int64_t>(conv_state_t.size(1)),
      stream);
}

}  // namespace

bool has_causal_conv1d_decode_specialization(int64_t batch_size,
                                             int64_t dim,
                                             bool has_silu) {
  const int32_t width = 4;
  CausalConv1dDecodeSpecialization spec =
      make_causal_conv1d_decode_specialization(
          CausalConv1dDecodeDim{static_cast<int32_t>(dim)},
          CausalConv1dDecodeWidth{width},
          CausalConv1dDecodeHasSilu{has_silu ? 1 : 0},
          CausalConv1dDecodeDType{TilelangDType::kBF16});
  return find_causal_conv1d_decode_kernel_entry(spec) != nullptr;
}

torch::Tensor causal_conv1d_decode(torch::Tensor& conv_state,
                                   const torch::Tensor& x,
                                   const torch::Tensor& weight,
                                   const torch::Tensor& bias,
                                   const torch::Tensor& init_indices,
                                   const torch::Tensor& current_indices,
                                   const torch::Tensor& initial_state_mode,
                                   bool has_silu) {
  check_decode_supported(x,
                         conv_state,
                         weight,
                         bias,
                         init_indices,
                         current_indices,
                         initial_state_mode,
                         has_silu);

  const int64_t batch_size = x.size(0);
  const int64_t dim = x.size(1);

  auto y = torch::empty({batch_size, dim}, x.options());

  run_tilelang_causal_conv1d_decode_once(x,
                                         conv_state,
                                         weight,
                                         bias,
                                         init_indices,
                                         current_indices,
                                         initial_state_mode,
                                         y,
                                         has_silu);

  return y;
}

}  // namespace xllm::kernel::npu::tilelang
