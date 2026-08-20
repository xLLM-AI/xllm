/* Copyright 2025-2026 The xLLM Authors.

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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "dispatch_registry.h"
#include "tilelang_ops_api.h"

#ifndef XLLM_TL_ROPE_REGISTRY_INC
#error "XLLM_TL_ROPE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_ROPE_REGISTRY_INC

void check_supported(const torch::Tensor& input,
                     const torch::Tensor& sin_cache,
                     const torch::Tensor& cos_cache) {
  CHECK(input.defined()) << "TileLang RoPE: input must be defined";
  CHECK(sin_cache.defined()) << "TileLang RoPE: sin_cache must be defined";
  CHECK(cos_cache.defined()) << "TileLang RoPE: cos_cache must be defined";

  CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
        sin_cache.device().type() == c10::DeviceType::PrivateUse1 &&
        cos_cache.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang RoPE: all tensors must be on NPU";

  CHECK_EQ(input.dtype(), sin_cache.dtype())
      << "TileLang RoPE: input/sin_cache dtype mismatch";
  CHECK_EQ(input.dtype(), cos_cache.dtype())
      << "TileLang RoPE: input/cos_cache dtype mismatch";
  [[maybe_unused]] auto dtype = to_tilelang_dtype(input.scalar_type());

  CHECK_EQ(input.dim(), 3) << "TileLang RoPE: input must be 3D [T, H, D]";
  CHECK_EQ(input.stride(2), 1)
      << "TileLang RoPE: input last dim stride must be 1";
  CHECK_EQ(input.stride(0), input.size(1) * input.stride(1))
      << "TileLang RoPE: unsupported input layout";

  CHECK_EQ(sin_cache.dim(), 2)
      << "TileLang RoPE: sin_cache must be 2D [T, rope_dim]";
  CHECK_EQ(cos_cache.dim(), 2)
      << "TileLang RoPE: cos_cache must be 2D [T, rope_dim]";
  CHECK_EQ(sin_cache.sizes(), cos_cache.sizes())
      << "TileLang RoPE: sin_cache/cos_cache shape mismatch";
  CHECK_EQ(sin_cache.size(1), input.size(2))
      << "TileLang RoPE: rope_dim mismatch between input and sin_cache";
  CHECK_EQ(sin_cache.size(0), input.size(0))
      << "TileLang RoPE: sin_cache token size must match input.size(0)";

  const int64_t row_count = input.size(0) * input.size(1);
  CHECK_GT(row_count, 0) << "TileLang RoPE: row_count must be > 0";
}

RopeSpecialization build_runtime_specialization(const torch::Tensor& x_rows) {
  CHECK_EQ(x_rows.dim(), 2) << "TileLang RoPE: x_rows must be 2D";
  CHECK_GT(x_rows.stride(0), 0) << "TileLang RoPE: x_rows stride must be > 0";
  CHECK_LE(x_rows.stride(0),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang RoPE: x_rows stride exceeds int range";
  CHECK_LE(x_rows.size(1),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang RoPE: rope_dim exceeds int range";

  return make_rope_specialization(
      RopeHeadDim{static_cast<int32_t>(x_rows.stride(0))},
      RopeRopeDim{static_cast<int32_t>(x_rows.size(1))},
      RopeDType{to_tilelang_dtype(x_rows.scalar_type())});
}

void run_tilelang_rope_once(torch::Tensor& x_rows,
                            const torch::Tensor& sin_rows,
                            const torch::Tensor& cos_rows) {
  CHECK_EQ(x_rows.dim(), 2) << "TileLang RoPE: x_rows must be 2D";
  CHECK_EQ(sin_rows.dim(), 2) << "TileLang RoPE: sin_rows must be 2D";
  CHECK_EQ(cos_rows.dim(), 2) << "TileLang RoPE: cos_rows must be 2D";
  CHECK_EQ(x_rows.size(0), sin_rows.size(0))
      << "TileLang RoPE: x_rows/sin_rows row mismatch";
  CHECK_EQ(x_rows.size(0), cos_rows.size(0))
      << "TileLang RoPE: x_rows/cos_rows row mismatch";
  CHECK_EQ(x_rows.size(1), sin_rows.size(1))
      << "TileLang RoPE: x_rows/sin_rows rope_dim mismatch";
  CHECK_EQ(x_rows.size(1), cos_rows.size(1))
      << "TileLang RoPE: x_rows/cos_rows rope_dim mismatch";

  CHECK_EQ(x_rows.dtype(), sin_rows.dtype())
      << "TileLang RoPE: x_rows/sin_rows dtype mismatch";
  CHECK_EQ(x_rows.dtype(), cos_rows.dtype())
      << "TileLang RoPE: x_rows/cos_rows dtype mismatch";

  CHECK_EQ(x_rows.stride(1), 1)
      << "TileLang RoPE: x_rows last dim stride must be 1";
  CHECK(sin_rows.is_contiguous())
      << "TileLang RoPE: sin_rows must be contiguous";
  CHECK(cos_rows.is_contiguous())
      << "TileLang RoPE: cos_rows must be contiguous";

  const int64_t row_count = x_rows.size(0);
  CHECK_LE(row_count, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang RoPE: row_count exceeds int range";

  const RopeSpecialization specialization =
      build_runtime_specialization(x_rows);
  const auto* entry = find_rope_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang RoPE: no compiled variant. Available variants: "
      << available_rope_variant_keys();
  CHECK_GE(specialization.head_dim, specialization.rope_dim)
      << "TileLang RoPE: compiled head_dim must be >= rope_dim";

  const int32_t device_id = x_rows.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  const int32_t num_tokens = static_cast<int32_t>(row_count);
  const int32_t x_stride = specialization.head_dim;

  entry->fn(reinterpret_cast<uint8_t*>(x_rows.data_ptr()),
            reinterpret_cast<uint8_t*>(const_cast<void*>(sin_rows.data_ptr())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(cos_rows.data_ptr())),
            reinterpret_cast<uint8_t*>(x_rows.data_ptr()),
            num_tokens,
            x_stride,
            stream);
}

}  // namespace

void rope_in_place(torch::Tensor& input,
                   const torch::Tensor& sin_cache,
                   const torch::Tensor& cos_cache) {
  check_supported(input, sin_cache, cos_cache);

  auto input_rows =
      input.as_strided({input.size(0) * input.size(1), input.size(2)},
                       {input.stride(1), input.stride(2)});
  auto sin_rows = sin_cache.unsqueeze(1)
                      .expand({input.size(0), input.size(1), sin_cache.size(1)})
                      .contiguous()
                      .view({input_rows.size(0), sin_cache.size(1)});
  auto cos_rows = cos_cache.unsqueeze(1)
                      .expand({input.size(0), input.size(1), cos_cache.size(1)})
                      .contiguous()
                      .view({input_rows.size(0), cos_cache.size(1)});
  run_tilelang_rope_once(input_rows, sin_rows, cos_rows);
}

}  // namespace xllm::kernel::npu::tilelang
