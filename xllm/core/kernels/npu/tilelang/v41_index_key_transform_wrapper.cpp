/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <acl/acl.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kIndexHeadDim = 128;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kMaxRows = 65536;

#include XLLM_TL_V41_INDEX_KEY_TRANSFORM_REGISTRY_INC

}  // namespace

torch::Tensor v41_index_key_transform(const torch::Tensor& key,
                                      const torch::Tensor& weight,
                                      const torch::Tensor& cos_sin,
                                      double eps) {
  CHECK_EQ(key.dim(), 2);
  CHECK_EQ(key.size(1), kIndexHeadDim);
  CHECK_EQ(weight.numel(), kIndexHeadDim);
  CHECK_EQ(cos_sin.dim(), 2);
  CHECK_EQ(cos_sin.size(0), key.size(0));
  CHECK_EQ(cos_sin.size(1), kRopeDim);
  CHECK_EQ(key.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cos_sin.scalar_type(), torch::kFloat32);
  CHECK(key.is_contiguous() && weight.is_contiguous() &&
        cos_sin.is_contiguous());
  CHECK(key.device() == weight.device() && key.device() == cos_sin.device());
  const int64_t num_rows = key.size(0);
  CHECK_LE(num_rows, kMaxRows);
  auto output = torch::empty_like(key);
  if (num_rows == 0) {
    return output;
  }

  const auto spec = make_v41_index_key_transform_specialization(
      V41IndexKeyTransformHeadDim{static_cast<int32_t>(kIndexHeadDim)});
  const auto* entry = find_v41_index_key_transform_kernel_entry(spec);
  CHECK(entry != nullptr) << "Missing V4.1 index key transform specialization";
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(key.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(key.data_ptr()),
            static_cast<uint8_t*>(weight.data_ptr()),
            static_cast<uint8_t*>(cos_sin.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()),
            static_cast<int32_t>(num_rows),
            static_cast<float>(eps),
            stream);
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
