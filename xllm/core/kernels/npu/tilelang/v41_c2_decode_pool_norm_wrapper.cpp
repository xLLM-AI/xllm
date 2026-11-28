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

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kHeadDim = 512;
constexpr int64_t kMaxGroups = 65536;
constexpr int64_t kMaxStateRows = 1 << 21;

#include XLLM_TL_V41_C2_DECODE_POOL_NORM_REGISTRY_INC

}  // namespace

torch::Tensor v41_c2_decode_pool_norm(const torch::Tensor& kv,
                                      const torch::Tensor& scores,
                                      const torch::Tensor& state_kv,
                                      const torch::Tensor& state_scores,
                                      const torch::Tensor& previous_rows,
                                      const torch::Tensor& weight,
                                      double eps) {
  CHECK_EQ(kv.dim(), 2);
  CHECK_EQ(kv.size(1), kHeadDim);
  CHECK(scores.sizes() == kv.sizes());
  CHECK_EQ(state_kv.dim(), 2);
  CHECK_EQ(state_kv.size(1), kHeadDim);
  CHECK(state_scores.sizes() == state_kv.sizes());
  CHECK_GT(state_kv.size(0), 0);
  CHECK_LE(state_kv.size(0), kMaxStateRows);
  CHECK_EQ(previous_rows.dim(), 1);
  CHECK_EQ(previous_rows.size(0), kv.size(0));
  CHECK_EQ(weight.numel(), kHeadDim);

  for (const auto& tensor : {kv, scores, state_kv, state_scores}) {
    CHECK_EQ(tensor.scalar_type(), torch::kFloat32);
    CHECK(tensor.is_contiguous() && tensor.device() == kv.device());
  }
  CHECK_EQ(previous_rows.scalar_type(), torch::kInt32);
  CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
  CHECK(previous_rows.is_contiguous() && previous_rows.device() == kv.device());
  CHECK(weight.is_contiguous() && weight.device() == kv.device());

  const int64_t groups = kv.size(0);
  CHECK_LE(groups, kMaxGroups);
  auto output = torch::empty({groups, kHeadDim}, weight.options());
  if (groups == 0) {
    return output;
  }

  const auto spec = make_v41_c2_decode_pool_norm_specialization(
      V41C2DecodePoolNormHeadDim{kHeadDim});
  const auto* entry = find_v41_c2_decode_pool_norm_kernel_entry(spec);
  CHECK(entry != nullptr) << "Missing V4.1 C2 decode pooling specialization";
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(kv.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(kv.data_ptr()),
            static_cast<uint8_t*>(scores.data_ptr()),
            static_cast<uint8_t*>(state_kv.data_ptr()),
            static_cast<uint8_t*>(state_scores.data_ptr()),
            static_cast<uint8_t*>(previous_rows.data_ptr()),
            static_cast<uint8_t*>(weight.data_ptr()),
            static_cast<uint8_t*>(output.data_ptr()),
            static_cast<int32_t>(groups),
            static_cast<float>(eps),
            stream);
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
