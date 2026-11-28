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

constexpr int32_t kHeadDim = 128;
constexpr int64_t kMaxRows = 4096;

#include XLLM_TL_V41_INDEX_QUERY_QUANT_REGISTRY_INC

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> v41_index_query_quant(
    const torch::Tensor& query) {
  CHECK_EQ(query.dim(), 3);
  const int64_t rows = query.size(0);
  const int64_t heads = query.size(1);
  CHECK_GT(rows, 0);
  CHECK_LE(rows, kMaxRows);
  CHECK(heads == 32 || heads == 64);
  CHECK_EQ(query.size(2), kHeadDim);
  CHECK_EQ(query.scalar_type(), torch::kBFloat16);
  CHECK(query.is_contiguous());

  auto codes = torch::empty(query.sizes(), query.options().dtype(torch::kInt8));
  auto scales =
      torch::empty({rows, heads}, query.options().dtype(torch::kFloat16));
  const auto spec = make_v41_index_query_quant_specialization(
      V41IndexQueryQuantHeadDim{kHeadDim});
  const auto* entry = find_v41_index_query_quant_kernel_entry(spec);
  CHECK(entry != nullptr)
      << "Missing V4.1 index query quantization specialization";
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(query.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(query.data_ptr()),
            static_cast<uint8_t*>(codes.data_ptr()),
            static_cast<uint8_t*>(scales.data_ptr()),
            static_cast<int32_t>(rows * heads),
            stream);
  return {codes, scales};
}

}  // namespace xllm::kernel::npu::tilelang
