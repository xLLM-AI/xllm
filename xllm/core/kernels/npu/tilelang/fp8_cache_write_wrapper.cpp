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

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <limits>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_FP8_CACHE_WRITE_REGISTRY_INC
#error "XLLM_TL_FP8_CACHE_WRITE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_FP8_CACHE_WRITE_REGISTRY_INC
}  // namespace

void fp8_cache_write(const torch::Tensor& slots,
                     const torch::Tensor& values,
                     torch::Tensor& cache) {
  CHECK(slots.defined() && values.defined() && cache.defined());
  CHECK(cache.device().type() == torch::kPrivateUse1);
  CHECK(slots.device() == cache.device() && values.device() == cache.device());
  CHECK(slots.is_contiguous() && values.is_contiguous() &&
        cache.is_contiguous());
  CHECK_EQ(slots.scalar_type(), torch::kInt64);
  CHECK_EQ(values.scalar_type(), torch::kUInt8);
  CHECK_EQ(cache.scalar_type(), torch::kUInt8);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(values.dim(), 2);
  CHECK_EQ(cache.dim(), 2);
  CHECK_EQ(values.size(0), slots.numel());
  CHECK_EQ(values.size(1), cache.size(1));
  CHECK(cache.size(1) == 64 || cache.size(1) == 128 || cache.size(1) == 512);
  if (slots.numel() == 0) {
    return;
  }
  CHECK_GT(cache.size(0), 0);
  CHECK_LE(cache.size(0),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const auto specialization = make_fp8_cache_write_specialization(
      Fp8CacheWriteHeadDim{static_cast<int32_t>(cache.size(1))});
  const auto* entry = find_fp8_cache_write_kernel_entry(specialization);
  CHECK(entry != nullptr) << "No compiled FP8 cache writer for head_dim="
                          << cache.size(1);
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(cache.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(slots.data_ptr()),
            static_cast<uint8_t*>(values.data_ptr()),
            static_cast<uint8_t*>(cache.data_ptr()),
            slots.numel(),
            cache.size(0),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
