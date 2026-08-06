/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_APPLY_TOKEN_BITMASK_REGISTRY_INC
#error "XLLM_TL_APPLY_TOKEN_BITMASK_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kMaxNumRows = 4096;
constexpr int64_t kMaxVocabSize = 262144;

#include XLLM_TL_APPLY_TOKEN_BITMASK_REGISTRY_INC

bool is_supported_logits_dtype(c10::ScalarType dtype) {
  return dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
         dtype == torch::kFloat32;
}

ApplyTokenBitmaskSpecialization build_runtime_specialization(
    const torch::Tensor& logits) {
  return make_apply_token_bitmask_specialization(
      ApplyTokenBitmaskDType{to_tilelang_dtype(logits.scalar_type())});
}

}  // namespace

bool can_apply_token_bitmask_inplace(const torch::Tensor& logits,
                                     const torch::Tensor& bitmask) {
  if (!logits.defined() || !bitmask.defined() || logits.dim() != 2 ||
      bitmask.dim() != 2) {
    return false;
  }
  if (logits.device().type() != c10::DeviceType::PrivateUse1 ||
      bitmask.device() != logits.device()) {
    return false;
  }
  if (!is_supported_logits_dtype(logits.scalar_type()) ||
      bitmask.scalar_type() != torch::kInt32) {
    return false;
  }
  if (!logits.is_contiguous() || !bitmask.is_contiguous()) {
    return false;
  }
  if (logits.size(0) != bitmask.size(0) || logits.size(0) <= 0 ||
      logits.size(0) > kMaxNumRows || logits.size(1) <= 0 ||
      logits.size(1) > kMaxVocabSize) {
    return false;
  }
  const int64_t expected_words = (logits.size(1) + 31) / 32;
  if (bitmask.size(1) != expected_words || logits.size(1) % 32 != 0) {
    return false;
  }
  return logits.size(0) <= std::numeric_limits<int32_t>::max() &&
         logits.size(1) <= std::numeric_limits<int32_t>::max() &&
         bitmask.size(1) <= std::numeric_limits<int32_t>::max();
}

void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask) {
  CHECK(can_apply_token_bitmask_inplace(logits, bitmask))
      << "TileLang apply_token_bitmask: unsupported tensor contract; logits="
      << logits.sizes() << ", bitmask=" << bitmask.sizes();

  const ApplyTokenBitmaskSpecialization specialization =
      build_runtime_specialization(logits);
  const auto* entry = find_apply_token_bitmask_kernel_entry(specialization);
  CHECK(entry != nullptr) << "TileLang apply_token_bitmask: no compiled "
                             "variant. Available variants: "
                          << available_apply_token_bitmask_variant_keys();

  const int32_t device_id = logits.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(reinterpret_cast<uint8_t*>(logits.data_ptr()),
            reinterpret_cast<uint8_t*>(const_cast<void*>(bitmask.data_ptr())),
            static_cast<int32_t>(logits.size(0)),
            static_cast<int32_t>(logits.size(1)),
            static_cast<int32_t>(bitmask.size(1)),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
