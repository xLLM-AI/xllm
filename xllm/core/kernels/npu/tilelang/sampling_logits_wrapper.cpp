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

#include <acl/acl.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SAMPLING_LOGITS_REGISTRY_INC
#error "XLLM_TL_SAMPLING_LOGITS_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_SAMPLING_LOGITS_REGISTRY_INC

constexpr int64_t kMaxRows = 4096;
constexpr int64_t kMaxVocab = 262144;
constexpr int64_t kMaxSpan = kMaxRows * kMaxVocab;

void check_vector(const torch::Tensor& input,
                  const torch::Tensor& logits,
                  torch::ScalarType dtype) {
  CHECK(input.defined());
  CHECK_EQ(input.device(), logits.device());
  CHECK_EQ(input.scalar_type(), dtype);
  CHECK_EQ(input.dim(), 1);
  CHECK_EQ(input.numel(), logits.size(0));
  CHECK(input.is_contiguous());
}

torch::Tensor run_sampling_logits(const torch::Tensor& logits,
                                  const torch::Tensor& seeds,
                                  const torch::Tensor& offsets,
                                  const torch::Tensor& uniforms,
                                  bool normalize_only) {
  CHECK(supports_sampling_logits(logits));
  const bool has_seeds = seeds.defined();
  CHECK_EQ(offsets.defined(), has_seeds);
  if (has_seeds) {
    check_vector(seeds, logits, torch::kInt64);
    check_vector(offsets, logits, torch::kInt64);
  }
  CHECK(normalize_only || has_seeds || uniforms.defined());
  if (uniforms.defined()) {
    check_vector(uniforms, logits, torch::kFloat32);
  }

  const c10_npu::NPUGuard guard(logits.device());
  torch::Tensor output =
      normalize_only ? torch::empty({logits.size(0), 2},
                                    logits.options().dtype(torch::kFloat32))
                     : torch::empty({logits.size(0)},
                                    logits.options().dtype(torch::kInt64));
  if (logits.size(0) == 0) {
    return output;
  }
  const auto* entry =
      find_sampling_logits_kernel_entry(make_sampling_logits_specialization(
          SamplingLogitsTaskCount{48},
          SamplingLogitsDType{to_tilelang_dtype(logits.scalar_type())},
          SamplingLogitsNormalizeOnly{static_cast<int32_t>(normalize_only)}));
  CHECK(entry != nullptr) << available_sampling_logits_variant_keys();
  const auto stream = c10_npu::getCurrentNPUStream(logits.device().index());
  entry->fn(
      reinterpret_cast<uint8_t*>(logits.data_ptr()),
      has_seeds ? reinterpret_cast<uint8_t*>(seeds.data_ptr()) : nullptr,
      has_seeds ? reinterpret_cast<uint8_t*>(offsets.data_ptr()) : nullptr,
      uniforms.defined() ? reinterpret_cast<uint8_t*>(uniforms.data_ptr())
                         : nullptr,
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      static_cast<int32_t>(logits.size(0)),
      static_cast<int32_t>(logits.size(1)),
      static_cast<int32_t>(logits.stride(0)),
      static_cast<int32_t>(has_seeds) | (uniforms.defined() ? 2 : 0),
      stream.stream());
  return output;
}

}  // namespace

bool supports_sampling_logits(const torch::Tensor& logits) {
  return logits.defined() && logits.device().type() == torch::kPrivateUse1 &&
         logits.dim() == 2 && logits.size(0) <= kMaxRows &&
         logits.size(1) > 0 && logits.size(1) <= kMaxVocab &&
         logits.stride(1) == 1 && logits.stride(0) >= logits.size(1) &&
         logits.stride(0) <= kMaxSpan &&
         (logits.size(0) == 0 ? 0 : logits.size(0) - 1) * logits.stride(0) +
                 logits.size(1) <=
             kMaxSpan &&
         (logits.scalar_type() == torch::kFloat32 ||
          logits.scalar_type() == torch::kBFloat16 ||
          logits.scalar_type() == torch::kFloat16);
}

torch::Tensor sampling_logits(const torch::Tensor& logits,
                              const torch::Tensor& seeds,
                              const torch::Tensor& offsets,
                              const torch::Tensor& uniforms) {
  return run_sampling_logits(logits,
                             seeds,
                             offsets,
                             uniforms,
                             /*normalize_only=*/false);
}

std::tuple<torch::Tensor, torch::Tensor> row_log_normalization(
    const torch::Tensor& logits) {
  const torch::Tensor values =
      run_sampling_logits(logits, {}, {}, {}, /*normalize_only=*/true);
  return {values.select(/*dim=*/1, /*index=*/0),
          values.select(/*dim=*/1, /*index=*/1)};
}

}  // namespace xllm::kernel::npu::tilelang
