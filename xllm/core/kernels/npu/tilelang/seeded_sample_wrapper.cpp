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

#ifndef XLLM_TL_SEEDED_SAMPLE_REGISTRY_INC
#error "XLLM_TL_SEEDED_SAMPLE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_SEEDED_SAMPLE_REGISTRY_INC

constexpr int64_t kMaxRows = 4096;
constexpr int64_t kMaxVocab = 262144;
constexpr int64_t kMaxSpan = kMaxRows * kMaxVocab;

void check_vector(const torch::Tensor& input,
                  const torch::Tensor& weights,
                  torch::ScalarType dtype) {
  CHECK(input.defined());
  CHECK_EQ(input.device(), weights.device());
  CHECK_EQ(input.scalar_type(), dtype);
  CHECK_EQ(input.dim(), 1);
  CHECK_EQ(input.numel(), weights.size(0));
  CHECK(input.is_contiguous());
}

}  // namespace

torch::Tensor seeded_sample(const torch::Tensor& weights,
                            const torch::Tensor& seeds,
                            const torch::Tensor& offsets,
                            const torch::Tensor& uniforms) {
  CHECK(weights.defined());
  CHECK_EQ(weights.device().type(), c10::DeviceType::PrivateUse1);
  CHECK_EQ(weights.scalar_type(), torch::kFloat32);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(weights.stride(1), 1);
  CHECK_LE(weights.size(0), kMaxRows);
  CHECK_GT(weights.size(1), 0);
  CHECK_LE(weights.size(1), kMaxVocab);
  CHECK_GE(weights.stride(0), weights.size(1));
  CHECK_LE(weights.stride(0), kMaxSpan);
  CHECK_LE(
      (weights.size(0) == 0 ? 0 : weights.size(0) - 1) * weights.stride(0) +
          weights.size(1),
      kMaxSpan);
  const bool has_seeds = seeds.defined();
  CHECK_EQ(offsets.defined(), has_seeds);
  CHECK(has_seeds || uniforms.defined());
  if (has_seeds) {
    check_vector(seeds, weights, torch::kInt64);
    check_vector(offsets, weights, torch::kInt64);
  }
  if (uniforms.defined()) {
    check_vector(uniforms, weights, torch::kFloat32);
  }

  const c10_npu::NPUGuard guard(weights.device());
  torch::Tensor output =
      torch::empty({weights.size(0)}, weights.options().dtype(torch::kInt64));
  if (weights.size(0) == 0) {
    return output;
  }
  const auto* entry = find_seeded_sample_kernel_entry(
      make_seeded_sample_specialization(SeededSampleTaskCount{48}));
  CHECK(entry != nullptr) << available_seeded_sample_variant_keys();
  const auto stream = c10_npu::getCurrentNPUStream(weights.device().index());
  entry->fn(
      reinterpret_cast<uint8_t*>(weights.data_ptr()),
      has_seeds ? reinterpret_cast<uint8_t*>(seeds.data_ptr()) : nullptr,
      has_seeds ? reinterpret_cast<uint8_t*>(offsets.data_ptr()) : nullptr,
      uniforms.defined() ? reinterpret_cast<uint8_t*>(uniforms.data_ptr())
                         : nullptr,
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      static_cast<int32_t>(weights.size(0)),
      static_cast<int32_t>(weights.size(1)),
      static_cast<int32_t>(weights.stride(0)),
      static_cast<int32_t>(has_seeds) | (uniforms.defined() ? 2 : 0),
      stream.stream());
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
