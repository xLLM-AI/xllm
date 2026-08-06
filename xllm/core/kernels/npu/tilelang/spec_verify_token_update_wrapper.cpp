/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SPEC_VERIFY_TOKEN_UPDATE_REGISTRY_INC
#error "XLLM_TL_SPEC_VERIFY_TOKEN_UPDATE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr size_t kDraftTokenSourceSlots = 5;

#include XLLM_TL_SPEC_VERIFY_TOKEN_UPDATE_REGISTRY_INC

void check_tokens(const torch::Tensor& tokens,
                  torch::ScalarType dtype,
                  int64_t expected_numel) {
  CHECK(tokens.defined() &&
        tokens.device().type() == c10::DeviceType::PrivateUse1)
      << "speculative verify token update requires NPU tensors";
  CHECK_EQ(tokens.scalar_type(), dtype);
  CHECK_EQ(tokens.numel(), expected_numel);
  CHECK(tokens.is_contiguous());
}

}  // namespace

bool has_spec_verify_token_update_specialization(int64_t spec_width) {
  if (spec_width <= 0 || !std::in_range<int32_t>(spec_width)) {
    return false;
  }
  const auto specialization = make_spec_verify_token_update_specialization(
      SpecVerifyTokenUpdateSpecWidth{static_cast<int32_t>(spec_width)});
  return find_spec_verify_token_update_kernel_entry(specialization) != nullptr;
}

void spec_verify_token_update(const torch::Tensor& base_tokens,
                              const std::vector<torch::Tensor>& draft_tokens,
                              torch::Tensor& persistent_tokens,
                              int64_t spec_width) {
  CHECK_EQ(static_cast<int64_t>(draft_tokens.size()) + 1, spec_width);
  CHECK_GT(spec_width, 0);
  CHECK_EQ(base_tokens.numel() % spec_width, 0);
  const int64_t batch_size = base_tokens.numel() / spec_width;
  CHECK_GT(batch_size, 0);
  CHECK_LE(batch_size, std::numeric_limits<int32_t>::max());
  check_tokens(base_tokens, torch::kInt32, batch_size * spec_width);
  CHECK(has_spec_verify_token_update_specialization(spec_width))
      << "speculative verify token update has no compiled variant for width "
      << spec_width << ": "
      << available_spec_verify_token_update_variant_keys();
  for (const auto& token : draft_tokens) {
    check_tokens(token, torch::kInt64, batch_size);
    CHECK_EQ(token.device(), base_tokens.device())
        << "all speculative verify tokens must be on the same NPU";
  }
  CHECK(persistent_tokens.defined() &&
        persistent_tokens.device().type() == c10::DeviceType::PrivateUse1);
  CHECK_EQ(persistent_tokens.scalar_type(), torch::kInt32);
  CHECK_GE(persistent_tokens.numel(), base_tokens.numel());
  CHECK(persistent_tokens.is_contiguous());
  CHECK_EQ(persistent_tokens.device(), base_tokens.device())
      << "persistent and source tokens must be on the same NPU";

  const auto specialization = make_spec_verify_token_update_specialization(
      SpecVerifyTokenUpdateSpecWidth{static_cast<int32_t>(spec_width)});
  const auto* entry =
      find_spec_verify_token_update_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "speculative verify token update has no compiled variant: "
      << available_spec_verify_token_update_variant_keys();

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(base_tokens.device().index()).stream();
  std::array<uint8_t*, kDraftTokenSourceSlots> draft_ptrs;
  CHECK_LE(draft_tokens.size(), draft_ptrs.size());
  draft_ptrs.fill(
      reinterpret_cast<uint8_t*>(const_cast<void*>(base_tokens.data_ptr())));
  for (size_t index = 0; index < draft_tokens.size(); ++index) {
    draft_ptrs[index] = reinterpret_cast<uint8_t*>(
        const_cast<void*>(draft_tokens[index].data_ptr()));
  }
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(base_tokens.data_ptr())),
      draft_ptrs[0],
      draft_ptrs[1],
      draft_ptrs[2],
      draft_ptrs[3],
      draft_ptrs[4],
      reinterpret_cast<uint8_t*>(persistent_tokens.data_ptr()),
      static_cast<int32_t>(batch_size),
      persistent_tokens.numel(),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
