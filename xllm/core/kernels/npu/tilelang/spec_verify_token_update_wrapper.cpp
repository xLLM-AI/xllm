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

#include "dispatch_registry.h"
#include "tilelang_ops_api.h"

#ifndef XLLM_TL_SPEC_VERIFY_TOKEN_UPDATE_REGISTRY_INC
#error "XLLM_TL_SPEC_VERIFY_TOKEN_UPDATE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_SPEC_VERIFY_TOKEN_UPDATE_REGISTRY_INC

void check_token(const torch::Tensor& token, torch::ScalarType dtype) {
  CHECK(token.defined() &&
        token.device().type() == c10::DeviceType::PrivateUse1)
      << "speculative verify token update requires NPU tensors";
  CHECK_EQ(token.scalar_type(), dtype);
  CHECK_EQ(token.numel(), 1);
  CHECK(token.is_contiguous());
}

}  // namespace

bool has_spec_verify_token_update_specialization(int64_t spec_width) {
  const auto specialization = make_spec_verify_token_update_specialization(
      SpecVerifyTokenUpdateSpecWidth{static_cast<int32_t>(spec_width)});
  return find_spec_verify_token_update_kernel_entry(specialization) != nullptr;
}

void spec_verify_token_update(const torch::Tensor& base_token,
                              const std::vector<torch::Tensor>& draft_tokens,
                              torch::Tensor& persistent_tokens) {
  const int64_t spec_width = static_cast<int64_t>(draft_tokens.size()) + 1;
  check_token(base_token, torch::kInt32);
  CHECK(has_spec_verify_token_update_specialization(spec_width))
      << "speculative verify token update has no compiled variant for width "
      << spec_width << ": "
      << available_spec_verify_token_update_variant_keys();
  for (const auto& token : draft_tokens) {
    check_token(token, torch::kInt64);
  }
  CHECK(persistent_tokens.defined() &&
        persistent_tokens.device().type() == c10::DeviceType::PrivateUse1);
  CHECK_EQ(persistent_tokens.scalar_type(), torch::kInt32);
  CHECK_GE(persistent_tokens.numel(), 8);
  CHECK(persistent_tokens.is_contiguous());

  const auto specialization = make_spec_verify_token_update_specialization(
      SpecVerifyTokenUpdateSpecWidth{static_cast<int32_t>(spec_width)});
  const auto* entry =
      find_spec_verify_token_update_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "speculative verify token update has no compiled variant: "
      << available_spec_verify_token_update_variant_keys();

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(base_token.device().index()).stream();
  std::array<uint8_t*, 5> draft_ptrs;
  draft_ptrs.fill(
      reinterpret_cast<uint8_t*>(const_cast<void*>(base_token.data_ptr())));
  for (size_t index = 0; index < draft_tokens.size(); ++index) {
    draft_ptrs[index] = reinterpret_cast<uint8_t*>(
        const_cast<void*>(draft_tokens[index].data_ptr()));
  }
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(base_token.data_ptr())),
      draft_ptrs[0],
      draft_ptrs[1],
      draft_ptrs[2],
      draft_ptrs[3],
      draft_ptrs[4],
      reinterpret_cast<uint8_t*>(persistent_tokens.data_ptr()),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
