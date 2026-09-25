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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_GREEDY_PREFIX_VERIFY_REGISTRY_INC
#error "XLLM_TL_GREEDY_PREFIX_VERIFY_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_GREEDY_PREFIX_VERIFY_REGISTRY_INC

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
constexpr int32_t kVecNum = 2;

constexpr int32_t compiled_task_limit() {
  int32_t limit = 0;
  for (const auto& entry : kGreedyPrefixVerifyRegistry) {
    limit = std::max(limit, entry.spec.task_count);
  }
  return limit;
}

constexpr int32_t kCompiledTaskLimit = compiled_task_limit();
static_assert(kCompiledTaskLimit > 0 && kCompiledTaskLimit % kVecNum == 0);

void check_input_metadata(const torch::Tensor& input, const char* name) {
  CHECK(input.defined()) << name << " must be defined";
  CHECK_EQ(input.device().type(), c10::DeviceType::PrivateUse1)
      << name << " must be on NPU";
  CHECK_EQ(input.dim(), 2) << name << " must be rank two";
  CHECK(input.scalar_type() == torch::kInt32 ||
        input.scalar_type() == torch::kInt64)
      << name << " must contain int32 or int64 token IDs";
  for (int64_t dim = 0; dim < input.dim(); ++dim) {
    CHECK_GE(input.size(dim), 0) << name;
    CHECK_LE(input.size(dim), kInt32Max) << name << " shape exceeds INT32";
    CHECK_GE(input.stride(dim), 0) << name << " requires nonnegative strides";
    CHECK_LE(input.stride(dim), kInt32Max) << name << " stride exceeds INT32";
  }

  // Offsets are relative to data_ptr(), which already includes storage_offset.
  int64_t span = 0;
  if (input.size(0) != 0 && input.size(1) != 0) {
    span = 1;
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
      const int64_t stride = input.stride(dim);
      if (stride != 0) {
        CHECK_LE(input.size(dim) - 1, (kInt32Max - span) / stride)
            << name << " physical span exceeds INT32";
        span += (input.size(dim) - 1) * stride;
      }
    }
  }
  CHECK_GE(input.storage_offset(), 0) << name;
  if (span != 0) {
    const uint64_t storage_elements =
        input.storage().nbytes() / input.element_size();
    const uint64_t offset = static_cast<uint64_t>(input.storage_offset());
    CHECK_LE(offset, storage_elements) << name << " offset exceeds storage";
    CHECK_LE(static_cast<uint64_t>(span), storage_elements - offset)
        << name << " physical span exceeds storage";
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> greedy_prefix_verify(
    const torch::Tensor& draft,
    const torch::Tensor& target,
    const torch::Tensor& bonus,
    bool mask_out_rejected_tokens) {
  check_input_metadata(draft, "draft");
  check_input_metadata(target, "target");
  check_input_metadata(bonus, "bonus");
  CHECK_EQ(draft.device(), target.device());
  CHECK_EQ(bonus.device(), target.device());
  CHECK_EQ(draft.sizes(), target.sizes());
  CHECK_EQ(bonus.size(0), target.size(0));
  CHECK_EQ(bonus.size(1), 1);

  const int64_t batch = target.size(0);
  const int64_t width = target.size(1);
  CHECK_LT(width, kInt32Max) << "output width exceeds INT32";
  CHECK_LE(batch, kInt32Max / (width + 1))
      << "output element count exceeds INT32";

  torch::Tensor full =
      torch::empty({batch, width + 1}, target.options().dtype(torch::kInt32));
  torch::Tensor masked;
  if (mask_out_rejected_tokens) {
    masked = torch::empty_like(full);
  }
  if (batch == 0) {
    return {full, masked};
  }

  const auto stream = c10_npu::getCurrentNPUStream(target.device().index());
  c10_npu::NPUStreamGuard stream_guard(stream);
  int32_t device_id = -1;
  CHECK_EQ(aclrtGetDevice(&device_id), ACL_SUCCESS);
  CHECK_EQ(device_id, target.device().index());
  int64_t vector_cores = 0;
  CHECK_EQ(aclrtGetDeviceInfo(static_cast<uint32_t>(device_id),
                              ACL_DEV_ATTR_VECTOR_CORE_NUM,
                              &vector_cores),
           ACL_SUCCESS)
      << "cannot query greedy prefix verify vector core count";
  CHECK_EQ(vector_cores, kCompiledTaskLimit)
      << "greedy prefix verify AOT task limit does not match device "
      << device_id << "; rebuild the registry for this device";

  const int32_t task_count =
      static_cast<int32_t>(std::min((batch + kVecNum - 1) / kVecNum * kVecNum,
                                    static_cast<int64_t>(kCompiledTaskLimit)));
  const auto specialization = make_greedy_prefix_verify_specialization(
      GreedyPrefixVerifyTaskCount{task_count},
      GreedyPrefixVerifyDraftBits{
          static_cast<int32_t>(draft.element_size() * 8)},
      GreedyPrefixVerifyTargetBits{
          static_cast<int32_t>(target.element_size() * 8)},
      GreedyPrefixVerifyBonusBits{
          static_cast<int32_t>(bonus.element_size() * 8)});
  const auto* entry = find_greedy_prefix_verify_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "greedy prefix verify has no compiled variant for tasks=" << task_count
      << ", draft/target/bonus bits=" << specialization.draft_bits << "/"
      << specialization.target_bits << "/" << specialization.bonus_bits << ": "
      << available_greedy_prefix_verify_variant_keys();

  // The generated ABI has no restrict promise; mask=false never accesses this
  // slot. Reuse full instead of allocating a second output or a dummy buffer.
  torch::Tensor& masked_buffer = mask_out_rejected_tokens ? masked : full;
  entry->fn(reinterpret_cast<uint8_t*>(draft.data_ptr()),
            reinterpret_cast<uint8_t*>(target.data_ptr()),
            reinterpret_cast<uint8_t*>(bonus.data_ptr()),
            reinterpret_cast<uint8_t*>(full.data_ptr()),
            reinterpret_cast<uint8_t*>(masked_buffer.data_ptr()),
            static_cast<int32_t>(batch),
            static_cast<int32_t>(width),
            static_cast<int32_t>(draft.stride(0)),
            static_cast<int32_t>(draft.stride(1)),
            static_cast<int32_t>(target.stride(0)),
            static_cast<int32_t>(target.stride(1)),
            static_cast<int32_t>(bonus.stride(0)),
            static_cast<int32_t>(mask_out_rejected_tokens),
            stream.stream());
  return {full, masked};
}

}  // namespace xllm::kernel::npu::tilelang
