/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */
#pragma once
#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace xllm_kernel::xlite {
inline void check_tensor(const torch::Tensor& value) {
  CHECK(value.device().type() == c10::DeviceType::PrivateUse1);
  CHECK(value.scalar_type() == torch::kBFloat16 ||
        value.scalar_type() == torch::kHalf);
  CHECK(at_npu::native::get_npu_format(value) == ACL_FORMAT_ND);
  CHECK(value.numel() <= std::numeric_limits<int32_t>::max());
}
inline void check_pair(const torch::Tensor& value, const torch::Tensor& other) {
  check_tensor(other);
  CHECK(value.device() == other.device());
  CHECK(value.scalar_type() == other.scalar_type());
}
inline uint32_t cores(const torch::Tensor& value, bool cube = false) {
  int64_t count = 0;
  CHECK(aclGetDeviceCapability(value.device().index(),
                               cube ? ACL_DEVICE_INFO_AI_CORE_NUM
                                    : ACL_DEVICE_INFO_VECTOR_CORE_NUM,
                               &count) == ACL_SUCCESS);
  CHECK(count > 0);
  return static_cast<uint32_t>(count);
}
inline int checked_status(uint32_t status) {
  CHECK(status == ACL_SUCCESS)
      << "xlite launch failed: " << aclGetRecentErrMsg();
  return static_cast<int>(status);
}
}  // namespace xllm_kernel::xlite
