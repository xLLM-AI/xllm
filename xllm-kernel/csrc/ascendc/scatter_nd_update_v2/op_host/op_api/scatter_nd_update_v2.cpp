/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
/*!
 * \file scatter_nd_update_v2.cpp
 * \brief
 */

#include "scatter_nd_update_v2.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
// #include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(XllmKernelScatterNdUpdateV2);
OP_TYPE_REGISTER(ScatterNdUpdateV2);

// AiCore支持的ScatterUpdate类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT16,
    op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType>
    ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                            op::DataType::DT_FLOAT16,
                                            op::DataType::DT_BOOL,
                                            op::DataType::DT_BF16,
                                            op::DataType::DT_INT64,
                                            op::DataType::DT_INT8};

inline static bool IsAiCoreSupport(const aclTensor* self) {
  // ScatterNdUpdateV2只需要判断self
  // ASCEND910B / ASCEND910_93 / ASCEND950
  // 均使用扩展dtype支持列表(含BF16/INT64/INT8等)，
  // 且这些平台只提供AiCore实现(见def中仅AddConfig AICore，无AICPU注册)，
  // 因此这些dtype必须走AiCore分支，否则会误分派到不存在的AiCPU kernel(errno
  // 361001)。
  const auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B ||
      socVersion == SocVersion::ASCEND910_93 ||
      socVersion == SocVersion::ASCEND950) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AiCore的执行逻辑
inline static const aclTensor* ScatterNdUpdateV2AiCore(
    const aclTensor* self,
    const aclTensor* indices,
    const aclTensor* updates,
    const aclIntArray* strides,
    bool use_locking,
    aclOpExecutor* executor) {
  L0_DFX(ScatterNdUpdateV2AiCore, self, indices, updates, use_locking);
  auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(XllmKernelScatterNdUpdateV2,
                                               OP_INPUT(self, indices, updates),
                                               OP_OUTPUT(self),
                                               OP_ATTR(strides, use_locking));
  CHECK_RET(retAicore == ACLNN_SUCCESS, nullptr);
  return self;
}

// AiCPU的执行逻辑
inline static const aclTensor* ScatterNdUpdateV2AiCPU(const aclTensor* self,
                                                      const aclTensor* indices,
                                                      const aclTensor* updates,
                                                      bool use_locking,
                                                      aclOpExecutor* executor) {
  L0_DFX(ScatterNdUpdateV2AiCPU, self, indices, updates, use_locking);

  static internal::AicpuTaskSpace space(
      "ScatterNdUpdateV2", ge::DEPEND_IN_SHAPE, true);
  space.SetRef(0);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
      ScatterNdUpdateV2,
      OP_ATTR_NAMES({"Tindices", "T", "use_locking"}),
      OP_INPUT(self, indices, updates),
      OP_OUTPUT(self),
      OP_ATTR(indices->GetDataType(), updates->GetDataType(), use_locking));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return self;
}

const aclTensor* XllmKernelScatterNdUpdateV2(const aclTensor* self,
                                             const aclTensor* indices,
                                             const aclTensor* updates,
                                             const aclIntArray* strides,
                                             bool use_locking,
                                             aclOpExecutor* executor) {
  if (IsAiCoreSupport(self)) {
    return ScatterNdUpdateV2AiCore(
        self, indices, updates, strides, use_locking, executor);
  } else {
    return ScatterNdUpdateV2AiCPU(
        self, indices, updates, use_locking, executor);
  }
}
}  // namespace l0op