/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
the terms and conditions of  * CANN Open Software License Agreement Version 2.0
(the "License").  * Please refer to the License for details. You may not use
this file except in compliance with the License.  * THIS SOFTWARE IS PROVIDED ON
AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,  *
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE.  * See LICENSE in the root of the software repository for
the full text of the License.  */

#include "moe_grouped_matmul_swiglu_quant_l0.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(XllmKernelMoeGroupedMatmulSwigluQuant);

const std::tuple<aclTensor*, aclTensor*> XllmKernelMoeGroupedMatmulSwigluQuant(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* perChannelScale,
    const aclTensor* perTokenScale,
    const aclTensor* groupList,
    double clamp_limit,
    bool use_split_arithmetic,
    aclOpExecutor* executor) {
  L0_DFX(XllmKernelMoeGroupedMatmulSwigluQuant,
         x,
         weight,
         perChannelScale,
         perTokenScale,
         groupList);
  if (x == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x is nullptr.");
    return std::tuple(nullptr, nullptr);
  }
  int64_t m = perTokenScale->GetViewShape().GetDim(0);
  int64_t n = perChannelScale->GetViewShape().GetDim(1);
  int64_t nAfterHalve = static_cast<int64_t>(n / 2);
  gert::Shape outShape({m, nAfterHalve});
  gert::Shape scaleOutShape({m});
  auto out = executor->AllocTensor(outShape, DataType::DT_INT8, ge::FORMAT_ND);
  auto scaleOut =
      executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT, ge::FORMAT_ND);

  void* outAddr = nullptr;
  void* scaleOutAddr = nullptr;
  // auto ret = INFER_SHAPE(XllmKernelMoeGroupedMatmulSwigluQuant,
  //                        OP_INPUT(x, weight, perChannelScale, perTokenScale,
  //                        groupList), OP_OUTPUT(out, scaleOut),
  //                        OP_ATTR(static_cast<float>(clamp_limit)));
  // if (ret != ACLNN_SUCCESS) {
  //     OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
  //     return std::tuple(nullptr, nullptr);
  // }

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
      XllmKernelMoeGroupedMatmulSwigluQuant,
      OP_INPUT(x, weight, perChannelScale, perTokenScale, groupList),
      OP_OUTPUT(out, scaleOut),
      OP_ATTR(static_cast<float>(clamp_limit), use_split_arithmetic));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return std::tuple(nullptr, nullptr);
  }
  return std::tie(out, scaleOut);
}

}  // namespace l0op
