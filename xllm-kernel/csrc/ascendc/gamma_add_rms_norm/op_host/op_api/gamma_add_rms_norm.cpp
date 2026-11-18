/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
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
 * \file gamma_add_rms_norm.cpp
 * \brief
 */
#include "gamma_add_rms_norm.h"

#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(XllmKernelGammaAddRmsNorm);

const std::array<aclTensor*, GAMMA_ADD_RMS_NORM_OUT_NUM>
XllmKernelGammaAddRmsNorm(const aclTensor* x1,
                          const aclTensor* x2,
                          const aclTensor* gamma,
                          double epsilon,
                          bool addGammaOffset,
                          int64_t mode,
                          aclOpExecutor* executor) {
  L0_DFX(XllmKernelGammaAddRmsNorm, x1, x2, gamma, epsilon, addGammaOffset);
  Shape dummyShape({0});
  if (mode == GAMMA_ADD_RMS_NORM_MODE) {
    Shape rstdShape;
    size_t x1DimNum = x1->GetViewShape().GetDimNum();
    size_t gammaDimNum = gamma->GetViewShape().GetDimNum();
    for (uint32_t i = 0; i < x1DimNum - gammaDimNum; i++) {
      rstdShape.AppendDim(x1->GetViewShape().GetDim(i));
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
      rstdShape.AppendDim(1);
    }
    dummyShape = rstdShape;
  }

  auto yOut = executor->AllocTensor(
      x1->GetViewShape(), x1->GetDataType(), x1->GetViewFormat());
  auto rstdOut = executor->AllocTensor(
      dummyShape, DataType::DT_FLOAT, x1->GetViewFormat());
  auto xOut = executor->AllocTensor(
      x1->GetViewShape(), x1->GetDataType(), x1->GetViewFormat());
  if (mode == PRE_RMS_NORM_MODE) {
    xOut = executor->AllocTensor(
        x1->GetViewShape(), x1->GetDataType(), x1->GetViewFormat());
  } else if (mode == POST_RMS_NORM_MODE) {
    xOut = executor->AllocTensor(
        dummyShape, x1->GetDataType(), x1->GetViewFormat());
  }

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
      XllmKernelGammaAddRmsNorm,
      OP_INPUT(x1, x2, gamma),
      OP_OUTPUT(yOut, rstdOut, xOut),
      OP_ATTR(static_cast<float>(epsilon), addGammaOffset));
  if (ret != ACL_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "XllmKernelGammaAddRmsNorm ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return {nullptr, nullptr, nullptr};
  }
  return {yOut, rstdOut, xOut};
}
}  // namespace l0op
