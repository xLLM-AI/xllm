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
 * \file gamma_add_rms_norm.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_GAMMA_ADD_RMS_NORM_H_
#define OP_API_INC_LEVEL0_GAMMA_ADD_RMS_NORM_H_

#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t GAMMA_ADD_RMS_NORM_OUT_NUM = 3;
constexpr int GAMMA_ADD_RMS_NORM_MODE = 0;
constexpr int PRE_RMS_NORM_MODE = 1;
constexpr int POST_RMS_NORM_MODE = 2;
const std::array<aclTensor*, GAMMA_ADD_RMS_NORM_OUT_NUM>
XllmKernelGammaAddRmsNorm(const aclTensor* x1,
                          const aclTensor* x2,
                          const aclTensor* gamma,
                          double epsilon,
                          bool addGammaOffset,
                          int64_t mode,
                          aclOpExecutor* executor);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_GAMMA_ADD_RMS_NORM_H_
