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
#include "aclnn_gamma_add_rms_norm.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "gamma_add_rms_norm.h"
#include "op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace GammaAddRmsNormACLNN {
constexpr int IDX_2 = 2;
constexpr int IDX_1 = 1;
constexpr int IDX_0 = 0;
constexpr int GAMMA_ADD_RMS_NORM_MODE = 0;
constexpr int PRE_RMS_NORM_MODE = 1;
constexpr int POST_RMS_NORM_MODE = 2;
const size_t MIN_SUPPORT_DIMS_NUMS = 1;
const size_t DIM_TWO = 2;

struct GammaAddRmsNormInputTensor {
  const aclTensor* x1;
  const aclTensor* x2;
  const aclTensor* gamma;
};

struct GammaAddRmsNormOutputTensor {
  aclTensor* yOut;
  aclTensor* rstdOut;
  aclTensor* xOut;
};

static const std::initializer_list<op::DataType> EXTEND_ATB_DTYPE_SUPPORT_LIST =
    {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> NORMAL_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,
    op::DataType::DT_FLOAT16,
    op::DataType::DT_BF16};

static bool CheckNotNull(GammaAddRmsNormInputTensor& inputTensor,
                         GammaAddRmsNormOutputTensor& outputTensor,
                         int64_t mode) {
  OP_CHECK_NULL(inputTensor.x1, return false);
  OP_CHECK_NULL(inputTensor.x2, return false);
  OP_CHECK_NULL(inputTensor.gamma, return false);
  OP_CHECK_NULL(outputTensor.yOut, return false);
  if (mode == GammaAddRmsNormACLNN::GAMMA_ADD_RMS_NORM_MODE) {
    OP_CHECK_NULL(outputTensor.rstdOut, return false);
    OP_CHECK_NULL(outputTensor.xOut, return false);
  } else if (mode == GammaAddRmsNormACLNN::PRE_RMS_NORM_MODE) {
    OP_CHECK_NULL(outputTensor.xOut, return false);
  }
  return true;
}

static bool CheckDtypeValid(GammaAddRmsNormInputTensor& inputTensor,
                            GammaAddRmsNormOutputTensor& outputTensor,
                            int64_t mode) {
  std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST =
      NORMAL_DTYPE_SUPPORT_LIST;
  if (mode == GammaAddRmsNormACLNN::PRE_RMS_NORM_MODE ||
      mode == GammaAddRmsNormACLNN::POST_RMS_NORM_MODE) {
    DTYPE_SUPPORT_LIST = EXTEND_ATB_DTYPE_SUPPORT_LIST;
  }

  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x1, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x2, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(
      inputTensor.gamma, DTYPE_SUPPORT_LIST, return false);

  OP_CHECK_DTYPE_NOT_SAME(inputTensor.x1, inputTensor.x2, return false);
  OP_CHECK_DTYPE_NOT_SAME(inputTensor.x1, inputTensor.gamma, return false);

  OP_CHECK_DTYPE_NOT_SUPPORT(
      outputTensor.yOut, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SAME(outputTensor.yOut, inputTensor.x1, return false);

  if (mode == GammaAddRmsNormACLNN::GAMMA_ADD_RMS_NORM_MODE) {
    OP_CHECK_DTYPE_NOT_SUPPORT(
        outputTensor.xOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(outputTensor.xOut, outputTensor.yOut, return false);
    OP_CHECK_DTYPE_NOT_MATCH(
        outputTensor.rstdOut, op::DataType::DT_FLOAT, return false);
  }
  if (mode == GammaAddRmsNormACLNN::PRE_RMS_NORM_MODE) {
    OP_CHECK_DTYPE_NOT_SUPPORT(
        outputTensor.xOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(outputTensor.xOut, outputTensor.yOut, return false);
  }
  return true;
}

static bool CheckShapeDim(GammaAddRmsNormInputTensor& inputTensor,
                          GammaAddRmsNormOutputTensor& outputTensor,
                          int64_t mode) {
  OP_CHECK_MAX_DIM(inputTensor.x1, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(inputTensor.x2, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(inputTensor.gamma, MAX_SUPPORT_DIMS_NUMS, return false);

  OP_CHECK_MIN_DIM(inputTensor.x1, MIN_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MIN_DIM(inputTensor.x2, MIN_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MIN_DIM(inputTensor.gamma, MIN_SUPPORT_DIMS_NUMS, return false);

  OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, inputTensor.x2, return false);

  OP_CHECK_MAX_DIM(outputTensor.yOut, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, outputTensor.yOut, return false);

  if (mode == GammaAddRmsNormACLNN::GAMMA_ADD_RMS_NORM_MODE) {
    OP_CHECK_MAX_DIM(outputTensor.rstdOut, MAX_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_MAX_DIM(outputTensor.xOut, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, outputTensor.xOut, return false);
  }
  if (mode == GammaAddRmsNormACLNN::PRE_RMS_NORM_MODE) {
    OP_CHECK_MAX_DIM(inputTensor.gamma, DIM_TWO, return false);
    OP_CHECK_MAX_DIM(outputTensor.xOut, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(inputTensor.x1, outputTensor.xOut, return false);
  }
  if (mode == GammaAddRmsNormACLNN::POST_RMS_NORM_MODE) {
    OP_CHECK_MAX_DIM(inputTensor.gamma, DIM_TWO, return false);
  }
  return true;
}

static aclnnStatus CheckParams(GammaAddRmsNormInputTensor& inputTensor,
                               GammaAddRmsNormOutputTensor& outputTensor,
                               int64_t& mode) {
  if (outputTensor.xOut != nullptr && outputTensor.rstdOut == nullptr) {
    mode = GammaAddRmsNormACLNN::PRE_RMS_NORM_MODE;  // Two outputs: pre-RMSNorm
                                                     // mode.
  } else if (outputTensor.xOut == nullptr && outputTensor.rstdOut == nullptr) {
    mode = GammaAddRmsNormACLNN::POST_RMS_NORM_MODE;  // One output:
                                                      // post-RMSNorm mode.
  }
  // 1. Check required input and output pointers.
  CHECK_RET(CheckNotNull(inputTensor, outputTensor, mode),
            ACLNN_ERR_PARAM_NULLPTR);

  // 2. Validate input and output data types.
  CHECK_RET(CheckDtypeValid(inputTensor, outputTensor, mode),
            ACLNN_ERR_PARAM_INVALID);

  // 3. Validate input and output shapes.
  CHECK_RET(CheckShapeDim(inputTensor, outputTensor, mode),
            ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus ComputeGammaAddRmsNorm(GammaAddRmsNormInputTensor& inputTensor,
                                   GammaAddRmsNormOutputTensor& outputTensor,
                                   double& epsilon,
                                   bool addGammaOffset,
                                   int64_t& mode,
                                   aclOpExecutor* executor) {
  aclTensor* yComputeOut = nullptr;
  aclTensor* rstdComputeOut = nullptr;
  aclTensor* xComputeOut = nullptr;

  auto GammaAddRmsNormOuts = l0op::XllmKernelGammaAddRmsNorm(inputTensor.x1,
                                                             inputTensor.x2,
                                                             inputTensor.gamma,
                                                             epsilon,
                                                             addGammaOffset,
                                                             mode,
                                                             executor);
  yComputeOut = std::get<IDX_0>(GammaAddRmsNormOuts);
  rstdComputeOut = std::get<IDX_1>(GammaAddRmsNormOuts);
  xComputeOut = std::get<IDX_2>(GammaAddRmsNormOuts);
  CHECK_RET(yComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // Copy yComputeOut to yOut.
  auto viewCopyYResult =
      l0op::ViewCopy(yComputeOut, outputTensor.yOut, executor);
  CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (mode == GammaAddRmsNormACLNN::GAMMA_ADD_RMS_NORM_MODE) {
    // Copy rstdComputeOut to rstdOut.
    auto viewCopyXResult =
        l0op::ViewCopy(rstdComputeOut, outputTensor.rstdOut, executor);
    CHECK_RET(viewCopyXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // Copy xComputeOut to xOut.
    auto viewCopyRstdResult =
        l0op::ViewCopy(xComputeOut, outputTensor.xOut, executor);
    CHECK_RET(viewCopyRstdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (mode == GammaAddRmsNormACLNN::PRE_RMS_NORM_MODE) {
    // Copy xComputeOut to xOut.
    auto viewCopyRstdResult =
        l0op::ViewCopy(xComputeOut, outputTensor.xOut, executor);
    CHECK_RET(viewCopyRstdResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}
}  // namespace GammaAddRmsNormACLNN

aclnnStatus aclnnGammaAddRmsNormGetWorkspaceSize(const aclTensor* x1,
                                                 const aclTensor* x2,
                                                 const aclTensor* gamma,
                                                 double epsilon,
                                                 bool addGammaOffset,
                                                 aclTensor* yOut,
                                                 aclTensor* rstdOut,
                                                 aclTensor* xOut,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor) {
  OP_LOGD("Enter aclnnGammaAddRmsNormGetWorkspaceSize.");
  L2_DFX_PHASE_1(aclnnGammaAddRmsNorm,
                 DFX_IN(x1, x2, gamma, epsilon, addGammaOffset),
                 DFX_OUT(yOut, rstdOut, xOut));

  // Create the operator executor.
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // Validate parameters.
  GammaAddRmsNormACLNN::GammaAddRmsNormInputTensor inputTensorOri = {
      x1, x2, gamma};
  GammaAddRmsNormACLNN::GammaAddRmsNormOutputTensor outputTensor = {
      yOut, rstdOut, xOut};

  int64_t mode =
      GammaAddRmsNormACLNN::GAMMA_ADD_RMS_NORM_MODE;  // 0: AddRmsNorm, 1:
                                                      // pre-RMSNorm, 2:
                                                      // post-RMSNorm.
  auto ret = CheckParams(inputTensorOri, outputTensor, mode);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // Support empty tensors.
  bool anyEmptyTensor = x1->IsEmpty() || gamma->IsEmpty();
  if (anyEmptyTensor) {
    OP_LOGW("Got empty tensor in aclnnGammaAddRmsNorm!");
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // Convert inputs to contiguous tensors. Optional inputs do not require null
  // checks here.
  auto x1Cont = l0op::Contiguous(x1, uniqueExecutor.get());
  auto x2Cont = l0op::Contiguous(x2, uniqueExecutor.get());
  auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());

  CHECK_RET(x1Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(x2Cont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

  GammaAddRmsNormACLNN::GammaAddRmsNormInputTensor inputTensor = {
      x1Cont, x2Cont, gammaCont};

  ret = GammaAddRmsNormACLNN::ComputeGammaAddRmsNorm(inputTensor,
                                                     outputTensor,
                                                     epsilon,
                                                     addGammaOffset,
                                                     mode,
                                                     uniqueExecutor.get());
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // Obtain the workspace size required for computation.
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  OP_LOGD("Finish aclnnGammaAddRmsNormGetWorkspaceSize.");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGammaAddRmsNorm(void* workspace,
                                 uint64_t workspaceSize,
                                 aclOpExecutor* executor,
                                 aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGammaAddRmsNorm);
  // Execute the operator through the framework executor.
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
