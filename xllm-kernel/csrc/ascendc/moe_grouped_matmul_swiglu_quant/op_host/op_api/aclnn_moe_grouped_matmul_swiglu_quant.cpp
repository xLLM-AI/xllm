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
#include "aclnn_moe_grouped_matmul_swiglu_quant.h"

#include <dlfcn.h>

#include <cmath>
#include <limits>
#include <new>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "moe_grouped_matmul_swiglu_quant_l0.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t SPLIT = 2L;
static constexpr int64_t K_LIMIT_A8W8 = 65536L;
static constexpr int64_t K_LIMIT_A8W4 = 20000L;
static constexpr int64_t N_LIMIT = 10240L;
static constexpr int64_t NZ_DIM_4_INT8 = 32L;
static constexpr int64_t NZ_DIM_4_INT4 = 64L;
static constexpr int64_t NZ_DIM_3 = 16L;
static constexpr int64_t OUTPUT_IDX_0 = 0L;
static constexpr int64_t OUTPUT_IDX_1 = 1L;
static constexpr int64_t DIM_IDX_0 = 0L;
static constexpr int64_t DIM_IDX_1 = 1L;
static constexpr int64_t DIM_IDX_2 = 2L;
static constexpr int64_t DIM_IDX_3 = 4L;
static constexpr size_t X_DIM_LIMIT = 2UL;
static constexpr size_t WEIGHT_ND_DIM_LIMIT = 3UL;
static constexpr size_t WEIGHT_NZ_DIM_LIMIT = 5UL;
static constexpr size_t WEIGHT_SCALE_DIM_LIMIT = 2UL;
static constexpr size_t WEIGHT_SCALE_PERGROUP_DIM_LIMIT = 3UL;
static constexpr size_t WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT = 2UL;
static constexpr size_t TOKEN_SCALE_DIM_LIMIT = 1UL;
static constexpr size_t BIAS_DIM_LIMIT = 2UL;
static constexpr size_t GROUP_LIST_DIM_LIMIT = 2UL;
static constexpr size_t QUANTOUT_DIM_LIMIT = 2UL;
static constexpr size_t QUANTSCALEOUT_DIM_LIMIT = 1UL;
static constexpr size_t INT4_PER_INT32 = 8UL;
bool isEnableWeightAssistanceMatrix = false;
int dequantMode = 0;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8};
static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8};
static const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,
    DataType::DT_FLOAT16,
    DataType::DT_BF16};
static const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT};
static const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT64};
static const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8};
static const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST =
    {DataType::DT_FLOAT};
static bool CheckNotNull(const aclTensor* x,
                         const aclTensor* weight,
                         const aclTensor* weightScale,
                         const aclTensor* xScale,
                         const aclTensor* groupList,
                         const aclTensor* output,
                         const aclTensor* outputScale) {
  OP_CHECK_NULL(x, return false);
  OP_CHECK_NULL(weight, return false);
  OP_CHECK_NULL(weightScale, return false);
  OP_CHECK_NULL(xScale, return false);
  OP_CHECK_NULL(groupList, return false);
  OP_CHECK_NULL(output, return false);
  OP_CHECK_NULL(outputScale, return false);
  return true;
}

static bool CheckInputOutDims_A8W8(const aclTensor* x,
                                   const aclTensor* weight,
                                   const aclTensor* weightScale,
                                   const aclTensor* xScale,
                                   const aclTensor* groupList,
                                   const aclTensor* output,
                                   const aclTensor* outputScale) {
  OP_CHECK_WRONG_DIMENSION(x, X_DIM_LIMIT, return false);
  op::Format weightViewFormat = weight->GetViewFormat();
  if (IsPrivateFormat(weightViewFormat) &&
      weight->GetViewShape().GetDimNum() != WEIGHT_ND_DIM_LIMIT) {
    OP_CHECK_WRONG_DIMENSION(weight, WEIGHT_NZ_DIM_LIMIT, return false);
  } else {
    OP_CHECK_WRONG_DIMENSION(weight, WEIGHT_ND_DIM_LIMIT, return false);
  }
  OP_CHECK_WRONG_DIMENSION(weightScale, WEIGHT_SCALE_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(xScale, TOKEN_SCALE_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(groupList, GROUP_LIST_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(output, QUANTOUT_DIM_LIMIT, return false);
  OP_CHECK_WRONG_DIMENSION(outputScale, QUANTSCALEOUT_DIM_LIMIT, return false);
  return true;
}

static bool CheckInputOutShape_A8W8(const aclTensor* x,
                                    const aclTensor* weight,
                                    const aclTensor* weightScale,
                                    const aclTensor* xScale,
                                    const aclTensor* groupList,
                                    const aclTensor* output,
                                    const aclTensor* outputScale) {
  int64_t m = x->GetViewShape().GetDim(0);
  int64_t k = x->GetViewShape().GetDim(1);
  int64_t n = weightScale->GetViewShape().GetDim(1);
  int64_t e = weight->GetViewShape().GetDim(0);
  if (n % SPLIT != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnGroupedMatmulSwiGluQuant, N is %ld , not an even number.",
            n);
    return false;
  }
  int64_t nAfterHalve = static_cast<int64_t>(n / SPLIT);
  // x的shape期望为[M, K]
  op::Shape xExpectShape = {m, k};
  // weight的NDshape期望为[E, K, N]
  op::Shape weightNDExpectShape = {e, k, n};
  // weight的NZshape期望为[E, N // 32, K // 16, 16, 32]
  op::Shape weightNZExpectShape = {e,
                                   static_cast<int64_t>(n / NZ_DIM_4_INT8),
                                   static_cast<int64_t>(k / NZ_DIM_3),
                                   NZ_DIM_3,
                                   NZ_DIM_4_INT8};
  // weightScale的shape期望为[E, N]
  op::Shape weightScaleExpectShape = {e, n};
  // xScale的shape期望为[E, N]
  op::Shape xScaleExpectShape = {m};
  // output的shape期望为[M, N]
  op::Shape outputExpectShape = {m, nAfterHalve};
  // outputScale的shape期望为[M]
  op::Shape outputScaleExpectShape = {m};
  op::Format weightViewFormat = weight->GetViewFormat();
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x, xExpectShape, return false);
  if (IsPrivateFormat(weightViewFormat) &&
      weight->GetViewShape().GetDimNum() != WEIGHT_ND_DIM_LIMIT) {
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(
        weight, weightNZExpectShape, return false);
  } else {
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(
        weight, weightNDExpectShape, return false);
  }
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(
      weightScale, weightScaleExpectShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(
      xScale, xScaleExpectShape, return false);

  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(
      output, outputExpectShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(
      outputScale, outputScaleExpectShape, return false);
  // groupList的长度应小于等于weight的专家数
  int64_t groupListLen = groupList->GetViewShape().GetDim(0);
  if (groupListLen > e) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnGroupedMatmulSwiGluQuant A8W8, Length of 'groupList' out of "
            "range (expected to be in range of [1, "
            "%ld], but got %ld)",
            e,
            groupListLen);
    return false;
  }
  if (n > N_LIMIT) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnGroupedMatmulSwiGluQuant A8W8: The current version does not "
            "support the scenario that "
            "N(%ld) is greater than %ld.",
            n,
            N_LIMIT);
    return false;
  }
  if (k >= K_LIMIT_A8W8) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnGroupedMatmulSwiGluQuant A8W8, The current version does not "
            "support the scenario."
            "The tail axis dimension of input0(x) is %ld, which need lower "
            "than %ld.",
            k,
            K_LIMIT_A8W8);
    return false;
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor* x,
                            const aclTensor* weight,
                            const aclTensor* weightScale,
                            const aclTensor* xScale,
                            const aclTensor* groupList,
                            const aclTensor* output,
                            const aclTensor* outputScale) {
  OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(
      weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(
      groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(output, QUANTOUT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(
      outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckFormat(const aclTensor* x,
                        const aclTensor* weight,
                        const aclTensor* output) {
  bool isNZ = weight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
  if (!isNZ) {
    // fp16 in fp32 out that is split k template, not precision-advanced now
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "aclnnGroupedMatmulSwiGluQuant, The current version does not "
            "support the scenario."
            "weight Format expect is FRACTAL_NZ, but got [%s].",
            op::ToString(weight->GetStorageFormat()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* x,
                               const aclTensor* weight,
                               const aclTensor* weightScale,
                               const aclTensor* xScale,
                               const aclTensor* groupList,
                               const aclTensor* output,
                               const aclTensor* outputScale) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(
                x, weight, weightScale, xScale, groupList, output, outputScale),
            ACLNN_ERR_PARAM_NULLPTR);
  // A8W8场景
  // 2. 校验输入、输出参数维度
  CHECK_RET(CheckInputOutDims_A8W8(
                x, weight, weightScale, xScale, groupList, output, outputScale),
            ACLNN_ERR_PARAM_INVALID);
  // 3. 校验输入、输出shape参数
  CHECK_RET(CheckInputOutShape_A8W8(
                x, weight, weightScale, xScale, groupList, output, outputScale),
            ACLNN_ERR_PARAM_INVALID);
  // 4. 检查输入的数据类型是否在支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(
                x, weight, weightScale, xScale, groupList, output, outputScale),
            ACLNN_ERR_PARAM_INVALID);

  // 5. 检查数据形状是否支持
  CHECK_RET(CheckFormat(x, weight, output), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static const aclTensor* SetTensorToNZFormat(const aclTensor* input,
                                            op::Shape& shape,
                                            aclOpExecutor* executor) {
  auto formatTensor =
      executor->CreateView(input, shape, input->GetViewOffset());
  formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
  formatTensor->SetOriginalFormat(input->GetViewFormat());
  formatTensor->SetViewShape(input->GetViewShape());
  return formatTensor;
}

static aclnnStatus
aclnnXllmKernelMoeGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* weightScale,
    const aclTensor* xScale,
    const aclTensor* groupList,
    double clamp_limit,
    bool use_split_arithmetic,
    aclTensor* output,
    aclTensor* outputScale,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  //
  op::Shape weightNzShape = weight->GetViewShape();
  weight = SetTensorToNZFormat(weight, weightNzShape, uniqueExecutor.get());

  auto ret = CheckParams(
      x, weight, weightScale, xScale, groupList, output, outputScale);

  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  if (output->IsEmpty() || groupList->IsEmpty() || outputScale->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  // 转连续
  x = l0op::Contiguous(x, uniqueExecutor.get());

  CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 若weight为私有格式，则不应该做连续性转换
  // （l0op::Contiguous接口会把viewShape赋值给storageShape）
  weight->SetOriginalShape(weight->GetViewShape());
  CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  weightScale = l0op::Contiguous(weightScale, uniqueExecutor.get());
  CHECK_RET(weightScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
  xScale = l0op::Contiguous(xScale, uniqueExecutor.get());
  CHECK_RET(xScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
  groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
  CHECK_RET(groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto ret_0 =
      l0op::XllmKernelMoeGroupedMatmulSwigluQuant(x,
                                                  weight,
                                                  weightScale,
                                                  xScale,
                                                  groupList,
                                                  clamp_limit,
                                                  use_split_arithmetic,
                                                  uniqueExecutor.get());
  CHECK_RET(ret_0 != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
  auto out0 = std::get<OUTPUT_IDX_0>(ret_0);
  auto ret_1 = l0op::ViewCopy(out0, output, uniqueExecutor.get());
  CHECK_RET(ret_1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto out1 = std::get<OUTPUT_IDX_1>(ret_0);
  auto ret_2 = l0op::ViewCopy(out1, outputScale, uniqueExecutor.get());
  CHECK_RET(ret_2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnXllmKernelMoeGroupedMatmulSwigluQuantGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* weightScale,
    const aclTensor* xScale,
    const aclTensor* groupList,
    aclTensor* output,
    aclTensor* outputScale,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnXllmKernelMoeGroupedMatmulSwigluQuant,
                 DFX_IN(x, weight, weightScale, xScale, groupList),
                 DFX_OUT(output, outputScale));
  // 固定写法，创建OpExecutor
  return aclnnXllmKernelMoeGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(
      x,
      weight,
      weightScale,
      xScale,
      groupList,
      0.0,
      false,
      output,
      outputScale,
      workspaceSize,
      executor);
}

aclnnStatus aclnnXllmKernelMoeGroupedMatmulSwigluQuantV2GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* weight_scale,
    const aclTensor* x_scale,
    const aclTensor* group_list,
    double clamp_limit,
    aclTensor* output,
    aclTensor* output_scale,
    uint64_t* workspace_size,
    aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspace_size, executor);
  CHECK_RET(
      std::isfinite(clamp_limit) && clamp_limit >= 0.0 &&
          clamp_limit <= static_cast<double>(std::numeric_limits<float>::max()),
      ACLNN_ERR_PARAM_INVALID);
  return aclnnXllmKernelMoeGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(
      x,
      weight,
      weight_scale,
      x_scale,
      group_list,
      clamp_limit,
      false,
      output,
      output_scale,
      workspace_size,
      executor);
}

aclnnStatus aclnnXllmKernelMoeGroupedMatmulSwigluQuantV3GetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* weight_scale,
    const aclTensor* x_scale,
    const aclTensor* group_list,
    double clamp_limit,
    bool use_split_arithmetic,
    aclTensor* output,
    aclTensor* output_scale,
    uint64_t* workspace_size,
    aclOpExecutor** executor) {
  OP_CHECK_COMM_INPUT(workspace_size, executor);
  CHECK_RET(
      std::isfinite(clamp_limit) && clamp_limit >= 0.0 &&
          clamp_limit <= static_cast<double>(std::numeric_limits<float>::max()),
      ACLNN_ERR_PARAM_INVALID);
  return aclnnXllmKernelMoeGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(
      x,
      weight,
      weight_scale,
      x_scale,
      group_list,
      clamp_limit,
      use_split_arithmetic,
      output,
      output_scale,
      workspace_size,
      executor);
}

aclnnStatus aclnnXllmKernelMoeGroupedMatmulSwigluQuantV3(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream) {
  return aclnnXllmKernelMoeGroupedMatmulSwigluQuantV2(
      workspace, workspace_size, executor, stream);
}

aclnnStatus aclnnXllmKernelMoeGroupedMatmulSwigluQuant(void* workspace,
                                                       uint64_t workspaceSize,
                                                       aclOpExecutor* executor,
                                                       aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnXllmKernelMoeGroupedMatmulSwigluQuant);
  CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
                 ACLNN_SUCCESS,
             ACLNN_ERR_INNER,
             "This is an error in GroupedMatmulSwigluQuant launch aicore");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnXllmKernelMoeGroupedMatmulSwigluQuantV2(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream) {
  return aclnnXllmKernelMoeGroupedMatmulSwigluQuant(
      workspace, workspace_size, executor, stream);
}

#ifdef __cplusplus
}
#endif
