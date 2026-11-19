/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ge {
namespace {
constexpr size_t kInputX = 0;
constexpr size_t kAttrGroupSize = 1;
constexpr size_t kAttrIsRmsNorm = 3;
constexpr size_t kOutputY = 0;
constexpr size_t kOutputMean = 1;
constexpr size_t kOutputRstd = 2;
}  // namespace

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(kInputX);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  gert::Shape* y_shape = context->GetOutputShape(kOutputY);
  gert::Shape* mean_shape = context->GetOutputShape(kOutputMean);
  gert::Shape* rstd_shape = context->GetOutputShape(kOutputRstd);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, mean_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);

  *y_shape = *x_shape;

  const int64_t dim_num = x_shape->GetDimNum();
  if (dim_num <= 0) {
    return GRAPH_FAILED;
  }
  int64_t full_n = 1;
  int64_t m = 1;
  full_n = x_shape->GetDim(dim_num - 1);
  for (int64_t i = 0; i < dim_num - 1; ++i) {
    m *= x_shape->GetDim(i);
  }

  int64_t group_size = full_n;
  auto attrs = context->GetAttrs();
  if (attrs != nullptr &&
      attrs->GetAttrPointer<int64_t>(kAttrGroupSize) != nullptr) {
    group_size = *attrs->GetAttrPointer<int64_t>(kAttrGroupSize);
  }
  if (group_size <= 0) {
    group_size = full_n;
  }
  if (full_n <= 0 || group_size <= 0 || full_n % group_size != 0) {
    return GRAPH_FAILED;
  }
  const int64_t ngroups = group_size > 0 ? full_n / group_size : 1;

  bool is_rms_norm = false;
  if (attrs != nullptr &&
      attrs->GetAttrPointer<bool>(kAttrIsRmsNorm) != nullptr) {
    is_rms_norm = *attrs->GetAttrPointer<bool>(kAttrIsRmsNorm);
  }

  mean_shape->SetDimNum(1);
  mean_shape->SetDim(0, is_rms_norm ? 0 : ngroups * m);
  rstd_shape->SetDimNum(1);
  rstd_shape->SetDim(0, ngroups * m);
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto input_dtype = context->GetInputDataType(kInputX);
  context->SetOutputDataType(kOutputY, input_dtype);
  context->SetOutputDataType(kOutputMean, ge::DT_FLOAT);
  context->SetOutputDataType(kOutputRstd, ge::DT_FLOAT);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XllmKernelLayerNormFwd)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
