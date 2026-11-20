/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_def_registry.h"

namespace {
constexpr size_t kZIndex = 3;
constexpr size_t kConvStateIndex = 5;
constexpr size_t kSsmCacheIndex = 12;
constexpr size_t kNormOutputIndex = 0;
constexpr size_t kConvStateOutputIndex = 1;
constexpr size_t kSsmCacheOutputIndex = 2;
}  // namespace

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* z_shape = context->GetInputShape(kZIndex);
  if (z_shape == nullptr || z_shape->GetDimNum() != 3) {
    return GRAPH_FAILED;
  }
  const gert::Shape* conv_state_shape = context->GetInputShape(kConvStateIndex);
  const gert::Shape* ssm_cache_shape = context->GetInputShape(kSsmCacheIndex);
  if (conv_state_shape == nullptr || ssm_cache_shape == nullptr) {
    return GRAPH_FAILED;
  }
  *context->GetOutputShape(kNormOutputIndex) = *z_shape;
  *context->GetOutputShape(kConvStateOutputIndex) = *conv_state_shape;
  *context->GetOutputShape(kSsmCacheOutputIndex) = *ssm_cache_shape;
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XllmKernelMegaGdnPrefillOp).InferShape(InferShape);
}  // namespace ge
