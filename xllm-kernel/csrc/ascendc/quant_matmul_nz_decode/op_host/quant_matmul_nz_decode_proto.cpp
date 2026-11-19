/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "quant_matmul_nz_decode_tiling.h"
#include "register/op_def_registry.h"

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(0);
  const gert::Shape* weight_shape = context->GetInputShape(1);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x_shape;
  y_shape->SetDim(y_shape->GetDimNum() - 1,
                  weight_shape->GetDim(weight_shape->GetDimNum() - 1));
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_BF16);
  return GRAPH_SUCCESS;
}

IMPL_OP(XllmKernelQuantMatmulNzDecode)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
