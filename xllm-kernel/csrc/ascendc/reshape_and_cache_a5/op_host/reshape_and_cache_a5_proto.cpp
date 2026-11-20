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

#include "register/op_def_registry.h"

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  *context->GetOutputShape(0) = *context->GetInputShape(2);
  *context->GetOutputShape(1) = *context->GetInputShape(3);
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(2));
  context->SetOutputDataType(1, context->GetInputDataType(3));
  return GRAPH_SUCCESS;
}

IMPL_OP(XllmKernelReshapeAndCacheA5)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
