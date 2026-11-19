/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <initializer_list>

#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kChunkSize = 128;

enum InputIndex {
  Q_INDEX = 0,
  K_INDEX,
  V_INDEX,
  G_INDEX,
  BETA_INDEX,
  MASK_LOWER_INDEX,
  MASK_FULL_INDEX,
  MINUS_IDENTITY_INDEX,
  CU_SEQLENS_INDEX,
  INITIAL_STATE_INDEX,
};

enum OutputIndex {
  OUT_INDEX = 0,
  G_SUM_INDEX,
  G_T_INDEX,
  BETA_T_INDEX,
  A_INDEX,
  A_INV_F32_INDEX,
  A_INV_INDEX,
  W_INDEX,
  U_INDEX,
  H_INDEX,
  V_NEW_INDEX,
  FINAL_STATE_INDEX,
};

enum AttrIndex {
  NUM_MATRICES_ATTR = 0,
  HAS_INITIAL_STATE_ATTR,
};

void SetShape(gert::Shape* shape, std::initializer_list<int64_t> dims) {
  shape->SetDimNum(dims.size());
  size_t index = 0;
  for (auto dim : dims) {
    shape->SetDim(index++, dim);
  }
}
}  // namespace

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* qShape = context->GetInputShape(Q_INDEX);
  const gert::Shape* vShape = context->GetInputShape(V_INDEX);
  const gert::Shape* gShape = context->GetInputShape(G_INDEX);
  const gert::Shape* cuShape = context->GetInputShape(CU_SEQLENS_INDEX);
  if (qShape == nullptr || vShape == nullptr || gShape == nullptr ||
      cuShape == nullptr) {
    return GRAPH_FAILED;
  }

  const int64_t totalTokens = qShape->GetDim(1);
  const int64_t numHeads = vShape->GetDim(2);
  int64_t numMatrices = 0;
  if (context->GetAttrs() != nullptr &&
      context->GetAttrs()->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR) !=
          nullptr) {
    numMatrices =
        *context->GetAttrs()->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR);
  }
  const int64_t inferredMatrices =
      numMatrices > 0
          ? numMatrices
          : ((totalTokens + kChunkSize - 1) / kChunkSize) * numHeads;
  const int64_t numSequences = cuShape->GetDim(0) - 1;

  *context->GetOutputShape(OUT_INDEX) = *vShape;
  *context->GetOutputShape(G_SUM_INDEX) = *gShape;
  SetShape(context->GetOutputShape(G_T_INDEX), {numHeads, totalTokens});
  SetShape(context->GetOutputShape(BETA_T_INDEX), {numHeads, totalTokens});
  SetShape(context->GetOutputShape(A_INDEX),
           {1, totalTokens, numHeads, kChunkSize});
  SetShape(context->GetOutputShape(A_INV_F32_INDEX),
           {1, totalTokens, numHeads, kChunkSize});
  SetShape(context->GetOutputShape(A_INV_INDEX),
           {1, totalTokens, numHeads, kChunkSize});
  *context->GetOutputShape(W_INDEX) = *vShape;
  *context->GetOutputShape(U_INDEX) = *vShape;
  SetShape(context->GetOutputShape(H_INDEX),
           {inferredMatrices, kHeadDim, kHeadDim});
  *context->GetOutputShape(V_NEW_INDEX) = *vShape;
  SetShape(context->GetOutputShape(FINAL_STATE_INDEX),
           {numSequences * numHeads, kHeadDim, kHeadDim});
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const ge::DataType vDtype = context->GetInputDataType(V_INDEX);
  const ge::DataType gDtype = context->GetInputDataType(G_INDEX);
  const ge::DataType betaDtype = context->GetInputDataType(BETA_INDEX);

  context->SetOutputDataType(OUT_INDEX, vDtype);
  context->SetOutputDataType(G_SUM_INDEX, gDtype);
  context->SetOutputDataType(G_T_INDEX, gDtype);
  context->SetOutputDataType(BETA_T_INDEX, betaDtype);
  context->SetOutputDataType(A_INDEX, betaDtype);
  context->SetOutputDataType(A_INV_F32_INDEX, ge::DT_FLOAT);
  context->SetOutputDataType(A_INV_INDEX, betaDtype);
  context->SetOutputDataType(W_INDEX, vDtype);
  context->SetOutputDataType(U_INDEX, vDtype);
  context->SetOutputDataType(H_INDEX, vDtype);
  context->SetOutputDataType(V_NEW_INDEX, vDtype);
  context->SetOutputDataType(FINAL_STATE_INDEX, vDtype);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XllmKernelMegaChunkGdn)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
