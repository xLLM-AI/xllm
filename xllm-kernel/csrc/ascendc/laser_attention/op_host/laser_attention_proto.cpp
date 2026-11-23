/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

namespace ops {

static ge::graphStatus LaserAttentionInferShape(
    gert::InferShapeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape* queryShape = context->GetInputShape(0);
  if (queryShape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  gert::Shape* softmaxOut = context->GetOutputShape(0);
  if (softmaxOut == nullptr) {
    return ge::GRAPH_FAILED;
  }
  int32_t queryDimNum = static_cast<int32_t>(queryShape->GetDimNum());
  if (queryDimNum < 4) {  // query dim num is 4
    return ge::GRAPH_FAILED;
  }
  softmaxOut->SetDimNum(queryDimNum - 1);
  softmaxOut->SetDim(0, queryShape->GetDim(0));
  softmaxOut->SetDim(1, queryShape->GetDim(1));
  softmaxOut->SetDim(2, queryShape->GetDim(2));  // index is 2

  gert::Shape* attnOut = context->GetOutputShape(1);
  if (attnOut == nullptr) {
    return ge::GRAPH_FAILED;
  }
  attnOut->SetDimNum(queryDimNum);
  attnOut->SetDim(0, queryShape->GetDim(0));
  attnOut->SetDim(1, queryShape->GetDim(1));
  attnOut->SetDim(2, queryShape->GetDim(2));  // index is 2
  attnOut->SetDim(3, queryShape->GetDim(3));  // index is 3

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LaserAttentionInferDtype(
    gert::InferDataTypeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const ge::DataType queryDtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, ge::DT_FLOAT);
  context->SetOutputDataType(1, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XllmKernelLaserAttention)
    .InferShape(LaserAttentionInferShape)
    .InferDataType(LaserAttentionInferDtype);

}  // namespace ops
