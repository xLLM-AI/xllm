/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "replace_token_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  ReplaceTokenTilingData tiling;
  auto aShape = context->GetInputShape(0)->GetOriginShape();
  auto bShape = context->GetInputShape(1)->GetOriginShape();
  uint32_t max_tokens = 10000;
  int32_t sequenceLength = aShape.GetDim(0);
  int32_t blength = bShape.GetDim(0);
  tiling.set_sequenceLength(sequenceLength);
  tiling.set_max_tokens(max_tokens);
  tiling.set_blength(blength);
  context->SetBlockDim(1);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(XllmKernelReplaceToken).Tiling(TilingFunc);
}  // namespace optiling