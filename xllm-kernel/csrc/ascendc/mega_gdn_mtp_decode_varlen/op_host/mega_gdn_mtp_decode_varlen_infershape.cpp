/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <array>
#include <cstddef>

#include "register/op_impl_registry.h"

namespace ops {
namespace {

// convOut <- qkv, convStateOut <- convState, ssmStateOut <- ssmState, out <- z.
constexpr std::array<size_t, 4> kOutputInputIndices = {0, 5, 8, 1};

ge::graphStatus infer_shape(gert::InferShapeContext* context) {
  for (size_t output_index = 0; output_index < kOutputInputIndices.size();
       ++output_index) {
    const gert::Shape* input =
        context->GetInputShape(kOutputInputIndices[output_index]);
    gert::Shape* output = context->GetOutputShape(output_index);
    if (input == nullptr || output == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *output = *input;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus infer_data_type(gert::InferDataTypeContext* context) {
  for (size_t output_index = 0; output_index < kOutputInputIndices.size();
       ++output_index) {
    context->SetOutputDataType(
        output_index,
        context->GetInputDataType(kOutputInputIndices[output_index]));
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_INFERSHAPE(XllmKernelMegaGdnMtpDecodeVarlen)
    .InferShape(infer_shape)
    .InferDataType(infer_data_type);

}  // namespace ops
