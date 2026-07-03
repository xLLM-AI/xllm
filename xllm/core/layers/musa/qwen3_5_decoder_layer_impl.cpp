/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/musa/qwen3_5_decoder_layer_impl.h"

#include <memory>

namespace xllm {
namespace layer {

Qwen3_5DecoderLayerImpl::Qwen3_5DecoderLayerImpl(const ModelContext& context,
                                                 int32_t layer_id)
    : Qwen3NextDecoderLayerImpl(context,
                                layer_id,
                                std::make_shared<Qwen3_5GatedDeltaNetImpl>(
                                    context.get_model_args(),
                                    context.get_quant_args(),
                                    context.get_parallel_args(),
                                    context.get_tensor_options())) {}

}  // namespace layer
}  // namespace xllm
