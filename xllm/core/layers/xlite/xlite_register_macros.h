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

// Registers "<ModelType>_xlite" model class + args loader.

#pragma once

#include <torch/torch.h>

#include "core/framework/model_context.h"
#include "core/layers/xlite/xlite_causal_lm_base.h"
#include "models/model_registry.h"

namespace xllm::xlite {

#define XLITE_REGISTER_MODEL(ModelType, AdapterClass, ArgsLambda)         \
  class ModelType##XliteForCausalLMImpl : public XliteCausalLMBase {      \
   public:                                                                \
    ModelType##XliteForCausalLMImpl(const ModelContext& ctx)              \
        : XliteCausalLMBase(ctx, std::make_unique<AdapterClass>()) {}     \
  };                                                                      \
  TORCH_MODULE(ModelType##XliteForCausalLM);                              \
  REGISTER_CAUSAL_MODEL_WITH_VARNAME(                                     \
      ModelType##_xlite, ModelType##_xlite, ModelType##XliteForCausalLM); \
  REGISTER_MODEL_ARGS_WITH_VARNAME(                                       \
      ModelType##_xlite, ModelType##_xlite, ArgsLambda)

}  // namespace xllm::xlite