/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <string>

#include "core/framework/model/model_family.h"

namespace xllm {

struct ResolvedModelDescriptor;

// Family-specific policies compose with the shared HF checkpoint I/O.
struct HFModelLoaderProfile {
  ModelFamily family = ModelFamily::UNKNOWN;
  bool load_media_preprocessor_args = false;
  bool load_rec_vocab = false;
};

// The requested execution contract takes precedence over a registration's
// default family: a registration may support both LLM and Rec execution.
// Factory availability is validated when creating the model context.
bool select_hf_model_loader_profile(const ResolvedModelDescriptor& descriptor,
                                    const std::string& requested_family,
                                    HFModelLoaderProfile* profile,
                                    std::string* error_message);

}  // namespace xllm
