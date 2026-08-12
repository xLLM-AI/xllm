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

#pragma once

#include <string>
#include <string_view>

namespace xllm {

class ModelArgs;

namespace runtime {
class Options;
}

namespace block_diffusion {

bool is_algorithm(std::string_view algorithm);

// Apply the common block-diffusion checkpoint contract, then delegate model
// registration names and backend-only capabilities to the platform policy.
void configure_model_args(ModelArgs& args,
                          const runtime::Options& options,
                          const std::string& model_weights_path);

}  // namespace block_diffusion
}  // namespace xllm
