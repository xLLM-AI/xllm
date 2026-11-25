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

#include <cstdint>
#include <optional>
#include <string>

#include "core/framework/request/dit_request_params.h"

namespace xllm {

class DiTMaster;

struct DiTGenerationCommand {
  std::string modality;
  std::string request_path;
  std::string output_path;
};

std::optional<DiTRequestParams> load_dit_generation_request(
    const DiTGenerationCommand& command);

int32_t generate_dit_output(DiTMaster& master,
                            DiTRequestParams request,
                            const std::string& output_path);

}  // namespace xllm
