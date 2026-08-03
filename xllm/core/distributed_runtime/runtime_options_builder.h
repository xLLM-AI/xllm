/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/common/options.h"
#include "core/runtime/options.h"

namespace xllm {

// Converts configuration shared by causal LLM and VLM engines into runtime
// options. Engine-specific settings are applied by their respective builders.
runtime::Options make_runtime_options(
    const Options& options,
    const std::vector<torch::Device>& devices);

// Adds draft-model and speculative-decoding settings while preserving the
// target backend selected by make_runtime_options().
runtime::Options make_speculative_runtime_options(
    const Options& options,
    const std::vector<torch::Device>& devices,
    const std::vector<torch::Device>& draft_devices);

}  // namespace xllm
