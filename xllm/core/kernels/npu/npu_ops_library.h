/* Copyright 2026 The xLLM Authors.

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

namespace xllm {

// Anchor symbol that forces the linker to keep the translation unit holding the
// TORCH_LIBRARY(xllm_ops) static registrations for NPU (PrivateUse1).
// See cuda_ops_library.h for the CUDA counterpart.
void ensure_xllm_ops_registered();

}  // namespace xllm
