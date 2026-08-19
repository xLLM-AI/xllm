/* Copyright 2025-2026 The xLLM Authors.

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

namespace xllm {

#ifdef XLLM_CAPI_WEAK
#define XLLM_CAPI_EXPORT \
  __attribute__((visibility("default"))) __attribute((weak))
#else
#define XLLM_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // XLLM_CAPI_WEAK

}  // namespace xllm
