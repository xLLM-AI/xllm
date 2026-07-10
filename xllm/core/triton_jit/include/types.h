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

#include <cstdint>

namespace xllm::triton_jit {

// Kernel launch grid (number of blocks in each dimension).
struct Grid {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// Triton compile / launch meta-parameters.
struct LaunchCfg {
  int32_t num_warps = 1;
  int32_t num_stages = 1;
};

}  // namespace xllm::triton_jit
