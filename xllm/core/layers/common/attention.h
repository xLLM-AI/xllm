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

#pragma once

#if defined(USE_MLU)
#include "layers/mlu/attention.h"
#elif defined(USE_NPU)
#include "layers/npu_torch/attention.h"
#elif defined(USE_MUSA)
#include "layers/musa/attention.h"
#elif defined(USE_CUDA)
#include "layers/cuda/attention.h"
#elif defined(USE_ILU)
#include "layers/ilu/attention.h"
#elif defined(USE_DCU)
#include "layers/dcu/attention.h"
#endif
