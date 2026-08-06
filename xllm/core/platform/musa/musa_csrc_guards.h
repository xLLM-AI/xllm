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

#define EIGEN_NO_CUDA
#define EIGEN_NO_GPU
// Include only from MUSA graph TUs that also pull ATen/cuda/* compatibility
// headers.
#define TORCH_MUSA_CSRC_CORE_MUSACACHINGALLOCATOR_H_
#define TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_