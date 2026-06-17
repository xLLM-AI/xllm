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

namespace lanczos {

void resize_8bpc(const uint8_t* src,
                 int32_t src_w,
                 int32_t src_h,
                 int32_t channels,
                 int32_t dst_w,
                 int32_t dst_h,
                 uint8_t* dst);

void resize_f32(const float* src,
                int32_t src_w,
                int32_t src_h,
                int32_t channels,
                int32_t dst_w,
                int32_t dst_h,
                float* dst);

}  // namespace lanczos