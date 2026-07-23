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

#include <cstdint>
#include <vector>

namespace xllm {

bool use_push_owner(int32_t src_tp_size, int32_t dst_tp_size);

std::vector<int32_t> get_dst_ranks(int32_t src_tp_rank,
                                   int32_t src_tp_size,
                                   int32_t dst_tp_size,
                                   int32_t dst_dp_rank);

std::vector<int32_t> get_src_tp_ranks(int32_t dst_tp_rank,
                                      int32_t src_tp_size,
                                      int32_t dst_tp_size);

}  // namespace xllm
