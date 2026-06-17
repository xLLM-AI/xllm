/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

namespace xllm {
namespace parallel_state {

// torch_npu ProcessGroupHCCL consumes ranktable through this HCCL environment
// variable. Keep it in one place so ATB and TORCH routes can share the same
// user-facing --rank_tablefile option.
void sync_torch_npu_rank_table_file_env(const std::string& rank_tablefile);

}  // namespace parallel_state
}  // namespace xllm
