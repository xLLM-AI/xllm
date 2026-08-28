/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

void pack_cache_blocks(const std::vector<torch::Tensor>& sources,
                       const torch::Tensor& block_ids,
                       const std::vector<torch::Tensor>& destinations) {
  CHECK_EQ(sources.size(), destinations.size());
  for (size_t index = 0; index < sources.size(); ++index) {
    destinations[index].copy_(
        sources[index].index_select(/*dim=*/0, block_ids));
  }
}

}  // namespace xllm::kernel::mlu
