/* Copyright 2026 The xLLM Authors.

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

#include "layers/mlu/dcp_batch_metadata.h"

#include <glog/logging.h>

#include <utility>

namespace xllm::layer {

std::pair<torch::Tensor, torch::Tensor> build_prefill_slot_order(
    const torch::Tensor& source_slots) {
  CHECK_EQ(source_slots.dim(), 1);
  CHECK(source_slots.scalar_type() == torch::kInt32 ||
        source_slots.scalar_type() == torch::kInt64);
  auto [slots, rows] = torch::sort(source_slots);
  const int64_t token_count = slots.numel();
  if (token_count > 1) {
    torch::Tensor repeated =
        (slots.slice(0, 1) == slots.slice(0, 0, -1)) & (slots.slice(0, 1) >= 0);
    repeated = torch::cat({torch::zeros({1}, repeated.options()), repeated});
    torch::Tensor indices = torch::arange(token_count, rows.options());
    // index_select checks bounds on MLU without copying a predicate to CPU.
    // A duplicate source has no unique K row and must fail, not pick a winner.
    rows = rows.index_select(0, torch::where(repeated, token_count, indices));
  }
  return {std::move(slots), std::move(rows)};
}

}  // namespace xllm::layer
