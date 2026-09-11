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

// torch::Tensor -> xlite XTensor bridge (XDtypeOf/TensorPtr/InitXTensor in
// xlite/xlite.h).

#pragma once

#include <torch/torch.h>
#include <xlite/xlite.h>

#include "core/framework/parallel_state/parallel_args.h"

namespace xllm::xlite {

// xlite symbols are global (::); bring into xllm::xlite.
using ::InitXTensor;
using ::TensorPtr;
using ::XDtypeOf;
using ::XTensor;

// Framework ParallelArgs::tp_size() is not populated; derive from
// world_size/dp_size.
inline uint32_t XliteTpSize(const ParallelArgs& pa) {
  int32_t dp = pa.dp_size();
  if (dp <= 0) {
    return 1u;
  }
  return static_cast<uint32_t>(pa.world_size() / dp);
}

inline uint32_t XliteTpRank(const ParallelArgs& pa) {
  uint32_t tp = XliteTpSize(pa);
  return tp > 0 ? static_cast<uint32_t>(pa.rank()) % tp : 0u;
}

}  // namespace xllm::xlite