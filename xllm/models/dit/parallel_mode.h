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

#include <glog/logging.h>

#include <cstdint>
#include <type_traits>

#include "core/framework/parallel_state/process_group.h"
#include "models/dit/sequence_parallel/sequence_parallel_mixin.h"

namespace xllm::dit {

enum class ParallelMode : int8_t {
  DEFAULT = 0,
  SEQUENCE_PARALLEL = 1,
};

template <typename ModelType>
ParallelMode resolve_parallel_mode(ProcessGroup* process_group) {
  if (process_group == nullptr || process_group->world_size() <= 1) {
    return ParallelMode::DEFAULT;
  }

  constexpr bool kSupportsSequenceParallel =
      std::is_base_of_v<SequenceParallelMixin, ModelType>;
  CHECK(kSupportsSequenceParallel)
      << "Sequence parallelism is enabled, but the model does not inherit "
         "SequenceParallelMixin";
  return ParallelMode::SEQUENCE_PARALLEL;
}

}  // namespace xllm::dit
