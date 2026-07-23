/* Copyright 2026 The xLLM Authors.

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

#include "flash_comm1_context.h"

namespace xllm {
namespace parallel_state {
namespace {

// Thread-local so a per-thread model forward can publish its own context
// without cross-thread interference (each worker runs its own forward).
thread_local FlashComm1Context t_flash_comm1_context;

}  // namespace

const FlashComm1Context& current_flash_comm1_context() {
  return t_flash_comm1_context;
}

bool flash_comm1_active() { return t_flash_comm1_context.enabled; }

FlashComm1Guard::FlashComm1Guard(bool enabled,
                                 int64_t num_tokens,
                                 int32_t tp_rank,
                                 int32_t tp_world_size,
                                 ProcessGroup* tp_group)
    : previous_(t_flash_comm1_context) {
  FlashComm1Context context;
  context.enabled = enabled;
  context.num_tokens = num_tokens;
  context.tp_rank = tp_rank;
  context.tp_world_size = tp_world_size;
  context.tp_group = tp_group;
  t_flash_comm1_context = context;
}

FlashComm1Guard::~FlashComm1Guard() { t_flash_comm1_context = previous_; }

}  // namespace parallel_state
}  // namespace xllm
