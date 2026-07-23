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

#pragma once

#include <cstdint>

namespace xllm {

class ProcessGroup;

namespace parallel_state {

// Per-forward FlashComm1 (sequence-parallel) state.
//
// FlashComm1 keeps the residual stream token-sharded across the TP group for
// the whole decoder stack. Each sequence-parallel boundary all-gathers back to
// the full token dimension before attention / MoE, and the RowParallelLinear
// tail all-reduce is replaced by a padded reduce-scatter(dim0) so the output
// returns to a token shard. Norm / hc / gate therefore run on 1/tp_world_size
// tokens.
//
// Layers such as RowParallelLinearImpl::forward() do not receive
// ModelInputParams, so the active context is published via a thread_local set
// by FlashComm1Guard at the top of each model forward().
struct FlashComm1Context {
  // True when a FlashComm1 sequence-parallel forward is active. The caller is
  // responsible for folding in the token-count threshold, tp_world_size > 1 and
  // any backend / graph checks before setting this.
  bool enabled = false;

  // Original (un-sharded, un-padded) number of tokens in this forward. Used to
  // unpad after an all-gather at sequence-parallel boundaries.
  int64_t num_tokens = 0;

  int32_t tp_rank = 0;
  int32_t tp_world_size = 1;

  // TP process group used for the sequence-parallel all-gather / reduce-scatter
  // boundaries. Populated by FlashComm1Guard so layers that only see the
  // context (e.g. the DSV4 decoder layer, RowParallelLinear) can reach the
  // group without extra plumbing.
  ProcessGroup* tp_group = nullptr;
};

// Returns the FlashComm1 context active on the current thread. When no guard is
// in scope the returned context has enabled == false.
const FlashComm1Context& current_flash_comm1_context();

// Convenience helper: true when a FlashComm1 context is active AND enabled.
bool flash_comm1_active();

// RAII guard that publishes a FlashComm1 context for the duration of a model
// forward() on the calling thread and restores the previous context on scope
// exit. Guards nest: the previous context is saved and restored, so an inner
// guard (e.g. a nested sub-model forward) does not clobber the outer state.
//
// `enabled` should already fold in the token-count threshold and the
// tp_world_size > 1 / backend checks decided by the caller.
class FlashComm1Guard final {
 public:
  FlashComm1Guard(bool enabled,
                  int64_t num_tokens,
                  int32_t tp_rank,
                  int32_t tp_world_size,
                  ProcessGroup* tp_group);
  ~FlashComm1Guard();

  FlashComm1Guard(const FlashComm1Guard&) = delete;
  FlashComm1Guard& operator=(const FlashComm1Guard&) = delete;
  FlashComm1Guard(FlashComm1Guard&&) = delete;
  FlashComm1Guard& operator=(FlashComm1Guard&&) = delete;

 private:
  FlashComm1Context previous_;
};

}  // namespace parallel_state
}  // namespace xllm
