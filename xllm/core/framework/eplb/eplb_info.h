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
#include <vector>

namespace xllm {

// EPLB per-step coordination payload flowing between EplbManager and the
// per-device EplbExecutor / worker forward path. Owns just the layer ids and
// the flat expert id list; deliberately POD so it can be trivially copied
// across the forward shared-memory ring buffer and the worker proto.
//
// Both layer id fields use `-1` as the sentinel for "no action this step".
// The field name comes from the caller's point of view:
//   prepare_layer_id : this step should START async loading the target layer's
//                      new expert weights (weights are staged, not yet live).
//   prepare_token    : unique token for this prepare attempt. Worker readiness
//                      reports echo this value so late completion from a timed
//                      out attempt cannot satisfy a newer attempt.
//   update_layer_id  : the previously prepared layer's staged weights are
//                      ready — activate them this step.
//   activation_token : identifies the activation attempt. The engine retains
//                      this token until the corresponding worker output is
//                      collected, so schedule overlap cannot acknowledge the
//                      command with an older step's load sample.
//   expert_ids       : flat, per-device shard, describing where each logical
//                      expert id lives after the pending plan is committed.
struct EplbInfo {
  // Target layer ID for new expert weight pre-loading (-1 = no pending load).
  int32_t prepare_layer_id = -1;
  int64_t prepare_token = -1;
  // Flat per-device shard of expert IDs describing the post-migration slot
  // map. Empty when prepare_layer_id == -1.
  std::vector<int32_t> expert_ids;
  // Layer ID whose pre-loaded weights are ready for activation this step
  // (-1 = no pending update).
  int32_t update_layer_id = -1;
  int64_t activation_token = -1;
};

}  // namespace xllm
