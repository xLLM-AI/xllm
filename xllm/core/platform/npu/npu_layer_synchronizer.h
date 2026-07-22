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

#include <acl/acl.h>

#include <atomic>
#include <vector>

namespace xllm {

class NPULayerSynchronizerImpl {
 public:
  NPULayerSynchronizerImpl(const int64_t num_layers,
                           const int32_t timeout = -1);
  virtual ~NPULayerSynchronizerImpl();

  aclrtEvent* get_event(const int64_t layer_index);
  std::atomic<bool>* get_event_flag(const int64_t layer_index);
  bool synchronize_layer(const int64_t layer_index);
  bool record_event(const int64_t layer_index, const int32_t device_index);
  // Unblock all pending synchronize_layer spins and make them report failure.
  void abort();
  uint32_t get_event_size() { return events_.size(); };

 private:
  std::vector<aclrtEvent> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
  std::atomic<bool> aborted_{false};
  const int32_t timeout_;
};

}  // namespace xllm
