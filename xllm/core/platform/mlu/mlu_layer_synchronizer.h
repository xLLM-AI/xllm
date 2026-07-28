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

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "platform/stream.h"

namespace xllm {

class MLULayerSynchronizerImpl final {
 public:
  explicit MLULayerSynchronizerImpl(int64_t num_layers);
  ~MLULayerSynchronizerImpl() = default;

  bool synchronize_layer(int64_t layer_index);
  bool record_stream(int64_t layer_index, Stream* stream);
  bool record_current(int64_t layer_index, int32_t device_index);
  void abort();
  uint32_t get_event_size() const {
    return static_cast<uint32_t>(events_.size());
  }

 private:
  bool valid_index(int64_t layer_index) const;

  std::vector<StreamEventPtr> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
  std::atomic<bool> aborted_{false};
};

}  // namespace xllm
