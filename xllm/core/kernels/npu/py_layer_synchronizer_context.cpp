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

#include "py_layer_synchronizer_context.h"

#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "platform/npu/npu_layer_synchronizer.h"

namespace xllm {

static thread_local std::shared_ptr<NPULayerSynchronizerImpl>
    tls_layer_synchronizer;

void set_current_layer_synchronizer(
    std::shared_ptr<NPULayerSynchronizerImpl> sync) {
  tls_layer_synchronizer = std::move(sync);
}

std::shared_ptr<NPULayerSynchronizerImpl> get_current_layer_synchronizer() {
  return tls_layer_synchronizer;
}

void record_current_layer_event(int64_t layer_id) {
  auto& sync = tls_layer_synchronizer;
  if (sync) {
    int32_t device_id =
        static_cast<int32_t>(c10_npu::getCurrentNPUStream().device_index());
    sync->record_event(layer_id, device_id);
  }
}

}  // namespace xllm
