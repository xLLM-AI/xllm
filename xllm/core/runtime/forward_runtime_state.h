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

#pragma once

#include <torch/types.h>

#include <vector>

#include "core/platform/stream_event.h"

namespace xllm {

// Buffer ownership and execution lifetime state shared by forward domains.
struct ForwardRuntimeState {
  // Own packed host staging storage and its corresponding device storage.
  // Model input tensors may reference regions within these buffers.
  torch::Tensor input_host_buffer;
  torch::Tensor device_input_buffer;
  bool input_host_buffer_has_layout = false;

  // True when all model input tensors already reference execution-device
  // views. Runtime preparation can reuse them without rebuilding or copying.
  bool device_tensors_ready = false;

  // Device-side readiness dependencies for inputs prepared on a different
  // stream. These are local runtime handles and are intentionally not included
  // in proto or shared-memory transport.
  StreamEventPtr metadata_ready_event;

  // Keep cross-stream metadata sources alive through no-sync execution. These
  // handles are local runtime state and are not serialized.
  std::vector<torch::Tensor> retained_device_tensors;
};

}  // namespace xllm
