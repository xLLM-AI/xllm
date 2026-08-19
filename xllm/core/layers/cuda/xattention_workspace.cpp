/* Copyright 2025-2026 The xLLM Authors.

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

#include "xattention_workspace.h"

#include <glog/logging.h>

namespace xllm::layer::xattention {

void XAttentionWorkspace::initialize(const torch::Device& device) {
  const bool is_already_initialized =
      int_workspace_buffer_.defined() &&
      page_locked_int_workspace_buffer_.defined() &&
      int_workspace_buffer_.device() == device;
  if (is_already_initialized) {
    return;
  }

  LOG(INFO) << "XAttentionWorkspace initialize on device: " << device;

  int_workspace_buffer_ = torch::empty(
      {8 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(device));
  page_locked_int_workspace_buffer_ = torch::empty(
      {int_workspace_buffer_.size(0)},
      torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
}

torch::Tensor XAttentionWorkspace::get_int_workspace_buffer() {
  return int_workspace_buffer_;
}

torch::Tensor XAttentionWorkspace::get_page_locked_int_workspace_buffer() {
  return page_locked_int_workspace_buffer_;
}

}  // namespace xllm::layer::xattention
