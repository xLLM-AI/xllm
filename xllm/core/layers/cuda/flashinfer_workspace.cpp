/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "flashinfer_workspace.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/model_config.h"

namespace xllm::layer::flashinfer {

void FlashinferWorkspace::initialize(const torch::Device& device) {
  LOG(INFO) << "FlashinferWorkspace initialize on device: " << device;
  float_workspace_buffer_ = torch::empty(
      {::xllm::ModelConfig::get_instance().flashinfer_workspace_buffer_size()},
      torch::dtype(torch::kUInt8).device(device));
  int_workspace_buffer_ = torch::empty(
      {8 * 1024 * 1024}, torch::dtype(torch::kUInt8).device(device));
  page_locked_int_workspace_buffer_ = torch::empty(
      {int_workspace_buffer_.size(0)},
      torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
  LOG(INFO)
      << "FlashinferWorkspace initialize end with "
         "flashinfer_workspace_buffer_size: "
      << ::xllm::ModelConfig::get_instance().flashinfer_workspace_buffer_size();
}

torch::Tensor FlashinferWorkspace::get_float_workspace_buffer() {
  return float_workspace_buffer_;
}

torch::Tensor FlashinferWorkspace::get_int_workspace_buffer() {
  return int_workspace_buffer_;
}

torch::Tensor FlashinferWorkspace::get_page_locked_int_workspace_buffer() {
  return page_locked_int_workspace_buffer_;
}

}  // namespace xllm::layer::flashinfer