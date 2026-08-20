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

#pragma once

#include <torch/torch.h>

#include "platform/vmm_api.h"

namespace xllm {

// Page ID type for physical page identification
using page_id_t = int64_t;

class PhyPage {
 public:
  // Constructor with page_id (-1 means unassigned)
  PhyPage(torch::Device device, page_id_t page_id = -1);

  ~PhyPage();

  const torch::Device& device() const { return device_; }

  PhyMemHandle get_phy_handle() const { return phy_handle_; }

  // Get the page ID
  page_id_t page_id() const { return page_id_; }

 private:
  torch::Device device_;
  PhyMemHandle phy_handle_;
  page_id_t page_id_;  // Unique identifier for this page in the pool
};
}  // namespace xllm
