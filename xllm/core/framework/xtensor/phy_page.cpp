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

#include "phy_page.h"

namespace xllm {
PhyPage::PhyPage(torch::Device device, page_id_t page_id)
    : device_(device), page_id_(page_id) {
  int32_t device_id = device_.index();

  // create a physical memory handle for the device
  vmm::create_phy_mem_handle(phy_handle_, device_id);
}

PhyPage::~PhyPage() { vmm::release_phy_mem_handle(phy_handle_); }
}  // namespace xllm
