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

#include "core/platform/cpu_binding.h"

namespace xllm::npu {

struct NpuIdentity {
  int32_t card_id = -1;
  int32_t chip_id = -1;
  int32_t logical_id = -1;
  int32_t physical_id = -1;
};

struct NpuCpuBindingInfo {
  CpuBindingPlan plan;
  NpuIdentity device;
};

std::vector<NpuIdentity> parse_npu_inventory(const std::string& text);
std::unordered_map<int32_t, std::vector<int32_t>> parse_npu_affinity(
    const std::string& text,
    const std::vector<NpuIdentity>& devices);
std::optional<int32_t> resolve_npu_logical_id(
    int32_t device_index,
    const std::string& visible_devices,
    const std::vector<NpuIdentity>& devices);
CpuBindingOptions npu_cpu_binding_options(bool global_slice);
std::optional<NpuCpuBindingInfo> get_npu_cpu_binding(
    int32_t device_index,
    const std::string& soc_name);
void bind_npu_irqs(const std::vector<int32_t>& cpus, const NpuIdentity& device);

}  // namespace xllm::npu
