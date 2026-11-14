/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

class DiTMapping;
class ProcessGroup;

// Borrowed references. DiTCollectiveCommunicator owns the three groups.
struct DiTSpQkvDomains {
  ProcessGroup* q = nullptr;
  ProcessGroup* k = nullptr;
  ProcessGroup* v = nullptr;
};

struct DiTSpQkvCommunicationOptions {
  bool create_domains = false;
};

struct DiTSpQkvDomainSpec {
  int32_t local_rank = 0;
  int32_t port = 0;
  std::vector<int32_t> ranks;
  std::string name;
};

DiTSpQkvCommunicationOptions make_dit_sp_qkv_communication_options(
    const std::string& execution_profile,
    bool requested);

std::array<DiTSpQkvDomainSpec, 3> build_dit_sp_qkv_domain_specs(
    const DiTMapping& mapping,
    int32_t ordinary_groups_end_port);

// Host metadata only. Called during qualified model setup, before the first
// projection or collective; this function never creates a communicator.
void validate_dit_sp_qkv_domains(const DiTSpQkvDomains& domains,
                                 const ProcessGroup* sequence_group);

}  // namespace xllm
