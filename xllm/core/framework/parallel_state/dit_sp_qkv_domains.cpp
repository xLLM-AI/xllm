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

#include "core/framework/parallel_state/dit_sp_qkv_domains.h"

#include <algorithm>
#include <stdexcept>

#include "core/framework/parallel_state/dit_mapping.h"
#include "core/framework/parallel_state/process_group.h"

namespace xllm {

DiTSpQkvCommunicationOptions make_dit_sp_qkv_communication_options(
    const std::string& execution_profile,
    bool requested) {
  DiTSpQkvCommunicationOptions options;
  if (!requested) {
    return options;
  }
  if (execution_profile != "a3_16card_tp2_u8_ref2va_v1") {
    throw std::invalid_argument(
        "DiT QKV domains require the qualified H3 profile");
  }
  options.create_domains = true;
  return options;
}

std::array<DiTSpQkvDomainSpec, 3> build_dit_sp_qkv_domain_specs(
    const DiTMapping& mapping,
    int32_t ordinary_groups_end_port) {
  const ParallelInfo& sp = mapping.get_parallel_info("sp");
  if (sp.group_size() != 8 || sp.num_group() != 2 ||
      mapping.get_parallel_info("tp").group_size() != 2 ||
      mapping.get_parallel_info("dp").group_size() != 1 ||
      mapping.get_parallel_info("cfg").group_size() != 1 ||
      sp.current_group_id() < 0 || sp.current_group_id() >= sp.num_group() ||
      sp.rank() < 0 || sp.rank() >= sp.group_size() ||
      static_cast<int32_t>(sp.rank_per_group().size()) != sp.num_group()) {
    throw std::invalid_argument(
        "DiT QKV domains require the TP2/SP8 world16 mapping");
  }
  const int64_t last_port =
      static_cast<int64_t>(ordinary_groups_end_port) + 64 + 3 * sp.num_group();
  if (ordinary_groups_end_port <= 0 || last_port > 65535) {
    throw std::invalid_argument("DiT QKV domain port range is invalid");
  }
  const std::vector<int32_t>& ranks =
      sp.rank_per_group()[sp.current_group_id()];
  if (ranks.size() != 8) {
    throw std::invalid_argument(
        "DiT QKV domain membership must contain eight ranks");
  }
  const std::array<std::string, 3> names = {
      "sp_q_domain", "sp_k_domain", "sp_v_domain"};
  std::array<DiTSpQkvDomainSpec, 3> specs;
  for (int32_t index = 0; index < 3; ++index) {
    specs[index].local_rank = sp.rank();
    specs[index].port = ordinary_groups_end_port + 64 + index * sp.num_group() +
                        sp.current_group_id() + 1;
    specs[index].ranks = ranks;
    specs[index].name = names[index];
  }
  return specs;
}

void validate_dit_sp_qkv_domains(const DiTSpQkvDomains& domains,
                                 const ProcessGroup* sequence_group) {
  if (sequence_group == nullptr || domains.q == nullptr ||
      domains.k == nullptr || domains.v == nullptr || domains.q == domains.k ||
      domains.q == domains.v || domains.k == domains.v ||
      domains.q == sequence_group || domains.k == sequence_group ||
      domains.v == sequence_group) {
    throw std::invalid_argument(
        "DiT QKV overlap requires three distinct domains");
  }
  for (const ProcessGroup* group : {domains.q, domains.k, domains.v}) {
    if (group->world_size() != 8 ||
        group->world_size() != sequence_group->world_size() ||
        group->rank() != sequence_group->rank() ||
        group->device() != sequence_group->device()) {
      throw std::invalid_argument(
          "DiT QKV domain rank, size or device mismatch");
    }
  }
}

}  // namespace xllm
