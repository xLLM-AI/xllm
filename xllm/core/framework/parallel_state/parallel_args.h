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

#include "core/common/macros.h"
#include "process_group.h"

#if defined(USE_NPU)
#include "hccl/hccl.h"
#include "xllm_atb_layers/models/base/param/mapping.h"
#endif

#include <nlohmann/json.hpp>
#include <string>

namespace xllm {

struct ParallelArgs {
  // Expose the parallel-dimension sizes (tp/dp/ep/cp/...) to the generic
  // property reflection layer so they can be forwarded to the embedded Python
  // model executor. Non-plain-data PROPERTY fields (mapping_data, vendor comm
  // handles) are skipped at compile time (see property_reflect.h).
  REFLECT_PROPERTIES(ParallelArgs);

  ParallelArgs(int32_t rank, int32_t world_size, ProcessGroup* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {}

  ParallelArgs(int32_t rank,
               int32_t world_size,
               int32_t dp_size,
               int32_t cp_size,
               ProcessGroup* process_group,
               int32_t ep_size)
      : rank_(rank),
        world_size_(world_size),
        dp_size_(dp_size),
        cp_size_(cp_size),
        process_group_(process_group),
        ep_size_(ep_size) {}

#if defined(USE_NPU)
  ParallelArgs(int32_t rank,
               int32_t world_size,
               int32_t dp_size,
               ProcessGroup* process_group,
               int32_t ep_size,
               int32_t cp_size,
               nlohmann::json mapping_data,
               atb_speed::base::Mapping mapping,
               std::string dispatchAndCombinecommDomain,
               HcclComm dispatchAndCombineHcclComm)
      : rank_(rank),
        world_size_(world_size),
        dp_size_(dp_size),
        process_group_(process_group),
        ep_size_(ep_size),
        cp_size_(cp_size),
        mapping_data_(mapping_data),
        mapping_(mapping),
        dispatchAndCombinecommDomain_(dispatchAndCombinecommDomain),
        dispatchAndCombineHcclComm_(dispatchAndCombineHcclComm) {}
#endif

  ParallelArgs(int32_t rank,
               int32_t world_size,
               int32_t dp_size,
               ProcessGroup* process_group)
      : rank_(rank),
        world_size_(world_size),
        dp_size_(dp_size),
        process_group_(process_group) {}

  ParallelArgs(int32_t rank,
               int32_t world_size,
               ProcessGroup* process_group,
               ProcessGroup* dp_local_process_group,
               int32_t dp_size)
      : rank_(rank),
        world_size_(world_size),
        process_group_(process_group),
        dp_local_process_group_(dp_local_process_group),
        dp_size_(dp_size) {}

  ParallelArgs(int32_t rank,
               int32_t world_size,
               int32_t dp_size,
               int32_t tp_size,
               int32_t sp_size,
               int32_t cfg_size,
               int32_t vae_size,
               ProcessGroup* process_group)
      : rank_(rank),
        world_size_(world_size),
        dp_size_(dp_size),
        tp_size_(tp_size),
        sp_size_(sp_size),
        cfg_size_(cfg_size),
        vae_size_(vae_size),
        process_group_(process_group) {}

  int32_t get_group_size_by_type(const std::string& group_type) const {
    if (group_type == "tp") {
      return tp_size();
    } else if (group_type == "sp") {
      return sp_size();
    } else if (group_type == "cfg") {
      return cfg_size();
    } else if (group_type == "dp") {
      return dp_size();
    } else if (group_type == "ep") {
      return ep_size();
    } else if (group_type == "vae") {
      return vae_size();
    } else if (group_type == "cp") {
      return cp_size();
    } else {
      LOG(FATAL) << "get unexpected group_type: " << group_type;
      return 1;
    }
    return 1;
  }

  // rank of current process
  PROPERTY(int32_t, rank) = 0;

  // world size
  PROPERTY(int32_t, world_size) = 0;

  // dp size
  PROPERTY(int32_t, dp_size) = 1;

  // ep size
  PROPERTY(int32_t, ep_size) = 1;

  // cp size
  PROPERTY(int32_t, cp_size) = 1;

  // Derived: CP rank of the current process within its DP group.
  // rank layout: dp_rank * (cp_size * tp_size) + cp_rank * tp_size + tp_rank
  [[nodiscard]] int32_t cp_rank() const noexcept {
    if (cp_size_ <= 1) {
      return 0;
    }
    int32_t tp_sz = world_size_ / dp_size_ / cp_size_;
    return (rank_ % (cp_size_ * tp_sz)) / tp_sz;
  }

  // KV-cache split width. 0 == "follow cp_size" (legacy). Use
  // `kv_split_size_effective()` instead of reading the raw value when computing
  // strides / block sizes; the raw setter is kept so the engine can override
  // the per-instance value (e.g. PD link negotiation) without touching gflags.
  PROPERTY(int32_t, kv_split_size) = 0;

  // Effective KV split width: kv_split_size_ if explicitly set, otherwise
  // cp_size_. When this equals 1 with cp_size_ > 1, each CP rank holds a
  // complete KV replica and the ATB prefix AllGather can be skipped.
  [[nodiscard]] int32_t kv_split_size_effective() const noexcept {
    return kv_split_size_ > 0 ? kv_split_size_ : cp_size_;
  }

  // KV-split rank: global rank block index over world_size / kv_split_size.
  // Aligns with MappingNPU::get_kv_split_group (get_dp_group stride) and ATB
  // kvSplitInfo.rankIds ordering used by the prefix AllGather.
  [[nodiscard]] int32_t kv_split_rank() const noexcept {
    const int32_t kv = kv_split_size_effective();
    if (kv <= 1) {
      return 0;
    }
    return rank_ / (world_size_ / kv);
  }

  // tp size
  PROPERTY(int32_t, tp_size) = 1;

  // sp size
  PROPERTY(int32_t, sp_size) = 1;

  // cfg size
  PROPERTY(int32_t, cfg_size) = 1;

  // cfg size
  PROPERTY(int32_t, vae_size) = 1;

  // atb hccl mapping json data
  PROPERTY(nlohmann::json, mapping_data);

#if defined(USE_NPU)
  // atb hccl mapping
  PROPERTY(atb_speed::base::Mapping, mapping);

  // atb hccl dispatchAndCombinecommDomain
  PROPERTY(std::string, dispatchAndCombinecommDomain);

  // atb hccl dispatchAndCombineHcclComm
  PROPERTY(HcclComm, dispatchAndCombineHcclComm);
#endif

  // the following pointers are unique pointers from CollectiveCommunicator
  //  So they are not owned by ParallelArgs.
  ProcessGroup* process_group_ = nullptr;
  ProcessGroup* dp_local_process_group_ = nullptr;
  ProcessGroup* tp_group_ = nullptr;
  // Optional output-projection group for models whose body and LMHead use
  // different tensor-parallel strategies.
  ProcessGroup* lm_head_group_ = nullptr;
  ProcessGroup* encoder_dp_group_ = nullptr;
  ProcessGroup* single_rank_group_ = nullptr;
  // Context-parallel communication group used by prefill attention.
  // The current MLU model-side CP path requires CP to span the full DP-local
  // rank set, so this temporarily aliases the TP group. Keep a distinct handle
  // for a future orthogonal CP x TP topology with a standalone CP group.
  ProcessGroup* cp_group_ = nullptr;
  ProcessGroup* moe_ep_group_ = nullptr;
  ProcessGroup* moe_tp_group_ = nullptr;

  // PyTorch creates its own TP process group. These fields only reserve the
  // TCPStore endpoint after the native process-group port range.
  std::string python_tp_rendezvous_host_;
  int32_t python_tp_rendezvous_port_ = 0;

  // ProcessGroups for DiT models
  ProcessGroup* dit_tp_group_ = nullptr;
  ProcessGroup* dit_sp_group_ = nullptr;
  ProcessGroup* dit_cfg_group_ = nullptr;
  ProcessGroup* dit_dp_group_ = nullptr;
  ProcessGroup* dit_vae_group_ = nullptr;
};

}  // namespace xllm
